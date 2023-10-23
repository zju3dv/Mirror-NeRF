import torch
from einops import rearrange, reduce, repeat

__all__ = ["render_rays"]


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: (N_rays, N_importance) the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps  # prevent division by zero (don't do inplace op!)
    pdf = weights / reduce(weights, "n1 n2 -> n1 1", "sum")  # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1)  # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (N_rays, N_samples_+1)
    # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = rearrange(
        torch.stack([below, above], -1), "n1 n2 c -> n1 (n2 c)", c=2
    )
    cdf_g = rearrange(torch.gather(cdf, 1, inds_sampled), "n1 (n2 c) -> n1 n2 c", c=2)
    bins_g = rearrange(torch.gather(bins, 1, inds_sampled), "n1 (n2 c) -> n1 n2 c", c=2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom < eps] = 1  # denom equals 0 means a bin has weight 0,
    # in which case it will not be sampled
    # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (
        bins_g[..., 1] - bins_g[..., 0]
    )
    return samples


def render_rays(
    models,
    embeddings,
    rays,
    N_samples=64,
    use_disp=False,
    perturb=0,
    noise_std=1,
    N_importance=0,
    chunk=1024 * 32,
    white_back=False,
    test_time=False,
    **kwargs,
):
    """
    Render rays by computing the output of @model applied on @rays
    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins and directions, near and far depths
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time
    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """

    def inference(results, model, typ, xyz, z_vals, test_time=False, **kwargs):
        """
        Helper function that performs model inference.
        Inputs:
            results: a dict storing all results
            model: NeRF model (coarse or fine)
            typ: 'coarse' or 'fine'
            xyz: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            test_time: test time or not
        Outputs:
            if weights_only:
                weights: (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
        """
        N_samples_ = xyz.shape[1]
        xyz_ = rearrange(xyz, "n1 n2 c -> (n1 n2) c")  # (N_rays*N_samples_, 3)
        compute_normal = kwargs.get("compute_normal", True)
        mirror_mask = kwargs.get("mirror_mask", None)
        mirror_mask_ = (
            None
            if mirror_mask is None
            else (mirror_mask.unsqueeze(-1).repeat(1, N_samples_).view(-1))
        )  # (N_rays*N_samples_)
        detach_density_outside_mirror_for_mask_loss = kwargs.get(
            "detach_density_outside_mirror_for_mask_loss", False
        )
        detach_density_for_mask_loss = kwargs.get("detach_density_for_mask_loss", False)
        detach_density_for_normal_loss = kwargs.get(
            "detach_density_for_normal_loss", False
        )

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        rgb_chunks = []
        sigma_chunks = []
        normal_chunks = []
        pred_normal_chunks = []
        is_mirror_chunks = []
        dir_embedded_ = repeat(dir_embedded, "n1 c -> (n1 n2) c", n2=N_samples_)
        # (N_rays*N_samples_, embed_dir_channels)
        for i in range(0, B, chunk):
            xyz_chunk = xyz_[i : i + chunk]
            mirror_mask_chunk = (
                None if mirror_mask_ is None else mirror_mask_[i : i + chunk]
            )
            if typ == "coarse" and test_time and "fine" in models:
                output = model(
                    xyz_chunk,
                    compute_normal=compute_normal,
                    sigma_only=True,
                    embedding_xyz=embedding_xyz,
                    embedding_dir=embedding_dir,
                    mirror_mask=mirror_mask_chunk,
                    detach_density_outside_mirror_for_mask_loss=detach_density_outside_mirror_for_mask_loss,
                    detach_density_for_mask_loss=detach_density_for_mask_loss,
                    detach_density_for_normal_loss=detach_density_for_normal_loss,
                )
            else:  # infer rgb and sigma and others
                xyzdir_embedded = torch.cat(
                    [xyz_chunk, dir_embedded_[i : i + chunk]], 1
                )
                output = model(
                    xyzdir_embedded,
                    compute_normal=compute_normal,
                    sigma_only=False,
                    embedding_xyz=embedding_xyz,
                    embedding_dir=embedding_dir,
                    mirror_mask=mirror_mask_chunk,
                    detach_density_outside_mirror_for_mask_loss=detach_density_outside_mirror_for_mask_loss,
                    detach_density_for_mask_loss=detach_density_for_mask_loss,
                    detach_density_for_normal_loss=detach_density_for_normal_loss,
                )
            sigma_chunks += [output["sigma"]]
            if "rgb" in output:
                rgb_chunks += [output["rgb"]]
            if "normal" in output:
                normal_chunks += [output["normal"]]
            if "pred_normal" in output:
                pred_normal_chunks += [output["pred_normal"]]
            if "is_mirror" in output:
                is_mirror_chunks += [output["is_mirror"]]
        sigmas = torch.cat(sigma_chunks, 0).view(N_rays, N_samples_)
        if len(rgb_chunks) > 0:
            rgbs = torch.cat(rgb_chunks, 0).view(N_rays, N_samples_, 3)
        if len(is_mirror_chunks) > 0:
            is_mirrors = torch.cat(is_mirror_chunks, 0).view(N_rays, N_samples_)

        # Convert these values using volume rendering (Section 4)
        deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(
            deltas[:, :1]
        )  # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # compute alpha by the formula (3)
        noise = torch.randn_like(sigmas) * noise_std
        alphas = 1 - torch.exp(
            -deltas * torch.relu(sigmas + noise)
        )  # (N_rays, N_samples_)

        alphas_shifted = torch.cat(
            [torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1
        )  # [1, 1-a1, 1-a2, ...]
        weights = alphas * torch.cumprod(
            alphas_shifted[:, :-1], -1
        )  # (N_rays, N_samples_)
        weights_sum = reduce(
            weights, "n1 n2 -> n1", "sum"
        )  # (N_rays), the accumulated opacity along the rays
        # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

        results[f"weights_{typ}"] = weights
        results[f"opacity_{typ}"] = weights_sum
        results[f"z_vals_{typ}"] = z_vals
        if test_time and typ == "coarse" and "fine" in models:
            return

        rgb_map = reduce(
            rearrange(weights, "n1 n2 -> n1 n2 1") * rgbs, "n1 n2 c -> n1 c", "sum"
        )
        depth_map = reduce(weights * z_vals, "n1 n2 -> n1", "sum")

        if white_back:
            rgb_map += 1 - weights_sum.unsqueeze(1)

        results[f"rgb_{typ}"] = rgb_map
        results[f"depth_{typ}"] = depth_map

        if len(is_mirror_chunks) > 0:
            if detach_density_for_mask_loss:
                mirror_mask_pred = reduce(
                    weights.detach() * is_mirrors, "n1 n2 -> n1", "sum"
                )
            elif (
                detach_density_outside_mirror_for_mask_loss
                and mirror_mask is not None
                and (mirror_mask < 0).any() == False
            ):
                mirror_mask = mirror_mask.clone().bool()
                weights_for_mirror = weights.clone()
                weights_for_mirror[~mirror_mask] = weights_for_mirror[
                    ~mirror_mask
                ].detach()
                mirror_mask_pred = reduce(
                    weights_for_mirror * is_mirrors, "n1 n2 -> n1", "sum"
                )
            else:
                mirror_mask_pred = reduce(weights * is_mirrors, "n1 n2 -> n1", "sum")
            results[f"mirror_mask_{typ}"] = mirror_mask_pred

        # normals
        weights_for_normal = (
            weights.detach() if detach_density_for_normal_loss else weights
        )
        if len(normal_chunks) > 0:
            normals = torch.cat(normal_chunks, 0).view(N_rays, N_samples_, 3)
            results[f"normal_{typ}"] = normals
            results[f"surface_normal_grad_{typ}"] = (
                normals * weights_for_normal.unsqueeze(-1)
            ).sum(1, keepdim=False)
        if len(pred_normal_chunks) > 0:
            pred_normals = torch.cat(pred_normal_chunks, 0).view(N_rays, N_samples_, 3)
            results[f"pred_normal_{typ}"] = pred_normals
            results[f"surface_normal_{typ}"] = (
                pred_normals * weights_for_normal.unsqueeze(-1)
            ).sum(1, keepdim=False)
        if len(normal_chunks) > 0 and len(pred_normal_chunks) > 0:
            normal_dif = torch.sum((normals - pred_normals) ** 2, dim=-1)
            results[f"normal_dif_{typ}"] = reduce(
                weights_for_normal * normal_dif, "n1 n2 -> n1", "sum"
            )

        return

    embedding_xyz, embedding_dir = embeddings["xyz"], embeddings["dir"]

    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, 1)
    # Embed direction
    dir_embedded = embedding_dir(
        kwargs.get("view_dir", rays_d)
    )  # (N_rays, embed_dir_channels)

    rays_o = rearrange(rays_o, "n1 c -> n1 1 c")
    rays_d = rearrange(rays_d, "n1 c -> n1 1 c")

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)  # (N_samples)
    if not use_disp:  # use linear sampling in depth space
        z_vals = near * (1 - z_steps) + far * z_steps
    else:  # use linear sampling in disparity space
        z_vals = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)

    if perturb > 0:  # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (
            z_vals[:, :-1] + z_vals[:, 1:]
        )  # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
        lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

        perturb_rand = perturb * torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse = rays_o + rays_d * rearrange(z_vals, "n1 n2 -> n1 n2 1")

    results = {}
    inference(
        results, models["coarse"], "coarse", xyz_coarse, z_vals, test_time, **kwargs
    )

    only_one_field = kwargs.get("only_one_field", False)
    if N_importance > 0:  # sample points of fine process

        def sample_fine_points(z_vals, weights, N_importance, perturb):
            z_vals_mid = 0.5 * (
                z_vals[:, :-1] + z_vals[:, 1:]
            )  # (N_rays, N_samples-1) interval mid points
            z_vals_ = sample_pdf(
                z_vals_mid,
                weights,
                N_importance,
                det=(perturb == 0),
            )
            # detach so that grad doesn't propogate to weights_coarse from here

            z_vals = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)[0]
            # combine coarse and fine samples
            return z_vals

        if only_one_field:
            if kwargs.get("current_epoch", 0) > kwargs.get(
                "only_one_field_fine_epoch", 2
            ):
                # inference coarse model use fine sample points.
                z_vals = sample_fine_points(
                    z_vals,
                    results["weights_coarse"][:, 1:-1].detach(),
                    N_importance,
                    perturb,
                )
                xyz_fine = rays_o + rays_d * rearrange(z_vals, "n1 n2 -> n1 n2 1")
                inference(
                    results,
                    models["coarse"],
                    "coarse",
                    xyz_fine,
                    z_vals,
                    test_time,
                    **kwargs,
                )
        else:
            # inference fine model use fine sample points.
            z_vals = sample_fine_points(
                z_vals,
                results["weights_coarse"][:, 1:-1].detach(),
                N_importance,
                perturb,
            )
            xyz_fine = rays_o + rays_d * rearrange(z_vals, "n1 n2 -> n1 n2 1")
            inference(
                results, models["fine"], "fine", xyz_fine, z_vals, test_time, **kwargs
            )

    # compute surface points
    for typ in ["coarse", "fine"]:
        if f"depth_{typ}" in results:
            results[f"x_surface_{typ}"] = rays_o.squeeze() + rays_d.squeeze() * results[
                f"depth_{typ}"
            ].unsqueeze(-1)

    return results
