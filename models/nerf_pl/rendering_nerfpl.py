import torch
from models.rendering import sample_pdf

__all__ = ["render_rays"]  # only render_rays is called outside

"""
Function dependencies: (-> means function calls)

@render_rays -> @inference -> @batched_inference

@render_rays -> @sample_pdf if there is fine model
"""


def batched_inference(model, inputs, chunk=1024 * 32):
    """
    Perform model inference by cutting input to smaller chunks due to memory issue.

    Inputs:
        model: model to run inference
        inputs: (B, ?) inputs to run inference
        chunk: the chunk size

    Outputs:
        out: (B, ?) having the same content as model(inputs)
    """
    B = inputs.shape[0]
    out_chunks = [model(inputs[i : i + chunk]) for i in range(0, B, chunk)]
    out = torch.cat(out_chunks, 0)

    return out


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
):
    """
    Render rays by computing the output of @model applied on @rays

    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins, directions in NDC and near, far depth bounds
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in @batchify
        white_back: whether the background is white (dataset dependent)

    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """

    def inference(model, embedding_xyz, embedding_dir, xyz_, dir_, z_vals):
        """
        Helper function that performs model inference.

        Inputs:
            model: NeRF model (coarse or fine)
            embedding_xyz: embedding module for xyz
            embedding_dir: embedding module for dir
            xyz_: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            dir_: (N_rays, 3) directions (not normalized)
            z_vals: (N_rays, N_samples_) depths of the sampled positions

        Outputs:
            rgb_final: (N_rays, 3) the final rgb image
            depth_final: (N_rays) depth map
            weights: (N_rays, N_samples_): weights fo each sample

        """
        N_samples_ = xyz_.shape[1]
        # Embed positions and directions
        xyz_ = xyz_.view(-1, 3)  # (N_rays*N_samples_, 3)
        xyz_embedded = embedding_xyz(xyz_)  # (N_rays*N_samples_, embed_xyz_channels)
        dir_normalized = dir_ / torch.norm(dir_, dim=1, keepdim=True)  # (N_rays, 3)
        dir_embedded = embedding_dir(dir_normalized)  # (N_rays, embed_dir_channels)
        dir_embedded = torch.repeat_interleave(dir_embedded, repeats=N_samples_, dim=0)
        # (N_rays*N_samples_, embed_dir_channels)

        xyzdir_embedded = torch.cat([xyz_embedded, dir_embedded], 1)

        # Perform model inference to get raw rgb sigma
        rgbsigma = batched_inference(model, xyzdir_embedded, chunk)
        rgbsigma = rgbsigma.view(N_rays, N_samples_, 4)

        # Convert these values using volume rendering (Section 4)
        rgbs = rgbsigma[..., :3]  # (N_rays, N_samples_, 3)
        sigmas = rgbsigma[..., 3]  # (N_rays, N_samples_)
        deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(
            deltas[:, :1]
        )  # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)

        noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std

        # compute alpha by the formula (3)
        alphas = 1 - torch.exp(
            -deltas * torch.relu(sigmas + noise)
        )  # (N_rays, N_samples_)
        alphas_shifted = torch.cat(
            [torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1
        )  # [1, a1, a2, ...]
        weights = (
            alphas * torch.cumprod(alphas_shifted, -1)[:, :-1]
        )  # (N_rays, N_samples_)
        weights_sum = weights.sum(1)  # (N_rays), the accumulated opacity along the rays

        # compute final weighted outputs
        rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)  # (N_rays, 3)
        depth_final = torch.sum(weights * z_vals, -1)  # (N_rays)

        if white_back:
            rgb_final = rgb_final + 1 - weights_sum.unsqueeze(-1)

        return rgb_final, depth_final, weights

    # Extract models from lists
    model_coarse = models[0]
    embedding_xyz = embeddings[0]
    embedding_dir = embeddings[1]

    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, 1)

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

        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays.device)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse_sampled = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(
        2
    )  # (N_rays, N_samples, 3)

    rgb_coarse, depth_coarse, weights_coarse = inference(
        model_coarse, embedding_xyz, embedding_dir, xyz_coarse_sampled, rays_d, z_vals
    )

    result = {
        "rgb_coarse": rgb_coarse,
        "depth_coarse": depth_coarse,
        "opacity_coarse": weights_coarse.sum(1),
    }

    if N_importance > 0:  # sample points for fine model
        z_vals_mid = 0.5 * (
            z_vals[:, :-1] + z_vals[:, 1:]
        )  # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(
            z_vals_mid, weights_coarse[:, 1:-1], N_importance, det=(perturb == 0)
        ).detach()
        # detach so that grad doesn't propogate to weights_coarse from here

        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

        xyz_fine_sampled = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(
            2
        )
        # (N_rays, N_samples+N_importance, 3)

        model_fine = models[1]
        rgb_fine, depth_fine, weights_fine = inference(
            model_fine, embedding_xyz, embedding_dir, xyz_fine_sampled, rays_d, z_vals
        )

        # print('rgb', rgb_fine[:5])

        result["rgb_fine"] = rgb_fine
        result["depth_fine"] = depth_fine
        result["opacity_fine"] = weights_fine.sum(1)

    return result
