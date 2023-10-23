import os
import cv2
from copy import deepcopy
import time

from collections import defaultdict
from tqdm import tqdm
import imageio

from models.rendering import render_rays
from models.mirror_nerf import *

from utils import load_ckpt
from utils.visualization import visualize_depth, visualize_rgb_map_global
from utils.func import l2_normalize
import metrics

from datasets import dataset_dict
from datasets.depth_utils import *

from train import NeRFSystem

import opt

torch.backends.cudnn.benchmark = True

all_depths_max = None
all_depths_min = None
all_depths_reflect_max = None
all_depths_reflect_min = None


def get_opt():
    parser = opt.get_opts(False)

    parser.add_argument("--split", type=str, default="test", help="test or test_train")
    parser.add_argument("--only_eval_idx", type=int, default=-1)
    parser.add_argument(
        "--not_save_depth",
        default=False,
        action="store_true",
        help="whether to save depth prediction",
    )
    parser.add_argument(
        "--depth_format",
        type=str,
        nargs="+",
        default=["png"],
        # choices=["png", "pfm", "bytes"],
        help="which format to save",
    )
    parser.add_argument(
        "--render_coarse_rgb",
        default=False,
        action="store_true",
    )

    # Applications
    parser.add_argument(
        "--app_control_mirror_roughness",
        default=False,
        action="store_true",
    )
    parser.add_argument("--trace_ray_times", type=int, default=4)
    parser.add_argument("--normal_noise_std", type=float, default=0.01)
    parser.add_argument(
        "--normal_noise_std_changes",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--app_reflection_substitution",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--substitution_ckpt_path",
        type=str,
        default=None,
        help="radiance field to be substituted in the mirror.",
    )

    parser.add_argument(
        "--app_place_new_mirror",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--plane_pos",
        type=str,
        default="plane_x",
        choices=["plane_x", "plane_y"],
    )

    parser.add_argument(
        "--app_reflect_newly_placed_objects",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--obj_ckpt_path",
        type=str,
        default=None,
        help="radiance field of object to be added into the scene.",
    )
    parser.add_argument(
        "--obj_model_type", type=str, default="d_nerf", choices=["nerf_pl", "d_nerf"]
    )

    return parser.parse_args()


@torch.no_grad()
def batched_inference(
    models, embeddings, rays, N_samples, N_importance, use_disp, chunk, **kwargs
):
    """Do batched inference on rays using chunk."""
    args = kwargs.get("args", None)
    system_substitution = kwargs.get("system_substitution", None)
    system_obj = kwargs.get("system_obj", None)
    render_kwargs_test_d_nerf = kwargs.get("render_kwargs_test_d_nerf", None)
    args_d_nerf = kwargs.get("args_d_nerf", None)
    frame_time = kwargs.get("frame_time", None)
    normal_noise_std = kwargs.get("normal_noise_std", 0)

    B = rays.shape[0]
    results = defaultdict(list)
    idx_cnt = 0
    for i in range(0, B, chunk):

        def render_rays_chunk_recursively(
            rays_chunk, mirror_mask_prev, idx_cnt, recur_level=0
        ):
            rendered_ray_chunks = render_rays(
                models,
                embeddings,
                rays_chunk,
                N_samples,
                use_disp,
                0,
                0,
                N_importance,
                chunk,
                dataset.white_back,
                test_time=kwargs.get("test_time", True),
                compute_normal=kwargs.get("trace_secondary_rays", False)
                and (not args.predict_normal),
                only_one_field=args.only_one_field,
                only_one_field_fine_epoch=args.only_one_field_fine_epoch,
                current_epoch=args.only_one_field_fine_epoch
                + 1,  # to guarantee inference coarse model use fine sample points when training only one field.
            )

            select_type = (
                "fine" if (N_importance > 0 and not args.only_one_field) else "coarse"
            )

            only_trace_rays_in_mirrors = not (recur_level < 1)

            # For visualization
            rendered_ray_chunks[f"rgb_{select_type}_reflect"] = torch.zeros_like(
                rendered_ray_chunks[f"rgb_{select_type}"]
            )
            rendered_ray_chunks[f"depth_{select_type}_reflect"] = torch.zeros_like(
                rendered_ray_chunks[f"depth_{select_type}"]
            ).to(rendered_ray_chunks[f"depth_{select_type}"].device)

            mask_valid_depth = (
                rendered_ray_chunks[f"depth_{select_type}"] > args.near
            )  # for scenes using white_back

            if args.app_reflect_newly_placed_objects:
                # transform rays for object radiance field
                rays_chunk_for_obj = rays_chunk.clone()
                # config
                pose_align = None
                translation = [0, 0, 0]
                scale = 1
                if "livingroom" in args.root_dir:
                    # translation = [0, 0, 0.5]
                    # scale = 1
                    translation = [0, 0, 0]
                    scale = 2
                elif "washroom" in args.root_dir:
                    translation = [-0.5, -0.5, 0]
                    scale = 2
                elif "office" in args.root_dir:
                    translation = [0, 3, 0.5]
                    scale = 2

                if pose_align is not None:
                    pose_ = torch.FloatTensor(pose_align).to(rays_chunk_for_obj.device)
                    pose_scale = pose_[:3, 0].clone()
                    for i in range(3):
                        pose_scale[i] = torch.norm(pose_[:3, i])
                    pose_rotation = pose_[:3, :3].clone()
                    for i in range(3):
                        pose_rotation[:3, i] /= pose_scale[i]

                    rays_chunk_for_obj[:, :3] = (
                        rays_chunk_for_obj[:, :3] @ pose_[:3, :3].T
                    )
                    rays_chunk_for_obj[:, :3] = (
                        rays_chunk_for_obj[:, :3] + pose_[:3, 3][None, :]
                    )
                    rays_chunk_for_obj[:, 3:6] = l2_normalize(
                        rays_chunk_for_obj[:, 3:6] @ pose_[:3, :3].T
                    )
                rays_chunk_for_obj[:, :3] = rays_chunk_for_obj[
                    :, :3
                ] * torch.FloatTensor([scale, scale, scale]).to(
                    rays_chunk_for_obj.device
                )
                rays_chunk_for_obj[:, :3] = rays_chunk_for_obj[
                    :, :3
                ] + torch.FloatTensor(translation).to(rays_chunk_for_obj.device)

                # rendering object
                if args.obj_model_type == "nerf_pl":
                    rendered_ray_chunks_obj = render_rays_obj(
                        system_obj.models,
                        system_obj.embeddings,
                        rays_chunk_for_obj,
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        dataset.white_back,
                    )
                elif args.obj_model_type == "d_nerf":
                    frame_time_ = frame_time * torch.ones_like(
                        rays_chunk_for_obj[..., :1]
                    )
                    rays_chunk_for_obj = torch.cat(
                        [rays_chunk_for_obj, frame_time_], -1
                    )
                    if args_d_nerf.use_viewdirs:
                        # provide ray directions as input
                        viewdirs = rays_chunk_for_obj[..., 3:6]
                        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
                        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
                        rays_chunk_for_obj = torch.cat(
                            [rays_chunk_for_obj, viewdirs], -1
                        )
                    rendered_ray_chunks_obj = render_rays_obj(
                        rays_chunk_for_obj, **render_kwargs_test_d_nerf
                    )
                    rendered_ray_chunks_obj[
                        f"depth_{select_type}"
                    ] = rendered_ray_chunks_obj["depth_map"]
                    rendered_ray_chunks_obj[
                        f"rgb_{select_type}"
                    ] = rendered_ray_chunks_obj["rgb_map"]
                    rendered_ray_chunks_obj[
                        f"opacity_{select_type}"
                    ] = rendered_ray_chunks_obj["acc_map"]

                rendered_ray_chunks_obj[f"depth_{select_type}"] = (
                    rendered_ray_chunks_obj[f"depth_{select_type}"]
                    / scale
                    / pose_scale[0]
                )

                # remove white bg
                mask_obj_depth = rendered_ray_chunks_obj[f"depth_{select_type}"] > 0
                mask_obj_opacity = (
                    rendered_ray_chunks_obj[f"opacity_{select_type}"] > 0.8
                )
                mask_obj = torch.logical_and(mask_obj_depth, mask_obj_opacity)

                # Deal with occlusion between newly placed objects and the original scene based on depth
                mask_obj_blocked_by_fg = (
                    rendered_ray_chunks_obj[f"depth_{select_type}"]
                    > rendered_ray_chunks[f"depth_{select_type}"]
                )
                mask_obj_blocked_by_fg = torch.logical_and(
                    mask_obj_blocked_by_fg, mask_valid_depth
                )
                mask_use_obj = torch.logical_and(
                    torch.logical_not(mask_obj_blocked_by_fg), mask_obj
                )
                rendered_ray_chunks[f"rgb_{select_type}"][
                    mask_use_obj
                ] = rendered_ray_chunks_obj[f"rgb_{select_type}"][mask_use_obj]
                rendered_ray_chunks[f"depth_{select_type}"][
                    mask_use_obj
                ] = rendered_ray_chunks_obj[f"depth_{select_type}"][mask_use_obj]
                rendered_ray_chunks["mirror_mask_fine"][mask_use_obj] = False

            #############################
            # Config mirror_mask
            if f"mirror_mask_{select_type}" in rendered_ray_chunks:
                mirror_mask = rendered_ray_chunks[f"mirror_mask_{select_type}"]
            elif "mirror_mask_fine" in rendered_ray_chunks:
                mirror_mask = rendered_ray_chunks["mirror_mask_fine"]
            elif "mirror_mask_coarse" in rendered_ray_chunks:
                mirror_mask = rendered_ray_chunks["mirror_mask_coarse"]
            else:
                mirror_mask = None
            if mirror_mask is not None:
                # hard clip
                mirror_mask[mirror_mask > 0.5] = 1
                mirror_mask[mirror_mask < 0.5] = 0
                mirror_mask = mirror_mask.bool()

            #############################
            # Config trace_secondary_rays
            trace_secondary_rays = (
                args.app_place_new_mirror
                or (
                    mirror_mask is not None
                    and mirror_mask.any()
                    and kwargs.get("trace_secondary_rays", False)
                )
            )
            if recur_level >= args.max_recursive_level:
                trace_secondary_rays = False

            #############################
            # Trace secondary rays
            if trace_secondary_rays:
                rays_o_chunk, rays_d_chunk = (
                    rays_chunk[:, 0:3],
                    rays_chunk[:, 3:6],
                )  # both (N_rays, 3)
                near_chunk, far_chunk = (
                    rays_chunk[:, 6:7],
                    rays_chunk[:, 7:8],
                )  # both (N_rays, 1)

                #############################
                # Get normal and secondary_rays_o
                secondary_rays_o = rendered_ray_chunks[f"x_surface_{select_type}"]
                if (
                    f"pred_normal_{select_type}" in rendered_ray_chunks
                ):  # use predict_normal
                    normal = (
                        rendered_ray_chunks[f"surface_normal_{select_type}"]
                        if f"surface_normal_{select_type}" in rendered_ray_chunks
                        else (
                            rendered_ray_chunks[f"pred_normal_{select_type}"]
                            * rendered_ray_chunks[f"weights_{select_type}"].unsqueeze(
                                -1
                            )
                        ).sum(1, keepdim=False)
                    )  # normal is correct, not -normal (n.dot(w) should be positive.)
                else:  # use grad normal
                    normal = (
                        rendered_ray_chunks[f"surface_normal_grad_{select_type}"]
                        if f"surface_normal_grad_{select_type}" in rendered_ray_chunks
                        else (
                            rendered_ray_chunks[f"normal_{select_type}"]
                            * rendered_ray_chunks[f"weights_{select_type}"].unsqueeze(
                                -1
                            )
                        ).sum(1, keepdim=False)
                    )

                #############################
                # Modify normal or secondary_rays_o or mirror_mask for applications
                if args.app_place_new_mirror:
                    # The plane equation of the new mirror: Based on the scene model, take a plane parallel to the z-axis, such as x=1 (or y=1).
                    # All vertices of the new mirror: for example, you can take an isosceles Right triangle [[1, 1, 1], [1, 0, 0], [1, 0, 1]], or a rectangle [[1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]].

                    # The following code segment outputs intersection_xyz and new_mirror_mask.
                    if args.plane_pos == "plane_x":
                        if "livingroom" in args.root_dir:
                            plane_x = 0
                            plane_normal = [-1, 0, 0]
                            # Note that `-normal` is equavalent to `normal` for calculating reflect direction.
                            rec_bound = [-1, 1, -0.5, 0.5]
                        elif "washroom" in args.root_dir:
                            plane_x = -1
                            plane_normal = [1, 0, 0]
                            rec_bound = [-1, 1, -1, 0.75]
                        elif "office" in args.root_dir:
                            plane_x = 1
                            plane_normal = [1, 0, 0]
                            rec_bound = [-1, 1, -1, 0.75]
                        else:
                            plane_x = -1
                            plane_normal = [1, 0, 0]
                            rec_bound = [-1, 1, -0.5, 0.5]

                        # Calculate the intersection point between the line where ray is located and the plane where the mirror is located.
                        # Line equation: (x-ray_o[0])/ray_d[0] = (y-ray_o[1])/ray_d[1] = (z-ray_o[2])/ray_d[2]
                        # Intersect with plane x=1: y = (1-ray_o[0])/ray_d[0] * ray_d[1] + ray_o[1]
                        #                           z = (1-ray_o[0])/ray_d[0] * ray_d[2] + ray_o[2]
                        y = (plane_x - rays_o_chunk[:, 0]) / rays_d_chunk[
                            :, 0
                        ] * rays_d_chunk[:, 1] + rays_o_chunk[:, 1]
                        z = (plane_x - rays_o_chunk[:, 0]) / rays_d_chunk[
                            :, 0
                        ] * rays_d_chunk[:, 2] + rays_o_chunk[:, 2]
                        x = torch.ones_like(y).to(y.device) * plane_x
                        # [N_rays]
                        intersection_xyz = torch.stack([x, y, z], dim=-1)  # [N_rays, 3]

                        # Determine if the intersection point is within the rectangle [[plane_x, 0, 0], [plane_x, 0, 1], [plane_x, 1, 1], [plane_x, 1, 0]]:
                        new_mirror_mask = torch.logical_not(
                            torch.logical_or(
                                torch.logical_or(
                                    torch.logical_or(
                                        y < rec_bound[0], z < rec_bound[2]
                                    ),
                                    y > rec_bound[1],
                                ),
                                z > rec_bound[3],
                            )
                        )  # [N_rays]

                    elif args.plane_pos == "plane_y":
                        if "washroom" in args.root_dir:
                            plane_y = 1.3
                            plane_normal = [0, -1, 0]
                            rec_bound = [-1, 1, -1, 1]
                        elif "livingroom" in args.root_dir:
                            plane_y = 1
                            plane_y = 1.65
                            plane_normal = [0, -1, 0]
                            rec_bound = [-1, 1, -0.5, 0.5]
                            rec_bound = [-0.3, 1.5, -0.5, 1]
                        elif "office" in args.root_dir:
                            plane_y = 0
                            plane_normal = [0, -1, 0]
                            rec_bound = [-1, 1, -0.5, 0.5]
                        else:
                            plane_y = 1
                            plane_normal = [0, -1, 0]
                            rec_bound = [-1, 1, -0.5, 0.5]

                        x = (plane_y - rays_o_chunk[:, 1]) / rays_d_chunk[
                            :, 1
                        ] * rays_d_chunk[:, 0] + rays_o_chunk[:, 0]
                        z = (plane_y - rays_o_chunk[:, 1]) / rays_d_chunk[
                            :, 1
                        ] * rays_d_chunk[:, 2] + rays_o_chunk[:, 2]
                        y = torch.ones_like(x).to(x.device) * plane_y
                        # [N_rays]
                        intersection_xyz = torch.stack([x, y, z], dim=-1)  # [N_rays, 3]

                        new_mirror_mask = torch.logical_not(
                            torch.logical_or(
                                torch.logical_or(
                                    torch.logical_or(
                                        x < rec_bound[0], z < rec_bound[2]
                                    ),
                                    x > rec_bound[1],
                                ),
                                z > rec_bound[3],
                            )
                        )  # [N_rays]

                    # Merge the normal of the new mirror into the original normal map.
                    normal[new_mirror_mask] = torch.tensor(
                        plane_normal, dtype=normal.dtype, device=normal.device
                    )
                    depth_new_mirror = torch.norm(
                        rays_o_chunk - intersection_xyz, dim=-1, keepdim=False
                    )  # The 3D distance from the origin of the ray to the intersection of the mirror surface  # [N_rays]
                    
                    # Deal with the case: Intersect on the ray, not on the reverse extension line of the ray
                    intersect_on_ray = (
                        torch.sum(
                            (intersection_xyz - rays_o_chunk) * rays_d_chunk, dim=-1
                        )
                        > 0
                    )
                    new_mirror_mask = torch.logical_and(
                        new_mirror_mask, intersect_on_ray
                    )

                    # Handling foreground occlusion
                    depth = rendered_ray_chunks[f"depth_{select_type}"]
                    mask_new_mirror_blocked_by_fg = depth_new_mirror > depth
                    mask_new_mirror_blocked_by_fg = torch.logical_and(
                        mask_new_mirror_blocked_by_fg, mask_valid_depth
                    )
                    new_mirror_mask[mask_new_mirror_blocked_by_fg] = False

                    secondary_rays_o_new_mirror = intersection_xyz
                    # Merge the secondary_rays_o of the new mirror into the original secondary_rays_o map.
                    secondary_rays_o[new_mirror_mask] = secondary_rays_o_new_mirror[
                        new_mirror_mask
                    ]
                    # Merge the mirror_mask of the new mirror into the original mirror_mask map.
                    mirror_mask[new_mirror_mask] = True

                    # for visualization
                    if "mirror_mask_fine" in rendered_ray_chunks:
                        rendered_ray_chunks["mirror_mask_fine"] = mirror_mask
                    elif "mirror_mask_coarse" in rendered_ray_chunks:
                        rendered_ray_chunks["mirror_mask_coarse"] = mirror_mask
                    depth[new_mirror_mask] = depth_new_mirror[new_mirror_mask]
                    if "depth_fine" in rendered_ray_chunks:
                        rendered_ray_chunks["depth_fine"] = depth
                    elif "depth_coarse" in rendered_ray_chunks:
                        rendered_ray_chunks["depth_coarse"] = depth
                # This segment is added for app_place_new_mirror, in case that new_mirror_mask is all False. We write like this because new_mirror_mask is calculated here (in trace_secondary_rays code segment).
                if args.app_place_new_mirror and mirror_mask.any() == False:
                    return rendered_ray_chunks

                if args.app_control_mirror_roughness:
                    noise = (torch.randn_like(normal) * normal_noise_std).to(
                        normal.device
                    )
                    normal_bkp = normal.clone()
                    normal = normal + noise
                
                #############################
                # Calculate reflect_direction
                normal = l2_normalize(normal)
                incident_inv_direction = l2_normalize(
                    -rays_d_chunk
                )  # note: incident_inv_direction should be outward
                cos = torch.sum(incident_inv_direction * normal, dim=-1)
                reflect_direction = (
                    2 * cos.unsqueeze(dim=-1).repeat(1, 3) * normal
                    - incident_inv_direction
                )  # (N_rays, 3)  # 2*n.dot(w)*n - w (normal and incident_direction must be normalized.)
                # For visualization
                rendered_ray_chunks[f"reflect_direction"] = reflect_direction

                #############################
                # Organize secondary_rays
                ray_forward_offset = 0.1  # the origin of the secondary ray should not be on the mirror surface.
                secondary_rays_near = torch.ones_like(far_chunk) * ray_forward_offset
                secondary_rays_far = far_chunk
                secondary_rays = torch.cat(
                    [
                        secondary_rays_o,
                        reflect_direction,
                        secondary_rays_near,
                        secondary_rays_far,
                    ],
                    dim=-1,
                )  # (N_rays, 8)

                #############################
                # Render reflected rays

                if only_trace_rays_in_mirrors:
                    secondary_rays = secondary_rays[
                        mirror_mask
                    ]  # [N_rays_in_mirror, 8]
                if secondary_rays.shape[0] > 0:
                    if args.app_reflection_substitution:
                        # transform rays
                        if "office" in args.root_dir:
                            translation = [0, 1, 0]
                            scale = 1
                        elif "market" in args.root_dir:
                            pose_align = [
                                [0, 1, 0, 0],
                                [-1, 0, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1],
                            ]
                            pose_ = torch.FloatTensor(pose_align).to(
                                secondary_rays.device
                            )
                            pose_scale = pose_[:3, 0].clone()
                            for i in range(3):
                                pose_scale[i] = torch.norm(pose_[:3, i])
                            pose_rotation = pose_[:3, :3].clone()
                            for i in range(3):
                                pose_rotation[:3, i] /= pose_scale[i]

                            secondary_rays[:, :3] = (
                                secondary_rays[:, :3] @ pose_[:3, :3].T
                            )
                            secondary_rays[:, :3] = (
                                secondary_rays[:, :3] + pose_[:3, 3][None, :]
                            )
                            secondary_rays[:, 3:6] = l2_normalize(
                                secondary_rays[:, 3:6] @ pose_[:3, :3].T
                            )

                            translation = [0, 0, 0]
                            scale = 1
                        else:
                            translation = [0, 0, 0]
                            scale = 1
                        secondary_rays[:, :3] = secondary_rays[
                            :, :3
                        ] * torch.FloatTensor([scale, scale, scale]).to(
                            secondary_rays.device
                        )
                        secondary_rays[:, :3] = secondary_rays[
                            :, :3
                        ] + torch.FloatTensor(translation).to(secondary_rays.device)
                        # query another radiance field
                        rendered_secondary_ray_chunks = render_rays(
                            system_substitution.models,
                            system_substitution.embeddings,
                            secondary_rays,
                            N_samples,
                            use_disp,
                            0,
                            0,
                            N_importance,
                            chunk,
                            dataset.white_back,
                            test_time=kwargs.get("test_time", True),
                            compute_normal=kwargs.get("trace_secondary_rays", False),
                            only_one_field=args.only_one_field,
                            only_one_field_fine_epoch=args.only_one_field_fine_epoch,
                            current_epoch=args.only_one_field_fine_epoch
                            + 1,  # to guarantee inference coarse model use fine sample points when training only one field.
                        )
                    else:
                        rendered_secondary_ray_chunks = render_rays_chunk_recursively(
                            secondary_rays,
                            mirror_mask,
                            idx_cnt,
                            recur_level=recur_level + 1,
                        )

                    # Implementing app_control_mirror_roughness here can reduce computation due to only tracing reflected rays in mirror.
                    if args.app_control_mirror_roughness:
                        trace_ray_times = args.trace_ray_times
                        for _ in range(trace_ray_times):
                            noise = (
                                torch.randn_like(normal_bkp) * normal_noise_std
                            ).to(normal_bkp.device)
                            normal = normal_bkp + noise
                            # Calculate reflect_direction
                            normal = l2_normalize(normal)
                            reflect_direction = (
                                2
                                * torch.sum(incident_inv_direction * normal, dim=-1)
                                .unsqueeze(dim=-1)
                                .repeat(1, 3)
                                * normal
                                - incident_inv_direction
                            )  # (N_rays, 3)  # 2*n.dot(w)*n - w (normal and incident_direction must be normalized.)
                            secondary_rays = torch.cat(
                                [
                                    secondary_rays_o,
                                    reflect_direction,
                                    secondary_rays_near,
                                    secondary_rays_far,
                                ],
                                dim=-1,
                            )  # (N_rays, 8)
                            # only trace rays in mirrors
                            secondary_rays = secondary_rays[
                                mirror_mask
                            ]  # [N_rays_in_mirror, 8]
                            rendered_secondary_ray_chunks_ = (
                                render_rays_chunk_recursively(
                                    secondary_rays,
                                    mirror_mask,
                                    idx_cnt,
                                    recur_level=recur_level + 1,
                                )
                            )
                            for typ in ["coarse", "fine"]:
                                if f"rgb_{typ}" in rendered_secondary_ray_chunks:
                                    rendered_secondary_ray_chunks[f"rgb_{typ}"] = (
                                        rendered_secondary_ray_chunks[f"rgb_{typ}"]
                                        + rendered_secondary_ray_chunks_[f"rgb_{typ}"]
                                    )

                        for typ in ["coarse", "fine"]:
                            if f"rgb_{typ}" in rendered_secondary_ray_chunks:
                                rendered_secondary_ray_chunks[
                                    f"rgb_{typ}"
                                ] = rendered_secondary_ray_chunks[f"rgb_{typ}"] / (
                                    trace_ray_times + 1
                                )

                    #############################
                    # Blend colors by mirror_mask
                    typ = select_type

                    base_color = rendered_ray_chunks[f"rgb_{typ}"]
                    if only_trace_rays_in_mirrors:
                        reflection_part = base_color.clone()
                        reflection_part[mirror_mask] = rendered_secondary_ray_chunks[
                            f"rgb_{typ}"
                        ]  # (N_rays, 3)
                    else:
                        reflection_part = rendered_secondary_ray_chunks[
                            f"rgb_{typ}"
                        ]  # (N_rays, 3)

                    mirror_mask_ = (
                        mirror_mask.float().unsqueeze(dim=-1).repeat(1, 3)
                    )  # (N_rays, 3)

                    rendered_ray_chunks[f"rgb_{typ}"] = (
                        mirror_mask_* reflection_part + (1 - mirror_mask_) * base_color
                    )

                    # For visualization
                    if only_trace_rays_in_mirrors:
                        rendered_ray_chunks[f"rgb_{typ}_reflect"] = torch.zeros_like(
                            rendered_ray_chunks[f"rgb_{typ}"]
                        )
                        rendered_ray_chunks[f"rgb_{typ}_reflect"][
                            mirror_mask.bool()
                        ] = rendered_secondary_ray_chunks[f"rgb_{typ}"]
                    else:
                        rendered_ray_chunks[
                            f"rgb_{typ}_reflect"
                        ] = rendered_secondary_ray_chunks[f"rgb_{typ}"]
                    if only_trace_rays_in_mirrors:
                        rendered_ray_chunks[
                            f"depth_{select_type}_reflect"
                        ] = torch.zeros_like(
                            rendered_ray_chunks[f"depth_{select_type}"]
                        )
                        rendered_ray_chunks[f"depth_{select_type}_reflect"][
                            mirror_mask.bool()
                        ] = rendered_secondary_ray_chunks[f"depth_{select_type}"]
                    else:
                        rendered_ray_chunks[
                            f"depth_{select_type}_reflect"
                        ] = rendered_secondary_ray_chunks[f"depth_{select_type}"]

            return rendered_ray_chunks

        rays_chunk = rays[i : i + chunk]
        mirror_mask_first = (
            torch.zeros(rays_chunk.shape[0]).bool().to(rays_chunk.device)
        )
        rendered_ray_chunks = render_rays_chunk_recursively(
            rays_chunk, mirror_mask_first, idx_cnt, recur_level=0
        )

        for k, v in rendered_ray_chunks.items():
            results[k] += [v.cpu()]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


def save_img_and_cal_psnr(
    typ,
    results,
    sample,
    imgs,
    mirror_masks,
    depth_maps,
    depth_reflect_maps,
    mirror_masks_float,
    psnrs,
    args,
    dir_name,
    mirror_mask_dir,
    depth_dir,
    normal_dir,
    depth_reflect_dir,
    x_surface_dir,
    i,
):
    if f"rgb_{typ}" in results:
        img_pred = np.clip(results[f"rgb_{typ}"].view(h, w, 3).cpu().numpy(), 0, 1)

        if args.not_save_depth == False:
            depth_pred = results[f"depth_{typ}"].view(h, w).cpu().numpy()
            depth_maps += [depth_pred]
            if "pfm" in args.depth_format:
                save_pfm(
                    os.path.join(depth_dir, f"depth_{typ}_{i:03d}.pfm"),
                    depth_pred,
                )
            if "png" in args.depth_format:
                global all_depths_max
                global all_depths_min
                if all_depths_max is None:
                    all_depths_max = np.max(depth_pred)
                    all_depths_min = np.min(depth_pred)
                else:
                    depth_pred_max = np.max(depth_pred)
                    depth_pred_min = np.min(depth_pred)
                    if depth_pred_max > all_depths_max:
                        all_depths_max = depth_pred_max
                    if depth_pred_min < all_depths_min:
                        all_depths_min = depth_pred_min

                canvas = visualize_depth(torch.as_tensor(depth_pred))  # (3, H, W)
                canvas = canvas.permute((1, 2, 0)).numpy()  # (H, W, 3)
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    os.path.join(depth_dir, f"depth_{typ}_{i:03d}.png"), canvas
                )
            if "bytes" in args.depth_format:
                with open(os.path.join(depth_dir, f"depth_{typ}_{i:03d}"), "wb") as f:
                    f.write(depth_pred.tobytes())

        img_pred_ = (img_pred * 255).astype(np.uint8)
        imgs += [img_pred_]
        imageio.imwrite(os.path.join(dir_name, f"rgb_{typ}_{i:03d}.png"), img_pred_)

        if "rgbs" in sample:
            rgbs = sample["rgbs"]
            img_gt = rgbs.view(h, w, 3)
            psnrs += [metrics.psnr(img_gt, img_pred).item()]

        if f"mirror_mask_{typ}" in results:
            mirror_mask_pred = np.clip(
                results[f"mirror_mask_{typ}"]
                .unsqueeze(-1)
                .repeat(1, 3)
                .view(h, w, 3)
                .cpu()
                .numpy(),
                0,
                1,
            )
            mirror_masks_float += [mirror_mask_pred]
            mirror_mask_pred_ = (mirror_mask_pred * 255).astype(np.uint8)
            mirror_masks += [mirror_mask_pred_]
            imageio.imwrite(
                os.path.join(mirror_mask_dir, f"mirror_mask_{typ}_{i:03d}.png"),
                mirror_mask_pred_,
            )

            if f"depth_{typ}_reflect" in results:
                depth_reflect = results[f"depth_{typ}_reflect"].view(h, w).cpu().numpy()
                depth_reflect_maps += [depth_reflect]

                global all_depths_reflect_max
                global all_depths_reflect_min
                if all_depths_reflect_max is None:
                    all_depths_reflect_max = np.max(depth_reflect)
                    all_depths_reflect_min = np.min(depth_reflect)
                else:
                    depth_reflect_max = np.max(depth_reflect)
                    depth_reflect_min = np.min(depth_reflect)
                    if depth_reflect_max > all_depths_reflect_max:
                        all_depths_reflect_max = depth_reflect_max
                    if depth_reflect_min < all_depths_reflect_min:
                        all_depths_reflect_min = depth_reflect_min

                canvas = visualize_depth(torch.as_tensor(depth_reflect))  # (3, H, W)
                canvas = canvas.permute((1, 2, 0)).numpy()  # (H, W, 3)
                canvas = (
                    canvas * mirror_mask_pred
                )  # mask depth outside mirror mask as black.
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    os.path.join(depth_reflect_dir, f"depth_reflect_{typ}_{i:03d}.png"),
                    canvas,
                )

        if f"surface_normal_grad_{typ}" in results:
            surface_normal_grad = np.clip(
                ((results[f"surface_normal_grad_{typ}"] + 1) / 2)
                .view(h, w, 3)
                .cpu()
                .numpy(),
                0,
                1,
            )
            surface_normal_grad = (surface_normal_grad * 255).astype(np.uint8)
            imageio.imwrite(
                os.path.join(normal_dir, f"surface_normal_grad_{typ}_{i:03d}.png"),
                surface_normal_grad,
            )
        if f"surface_normal_{typ}" in results:
            surface_normal = np.clip(
                ((results[f"surface_normal_{typ}"] + 1) / 2)
                .view(h, w, 3)
                .cpu()
                .numpy(),
                0,
                1,
            )
            surface_normal = (surface_normal * 255).astype(np.uint8)
            imageio.imwrite(
                os.path.join(normal_dir, f"surface_normal_{typ}_{i:03d}.png"),
                surface_normal,
            )

        if f"x_surface_{typ}" in results:
            x_surface = np.clip(
                (visualize_rgb_map_global(results[f"x_surface_{typ}"]))
                .view(h, w, 3)
                .cpu()
                .numpy(),
                0,
                1,
            )
            x_surface = (x_surface * 255).astype(np.uint8)
            imageio.imwrite(
                os.path.join(x_surface_dir, f"x_surface_{typ}_{i:03d}.png"), x_surface
            )


def save_gif_and_print_mean_psnr(
    typ, dir_name, args, imgs, mirror_masks, depth_maps, psnrs
):
    FPS = 15  # 30
    imageio.mimsave(
        os.path.join(dir_name, f"{args.exp_name}_rgb_{typ}.gif"), imgs, fps=FPS
    )

    if len(mirror_masks) > 0:
        imageio.mimsave(
            os.path.join(dir_name, f"{args.exp_name}_mirror_mask_{typ}.gif"),
            mirror_masks,
            fps=FPS,
        )

    if args.not_save_depth == False:
        depth_imgs = (depth_maps - np.min(depth_maps)) / (
            max(np.max(depth_maps) - np.min(depth_maps), 1e-8)
        )
        depth_imgs_ = [
            cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_JET)
            for img in depth_imgs
        ]
        imageio.mimsave(
            os.path.join(dir_name, f"{args.exp_name}_depth_{typ}.gif"),
            depth_imgs_,
            fps=FPS,
        )

    if psnrs:
        mean_psnr = np.mean(psnrs)
        print(f"Mean PSNR ({typ}): {mean_psnr:.2f}")


def save_depth_unified_normalization(
    typ,
    args,
    depth_unified_normalization_dir,
    depth_reflect_unified_normalization_dir,
    depth_maps,
    depth_reflect_maps,
    mirror_masks_float,
):
    for i, depth_pred in enumerate(depth_maps):
        if "png" in args.depth_format:
            global all_depths_max
            global all_depths_min
            if all_depths_max is not None:
                canvas = visualize_depth(
                    torch.as_tensor(depth_pred),
                    vmin=all_depths_min,
                    vmax=all_depths_max,
                )  # (3, H, W)
                canvas = canvas.permute((1, 2, 0)).numpy()  # (H, W, 3)
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    os.path.join(
                        depth_unified_normalization_dir, f"depth_{typ}_{i:03d}.png"
                    ),
                    canvas,
                )
    for i, depth_reflect in enumerate(depth_reflect_maps):
        global all_depths_reflect_max
        global all_depths_reflect_min
        if all_depths_reflect_max is not None:
            canvas = visualize_depth(
                torch.as_tensor(depth_reflect),
                vmin=all_depths_reflect_min,
                vmax=all_depths_reflect_max,
            )  # (3, H, W)
            canvas = canvas.permute((1, 2, 0)).numpy()  # (H, W, 3)
            canvas = (
                canvas * mirror_masks_float[i]
            )  # mask depth outside mirror mask as black.
            canvas = (canvas * 255).astype(np.uint8)
            imageio.imwrite(
                os.path.join(
                    depth_reflect_unified_normalization_dir,
                    f"depth_reflect_{typ}_{i:03d}.png",
                ),
                canvas,
            )

if __name__ == "__main__":
    args = get_opt()
    w, h = args.img_wh

    kwargs = {
        "root_dir": args.root_dir,
        "split": args.split,
        "img_wh": tuple(args.img_wh),
        "hparams": args,
    }

    if args.dataset_name == "llff":
        kwargs["spheric_poses"] = args.spheric_poses
    dataset = dataset_dict[args.dataset_name](**kwargs)

    system = NeRFSystem(args)
    for key in system.models.keys():
        if key == "coarse" or key == "fine":
            load_ckpt(system.models[key], args.ckpt_path, model_name="nerf_" + key)
        else:
            load_ckpt(system.models[key], args.ckpt_path, model_name=key)
        system.models[key].cuda().eval()

    system_substitution = None
    if args.app_reflection_substitution:
        if args.substitution_ckpt_path is None:
            print(
                "[Error] substitution_ckpt_path should be appointed in app_reflection_substitution."
            )
            sys.exit(1)
        args_substitution = deepcopy(args)
        args_substitution.bound = 6
        system_substitution = NeRFSystem(args_substitution)
        print(
            "[info] Load another radiance field of scene to substitute from ckpt:",
            args.substitution_ckpt_path,
        )
        for key in system_substitution.models.keys():
            if key == "coarse" or key == "fine":
                load_ckpt(
                    system_substitution.models[key],
                    args.substitution_ckpt_path,
                    model_name="nerf_" + key,
                )
            else:
                load_ckpt(
                    system_substitution.models[key],
                    args.substitution_ckpt_path,
                    model_name=key,
                )
            system_substitution.models[key].cuda().eval()

    system_obj = None
    args_d_nerf = None
    render_kwargs_test_d_nerf = None
    if args.app_reflect_newly_placed_objects:
        if args.obj_ckpt_path is None:
            print(
                "[Error] obj_ckpt_path should be appointed in app_reflect_newly_placed_objects."
            )
            sys.exit(1)
        if args.obj_model_type == "nerf_pl":
            from models.nerf_pl.rendering_nerfpl import render_rays as render_rays_obj
            from models.nerf_pl.train_nerfpl import NeRFSystem as NeRFSystem_nerfpl

            system_obj = NeRFSystem_nerfpl(args)
            print("[info] Load object radiance field from ckpt:", args.obj_ckpt_path)
            load_ckpt(
                system_obj.nerf_coarse,
                args.obj_ckpt_path,
                "nerf_coarse",
                args.prefixes_to_ignore,
            )
            system_obj.nerf_coarse.cuda().eval()
            if args.N_importance > 0:
                load_ckpt(
                    system_obj.nerf_fine,
                    args.obj_ckpt_path,
                    "nerf_fine",
                    args.prefixes_to_ignore,
                )
                system_obj.nerf_fine.cuda().eval()
        elif args.obj_model_type == "d_nerf":
            from models.d_nerf.run_dnerf import config_parser as config_parser_d_nerf
            from models.d_nerf.run_dnerf import create_nerf as create_d_nerf
            from models.d_nerf.run_dnerf import render_rays as render_rays_obj

            # get config file
            parser = config_parser_d_nerf()
            d_nerf_config_file = os.path.join(
                os.path.split(args.obj_ckpt_path)[0], "config.txt"
            )
            args_d_nerf = parser.parse_args(f"--config {d_nerf_config_file}")
            args_d_nerf.basedir = os.path.split(os.path.split(args.obj_ckpt_path)[0])[0]
            print("args_d_nerf.basedir:", args_d_nerf.basedir)
            # set render params
            _, render_kwargs_test_d_nerf, _, _, _ = create_d_nerf(args_d_nerf)
            render_kwargs_test_d_nerf.update({"near": 2.0, "far": 6.0})

    (
        imgs,
        mirror_masks,
        depth_maps,
        depth_reflect_maps,
        mirror_masks_float,
        psnrs,
    ) = ([], [], [], [], [], [])
    (
        imgs_coarse,
        mirror_masks_coarse,
        depth_maps_coarse,
        depth_reflect_maps_coarse,
        mirror_masks_float_coarse,
        psnrs_coarse,
    ) = ([], [], [], [], [], [])
    dir_name = f"results/{args.dataset_name}/{args.exp_name}"
    print(f"[info] Results saved to dir {dir_name}.")
    os.makedirs(dir_name, exist_ok=True)
    depth_dir = os.path.join(dir_name, "depth")
    depth_unified_normalization_dir = os.path.join(
        dir_name, "depth_unified_normalization"
    )
    if args.not_save_depth == False:
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(depth_unified_normalization_dir, exist_ok=True)
    mirror_mask_dir = os.path.join(dir_name, "mirror_mask")
    os.makedirs(mirror_mask_dir, exist_ok=True)
    normal_dir = os.path.join(dir_name, "normal")
    os.makedirs(normal_dir, exist_ok=True)
    depth_reflect_dir = os.path.join(dir_name, "depth_reflect")
    os.makedirs(depth_reflect_dir, exist_ok=True)
    depth_reflect_unified_normalization_dir = os.path.join(
        dir_name, "depth_reflect_unified_normalization"
    )
    os.makedirs(depth_reflect_unified_normalization_dir, exist_ok=True)
    x_surface_dir = os.path.join(dir_name, "x_surface")
    os.makedirs(x_surface_dir, exist_ok=True)

    typ = ""

    total_time_dataset = 0

    for i in tqdm(range(len(dataset))):
        # # quickly inference one specified view.
        if args.only_eval_idx >= 0 and i != args.only_eval_idx:
            continue

        sample = dataset[i]
        rays = sample["rays"].cuda()

        progress = i / len(dataset)
        progress_cycle = progress * 2 if progress < 0.5 else 1 - (progress - 0.5) * 2
        normal_noise_std = (
            args.normal_noise_std * progress_cycle
            if args.normal_noise_std_changes
            else args.normal_noise_std
        )

        results = batched_inference(
            system.models,
            system.embeddings,
            rays,
            args.N_samples,
            args.N_importance,
            args.use_disp,
            args.chunk,
            trace_secondary_rays=args.trace_secondary_rays,
            predict_normal=args.predict_normal,
            test_time=(not args.render_coarse_rgb),
            args=args,
            system_substitution=system_substitution,
            system_obj=system_obj,
            render_kwargs_test_d_nerf=render_kwargs_test_d_nerf,
            args_d_nerf=args_d_nerf,
            frame_time=progress,  # range in [0,1]. Set to a fixed value => static scene.
            normal_noise_std=normal_noise_std,
        )

        typ = "fine" if "rgb_fine" in results else "coarse"
        save_img_and_cal_psnr(
            typ,
            results,
            sample,
            imgs,
            mirror_masks,
            depth_maps,
            depth_reflect_maps,
            mirror_masks_float,
            psnrs,
            args,
            dir_name,
            mirror_mask_dir,
            depth_dir,
            normal_dir,
            depth_reflect_dir,
            x_surface_dir,
            i,
        )
        if typ != "coarse" and args.render_coarse_rgb:
            save_img_and_cal_psnr(
                "coarse",
                results,
                sample,
                imgs_coarse,
                mirror_masks_coarse,
                depth_maps_coarse,
                depth_reflect_maps_coarse,
                mirror_masks_float_coarse,
                psnrs_coarse,
                args,
                dir_name,
                mirror_mask_dir,
                depth_dir,
                normal_dir,
                depth_reflect_dir,
                x_surface_dir,
                i,
            )

    if len(imgs) > 0:
        save_gif_and_print_mean_psnr(
            typ, dir_name, args, imgs, mirror_masks, depth_maps, psnrs
        )
        save_depth_unified_normalization(
            typ,
            args,
            depth_unified_normalization_dir,
            depth_reflect_unified_normalization_dir,
            depth_maps,
            depth_reflect_maps,
            mirror_masks_float,
        )
        if typ != "coarse" and args.render_coarse_rgb and len(imgs_coarse) > 0:
            save_gif_and_print_mean_psnr(
                "coarse",
                dir_name,
                args,
                imgs_coarse,
                mirror_masks_coarse,
                depth_maps_coarse,
                psnrs_coarse,
            )
            save_depth_unified_normalization(
                "coarse",
                args,
                depth_unified_normalization_dir,
                depth_reflect_unified_normalization_dir,
                depth_maps_coarse,
                depth_reflect_maps_coarse,
                mirror_masks_float_coarse,
            )
