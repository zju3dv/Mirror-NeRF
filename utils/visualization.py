import torchvision.transforms as T
import torch
import numpy as np
import cv2
from PIL import Image
import os
from torchvision import utils as vutils


def visualize_depth(depth, cmap=cv2.COLORMAP_JET, vmin=None, vmax=None):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x)  # change nan to 0
    mi = np.min(x) if vmin == None else vmin  # get minimum depth
    ma = np.max(x) if vmax == None else vmax
    x = np.clip(x, mi, ma)
    x = (x - mi) / max(ma - mi, 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_


def visualize_val_image(img_wh, batch, results, add_text=True):
    W, H = img_wh
    stack_images = []

    vis_min, vis_max = None, None  # for depth

    img_gt = batch["rgbs"].view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
    stack_images += [add_text_to_tensor_image(img_gt, "gt_img", add_text=add_text)]

    for typ in ["fine", "coarse"]:
        if f"rgb_{typ}" in results:
            img = (
                results[f"rgb_{typ}"].view(H, W, 3).permute(2, 0, 1).cpu()
            )  # (3, H, W)
            stack_images += [
                add_text_to_tensor_image(img, f"img_{typ}", add_text=add_text)
            ]
    for typ in ["fine", "coarse"]:
        if f"rgb_{typ}_reflect" in results:
            img = (
                results[f"rgb_{typ}_reflect"].view(H, W, 3).permute(2, 0, 1).cpu()
            )  # (3, H, W)
            stack_images += [
                add_text_to_tensor_image(img, f"img_reflect_{typ}", add_text=add_text)
            ]
    for typ in ["fine", "coarse"]:
        if f"rgb_{typ}_direct" in results:
            img = (
                results[f"rgb_{typ}_direct"].view(H, W, 3).permute(2, 0, 1).cpu()
            )  # (3, H, W)
            stack_images += [
                add_text_to_tensor_image(img, f"img_direct_{typ}", add_text=add_text)
            ]

    for typ in ["fine", "coarse"]:
        if f"depth_{typ}" in results:
            depth = visualize_depth(
                results[f"depth_{typ}"].view(H, W), vmin=vis_min, vmax=vis_max
            )  # (3, H, W)
            stack_images += [
                add_text_to_tensor_image(depth, f"depth_{typ}", add_text=add_text)
            ]
    for typ in ["fine", "coarse"]:
        if f"depth_{typ}_reflect" in results:
            depth_reflect = visualize_depth(
                results[f"depth_{typ}_reflect"].view(H, W), vmin=vis_min, vmax=vis_max
            )  # (3, H, W)
            stack_images += [
                add_text_to_tensor_image(
                    depth_reflect, f"depth_reflect_{typ}", add_text=add_text
                )
            ]

    if "mirror_mask" in batch:
        mirror_mask = (
            batch["mirror_mask"]
            .squeeze()
            .unsqueeze(dim=-1)
            .repeat(1, 3)
            .view(H, W, 3)
            .permute(2, 0, 1)
            .cpu()
        )  # (3, H, W)
        stack_images += [
            add_text_to_tensor_image(mirror_mask, "gt_mirror_mask", add_text=add_text)
        ]

    for typ in ["fine", "coarse"]:
        if f"mirror_mask_{typ}" in results:
            mirror_mask_pred = (
                results[f"mirror_mask_{typ}"]
                .squeeze()
                .unsqueeze(dim=-1)
                .repeat(1, 3)
                .view(H, W, 3)
                .permute(2, 0, 1)
                .cpu()
            )  # (3, H, W)
            stack_images += [
                add_text_to_tensor_image(
                    mirror_mask_pred, f"mirror_mask_pred_{typ}", add_text=add_text
                )
            ]

    for typ in ["fine", "coarse"]:
        if f"surface_normal_{typ}" in results:
            surface_normal = (results[f"surface_normal_{typ}"] + 1) / 2
            vis = surface_normal.view(H, W, 3).permute(2, 0, 1).cpu()
            stack_images += [
                add_text_to_tensor_image(vis, f"normal_pred_{typ}", add_text=add_text)
            ]

        if f"surface_normal_grad_{typ}" in results:
            surface_normal_grad = (results[f"surface_normal_grad_{typ}"] + 1) / 2
            vis = surface_normal_grad.view(H, W, 3).permute(2, 0, 1).cpu()
            stack_images += [
                add_text_to_tensor_image(vis, f"normal_grad_{typ}", add_text=add_text)
            ]

    if f"secondary_rays_o" in results:
        secondary_rays_o = (
            results[f"secondary_rays_o"].view(H, W, 3).permute(2, 0, 1).cpu()
        )  # (3, H, W)
        stack_images += [
            add_text_to_tensor_image(
                secondary_rays_o, f"secondary_rays_o", add_text=add_text
            )
        ]
        secondary_rays_o_vis = (
            visualize_rgb_map_global(results[f"secondary_rays_o"])
            .view(H, W, 3)
            .permute(2, 0, 1)
            .cpu()
        )  # (3, H, W)
        stack_images += [
            add_text_to_tensor_image(
                secondary_rays_o_vis, f"secondary_rays_o_vis", add_text=add_text
            )
        ]

    if f"reflect_direction" in results:
        reflect_direction = (
            results[f"reflect_direction"].view(H, W, 3).permute(2, 0, 1).cpu()
        )  # (3, H, W)
        stack_images += [
            add_text_to_tensor_image(
                reflect_direction, f"reflect_direction", add_text=add_text
            )
        ]
        reflect_direction_vis = (
            visualize_rgb_map_global(results[f"reflect_direction"])
            .view(H, W, 3)
            .permute(2, 0, 1)
            .cpu()
        )  # (3, H, W)
        stack_images += [
            add_text_to_tensor_image(
                reflect_direction_vis, f"reflect_direction_vis", add_text=add_text
            )
        ]

    for typ in ["fine", "coarse"]:
        if f"x_surface_{typ}" in results:
            x_surface = (
                visualize_rgb_map_global(results[f"x_surface_{typ}"])
                .view(H, W, 3)
                .permute(2, 0, 1)
                .cpu()
            )  # (3, H, W)
            stack_images += [
                add_text_to_tensor_image(
                    x_surface, f"x_surface_{typ}", add_text=add_text
                )
            ]

    stack = (
        torch.stack(stack_images) if len(stack_images) > 0 else None
    )  # (N_pics, 3, H, W)
    return stack


def add_text_to_tensor_image(
    image: torch.Tensor, text: str, pos: tuple = (20, 20), add_text=True
):
    """
    image: torch.FloatTensor [3, H, W] in the form of [0, 1]
    """
    if not add_text:
        return image
    image_np = (image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8).copy()
    image_np = cv2.putText(
        image_np,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,  # font scale
        (255, 0, 0),  # color
        2,  # thickness
    )
    return torch.from_numpy(image_np.astype(np.float32)).permute(2, 0, 1) * 255


def visualize_rgb_map_global(tensor, eps=1e-8):
    """Normalize a map to be in range [0,1].
    tensor: torch.Tensor
    """
    output = tensor.clone()
    min = torch.min(output)
    max = torch.max(output)
    if min == max:
        return torch.ones_like(output)
    rg = max - min
    if rg < eps:
        print("Warning: divisor is close to 0 (in visualize_rgb_map_global).")
    output = (output - min) / rg
    return output
