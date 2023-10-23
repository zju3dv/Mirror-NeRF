from torch import nn
import torch
import math
from utils.func import binary_cross_entropy


class ColorLoss(nn.Module):
    def __init__(self, coef=1, woMaskRGBtoBlack=False):
        super().__init__()
        self.woMaskRGBtoBlack = woMaskRGBtoBlack
        self.coef = coef
        self.loss = nn.MSELoss(reduction="mean")

    def forward(self, inputs, batch, train_geometry_stage=False):
        loss = 0
        targets = batch["rgbs"].view(-1, 3)
        if (
            train_geometry_stage
            and "mirror_mask" in batch
            and (batch["mirror_mask"] < 0).any()
        ):
            if "mirror_mask_fine" in inputs or "mirror_mask_coarse" in inputs:
                mirror_mask = (
                    inputs["mirror_mask_fine"].detach()
                    if "mirror_mask_fine" in inputs
                    else inputs["mirror_mask_coarse"].detach()
                )
                mirror_mask[mirror_mask > 0.5] = 1
                mirror_mask[mirror_mask < 0.5] = 0
                mirror_mask = mirror_mask.bool().detach()
                for typ in ["coarse", "fine"]:
                    if f"rgb_{typ}" in inputs:
                        loss += self.loss(
                            inputs[f"rgb_{typ}"][~mirror_mask], targets[~mirror_mask]
                        )
            else:
                loss = 0
        elif train_geometry_stage and "mirror_mask" in batch and self.woMaskRGBtoBlack:
            # valid GT mirror mask
            mirror_mask = batch["mirror_mask"].squeeze().bool()
            for typ in ["coarse", "fine"]:
                if f"rgb_{typ}" in inputs:
                    loss += self.loss(
                        inputs[f"rgb_{typ}"][~mirror_mask], targets[~mirror_mask]
                    )
        else:
            for typ in ["coarse", "fine"]:
                if f"rgb_{typ}" in inputs:
                    loss += self.loss(inputs[f"rgb_{typ}"], targets)

        return self.coef * loss


class NormalLoss(nn.Module):
    def __init__(self, coef=1e-4, normal_loss_only_inside_mirror=False):
        super().__init__()
        self.coef = coef
        self.normal_loss_only_inside_mirror = normal_loss_only_inside_mirror

    def forward(self, inputs, batch):
        loss = 0
        if "mirror_mask" in batch and (batch["mirror_mask"] < 0).any() == False:
            mirror_mask = batch["mirror_mask"].squeeze().bool()
        else:
            mirror_mask = None

        if mirror_mask is not None:
            for typ in ["coarse", "fine"]:
                if f"normal_dif_{typ}" in inputs:
                    if not self.normal_loss_only_inside_mirror:
                        loss += inputs[f"normal_dif_{typ}"][~mirror_mask].mean()
                    loss += inputs[f"normal_dif_{typ}"][mirror_mask].mean() * 100
        else:
            for typ in ["coarse", "fine"]:
                if f"normal_dif_{typ}" in inputs:
                    loss += inputs[f"normal_dif_{typ}"].mean()
        return self.coef * loss


class PlaneConsistentLoss(nn.Module):
    def __init__(
        self,
        coef=0.1,
    ):
        super().__init__()
        self.coef = coef

    def cal_plane_consistent_loss(self, inputs, masks=None):
        # inputs: [N_rays, 3]
        # masks: [N_rays]
        if masks is None:
            masks = torch.ones_like(inputs).bool().to(inputs.device)
        loss = 0
        inputs_in_mask = inputs[masks]
        N_rays_in_mask = inputs_in_mask.shape[0]
        times = N_rays_in_mask // 4
        if times > 0:
            for i in range(times):
                select_pts = [
                    inputs_in_mask[
                        torch.randint(high=N_rays_in_mask, size=(1,))[0].item()
                    ]
                    for _ in range(4)
                ]
                loss_ = torch.cross(
                    select_pts[1] - select_pts[0], select_pts[2] - select_pts[0], dim=-1
                )
                loss_ = torch.sum(loss_ * (select_pts[3] - select_pts[0]), dim=-1)
                loss += torch.abs(loss_)
            loss = loss / times
        return loss

    def forward(self, inputs, batch):
        loss = 0
        if "mirror_mask" in batch and (batch["mirror_mask"] < 0).any() == False:
            mirror_mask = batch["mirror_mask"].squeeze().bool()
        else:
            mirror_mask = None

        if mirror_mask is not None:
            for typ in ["fine", "coarse"]:
                if f"x_surface_{typ}" in inputs:
                    loss += self.cal_plane_consistent_loss(
                        inputs[f"x_surface_{typ}"],
                        mirror_mask,
                    )
        return self.coef * loss



class NormalRegLoss(nn.Module):
    """
    normal should back-facing the ray_d, similar to Ref-NeRF
    """

    def __init__(self, coef=1, ext_supervise_grad_normal=True):
        super().__init__()
        self.coef = coef
        self.ext_supervise_grad_normal = ext_supervise_grad_normal

    def forward(self, inputs, batch):
        rays_d = batch["rays"][..., 3:6].view(-1, 3)
        mask = (
            batch["valid_mask"].view(-1)
            if "valid_mask" in batch
            else torch.ones(rays_d.shape[0]).bool().to(rays_d.device)
        )  # (N_rays)
        loss = 0
        for typ in ["coarse", "fine"]:
            if f"pred_normal_{typ}" in inputs:  # (N_rays, N_samples, 3)
                loss += (
                    (
                        torch.relu(
                            inputs[f"pred_normal_{typ}"][mask]
                            * rays_d[mask].unsqueeze(1)
                        )
                    ).sum(-1)
                    * inputs[f"weights_{typ}"][mask]
                ).mean()
        if self.ext_supervise_grad_normal:
            for typ in ["fine"]:
                if f"normal_{typ}" in inputs:
                    loss += (
                        (
                            torch.relu(
                                inputs[f"normal_{typ}"][mask]
                                * rays_d[mask].unsqueeze(1)
                            )
                        ).sum(-1)
                        * inputs[f"weights_{typ}"][mask]
                    ).mean()
        return self.coef * loss


class MirrorMaskLoss(nn.Module):
    def __init__(self, coef=1, model_type="nerf"):
        super().__init__()
        self.coef = coef
        self.loss = (
            binary_cross_entropy
            if model_type == "nerf_tcnn"
            else nn.BCELoss(reduction="none")
        )

    def forward(self, inputs, batch):
        loss = 0
        for typ in ["coarse", "fine"]:
            if (
                f"mirror_mask_{typ}" in inputs
                and "mirror_mask" in batch
            ):
                mirror_mask_pred = inputs[f"mirror_mask_{typ}"]
                mirror_mask_pred = torch.clamp(mirror_mask_pred, min=1e-7, max=1 - 1e-7)
                mirror_mask_gt = batch["mirror_mask"].squeeze().float()
                valid_mirror_mask = mirror_mask_gt >= 0
                loss_ = self.loss(mirror_mask_pred, mirror_mask_gt)
                loss += (loss_ * valid_mirror_mask.int().detach()).mean()
        return self.coef * loss


class TotalLoss(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.color_loss = ColorLoss(
            coef=hparams.color_loss_weight, woMaskRGBtoBlack=hparams.woMaskRGBtoBlack
        )
        self.normal_loss = NormalLoss(
            coef=hparams.normal_loss_weight,
            normal_loss_only_inside_mirror=hparams.normal_loss_only_inside_mirror,
        )
        self.normal_reg_loss = NormalRegLoss(coef=hparams.normal_reg_loss_weight)
        self.mirror_mask_loss = MirrorMaskLoss(
            coef=hparams.mirror_mask_loss_weight, model_type=self.hparams.model_type
        )
        if self.hparams.use_plane_consistent_loss:
            self.plane_consistent_loss = PlaneConsistentLoss(
                coef=hparams.plane_consistent_loss_weight
            )

    def forward(self, inputs, batch, train_geometry_stage=False, epoch=-1):
        loss_dict = dict()

        # color loss
        loss_dict["color_loss"] = self.color_loss(
            inputs,
            batch,
            train_geometry_stage,
        )

        # mirror mask loss
        if (
            not train_geometry_stage
            or epoch >= self.hparams.train_mirror_mask_start_epoch
        ):
            # equal to: if not (self.hparams.train_geometry_stage and epoch < self.hparams.train_mirror_mask_start_epoch):
            loss_dict["mirror_mask_loss"] = self.mirror_mask_loss(inputs, batch)

        if epoch >= self.hparams.smooth_mirror_start_epoch:
            if self.hparams.use_plane_consistent_loss:
                loss_dict["plane_consistent_loss"] = self.plane_consistent_loss(
                    inputs, batch
                )
        
        # normal loss
        if not train_geometry_stage or epoch >= self.hparams.train_normal_start_epoch:
            loss_dict["normal_loss"] = self.normal_loss(inputs, batch)
            loss_dict["normal_reg_loss"] = self.normal_reg_loss(inputs, batch)

        # remove unused loss
        loss_dict = {k: v for k, v in loss_dict.items() if v != None}

        loss_sum = sum(list(loss_dict.values()))

        return loss_sum, loss_dict


def get_loss(hparams):
    return TotalLoss(hparams)
