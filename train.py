import os
import time
import json

from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.mirror_nerf import *
from models.rendering import *

# optimizer, scheduler, visualization
from utils import *
from utils.func import l2_normalize

# losses
from losses import get_loss

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        if self.hparams.for_vis:
            self.hparams.noise_std = 0
            self.hparams.perturb = 0
        self.train_geometry_stage = self.hparams.train_geometry_stage  # record state

        self.loss = get_loss(self.hparams)

        if self.hparams.model_type == "nerf":
            self.embedding_xyz = Embedding(self.hparams.N_emb_xyz)
            self.embedding_dir = Embedding(self.hparams.N_emb_dir)
            self.embeddings = {"xyz": self.embedding_xyz, "dir": self.embedding_dir}

            self.nerf_coarse = MirrorNeRF(
                in_channels_xyz=6 * self.hparams.N_emb_xyz + 3,
                in_channels_dir=6 * self.hparams.N_emb_dir + 3,
                predict_normal=self.hparams.predict_normal,
                predict_mirror_mask=self.hparams.predict_mirror_mask,
            )
            self.models = {"coarse": self.nerf_coarse}
            load_ckpt(self.nerf_coarse, self.hparams.weight_path, "nerf_coarse")

            if self.hparams.N_importance > 0 and not self.hparams.only_one_field:
                self.nerf_fine = MirrorNeRF(
                    in_channels_xyz=6 * self.hparams.N_emb_xyz + 3,
                    in_channels_dir=6 * self.hparams.N_emb_dir + 3,
                    predict_normal=self.hparams.predict_normal,
                    predict_mirror_mask=self.hparams.predict_mirror_mask,
                )
                self.models["fine"] = self.nerf_fine
                load_ckpt(self.nerf_fine, self.hparams.weight_path, "nerf_fine")
        elif self.hparams.model_type == "nerf_tcnn":
            from models.mirror_nerf_tcnn import MirrorNeRFTcnn
            self.embedding_xyz = Embedding(0)
            self.embedding_dir = Embedding(0)
            self.embeddings = {"xyz": self.embedding_xyz, "dir": self.embedding_dir}

            self.nerf_coarse = MirrorNeRFTcnn(
                encoding="hashgrid",
                bound=self.hparams.bound,
                cuda_ray=False,
                density_scale=1,
                min_near=0.2,
                density_thresh=10,
                bg_radius=False,
                predict_normal=self.hparams.predict_normal,
                predict_mirror_mask=self.hparams.predict_mirror_mask,
            )
            self.models = {"coarse": self.nerf_coarse}
            load_ckpt(self.nerf_coarse, self.hparams.weight_path, "nerf_coarse")

            if self.hparams.N_importance > 0 and not self.hparams.only_one_field:
                self.nerf_fine = MirrorNeRFTcnn(
                    encoding="hashgrid",
                    bound=self.hparams.bound,
                    cuda_ray=False,
                    density_scale=1,
                    min_near=0.2,
                    density_thresh=10,
                    bg_radius=False,
                    predict_normal=self.hparams.predict_normal,
                    predict_mirror_mask=self.hparams.predict_mirror_mask,
                )
                self.models["fine"] = self.nerf_fine
                load_ckpt(self.nerf_fine, self.hparams.weight_path, "nerf_fine")
    
    def forward(self, rays, extra=dict()):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            extra_chunk = dict()
            for k, v in extra.items():
                if isinstance(v, torch.Tensor):
                    extra_chunk[k] = v[i : i + self.hparams.chunk]
                else:
                    extra_chunk[k] = v

            rays_chunk = rays[i : i + self.hparams.chunk]
            mirror_mask_first = (
                torch.ones(rays_chunk.shape[0]).bool().to(rays_chunk.device)
            )
            rendered_ray_chunks = self.render_rays_chunk_recursively(
                rays_chunk, mirror_mask_first, recur_level=0, **extra_chunk
            )

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def render_rays_chunk_recursively(
        self, rays_chunk, mirror_mask_prev, recur_level=0, **extra_chunk
    ):
        rendered_ray_chunks = render_rays(
            self.models,
            self.embeddings,
            rays_chunk,
            self.hparams.N_samples,
            self.hparams.use_disp,
            self.hparams.perturb,
            self.hparams.noise_std,
            self.hparams.N_importance,
            self.hparams.chunk,  # chunk size is effective in val mode
            self.train_dataset.white_back,
            compute_normal=self.hparams.trace_secondary_rays,
            **extra_chunk,
        )

        select_type = (
            "fine"
            if (self.hparams.N_importance > 0 and not self.hparams.only_one_field)
            else "coarse"
        )

        #############################
        # Config mirror_mask
        mirror_mask = extra_chunk["mirror_mask"].clone()
        # if GT mirror mask not available (invalid GT mirror mask or for traced rays)
        if (mirror_mask < 0).any() or recur_level > 0:
            if "mirror_mask_fine" in rendered_ray_chunks:
                mirror_mask = rendered_ray_chunks["mirror_mask_fine"].detach()
            elif "mirror_mask_coarse" in rendered_ray_chunks:
                mirror_mask = rendered_ray_chunks["mirror_mask_coarse"].detach()
            else:
                mirror_mask = 0
                # Then mirror_mask.bool().any()=False, then trace_secondary_rays=False.
            mirror_mask[mirror_mask > 0.5] = 1
            mirror_mask[mirror_mask < 0.5] = 0
        if (not self.hparams.only_trace_rays_in_mirrors) and recur_level > 0:
            mirror_mask = mirror_mask * mirror_mask_prev.detach()

        #############################
        # Config trace_secondary_rays
        trace_secondary_rays = (
            self.hparams.trace_secondary_rays
            and (not self.train_geometry_stage)
            and (mirror_mask.bool().any() or self.hparams.for_vis)
        )  # use mirror_mask.any() to reduce computation cost.
        if recur_level >= self.hparams.max_recursive_level:
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
                        * rendered_ray_chunks[f"weights_{select_type}"].unsqueeze(-1)
                    ).sum(1, keepdim=False)
                )  # not detach() to jointly optimize.
                # normal is correct, not -normal (n.dot(w) should be positive.)
            else:  # use grad normal
                normal = (
                    rendered_ray_chunks[f"surface_normal_grad_{select_type}"]
                    if f"surface_normal_grad_{select_type}" in rendered_ray_chunks
                    else (
                        rendered_ray_chunks[f"normal_{select_type}"]
                        * rendered_ray_chunks[f"weights_{select_type}"].unsqueeze(-1)
                    ).sum(1, keepdim=False)
                )

            #############################
            # Calculate reflect_direction
            normal = l2_normalize(
                normal.detach() if self.hparams.detach_normal_in_reflection else normal
            )
            incident_inv_direction = l2_normalize(
                -rays_d_chunk
            )  # note: incident_inv_direction should be outward
            cos = torch.sum(incident_inv_direction * normal, dim=-1)
            reflect_direction = (
                2 * cos.unsqueeze(dim=-1).repeat(1, 3) * normal - incident_inv_direction
            )  # (N_rays, 3)  # 2*n.dot(w)*n - w (normal and incident_direction must be normalized.)

            #############################
            # Organize secondary_rays
            ray_forward_offset = 0.1  # 0.05  # the origin of the secondary ray should not be on the mirror surface.
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

            only_trace_rays_in_mirrors = self.hparams.only_trace_rays_in_mirrors
            if only_trace_rays_in_mirrors:
                secondary_rays = secondary_rays[
                    mirror_mask.bool()
                ]  # [N_rays_in_mirror, 8]
            if secondary_rays.shape[0] > 0:
                rendered_secondary_ray_chunks = self.render_rays_chunk_recursively(
                    secondary_rays,
                    mirror_mask,
                    recur_level=recur_level + 1,
                    **extra_chunk,
                )

                #############################
                # Blend colors by mirror_mask
                for typ in ["coarse", "fine"]:
                    if (
                        f"rgb_{typ}" in rendered_ray_chunks
                        and f"rgb_{typ}" in rendered_secondary_ray_chunks
                    ):
                        rendered_ray_chunks[f"rgb_{typ}_direct"] = rendered_ray_chunks[
                            f"rgb_{typ}"
                        ]

                        base_color = rendered_ray_chunks[f"rgb_{typ}"]
                        if only_trace_rays_in_mirrors:
                            reflection_part = base_color.clone().detach()
                            reflection_part[
                                mirror_mask.bool()
                            ] = rendered_secondary_ray_chunks[
                                f"rgb_{typ}"
                            ]  # (N_rays, 3)
                        else:
                            reflection_part = rendered_secondary_ray_chunks[
                                f"rgb_{typ}"
                            ]  # (N_rays, 3)
                        if (
                            self.hparams.detach_ref_color_for_blend
                            and self.current_epoch
                            >= self.hparams.train_geometry_stage_end_epoch + 1
                        ):
                            reflection_part = reflection_part.detach()
                        mirror_mask_ = (
                            mirror_mask.float().unsqueeze(dim=-1).repeat(1, 3)
                        )  # (N_rays, 3)

                        rendered_ray_chunks[f"rgb_{typ}"] = (
                            mirror_mask_ * reflection_part + (1 - mirror_mask_) * base_color
                        )

                        # for visualization
                        if extra_chunk["is_eval"]:
                            if only_trace_rays_in_mirrors:
                                rendered_ray_chunks[
                                    f"rgb_{typ}_reflect"
                                ] = torch.zeros_like(rendered_ray_chunks[f"rgb_{typ}"])
                                rendered_ray_chunks[f"rgb_{typ}_reflect"][
                                    mirror_mask.bool()
                                ] = rendered_secondary_ray_chunks[f"rgb_{typ}"]
                            else:
                                rendered_ray_chunks[
                                    f"rgb_{typ}_reflect"
                                ] = rendered_secondary_ray_chunks[f"rgb_{typ}"]
                if extra_chunk["is_eval"]:
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
                    rendered_ray_chunks[f"secondary_rays_o"] = secondary_rays_o
                    rendered_ray_chunks[f"reflect_direction"] = reflect_direction
        else:  # this chunk of mirror_mask is all "False"
            # for visualization
            if extra_chunk["is_eval"]:
                for typ in ["coarse", "fine"]:
                    if f"rgb_{typ}" in rendered_ray_chunks:
                        rendered_ray_chunks[f"rgb_{typ}_reflect"] = torch.zeros_like(
                            rendered_ray_chunks[f"rgb_{typ}"]
                        )
                        rendered_ray_chunks[f"rgb_{typ}_direct"] = torch.zeros_like(
                            rendered_ray_chunks[f"rgb_{typ}"]
                        )
                rendered_ray_chunks[f"depth_{select_type}_reflect"] = torch.zeros_like(
                    rendered_ray_chunks[f"depth_{select_type}"]
                )
                rendered_ray_chunks[f"secondary_rays_o"] = torch.zeros_like(
                    rendered_ray_chunks[f"rgb_{select_type}"]
                )
                rendered_ray_chunks[f"reflect_direction"] = torch.zeros_like(
                    rendered_ray_chunks[f"rgb_{select_type}"]
                )

        return rendered_ray_chunks

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {
            "root_dir": self.hparams.root_dir,
            "img_wh": tuple(self.hparams.img_wh),
            "hparams": self.hparams,
        }
        if self.hparams.dataset_name == "llff":
            kwargs["spheric_poses"] = self.hparams.spheric_poses
            kwargs["val_num"] = self.hparams.num_gpus
        self.train_dataset = dataset(split="train", **kwargs)
        self.val_dataset = dataset(split="val", **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        self.scheduler = get_scheduler(self.hparams, self.optimizer)
        return [self.optimizer], [self.scheduler]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=4,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=4,
            batch_size=1,  # validate one image (H*W rays) at a time
            pin_memory=True,
        )

    def training_step(self, batch, batch_nb):
        if (
            self.train_geometry_stage
            and self.current_epoch >= self.hparams.train_geometry_stage_end_epoch
        ):
            self.train_geometry_stage = False
        if (
            self.train_dataset.train_geometry_stage
            and self.current_epoch >= self.hparams.train_geometry_stage_end_epoch - 1
        ):
            self.train_dataset.train_geometry_stage = False
            # Wait for reloading dataloader. Note that train_dataloader() is called before training_step in one epoch, and reload_dataloaders_every_n_epochs=hparams.train_geometry_stage_end_epoch.
        if (
            self.val_dataset.train_geometry_stage
            and self.current_epoch >= self.hparams.train_geometry_stage_end_epoch - 1
        ):
            self.val_dataset.train_geometry_stage = False
            # Wait for reloading dataloader. Note that train_dataloader() is called before training_step in one epoch, and reload_dataloaders_every_n_epochs=hparams.train_geometry_stage_end_epoch.

        if (
            batch["mirror_mask"] < 0
        ).any() and self.current_epoch <= self.hparams.train_mirror_mask_start_epoch:
            return None  # (the step will be skipped)

        # Mask RGB inside mirror in train_geometry_stage.
        if (
            self.train_geometry_stage
            and not (batch["mirror_mask"] < 0).any()
            and not self.hparams.woMaskRGBtoBlack
        ):
            batch["rgbs"][batch["mirror_mask"].bool()] = 0

        rays, rgbs = batch["rays"].squeeze(), batch["rgbs"].squeeze()

        extra_info = dict()
        extra_info["is_eval"] = False
        extra_info["mirror_mask"] = batch["mirror_mask"].squeeze()
        extra_info["only_one_field"] = self.hparams.only_one_field
        extra_info["only_one_field_fine_epoch"] = self.hparams.only_one_field_fine_epoch
        extra_info["current_epoch"] = self.current_epoch
        extra_info["train_geometry_stage"] = self.train_geometry_stage
        extra_info[
            "detach_density_outside_mirror_for_mask_loss"
        ] = self.hparams.detach_density_outside_mirror_for_mask_loss
        extra_info[
            "detach_density_for_mask_loss"
        ] = self.hparams.detach_density_for_mask_loss
        extra_info[
            "detach_density_for_normal_loss"
        ] = self.hparams.detach_density_for_normal_loss

        results = self(rays, extra_info)

        loss_sum, loss_dict = self.loss(
            results,
            batch,
            self.train_geometry_stage,
            self.current_epoch,
        )

        with torch.no_grad():
            typ = "fine" if "rgb_fine" in results else "coarse"
            psnr_ = psnr(results[f"rgb_{typ}"], rgbs)
            psnr_coarse = psnr(results[f"rgb_coarse"], rgbs)

        self.log("lr", get_learning_rate(self.optimizer))
        self.log("train/loss", loss_sum)
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v)
        self.log("train/psnr", psnr_, prog_bar=True)
        self.log("train/psnr_coarse", psnr_coarse)

        return loss_sum

    def validation_step(self, batch, batch_nb):
        rays, rgbs = batch["rays"], batch["rgbs"]
        if (
            self.train_geometry_stage
            and self.current_epoch >= self.hparams.train_geometry_stage_end_epoch
        ):
            self.train_geometry_stage = False
        if (
            self.train_dataset.train_geometry_stage
            and self.current_epoch >= self.hparams.train_geometry_stage_end_epoch - 1
        ):
            self.train_dataset.train_geometry_stage = False
            # Wait for reloading dataloader. Note that train_dataloader() is called before training_step in one epoch, and reload_dataloaders_every_n_epochs=hparams.train_geometry_stage_end_epoch.
        if (
            self.val_dataset.train_geometry_stage
            and self.current_epoch >= self.hparams.train_geometry_stage_end_epoch - 1
        ):
            self.val_dataset.train_geometry_stage = False
            # Wait for reloading dataloader. Note that train_dataloader() is called before training_step in one epoch, and reload_dataloaders_every_n_epochs=hparams.train_geometry_stage_end_epoch.

        # Mask RGB inside mirror in train_geometry_stage.
        if (
            self.train_geometry_stage
            and not (batch["mirror_mask"] < 0).any()
            and not self.hparams.woMaskRGBtoBlack
        ):
            batch["rgbs"][batch["mirror_mask"].bool()] = 0

        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)

        extra_info = dict()
        extra_info["is_eval"] = True
        extra_info["mirror_mask"] = batch["mirror_mask"].squeeze()
        extra_info["only_one_field"] = self.hparams.only_one_field
        extra_info["only_one_field_fine_epoch"] = self.hparams.only_one_field_fine_epoch
        extra_info["current_epoch"] = self.current_epoch
        extra_info["train_geometry_stage"] = self.train_geometry_stage
        extra_info[
            "detach_density_outside_mirror_for_mask_loss"
        ] = self.hparams.detach_density_outside_mirror_for_mask_loss
        extra_info[
            "detach_density_for_mask_loss"
        ] = self.hparams.detach_density_for_mask_loss
        extra_info[
            "detach_density_for_normal_loss"
        ] = self.hparams.detach_density_for_normal_loss

        results = self(rays, extra_info)

        loss_sum, loss_dict = self.loss(
            results,
            batch,
            self.train_geometry_stage,
        )
        for k, v in loss_dict.items():
            self.log(f"val/{k}", v)
        log = {"val_loss": loss_sum}
        log.update(loss_dict)
        typ = "fine" if "rgb_fine" in results else "coarse"

        if batch_nb == 0:
            stack_image = visualize_val_image(
                self.hparams.img_wh, batch, results, add_text=(not self.hparams.for_vis)
            )  # (N_pics, 3, H, W)
            self.logger.experiment.add_images(
                "val/GT_pred_depth", stack_image, self.global_step
            )

        psnr_ = psnr(results[f"rgb_{typ}"], rgbs)
        log["val_psnr"] = psnr_
        psnr_coarse = psnr(results[f"rgb_coarse"], rgbs)
        log["val_psnr_coarse"] = psnr_coarse

        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        mean_psnr = torch.stack([x["val_psnr"] for x in outputs]).mean()
        mean_psnr_coarse = torch.stack([x["val_psnr_coarse"] for x in outputs]).mean()

        self.log("val/loss", mean_loss)
        self.log("val/psnr", mean_psnr, prog_bar=True)
        self.log("val/psnr_coarse", mean_psnr_coarse)

def main(hparams):
    set_rand_seed()
    exp_name = get_timestamp() + "_" + hparams.exp_name
    print(f"Start with exp_name: {exp_name}.")
    log_path = f"logs/{exp_name}"
    hparams.log_path = log_path
    hparams.debug_dir = log_path + "/debug"

    system = NeRFSystem(hparams)
    ckpt_cb = ModelCheckpoint(
        dirpath=log_path,
        filename="{epoch:d}",
        monitor="val/psnr",
        mode="max",
        # save_top_k=5,
        save_top_k=-1,
        save_last=True,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [ckpt_cb, pbar]

    logger = TensorBoardLogger(save_dir="logs", name=exp_name, default_hp_metric=False)

    if hparams.train_geometry_stage:
        trainer = Trainer(
            max_epochs=hparams.num_epochs,
            callbacks=callbacks,
            resume_from_checkpoint=hparams.ckpt_path,
            logger=logger,
            enable_model_summary=False,
            accelerator="auto",
            devices=hparams.num_gpus,
            num_sanity_val_steps=1,
            benchmark=True,
            profiler="simple" if hparams.num_gpus == 1 else None,
            strategy=DDPPlugin(find_unused_parameters=False)
            if hparams.num_gpus > 1
            else None,
            val_check_interval=0.25,
            precision=16 if hparams.model_type == "nerf_tcnn" else 32,
            reload_dataloaders_every_n_epochs=hparams.train_geometry_stage_end_epoch,
        )
    else:
        trainer = Trainer(
            max_epochs=hparams.num_epochs,
            callbacks=callbacks,
            resume_from_checkpoint=hparams.ckpt_path,
            logger=logger,
            enable_model_summary=False,
            accelerator="auto",
            devices=hparams.num_gpus,
            num_sanity_val_steps=1,
            benchmark=True,
            profiler="simple" if hparams.num_gpus == 1 else None,
            strategy=DDPPlugin(find_unused_parameters=False)
            if hparams.num_gpus > 1
            else None,
            val_check_interval=0.25,
            precision=16 if hparams.model_type == "nerf_tcnn" else 32,
        )

    make_source_code_snapshot(f"logs/{exp_name}")

    trainer.fit(system)


if __name__ == "__main__":
    hparams = get_opts()
    main(hparams)
