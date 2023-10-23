import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
import cv2

from .ray_utils import *
from .geo_utils import *
from .colmap_utils import (
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
    read_dense_bin_array,
)


class RealDatasetColmap(Dataset):
    def __init__(self, root_dir, split="train", img_wh=(800, 800), hparams=None):
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.hparams = hparams
        self.define_transforms()
        self.wo_full_gt_mirror_masks = False
        self.train_geometry_stage = self.hparams.train_geometry_stage
        self.white_back = False
        self.spheric_poses = True

        self.read_meta()

    def define_transforms(self):
        self.transform = T.ToTensor()

    def read_frame_data(self, c2w, image_path, no_data_when_test=False):
        # generate rays
        rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
        if not self.spheric_poses:
            near, far = 0, 1
            rays_o, rays_d = get_ndc_rays(
                self.img_wh[1], self.img_wh[0], self.focal, 1.0, rays_o, rays_d
            )
            # near plane is always at 1.0
            # near and far in NDC are always 0 and 1
            # See https://github.com/bmild/nerf/issues/34
        else:
            # near = self.bounds.min()
            # far = min(8 * near, self.bounds.max())  # focus on central object only
            near = self.hparams.near / self.hparams.scale_factor
            far = self.hparams.far / self.hparams.scale_factor

        rays = torch.cat(
            [
                rays_o,
                rays_d,
                near * torch.ones_like(rays_o[:, :1]),
                far * torch.ones_like(rays_o[:, :1]),
            ],
            1,
        )  # (h*w, 8)

        if no_data_when_test:
            sample = {
                "rays": rays,
                "c2w": c2w,
            }
        else:
            # read image
            img = Image.open(image_path).convert("RGB")
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (3, h, w)
            img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB

            # read in mirror mask
            img_file_name = os.path.split(image_path)[-1]
            self.mirror_mask_dir = os.path.join(self.root_dir, "masks")
            mirror_mask_path = os.path.join(self.mirror_mask_dir, img_file_name)
            mirror_mask = cv2.imread(mirror_mask_path, cv2.IMREAD_ANYDEPTH)
            if mirror_mask is None:
                print(f"[warning] mirror_mask not exist:{mirror_mask_path}")
                self.wo_full_gt_mirror_masks = True
                # use -1 to mark invalid GT mirror mask
                mirror_mask = np.ones((self.img_wh[1], self.img_wh[0])) * -1
                mirror_mask = self.transform(mirror_mask)
            else:
                mirror_mask = cv2.resize(
                    mirror_mask, self.img_wh, interpolation=cv2.INTER_NEAREST
                )
                mirror_mask = self.transform(mirror_mask)
                mirror_mask[mirror_mask < 0.5] = 0
                mirror_mask[mirror_mask > 0.5] = 1
            mirror_mask = mirror_mask.squeeze()  # (h, w) # float
            mirror_mask = mirror_mask.view(-1)  # (h*w)

            sample = {
                "rays": rays,
                "c2w": c2w,
                "rgbs": img,
                "mirror_mask": mirror_mask,
            }
        return sample

    def read_meta(self):
        # Step 1: rescale focal length according to training resolution
        camdata = read_cameras_binary(os.path.join(self.root_dir, "sparse/cameras.bin"))
        H = camdata[1].height
        W = camdata[1].width
        self.focal = camdata[1].params[0] * self.img_wh[0] / W
        # Step 2: correct poses
        # read extrinsics (of successfully reconstructed images)
        imdata = read_images_binary(os.path.join(self.root_dir, "sparse/images.bin"))
        perm = np.argsort([imdata[k].name for k in imdata])
        # read successfully reconstructed images and ignore others
        self.image_paths = [
            os.path.join(self.root_dir, "images", name)
            for name in sorted([imdata[k].name for k in imdata])
        ]
        print("[Info] len(image_paths):", len(self.image_paths))
        w2c_mats = []
        bottom = np.array([0, 0, 0, 1.0]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat()
            t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        w2c_mats = np.stack(w2c_mats, 0)
        poses = np.linalg.inv(w2c_mats)[:, :3]  # (N_images, 3, 4) cam2world matrices

        # read bounds
        self.bounds = np.zeros((len(poses), 2))  # (N_images, 2)

        # # if reconstructed_camera_poses from colmap are not complete, do not use points3D.bin.
        # pts3d = read_points3d_binary(os.path.join(self.root_dir, "sparse/points3D.bin"))
        # pts_world = np.zeros((1, 3, len(pts3d)))  # (1, 3, N_points)
        # visibilities = np.zeros((len(poses), len(pts3d)))  # (N_images, N_points)
        # for i, k in enumerate(pts3d):
        #     pts_world[0, :, i] = pts3d[k].xyz
        #     for j in pts3d[k].image_ids:
        #         visibilities[j - 1, i] = 1
        # # calculate each point's depth w.r.t. each camera
        # # it's the dot product of "points - camera center" and "camera frontal axis"
        # depths = ((pts_world - poses[..., 3:4]) * poses[..., 2:3]).sum(
        #     1
        # )  # (N_images, N_points)
        # for i in range(len(poses)):
        #     visibility_i = visibilities[i]
        #     zs = depths[i][visibility_i == 1]
        #     self.bounds[i] = [np.percentile(zs, 0.1), np.percentile(zs, 99.9)]

        # permute the matrices to increasing order
        poses = poses[perm]
        self.bounds = self.bounds[perm]
        # use user-defined near far to avoid being affected by outliers from colmap.
        self.bounds[:, 0] = self.hparams.near
        self.bounds[:, 1] = self.hparams.far

        # COLMAP poses has rotation in form "right down front", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 0:1], -poses[..., 1:3], poses[..., 3:4]], -1)
        self.poses, self.pose_avg = center_poses(poses)  # self.poses = poses
        # distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        # val_idx = np.argmin(
        #     distances_from_center
        # )  # choose val image as the closest to center image
        val_idx = self.hparams.val_idx

        # # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # # See https://github.com/bmild/nerf/issues/34
        # near_original = self.bounds.min()
        # scale_factor = near_original * 0.75  # 0.75 is the default parameter
        # # the nearest depth is at 1/0.75=1.33
        scale_factor = self.hparams.scale_factor
        self.bounds /= scale_factor
        self.poses[..., 3] /= scale_factor

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(
            self.img_wh[1], self.img_wh[0], self.focal
        )  # (H, W, 3)

        if self.split == "train":  # create data buffer of all rays
            # skip frames
            if self.hparams.train_skip_step != 1:
                self.image_paths = [
                    self.image_paths[i]
                    for i in np.arange(
                        0, len(self.image_paths), self.hparams.train_skip_step
                    )
                ]
                self.poses = self.poses[:: self.hparams.train_skip_step, ...]
                self.bounds = self.bounds[:: self.hparams.train_skip_step, ...]

            # use first N_images-1 to train, the LAST is val
            self.all_rays = []
            self.all_rgbs = []
            self.all_mirror_masks = []
            self.all_depths = []

            self.rays_wmask = []
            self.rgbs_wmask = []
            self.mirror_masks_wmask = []
            self.depths_wmask = []

            len_images = len(self.image_paths)
            for i, image_path in enumerate(self.image_paths):
                # exclude the val image  # val_idx is reletive to the skipped training set.
                if i == val_idx:
                    continue
                print(
                    "\rRead meta {:05d} : {:05d}".format(
                        i,
                        len_images - 1,
                    ),
                    end="",
                )

                c2w = torch.FloatTensor(self.poses[i])
                sample = self.read_frame_data(c2w, image_path)

                self.all_rays += [sample["rays"]]
                self.all_rgbs += [sample["rgbs"]]
                self.all_mirror_masks += [sample["mirror_mask"]]

                if (sample["mirror_mask"] < 0).any() == False:
                    self.rgbs_wmask += [sample["rgbs"]]
                    self.rays_wmask += [sample["rays"]]
                    self.mirror_masks_wmask += [sample["mirror_mask"]]

            self.all_rays = torch.cat(self.all_rays, 0)  # ((N_images)*h*w, 8)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # ((N_images)*h*w, 3)
            self.all_mirror_masks = torch.cat(
                self.all_mirror_masks, 0
            )  # ((N_images)*h*w)

            self.rays_wmask = torch.cat(self.rays_wmask, 0)
            self.rgbs_wmask = torch.cat(self.rgbs_wmask, 0)
            self.mirror_masks_wmask = torch.cat(self.mirror_masks_wmask, 0)

        elif self.split == "val":
            print("val image is", self.image_paths[val_idx])
            self.val_idx = val_idx

        elif (
            self.split == "test" or self.split == "test_train"
        ):  # for testing, create a parametric rendering path
            if self.split.endswith("train"):  # test on training set
                self.poses_test = self.poses
            elif not self.spheric_poses:
                focus_depth = 3.5  # hardcoded, this is numerically close to the formula
                # given in the original repo. Mathematically if near=1
                # and far=infinity, then this number will converge to 4
                radii = np.percentile(np.abs(self.poses[..., 3]), 90, axis=0)
                self.poses_test = create_spiral_poses(radii, focus_depth)
            else:
                radius = 1.1 * self.bounds.min()
                self.poses_test = create_spheric_poses(radius)

    def __len__(self):
        if self.split == "train":
            return (
                len(self.rays_wmask)
                if self.train_geometry_stage
                else len(self.all_rays)
            )
        elif self.split == "val":
            return 1  # only validate 8 images (to support <=8 gpus)
        elif self.split == "test_train":
            return len(self.poses)
        elif self.split == "test":
            return len(self.poses_test)
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.split == "train":  # use data in the buffers
            if self.train_geometry_stage:
                sample = {
                    "rays": self.rays_wmask[idx],
                    "rgbs": self.rgbs_wmask[idx],
                    "mirror_mask": self.mirror_masks_wmask[idx],
                }
            else:
                sample = {
                    "rays": self.all_rays[idx],
                    "rgbs": self.all_rgbs[idx],
                    "mirror_mask": self.all_mirror_masks[idx],
                }
        else:
            if self.split == "val":
                c2w = torch.FloatTensor(self.poses[self.val_idx])
            elif self.split == "test":
                c2w = torch.FloatTensor(self.poses_test[idx])
            else:
                c2w = torch.FloatTensor(self.poses[idx])

            if self.split == "test":
                sample = self.read_frame_data(c2w, None, no_data_when_test=True)
            else:
                if self.split == "val":
                    idx = self.val_idx
                sample = self.read_frame_data(c2w, self.image_paths[idx])

        return sample
