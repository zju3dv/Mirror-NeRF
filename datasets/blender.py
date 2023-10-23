import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
import cv2

from .ray_utils import *


class BlenderDataset(Dataset):
    def __init__(self, root_dir, split="train", img_wh=(800, 800), hparams=None):
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.hparams = hparams
        self.define_transforms()
        self.wo_full_gt_mirror_masks = False
        self.train_geometry_stage = self.hparams.train_geometry_stage

        self.read_meta()
        self.white_back = False  # True for object

    def read_meta(self):
        with open(
            os.path.join(self.root_dir, f"transforms_{self.split}.json"), "r"
        ) as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = (
            0.5 * 800 / np.tan(0.5 * self.meta["camera_angle_x"])
        )  # original focal length when W=800

        self.focal *= (
            self.img_wh[0] / 800
        )  # modify focal length to match size self.img_wh

        # bounds, common for all scenes
        self.near = self.hparams.near
        self.far = self.hparams.far
        self.bounds = np.array([self.near, self.far])

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, self.focal)  # (h, w, 3)

        self.direction_orig_norm = torch.norm(self.directions, dim=-1, keepdim=True)

        if self.split == "train":  # create buffer of all rays and rgb data
            # skip frames
            self.meta["frames"] = [
                self.meta["frames"][i]
                for i in np.arange(
                    0, len(self.meta["frames"]), self.hparams.train_skip_step
                )
            ]

            # self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            self.all_mirror_masks = []
            self.all_depths = []

            self.poses_wmask = []
            self.rays_wmask = []
            self.rgbs_wmask = []
            self.mirror_masks_wmask = []
            self.depths_wmask = []

            len_frames = len(self.meta["frames"])
            for idx, frame in enumerate(self.meta["frames"]):
                print(
                    "\rRead meta {:05d} : {:05d}".format(
                        idx,
                        len_frames - 1,
                    ),
                    end="",
                )

                sample = self.read_frame_data(frame)

                self.poses += [sample["pose"]]
                # self.image_paths += [sample["image_path"]]
                self.all_rgbs += [sample["rgbs"]]
                self.all_rays += [sample["rays"]]
                self.all_mirror_masks += [sample["mirror_mask"]]

                if (sample["mirror_mask"] < 0).any() == False:
                    self.poses_wmask += [sample["pose"]]
                    self.rgbs_wmask += [sample["rgbs"]]
                    self.rays_wmask += [sample["rays"]]
                    self.mirror_masks_wmask += [sample["mirror_mask"]]

            self.all_rays = torch.cat(
                self.all_rays, 0
            )  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(
                self.all_rgbs, 0
            )  # (len(self.meta['frames])*h*w, 3)
            self.all_mirror_masks = torch.cat(
                self.all_mirror_masks, 0
            )  # (len(self.meta['frames])*h*w)
            self.rays_wmask = torch.cat(self.rays_wmask, 0)
            self.rgbs_wmask = torch.cat(self.rgbs_wmask, 0)
            self.mirror_masks_wmask = torch.cat(self.mirror_masks_wmask, 0)

        elif self.split == "val":
            self.val_idx = self.hparams.val_idx

    def define_transforms(self):
        self.transform = T.ToTensor()

    def read_frame_data(self, frame):
        # read camera pose
        pose = np.array(frame["transform_matrix"])
        c2w = torch.FloatTensor(pose)[:3, :4]

        # read image
        image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
        if not os.path.exists(image_path):
            print("Skip file which does not exist:", image_path)
            return None
        img = Image.open(image_path)
        img = img.resize(self.img_wh, Image.LANCZOS)
        img = self.transform(img)  # (c, h, w)
        valid_mask = (img[-1] > 0).flatten()  # (H*W) valid color area
        dim = img.shape[0]
        img = img.view(dim, -1).permute(1, 0)  # (h*w, c)
        if dim == 4:  # RGBA
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB

        # read in mirror mask
        img_file_name = os.path.split(frame["file_path"])[-1]
        self.mirror_mask_dir = os.path.join(self.root_dir, "masks")
        mirror_mask_path = os.path.join(
            self.mirror_mask_dir, f"MirrorMask_{img_file_name[6:]}.png"
        )
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
        mirror_mask = mirror_mask.squeeze()  # (H, W) # float
        mirror_mask = mirror_mask.view(-1)  # (H*W)

        # generate rays
        rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
        rays = torch.cat(
            [
                rays_o,
                rays_d,
                self.near * torch.ones_like(rays_o[:, :1]),
                self.far * torch.ones_like(rays_o[:, :1]),
            ],
            1,
        )  # (H*W, 8)

        sample = {
            "rays": rays,
            "rgbs": img,
            "pose": pose,
            "c2w": c2w,
            "valid_mask": valid_mask,
            "mirror_mask": mirror_mask,
        }
        return sample

    def __len__(self):
        if self.split == "train":
            return (
                len(self.rays_wmask)
                if self.train_geometry_stage
                else len(self.all_rays)
            )
        if self.split == "val":
            return 1  # only validate 8 images (to support <=8 gpus)
        return len(self.meta["frames"])

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
        else:  # create data for each image separately
            if self.split == "val":
                frame = self.meta["frames"][self.val_idx]
            else:
                frame = self.meta["frames"][idx]
            sample = self.read_frame_data(frame)

        return sample
