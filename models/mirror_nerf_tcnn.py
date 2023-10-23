import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import get_encoder

from utils.func import l2_normalize, gradient

import numpy as np
import tinycudann as tcnn


class MirrorNeRFTcnn(nn.Module):
    def __init__(
        self,
        encoding="hashgrid",
        encoding_dir="sphere_harmonics",
        encoding_bg="hashgrid",
        num_layers=2,
        hidden_dim=64,
        geo_feat_dim=15,
        num_layers_color=3,
        hidden_dim_color=64,
        num_layers_bg=2,
        hidden_dim_bg=64,
        bound=1,
        **kwargs,
    ):
        super().__init__()
        self.bound = bound

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        n_levels = 16
        n_features_per_level = 2
        per_level_scale = np.exp2(np.log2(2048 * bound / n_levels) / (n_levels - 1))
        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": per_level_scale,
            },
        )
        self.in_dim = n_levels * n_features_per_level

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim  # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim

            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir)

        color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir + self.geo_feat_dim
            else:
                in_dim = hidden_dim_color

            if l == num_layers_color - 1:
                out_dim = 3  # 3 rgb
            else:
                out_dim = hidden_dim_color

            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)

        # background network
        if kwargs.get("bg_radius", 0):
            self.num_layers_bg = num_layers_bg
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(
                encoding_bg,
                input_dim=2,
                num_levels=4,
                log2_hashmap_size=19,
                desired_resolution=2048,
            )  # much smaller hashgrid

            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.in_dim_bg + self.in_dim_dir
                else:
                    in_dim = hidden_dim_bg

                if l == num_layers_bg - 1:
                    out_dim = 3  # 3 rgb
                else:
                    out_dim = hidden_dim_bg

                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None

        # normal network
        self.predict_normal = kwargs.get("predict_normal", False)
        if self.predict_normal:
            self.num_layers_normal = num_layers
            self.hidden_dim_normal = hidden_dim
            normal_net = []
            for l in range(self.num_layers_normal):
                if l == 0:
                    in_dim = self.geo_feat_dim
                else:
                    in_dim = self.hidden_dim_normal

                if l == self.num_layers_normal - 1:
                    out_dim = 3
                else:
                    out_dim = self.hidden_dim_normal

                normal_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.normal_net = nn.ModuleList(normal_net)

        # mirror_mask network
        self.predict_mirror_mask = kwargs.get("predict_mirror_mask", False)
        if self.predict_mirror_mask:
            self.hidden_dim_is_mirror = hidden_dim // 2
            self.is_mirror_net = nn.Sequential(
                nn.Linear(self.geo_feat_dim, self.hidden_dim_is_mirror),
                nn.LeakyReLU(inplace=True),
                nn.Linear(self.hidden_dim_is_mirror, 1),
                nn.Sigmoid(),
            )

    def forward(
        self,
        x,
        compute_normal=True,
        sigma_only=False,
        embedding_xyz=None,
        embedding_dir=None,
        mirror_mask=None,
        detach_density_outside_mirror_for_mask_loss=False,
        detach_density_for_mask_loss=False,
        detach_density_for_normal_loss=False,
    ):
        output_dict = {}

        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        if not sigma_only:
            x, d = torch.split(x, [3, 3], dim=-1)
        else:
            x = x

        if compute_normal:
            # grad normal
            x.requires_grad_(True)
            with torch.enable_grad():
                sigma, geo_feat = self.forward_density(x)
            grad_density = gradient(x, sigma, normalize=False)
            normal = l2_normalize(-grad_density)
            output_dict["normal"] = normal
        else:
            sigma, geo_feat = self.forward_density(x)
        output_dict["sigma"] = sigma
        output_dict["geo_feat"] = geo_feat

        if self.predict_normal:
            # pred normal
            predict_normal = l2_normalize(
                self.forward_normal(
                    geo_feat.detach() if detach_density_for_normal_loss else geo_feat
                )
            )
            output_dict["pred_normal"] = predict_normal

        if not sigma_only:
            # color
            output_dict["rgb"] = self.forward_color(geo_feat, d)

            # mirror mask
            if hasattr(self, "is_mirror_net"):
                if detach_density_for_mask_loss:
                    output_dict["is_mirror"] = self.forward_is_mirror(geo_feat.detach())
                elif (
                    detach_density_outside_mirror_for_mask_loss
                    and mirror_mask is not None
                    and (mirror_mask < 0).any() == False
                ):
                    mirror_mask = mirror_mask.clone().bool()
                    geo_feat_for_mirror = geo_feat.clone()
                    geo_feat_for_mirror[~mirror_mask] = geo_feat_for_mirror[
                        ~mirror_mask
                    ].detach()
                    output_dict["is_mirror"] = self.forward_is_mirror(
                        geo_feat_for_mirror
                    )
                else:
                    output_dict["is_mirror"] = self.forward_is_mirror(geo_feat)

        return output_dict

    def forward_density(self, x):
        # x: [N, 3], in [-bound, bound]

        # sigma
        x = (x + self.bound) / (2 * self.bound)  # to [0, 1]
        h = self.encoder(
            x
        ).float()  # same as https://github.dev/bennyguo/instant-nsr-pl
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        # sigma = F.relu(h[..., 0])
        sigma = h[..., 0]
        geo_feat = h[..., 1:]
        return sigma, geo_feat

    def forward_color(self, geo_feat, d):
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        # sigmoid activation for rgb
        color = torch.sigmoid(h)
        return color

    def forward_normal(self, geo_feat):
        nh = geo_feat
        for l in range(self.num_layers_normal):
            nh = self.normal_net[l](nh)
            if l != self.num_layers_normal - 1:
                nh = F.relu(nh, inplace=True)
        return nh

    def forward_is_mirror(self, geo_feat):
        is_mirror = self.is_mirror_net(geo_feat)
        return is_mirror

    # optimizer utils
    def get_params(self, lr):
        params = [
            {"params": self.encoder.parameters(), "lr": lr},
            {"params": self.sigma_net.parameters(), "lr": lr},
            {"params": self.encoder_dir.parameters(), "lr": lr},
            {"params": self.color_net.parameters(), "lr": lr},
        ]
        if self.bg_radius > 0:
            params.append({"params": self.encoder_bg.parameters(), "lr": lr})
            params.append({"params": self.bg_net.parameters(), "lr": lr})
        if self.predict_normal:
            params.append({"params": self.normal_net.parameters(), "lr": lr})
        if self.predict_mirror_mask:
            params.append({"params": self.is_mirror_net.parameters(), "lr": lr})
        return params
