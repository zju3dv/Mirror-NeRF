import torch
from torch import nn
from utils.func import l2_normalize, gradient


class Embedding(nn.Module):
    def __init__(self, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super().__init__()
        self.N_freqs = N_freqs
        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, f)

        Outputs:
            out: (B, 2*f*N_freqs+f)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]

        return torch.cat(out, -1)


class MirrorNeRF(nn.Module):
    def __init__(
        self, D=8, W=256, in_channels_xyz=63, in_channels_dir=27, skips=[4], **kwargs
    ):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(MirrorNeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W + in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.geo_feat_dim = W
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
            nn.Linear(W + in_channels_dir, W // 2), nn.ReLU(True)
        )

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Sequential(nn.Linear(W // 2, 3), nn.Sigmoid())

        # normal network
        self.predict_normal = kwargs.get("predict_normal", False)
        if self.predict_normal:
            self.hidden_dim_normal = W // 2
            self.normal_net = nn.Sequential(
                nn.Linear(self.geo_feat_dim, self.hidden_dim_normal),
                nn.Linear(self.hidden_dim_normal, 3),
            )

        # mirror_mask network
        self.predict_mirror_mask = kwargs.get("predict_mirror_mask", False)
        if self.predict_mirror_mask:
            self.hidden_dim_is_mirror = W // 2
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
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        output_dict = {}
        if not sigma_only:
            in_xyz, input_dir = torch.split(x, [3, self.in_channels_dir], dim=-1)
        else:
            in_xyz = x

        global time1, time2, time1_cnt, time2_cnt
        if compute_normal:
            # grad normal
            in_xyz.requires_grad_(True)
            with torch.enable_grad():
                input_xyz = (
                    embedding_xyz(in_xyz) if embedding_xyz is not None else in_xyz
                )
                sigma, geo_feat = self.forward_density(input_xyz)
            grad_density = gradient(in_xyz, sigma, normalize=False)
            normal = l2_normalize(-grad_density)
            output_dict["normal"] = normal
        else:
            input_xyz = embedding_xyz(in_xyz) if embedding_xyz is not None else in_xyz
            sigma, geo_feat = self.forward_density(input_xyz)

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
            output_dict["rgb"] = self.forward_color(geo_feat, input_dir)

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

    def forward_density(self, input_xyz):
        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)
        geo_feat = xyz_
        sigma = self.sigma(xyz_)
        return sigma, geo_feat

    def forward_color(self, geo_feat, input_dir):
        xyz_encoding_final = self.xyz_encoding_final(geo_feat)
        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)
        return rgb

    def forward_normal(self, geo_feat):
        normal = self.normal_net(geo_feat)
        return normal

    def forward_is_mirror(self, geo_feat):
        is_mirror = self.is_mirror_net(geo_feat)
        return is_mirror
