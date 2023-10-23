from argparse import ArgumentParser
import os
import json
from PIL import Image
from torchvision import transforms as T

from skimage import metrics
import lpips


def get_opts():
    parser = ArgumentParser()
    parser.add_argument("--split_path", type=str)
    # parser.add_argument("--gt_img_dir", type=str)
    parser.add_argument("--res_img_dir", type=str)
    return parser.parse_args()


def cal_psnr_ssim(p, t):
    """Compute PSNR of model image predictions.
    :param prediction: Return value of forward pass.
    :param ground_truth: Ground truth.
    :return: (psnr, ssim): tuple of floats
    """
    ssim = metrics.structural_similarity(
        p, t, multichannel=True, channel_axis=-1, data_range=1
    )
    psnr = metrics.peak_signal_noise_ratio(p, t, data_range=1)
    return ssim, psnr


loss_fn_alex = lpips.LPIPS(net="alex") 

def load_image_to_tensor(path, resize_wh=None):  # return img in [0,1]
    if not os.path.exists(path):
        print("cannot open", path)

    img = Image.open(path)
    if resize_wh != None:
        img = img.resize(resize_wh, Image.LANCZOS)
    img = T.ToTensor()(img)
    img_wh = (img.shape[2], img.shape[1])
    dim = img.shape[0]
    if dim == 4:  # RGBA
        img = img[:3, ...] * img[-1:, ...] + (1 - img[-1:, ...])  # blend A to RGB
    return img, img_wh


if __name__ == "__main__":
    args = get_opts()
    with open(os.path.join(args.split_path), "r") as f:
        meta = json.load(f)
    root_dir = os.path.split(args.split_path)[0]
    frames = meta["frames"]
    all_psnr = []
    all_ssim = []
    all_lpips = []
    N = 0
    for idx, frame in enumerate(frames):
        res_img_path = os.path.join(args.res_img_dir, f"rgb_fine_{idx:03d}.png")
        res_img, res_img_wh = load_image_to_tensor(res_img_path)

        file_path = (
            f"{frame['file_path']}.png"
            if "mirror_syn_scene" in root_dir
            else frame["file_path"]
        )
        gt_img_path = os.path.join(root_dir, file_path)
        gt_img, _ = load_image_to_tensor(gt_img_path, resize_wh=res_img_wh)

        ssim, psnr = cal_psnr_ssim(
            res_img.permute(1, 2, 0).numpy(), gt_img.permute(1, 2, 0).numpy()
        )  # input img should be in [0,1]
        all_psnr.append(psnr)
        all_ssim.append(ssim)
        all_lpips.append(
            loss_fn_alex(res_img * 2 - 1, gt_img * 2 - 1).squeeze().item()
        )  # input img should be in [-1,1]
        N += 1

    print(
        "Mean PSNR {} SSIM {} LPIPS {}".format(
            sum(all_psnr) / N, sum(all_ssim) / N, sum(all_lpips) / N
        )
    )
