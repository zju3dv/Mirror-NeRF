import argparse


def get_opts(b_parse_args=True):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/ubuntu/data/nerf_example_data/nerf_synthetic/lego",
        help="root directory of dataset",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="blender",
        choices=["blender", "llff", "real_colmap", "real_arkit"],
        help="which dataset to train/val",
    )
    parser.add_argument(
        "--img_wh",
        nargs="+",
        type=int,
        default=[800, 800],
        help="resolution (img_w, img_h) of the image",
    )
    parser.add_argument(
        "--spheric_poses",
        default=False,
        action="store_true",
        help="whether images are taken in spheric poses (for llff)",
    )

    parser.add_argument(
        "--N_emb_xyz",
        type=int,
        default=10,
        help="number of frequencies in xyz positional encoding",
    )
    parser.add_argument(
        "--N_emb_dir",
        type=int,
        default=4,
        help="number of frequencies in dir positional encoding",
    )
    parser.add_argument(
        "--N_samples", type=int, default=64, help="number of coarse samples"
    )
    parser.add_argument(
        "--N_importance",
        type=int,
        default=128,
        help="number of additional fine samples",
    )
    parser.add_argument(
        "--use_disp",
        default=False,
        action="store_true",
        help="use disparity depth sampling",
    )
    parser.add_argument(
        "--perturb",
        type=float,
        default=1.0,
        help="factor to perturb depth sampling points",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=1.0,
        help="std dev of noise added to regularize sigma",
    )

    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument(
        "--chunk",
        type=int,
        default=32 * 1024,
        help="chunk size to split the input to avoid OOM",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=16, help="number of training epochs"
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpus")

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="pretrained checkpoint to load (including optimizers, etc)",
    )
    parser.add_argument(
        "--prefixes_to_ignore",
        nargs="+",
        type=str,
        default=["loss"],
        help="the prefixes to ignore in the checkpoint state dict",
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        default=None,
        help="pretrained model weight to load (do not load optimizers, etc)",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="optimizer type",
        choices=["sgd", "adam", "radam", "ranger"],
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="learning rate momentum"
    )
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="steplr",
        help="scheduler type",
        choices=["steplr", "cosine", "poly"],
    )
    #### params for warmup, only applied when optimizer == 'sgd' or 'adam'
    parser.add_argument(
        "--warmup_multiplier",
        type=float,
        default=1.0,
        help="lr is multiplied by this factor after --warmup_epochs",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=0,
        help="Gradually warm-up(increasing) learning rate in optimizer",
    )
    ###########################
    #### params for steplr ####
    parser.add_argument(
        "--decay_step", nargs="+", type=int, default=[20], help="scheduler decay step"
    )
    parser.add_argument(
        "--decay_gamma", type=float, default=0.1, help="learning rate decay amount"
    )
    ###########################
    #### params for poly ####
    parser.add_argument(
        "--poly_exp",
        type=float,
        default=0.9,
        help="exponent for polynomial learning rate decay",
    )
    ###########################

    parser.add_argument("--exp_name", type=str, default="exp", help="experiment name")

    ##################################
    ##### User Defined Configs #######
    ##################################

    # model (should be consistent in training, evaluating and validating loaded ckpt in one script.)
    parser.add_argument(
        "--model_type", type=str, default="nerf", choices=["nerf", "nerf_tcnn"]
    )
    parser.add_argument("--predict_normal", action="store_true", default=False)
    parser.add_argument("--predict_mirror_mask", action="store_true", default=False)
    parser.add_argument("--trace_secondary_rays", action="store_true", default=False)
    parser.add_argument("--only_one_field", action="store_true", default=False)
    parser.add_argument("--only_one_field_fine_epoch", type=int, default=2)

    # dataset
    parser.add_argument(
        "--bound",
        type=float,
        default=1.0,
        help="radius of bounding sphere of the scene.",
    )
    parser.add_argument("--near", type=float, default=0.05)
    parser.add_argument("--far", type=float, default=8.0)
    parser.add_argument("--scale_factor", type=float, default=1)
    parser.add_argument("--val_idx", type=int, default=0)
    parser.add_argument("--train_skip_step", type=int, default=1)

    # training strategy
    parser.add_argument("--max_recursive_level", type=int, default=1)
    parser.add_argument(
        "--only_trace_rays_in_mirrors", action="store_true", default=False
    )
    parser.add_argument(
        "--for_vis",
        action="store_true",
        default=False,
        help="True to trace all rays for visualization. False to trace rays in the chunk containing mirror.",
    )
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument(
        "--train_geometry_stage",
        action="store_true",
        default=False,
        help="For the part in mirror, train geometry of mirror only, not care the color in mirror.",
    )

    parser.add_argument(
        "--train_geometry_stage_end_epoch", type=int, default=4
    )
    parser.add_argument("--smooth_mirror_start_epoch", type=int, default=2)
    parser.add_argument("--train_mirror_mask_start_epoch", type=int, default=2)
    parser.add_argument("--train_normal_start_epoch", type=int, default=1)

    parser.add_argument(
        "--detach_density_outside_mirror_for_mask_loss",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--detach_density_for_mask_loss", action="store_true", default=False
    )
    parser.add_argument(
        "--detach_density_for_normal_loss", action="store_true", default=False
    )
    parser.add_argument(
        "--detach_normal_in_reflection", action="store_true", default=False
    )
    parser.add_argument("--woMaskRGBtoBlack", action="store_true", default=False)
    parser.add_argument(
        "--detach_ref_color_for_blend", action="store_true", default=False
    )

    # loss
    parser.add_argument(
        "--normal_loss_only_inside_mirror", action="store_true", default=False
    )
    parser.add_argument(
        "--use_plane_consistent_loss", action="store_true", default=False
    )

    # loss weight
    parser.add_argument("--color_loss_weight", type=float, default=1)
    parser.add_argument("--normal_loss_weight", type=float, default=1e-4)
    parser.add_argument("--normal_reg_loss_weight", type=float, default=0.1)
    parser.add_argument("--mirror_mask_loss_weight", type=float, default=0.1)
    parser.add_argument("--plane_consistent_loss_weight", type=float, default=0.1)

    if b_parse_args == True:
        return parser.parse_args()
    else:
        return parser
