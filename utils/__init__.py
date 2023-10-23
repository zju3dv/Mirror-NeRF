import random

import torch

# optimizer
from torch.optim import SGD, Adam
import torch_optimizer as optim

# scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from .warmup_scheduler import GradualWarmupScheduler

from .visualization import *

from datetime import datetime
from shutil import copytree, ignore_patterns


def get_named_parameters(models):
    """Get all model parameters recursively."""
    parameters = []
    if isinstance(models, list):
        for model in models:
            parameters += get_named_parameters(model)
    elif isinstance(models, dict):
        for model in models.values():
            parameters += get_named_parameters(model)
    else:  # models is actually a single pytorch model
        parameters += list(models.named_parameters())
    return parameters


def get_parameters(models):
    """Get all model parameters recursively."""
    parameters = []
    if isinstance(models, list):
        for model in models:
            parameters += get_parameters(model)
    elif isinstance(models, dict):
        for model in models.values():
            parameters += get_parameters(model)
    else:  # models is actually a single pytorch model
        parameters += list(models.parameters())
    return parameters


def get_optimizer(hparams, models, extra_parameters=None):
    eps = 1e-8
    parameters = get_parameters(models)
    if extra_parameters is not None:
        parameters += extra_parameters
    if hparams.optimizer == "sgd":
        optimizer = SGD(
            parameters,
            lr=hparams.lr,
            momentum=hparams.momentum,
            weight_decay=hparams.weight_decay,
        )
    elif hparams.optimizer == "adam":
        optimizer = Adam(
            parameters, lr=hparams.lr, eps=eps, weight_decay=hparams.weight_decay
        )
    elif hparams.optimizer == "radam":
        optimizer = optim.RAdam(
            parameters, lr=hparams.lr, eps=eps, weight_decay=hparams.weight_decay
        )
    elif hparams.optimizer == "ranger":
        optimizer = optim.Ranger(
            parameters, lr=hparams.lr, eps=eps, weight_decay=hparams.weight_decay
        )
    else:
        raise ValueError("optimizer not recognized!")

    return optimizer


def get_scheduler(hparams, optimizer):
    eps = 1e-8
    if hparams.lr_scheduler == "steplr":
        scheduler = MultiStepLR(
            optimizer, milestones=hparams.decay_step, gamma=hparams.decay_gamma
        )
    elif hparams.lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=hparams.num_epochs, eta_min=eps)
    elif hparams.lr_scheduler == "poly":
        scheduler = LambdaLR(
            optimizer,
            lambda epoch: (1 - epoch / hparams.num_epochs) ** hparams.poly_exp,
        )
    else:
        raise ValueError("scheduler not recognized!")

    if hparams.warmup_epochs > 0 and hparams.optimizer not in ["radam", "ranger"]:
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=hparams.warmup_multiplier,
            total_epoch=hparams.warmup_epochs,
            after_scheduler=scheduler,
        )

    return scheduler


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def extract_model_state_dict(ckpt_path, model_name="model", prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    checkpoint_ = {}
    if "state_dict" in checkpoint:  # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint["state_dict"]
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        k = k[len(model_name) + 1 :]
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                print("ignore", k)
                break
        else:
            checkpoint_[k] = v
    return checkpoint_


def load_ckpt(model, ckpt_path, model_name="model", prefixes_to_ignore=[]):
    if not ckpt_path:
        return
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    assert len(checkpoint_) > 0, "[Error] can not find {} in checkpoint".format(
        model_name
    )
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict, strict=False)


def get_timestamp():
    return datetime.now().strftime(r"%y%m%d_%H%M%S")


def copy_files(src_dir, dst_dir, *ignores):
    copytree(src_dir, dst_dir, ignore=ignore_patterns(*ignores))


def make_source_code_snapshot(log_dir):
    copy_files(
        ".",
        f"{log_dir}/source",
        "saved",
        "__pycache__",
        "data",
        "logs",
        "scans",
        ".vscode",
        "*.so",
        "*.a",
        ".ipynb_checkpoints",
        "build",
        "bin",
        "*.ply",
        "eigen",
        "pybind11",
        "*.npy",
        "*.pth",
        ".git",
        "debug",
        "ckpts",
        "results",
    )


def set_rand_seed(seed=1):
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # 保证每次返回的卷积算法是确定的
