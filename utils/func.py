import torch
import math


def l2_normalize(x, eps=torch.as_tensor(torch.finfo(torch.float32).eps)):
    """Normalize x to unit length along last axis."""
    return x / torch.sqrt(torch.maximum(torch.sum(x**2, axis=-1, keepdims=True), eps))


def gradient(inputs: torch.Tensor, outputs: torch.Tensor, normalize=True):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        # allow_unused=True, #
        only_inputs=True,
    )
    # points_grad = points_grad[0][..., -3:]
    points_grad = points_grad[0]
    if normalize:
        points_grad = l2_normalize(points_grad, dim=-1)
    return points_grad


def binary_cross_entropy(input, target, reduction="none"):
    """F.binary_cross_entropy is not numerically stable in mixed-precision training."""
    if reduction == "none":
        return -(target * torch.log(input) + (1 - target) * torch.log(1 - input))
    if reduction == "mean":
        return -(target * torch.log(input) + (1 - target) * torch.log(1 - input)).mean()
