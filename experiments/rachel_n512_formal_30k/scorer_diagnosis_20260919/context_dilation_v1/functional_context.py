"""Functional context intervention; never edit the supplied module or weights."""
import torch
from torch import nn
from torch.nn import functional as F


def circular_conv(module, value, dilation):
    if (type(dilation) is not int or dilation not in (1, 2) or not isinstance(module, nn.Conv1d)
            or module.padding_mode != "circular" or module.stride != (1,)
            or module.dilation != (1,) or module.kernel_size[0] % 2 != 1
            or module.padding != (module.kernel_size[0] // 2,)):
        raise ValueError("requires original odd-kernel circular stride1/dilation1 Conv1d and dilation1/2")
    radius = dilation * (module.kernel_size[0] // 2)
    padded = F.pad(value, (radius, radius), mode="circular")
    return F.conv1d(padded, module.weight, module.bias, stride=1, padding=0,
                    dilation=dilation, groups=module.groups)


def within(context, value, valid, normalized_rc, *, dilation):
    result = value + context.position(normalized_rc)
    result = torch.where(valid[:, :, None], result, torch.zeros_like(result))
    for block in context.blocks:
        if [type(layer) for layer in block] != [nn.Conv1d, nn.GroupNorm, nn.SiLU, nn.Conv1d, nn.GroupNorm]:
            raise ValueError("unexpected production context block; do not silently approximate")
        update = result.transpose(1, 2)
        for layer in block:
            update = circular_conv(layer, update, dilation) if isinstance(layer, nn.Conv1d) else layer(update)
        result = result + update.transpose(1, 2)
        result = torch.where(valid[:, :, None], result, torch.zeros_like(result))
    return result


def context_forward(context, first, second, valid_a, valid_b, points_a_rc, points_b_rc,
                    canvas_size, *, dilation):
    """Same position/GN/landmark attention/feedforward; only conv sampling changes."""
    scale = 2.0 / float(canvas_size - 1)
    a = within(context, first, valid_a, points_a_rc * scale - 1.0, dilation=dilation)
    b = within(context, second, valid_b, points_b_rc * scale - 1.0, dilation=dilation)
    return context._cross(a, valid_a, b, valid_b), context._cross(b, valid_b, a, valid_a)
