import os

import torch
from torch import autograd

from torchdrug import utils


path = os.path.dirname(__file__)


if torch.cuda.is_available():
    spmm_ext = utils.load_extension(
        "generalized_spmm",
        [os.path.join(path, "generalized_spmm.cpp"), os.path.join(path, "generalized_spmm.cu")],
        extra_cflags=["-g", "-Ofast", "-DCUDA_OP"], extra_cuda_cflags=["-O3"])
else:
    spmm_ext = utils.load_extension(
        "generalized_spmm",
        [os.path.join(path, "generalized_spmm.cpp")],
        extra_cflags=["-g", "-Ofast"])


class SPMMMax(autograd.Function):

    @staticmethod
    def forward(ctx, mask, input):
        mask = mask.coalesce()
        if input.device.type == "cuda":
            forward = spmm_ext.spmm_max_cuda_forward
        else:
            forward = spmm_ext.spmm_max_cpu_forward
        max, max_indices = forward(mask, input)
        ctx.save_for_backward(mask, max_indices)
        return max, max_indices

    @staticmethod
    def backward(ctx, output_grad, indice_grad):
        if output_grad.device.type == "cuda":
            backward = spmm_ext.spmm_max_cuda_backward
        else:
            backward = spmm_ext.spmm_max_cpu_backward
        mask_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        return mask_grad, input_grad


def generalized_spmm(x, y, addition="sum", multiplication="mul"):
    assert x.is_sparse

    if addition == "sum" and multiplication == "mul":
        x = torch.spmm(x, y)
        return x
    elif addition == "max" and multiplication == "mul":
        return SPMMMax.apply(x, y)[0]
    elif addition == "min" and multiplication == "mul":
        return -SPMMMax.apply(x, -y)[0]
    else:
        raise ValueError("Can't perform generalized spmm with addition `%s` and multiplication `%s`"
                         % (addition, multiplication))
