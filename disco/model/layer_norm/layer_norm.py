# Copyright 2024 ByteDance and/or its affiliates.
# This file was modified in 2026 by Jarrid Rector-Brooks, Marta Skreta, Chenghao Liu, Xi Zhang, and Alexander Tong
#
# Copyright 2021- HPC-AI Technology Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
import numbers
import os
import sys

import torch
from torch.nn.parameter import Parameter

sys.path.append(os.path.dirname(__file__))

try:
    fastfold_layer_norm_cuda = importlib.import_module("fastfold_layer_norm_cuda")
except ImportError:
    from disco.model.layer_norm.torch_ext_compile import compile

    current_dir = os.path.dirname(__file__)
    fastfold_layer_norm_cuda = compile(
        name="fastfold_layer_norm_cuda",
        sources=[
            os.path.join(f"{current_dir}/kernel", file)
            for file in ["layer_norm_cuda.cpp", "layer_norm_cuda_kernel.cu"]
        ],
        extra_include_paths=[f"{current_dir}/kernel"],
        build_directory=current_dir,
    )


class FusedLayerNormAffineFunction(torch.autograd.Function):
    """CUDA kernel-based fused layer normalization with affine transform.

    Implements forward and backward passes using a custom CUDA kernel for
    layer normalization with learnable weight and bias parameters. Handles
    bfloat16 inputs by disabling autocast during kernel calls.
    """

    @staticmethod
    def forward(ctx, input, weight, bias, normalized_shape, eps):
        d = input.dtype
        if d is torch.bfloat16:
            with torch.amp.autocast("cuda", enabled=False):
                ctx.normalized_shape = normalized_shape
                ctx.eps = eps
                input_ = input.contiguous()
                weight_ = weight.contiguous().to(dtype=d)
                bias_ = bias.contiguous().to(dtype=d)
                output, mean, invvar = fastfold_layer_norm_cuda.forward_affine(
                    input_, ctx.normalized_shape, weight_, bias_, ctx.eps
                )
                ctx.save_for_backward(input_, weight_, bias_, mean, invvar)
        else:
            ctx.normalized_shape = normalized_shape
            ctx.eps = eps
            input_ = input.contiguous()
            weight_ = weight.contiguous()
            bias_ = bias.contiguous()
            output, mean, invvar = fastfold_layer_norm_cuda.forward_affine(
                input_, ctx.normalized_shape, weight_, bias_, ctx.eps
            )
            ctx.save_for_backward(input_, weight_, bias_, mean, invvar)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        d = grad_output.dtype
        if d is torch.bfloat16:
            with torch.amp.autocast("cuda", enabled=False):
                input_, weight_, bias_, mean, invvar = ctx.saved_tensors
                grad_input = grad_weight = grad_bias = None
                (
                    grad_input,
                    grad_weight,
                    grad_bias,
                ) = fastfold_layer_norm_cuda.backward_affine(
                    grad_output.contiguous(),
                    mean,
                    invvar,
                    input_,
                    ctx.normalized_shape,
                    weight_.to(dtype=d),
                    bias_.to(dtype=d),
                    ctx.eps,
                )
        else:
            input_, weight_, bias_, mean, invvar = ctx.saved_tensors
            grad_input = grad_weight = grad_bias = None
            (
                grad_input,
                grad_weight,
                grad_bias,
            ) = fastfold_layer_norm_cuda.backward_affine(
                grad_output.contiguous(),
                mean,
                invvar,
                input_,
                ctx.normalized_shape,
                weight_,
                bias_,
                ctx.eps,
            )

        return grad_input, grad_weight, grad_bias, None, None


class FusedLayerNorm(torch.nn.Module):
    """Fused layer normalization module with CUDA kernel acceleration.

    Drop-in replacement for torch.nn.LayerNorm that uses a custom CUDA kernel
    for the forward pass, providing improved performance on GPU.

    Args:
        normalized_shape: Input shape from an expected input of size.
        eps: Value added to the denominator for numerical stability.
    """

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = Parameter(torch.ones(*normalized_shape))
        self.bias = Parameter(torch.ones(*normalized_shape))
        self.reset_parameters()

    def reset_parameters(self):
        """Initializes weight to ones and bias to zeros."""
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, input):
        return self.kernel_forward(input)

    def kernel_forward(self, input):
        """Forward pass using the CUDA kernel for fused layer normalization.

        Args:
            input: Input tensor to normalize.

        Returns:
            Normalized tensor with affine transform applied.
        """
        return FusedLayerNormAffineFunction.apply(
            input, self.weight, self.bias, self.normalized_shape, self.eps
        )
