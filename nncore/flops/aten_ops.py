"""
Copyright (C) 2023 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
"""

from typing import Any, List

import torch

aten = torch.ops.aten


def prod(x: torch.Size) -> int:
    res = 1
    for i in x:
        res *= i
    return res


def matmul_flop(inputs: List[Any], outputs: List[Any]) -> int:
    """
    Count flops for matmul.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two matrices.
    input_shapes = [v.shape for v in inputs]
    assert len(input_shapes) == 2, input_shapes
    assert input_shapes[0][-1] == input_shapes[1][-2], input_shapes
    # For matix [m, k] x [k, n], the flop number is m x k x n x 2.
    # GFLOPS = 2 * GMACs: https://github.com/sovrasov/flops-counter.pytorch/issues/16
    # Reference: https://arxiv.org/pdf/2205.05198
    flop = prod(input_shapes[0]) * input_shapes[1][-1] * 2
    return flop


def addmm_flop(inputs: List[Any], outputs: List[Any]) -> int:
    """
    Count flops for fully connected layers (nn.Linear).
    Bias is considered if exists.
    """
    # inputs: bias, input, weight
    input_shapes = [v.shape for v in inputs[1:3]]
    # input_shapes[0]: [batch size, input feature dimension]
    # input_shapes[1]: [batch size, output feature dimension]
    assert len(input_shapes[0]) == 2, input_shapes[0]
    assert len(input_shapes[1]) == 2, input_shapes[1]
    batch_size, input_dim = input_shapes[0]
    output_dim = input_shapes[1][1]
    flops = batch_size * input_dim * output_dim * 2

    if inputs[0] is not None:
        flops += batch_size * output_dim

    return flops


def bmm_flop(inputs: List[Any], outputs: List[Any]) -> int:
    """
    Count flops for the bmm operation.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two tensors.
    assert len(inputs) == 2, len(inputs)
    input_shapes = [v.shape for v in inputs]
    bs, m, k = input_shapes[0]
    n = input_shapes[-1][-1]
    flop = bs * m * k * n * 2
    return flop


def conv_flop_count(
    x_shape: torch.Size,
    w_shape: torch.Size,
    out_shape: torch.Size,
    transposed: bool = False,
    bias: bool = False,
) -> int:
    """
    Count MACs for convolution.
    Summation is ignored when applying conv kernel, but counted for bias.
    Args:
        x_shape (list(int)): The input shape before convolution.
        w_shape (list(int)): The filter shape.
        out_shape (list(int)): The output shape after convolution.
        transposed (bool): is the convolution transposed
        bias (bool): is the bias counted
    Returns:
        int: the number of MACs
    """
    batch_size = x_shape[0]
    conv_shape = (x_shape if transposed else out_shape)[2:]
    flop = batch_size * prod(w_shape) * prod(conv_shape)
    if bias:
        flop += batch_size * out_shape[1] * prod(out_shape[2:])
    return flop


def conv_flop(inputs: List[Any], outputs: List[Any]) -> int:
    """
    Count flops for convolution.
    """
    (input, w, b, stride, pad, dilation, transposed, _, groups) = inputs
    output = outputs[0]
    return conv_flop_count(
        input.shape, w.shape, output.shape, transposed=transposed, bias=b is not None
    )


def attn_flop(inputs: List[Any], outputs: List[Any]) -> int:
    """
    Count flops for attention.
    """
    assert len(inputs) == 3, len(inputs)
    input_shapes = [v.shape for v in inputs]
    bs, num_head, seq_len, head_dim = inputs[0].shape
    assert input_shapes[0] == input_shapes[1] == input_shapes[2], input_shapes

    attn_score_flops = bs * num_head * seq_len * head_dim * seq_len * 2
    attn_flops = bs * num_head * seq_len * seq_len * head_dim * 2

    return attn_score_flops + attn_flops


ATEN_OPS_MAPPING = {
    aten.mm: matmul_flop,
    aten.matmul: matmul_flop,
    aten.addmm: addmm_flop,
    aten.bmm: bmm_flop,
    aten.convolution: conv_flop,
    aten._convolution: conv_flop,
    aten._scaled_dot_product_flash_attention: attn_flop,
    aten._scaled_dot_product_flash_attention_for_cpu: attn_flop,
}
