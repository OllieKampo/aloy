
"""
Module containing utility functions for convolutional neural network layers.
"""


import torch.nn as nn


def calc_output_len(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int
) -> int:
    """
    Calculate the output length over one dimension of a convolutional layer.

    See: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    Stride: https://deepai.org/machine-learning-glossary-and-terms/stride
    Padding: https://deepai.org/machine-learning-glossary-and-terms/padding
    """
    return (((input_size
              + (2 * padding)
              - (dilation * (kernel_size - 1))
              - 1)
             // stride)
            + 1)


def calc_output_shape(
    input_size: tuple[int, ...],
    kernel_size: tuple[int, ...] | int,
    stride: tuple[int, ...] | int,
    padding: tuple[int, ...] | int,
    dilation: tuple[int, ...] | int
) -> tuple[int, ...]:
    """
    Calculate the output shape of a 2d or 3d convolutional layer (per channel).

    See: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    dimensions = len(input_size)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * dimensions
    if isinstance(stride, int):
        stride = (stride,) * dimensions
    if isinstance(padding, int):
        padding = (padding,) * dimensions
    if isinstance(dilation, int):
        dilation = (dilation,) * dimensions
    return tuple(
        calc_output_len(input_, kernel_, stride_, padding_, dilation_)
        for input_, kernel_, stride_, padding_, dilation_
        in zip(input_size, kernel_size, stride, padding, dilation)
    )


def calc_output_shape_from(
    input_size: tuple[int, ...],
    layer: nn.Conv2d | nn.Conv3d
) -> tuple[int, ...]:
    """
    Calculate the output shape of a 2d or 3d convolutional layer (per channel).

    See: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    return calc_output_shape(
        input_size=input_size,
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation
    )


def size_of_flat_layer(
    output_shape: tuple[int, int],
    num_channels: int
) -> int:
    """
    Calculate the size of a flattened layer.

    See: https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html
    """
    return output_shape[0] * output_shape[1] * num_channels