
"""
Module containing utility functions for convolutional neural network layers.
"""


from typing import TypeAlias
import torch
import torch.nn as nn


def calc_conv_output_len(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int
) -> int:
    """
    Calculate the output length over one dimension of a convolutional layer
    1d, 2d, or 3d.

    Formula is:
    ```
    len_ = (((input_size
              + (2 * padding)
              - (dilation * (kernel_size - 1))
              - 1)
             // stride)
            + 1)
    ```

    Stride: https://deepai.org/machine-learning-glossary-and-terms/stride

    Padding: https://deepai.org/machine-learning-glossary-and-terms/padding
    """
    return (((input_size
              + (2 * padding)
              - (dilation * (kernel_size - 1))
              - 1)
             // stride)
            + 1)


def calc_conv_transpose_output_len(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    output_padding: int
) -> int:
    """
    Calculate the output length over one dimension of a convolutional
    transpose layer 1d, 2d, or 3d.

    Formula is:
    ```
    len_ = (stride * (input_size - 1)
            + (dilation * (kernel_size - 1))
            - (2 * padding)
            + output_padding
            + 1)
    ```

    Stride: https://deepai.org/machine-learning-glossary-and-terms/stride

    Padding: https://deepai.org/machine-learning-glossary-and-terms/padding
    """
    return (stride * (input_size - 1)
            + (dilation * (kernel_size - 1))
            - (2 * padding)
            + output_padding
            + 1)


def calc_conv_output_shape(
    input_size: tuple[int, ...],
    kernel_size: tuple[int, ...] | int,
    stride: tuple[int, ...] | int,
    padding: tuple[int, ...] | int,
    dilation: tuple[int, ...] | int
) -> tuple[int, ...]:
    """
    Calculate the output shape of a 2d or 3d convolutional layer (per channel).
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
        calc_conv_output_len(input_, kernel_, stride_, padding_, dilation_)
        for input_, kernel_, stride_, padding_, dilation_
        in zip(input_size, kernel_size, stride, padding, dilation)
    )


def calc_conv_transpose_output_shape(
    input_size: tuple[int, ...],
    kernel_size: tuple[int, ...] | int,
    stride: tuple[int, ...] | int,
    padding: tuple[int, ...] | int,
    dilation: tuple[int, ...] | int,
    output_padding: tuple[int, ...] | int
) -> tuple[int, ...]:
    """
    Calculate the output shape of a 2d or 3d convolutional transpose layer (per channel).
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
    if isinstance(output_padding, int):
        output_padding = (output_padding,) * dimensions
    return tuple(
        calc_conv_transpose_output_len(
            input_, kernel_, stride_, padding_, dilation_, output_padding_
        )
        for input_, kernel_, stride_, padding_, dilation_, output_padding_
        in zip(input_size, kernel_size, stride, padding, dilation,
               output_padding)
    )


ConvolutionalLayer: TypeAlias = (
    nn.Conv1d |
    nn.Conv2d |
    nn.Conv3d |
    nn.ConvTranspose1d |
    nn.ConvTranspose2d |
    nn.ConvTranspose3d
)


def calc_conv_output_shape_from(
    input_size: tuple[int, ...],
    layer: ConvolutionalLayer
) -> tuple[int, ...]:
    """
    Calculate the output shape of a 1d, 2d, or 3d standard or transpose
    convolutional layer (per channel).
    """
    return calc_conv_output_shape(
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


def pad_circular(
    input_: torch.Tensor,
    padding: tuple[int, ...] | int
) -> torch.Tensor:
    """
    Pad a tensor with circular padding.

    Essentially, this copies the top `padding` rows and adds them to the
    bottom, the bottom and adds them to the top, the left columns and adds
    them to the right, and the right column and adds it to the left.
    """
    if isinstance(padding, int):
        padding = (padding,) * 4
    return torch.nn.functional.pad(
        input=input_,
        pad=padding,
        mode="circular"
    )
