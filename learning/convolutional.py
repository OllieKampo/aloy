###########################################################################
###########################################################################
## Module containing functions for convolutional neural network layers.  ##
##                                                                       ##
## Copyright (C) 2023 Oliver Michael Kamperis                            ##
##                                                                       ##
## This program is free software: you can redistribute it and/or modify  ##
## it under the terms of the GNU General Public License as published by  ##
## the Free Software Foundation, either version 3 of the License, or     ##
## any later version.                                                    ##
##                                                                       ##
## This program is distributed in the hope that it will be useful,       ##
## but WITHOUT ANY WARRANTY; without even the implied warranty of        ##
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          ##
## GNU General Public License for more details.                          ##
##                                                                       ##
## You should have received a copy of the GNU General Public License     ##
## along with this program. If not, see <https://www.gnu.org/licenses/>. ##
###########################################################################
###########################################################################

"""
Module containing functions for convolutional neural network layers.
"""


from typing import TypeAlias
import torch
import torch.nn as nn

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"


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

    Parameters
    ----------
    `input_size: int` - The length of the input over the dimension.

    `kernel_size: int` - The length of the kernel over the dimension.

    `stride: int` - The stride of the convolution over the dimension.
    Stride is the number of elements to skip between each convolution.
    See: https://deepai.org/machine-learning-glossary-and-terms/stride

    `padding: int` - The padding of the convolution over the dimension.
    Padding is the number of elements to add to the input on each side.
    See: https://deepai.org/machine-learning-glossary-and-terms/padding

    `dilation: int` - The dilation of the convolution over the dimension.
    Dilation is the number of elements of space between each kernel element.
    See: https://www.geeksforgeeks.org/dilated-convolution/

    Returns
    -------
    `int` - The length of the output over the dimension.
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

    Parameters
    ----------
    `input_size: int` - The length of the input over the dimension.

    `kernel_size: int` - The length of the kernel over the dimension.

    `stride: int` - The stride of the convolution over the dimension.
    Stride is the number of elements to skip between each convolution.
    See: https://deepai.org/machine-learning-glossary-and-terms/stride

    `padding: int` - The padding of the convolution over the dimension.
    Padding is the number of elements to add to the input on each side.
    See: https://deepai.org/machine-learning-glossary-and-terms/padding

    `dilation: int` - The dilation of the convolution over the dimension.
    Dilation is the number of elements of space between each kernel element.
    See: https://www.geeksforgeeks.org/dilated-convolution/

    Returns
    -------
    `int` - The length of the output over the dimension.
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

    Parameters
    ----------
    `input_size: tuple[int, ...]` - The shape of the input tensor.

    `kernel_size: tuple[int, ...] | int` - The size of the kernel.

    `stride: tuple[int, ...] | int` - The stride of the convolution.

    `padding: tuple[int, ...] | int` - The padding of the convolution.

    `dilation: tuple[int, ...] | int` - The dilation of the convolution.

    If an int is passed for any of; `kernel_size`, `stride`, `padding`,
    or `dilation`, the same value is used for all dimensions.

    Returns
    -------
    `tuple[int, ...]` - The shape of the output tensor.
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
    Calculate the output shape of a 2d or 3d convolutional transpose layer
    (per channel).

    Parameters
    ----------
    `input_size: tuple[int, ...]` - The shape of the input tensor.

    `kernel_size: tuple[int, ...] | int` - The size of the kernel.

    `stride: tuple[int, ...] | int` - The stride of the convolution.

    `padding: tuple[int, ...] | int` - The padding of the convolution.

    `dilation: tuple[int, ...] | int` - The dilation of the convolution.

    If an int is passed for any of; `kernel_size`, `stride`, `padding`,
    or `dilation`, the same value is used for all dimensions.

    Returns
    -------
    `tuple[int, ...]` - The shape of the output tensor.
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
    convolutional layer (per channel), from the input size and a layer
    object.

    Parameters
    ----------
    `input_size: tuple[int, ...]` - The shape of the input tensor.

    `layer: ConvolutionalLayer` - The layer object, can be any of the
    following types; `nn.Conv1d`, `nn.Conv2d`, `nn.Conv3d`,
    `nn.ConvTranspose1d`, `nn.ConvTranspose2d`, or `nn.ConvTranspose3d`.

    Returns
    -------
    `tuple[int, ...]` - The shape of the output tensor.
    """
    if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return calc_conv_output_shape(
            input_size=input_size,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,  # TODO: This can be a string? Check this.
            dilation=layer.dilation
        )
    elif isinstance(layer, (nn.ConvTranspose1d, nn.ConvTranspose2d,
                            nn.ConvTranspose3d)):
        return calc_conv_transpose_output_shape(
            input_size=input_size,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,  # TODO: This can be a string? Check this.
            dilation=layer.dilation,
            output_padding=layer.output_padding
        )
    else:
        raise TypeError(
            "Layer must be convolutional. "
            f"Got; {layer!r} of type {type(layer)!r}."
        )


def size_of_flat_layer(
    output_shape: tuple[int, ...],
    num_channels: int
) -> int:
    """
    Calculate the size of output of a convolutional layer when flattened.
    Useful for calculating the size of the input to a linear layer.

    Parameters
    ----------
    `output_shape: tuple[int, ...]` - The shape of the output tensor.

    `num_channels: int` - The number of channels in the output tensor.

    Returns
    -------
    `int` - The size of the output tensor when flattened.
    """
    shape_multiple = 1
    for dim in output_shape:
        shape_multiple *= dim
    return shape_multiple * num_channels


def pad_circular(
    input_: torch.Tensor,
    padding: tuple[int, ...] | int
) -> torch.Tensor:
    """
    Pad a tensor with circular padding.

    Essentially, this copies the top `padding` rows and adds them to the
    bottom, the bottom and adds them to the top, the left columns and adds
    them to the right, and the right column and adds it to the left.

    Parameters
    ----------
    `input_: torch.Tensor` - The tensor to pad.

    `padding: tuple[int, ...] | int` - The padding to apply to each
    dimension. If an int is passed, the same padding is applied to all
    dimensions.

    Returns
    -------
    `torch.Tensor` - The padded tensor.
    """
    if isinstance(padding, int):
        padding = (padding,) * 4
    return torch.nn.functional.pad(
        input=input_,
        pad=padding,
        mode="circular"
    )
