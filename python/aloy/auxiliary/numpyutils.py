###########################################################################
###########################################################################
## Module defining utility functions for numpy.                          ##
##                                                                       ##
## Copyright (C)  2022  Oliver Michael Kamperis                          ##
## Email: o.m.kamperis@gmail.com                                         ##
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

"""Module defining utility functions for numpy."""

from typing import Callable
import numpy as np
from numba import njit
from scipy import spatial as scipy_spatial


def arg_first(
    array: np.ndarray,
    value: np.number | np.ndarray,
    axis: int = 0,
    default: np.number = -1
) -> int | np.ndarray:
    """
    Get the index of the first element in the array that is equal to the
    given value.
    """
    axis_view = np.moveaxis(array, axis, 0)
    return __arg_first(axis_view, value, default)


@njit
def __arg_first(
    arr: np.ndarray,
    value: np.number,
    default: np.number
) -> int | np.ndarray:
    """
    Get the index of the first element in the array's first axis that is equal
    to the given value.
    """
    if arr.ndim > 1:
        arr = np.empty(arr.shape[:-1], dtype=arr.dtype)
        for i in range(arr.shape[0]):
            arr[i] = __arg_first(arr[i], value, default)
        return arr
    for i in range(arr.shape[0]):
        if arr[i] == value:
            return i
    return default


def arg_nth(
    value: np.number | np.ndarray,
    array: np.ndarray,
    n: int,
    axis: int = 0,
    default: np.number = -1
) -> int:
    """
    Get the index of the nth element in the array that is equal to the given
    value.
    """
    axis_view = np.moveaxis(array, axis, 0)
    return __arg_nth(value, axis_view, n, default)


@njit
def __arg_nth(
    value: np.number,
    array: np.ndarray,
    n: int,
    default: np.number
) -> int:
    """
    Get the index of the nth element in the array's first axis that is equal
    to the given value.
    """
    if array.ndim > 1:
        arr = np.empty(array.shape[:-1], dtype=array.dtype)
        for i in range(array.shape[0]):
            arr[i] = __arg_nth(value, array[i], n, default)
        return arr
    count: int = 0
    for i in range(array.shape[0]):
        if array[i] == value:
            count += 1
            if count == n:
                return i
    return default


@njit
def arg_nearest(value: np.number, array: np.ndarray, axis: int = 0) -> int:
    """
    Get the index of the element in the array that is nearest to the given
    value.
    """
    return np.argmin(np.absolute(array - value), axis=axis)


def arg_first_where(
    condition: Callable[[np.ndarray | np.number], np.ndarray],
    array: np.ndarray,
    axis: int = 0,
    invalid_val: np.number = -1
) -> int | np.ndarray:
    """
    Get the index of the first element in the array that satisfies the
    condition.

    The condition is a function that takes an array and returns a boolean
    array.
    """
    mask = condition(array)
    if mask.ndim == 1:
        if axis != 0:
            raise ValueError("Axis must be 0 for 1-dimensional arrays.")
        return mask.argmax() if mask.any() else invalid_val
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def euclidean_distances_from(
    point_cloud: np.ndarray,
    point: np.ndarray
) -> np.ndarray:
    """
    Calculate the Euclidean distance between a given point and other points in
    a cloud.
    """
    return np.apply_along_axis(
        scipy_spatial.distance.euclidean,
        axis=0, arr=point_cloud, v=point
    )


def get_stationary_points(array: np.ndarray) -> np.ndarray:
    """
    Get indices of the stationary points of a curve.

    A stationary point is a point on a curve where the gradient is zero.
    A stationary point may be a turning point or an inflection point (but not
    all inflection points are stationary points).

    The behaviour of this function is undefined for lines of the form `y = c`.
    """
    return np.unique(
        np.concatenate(
            (get_turning_points(array),
             get_inflection_points(array))  # TODO: This is not correct, not all inflection points are stationary points.
        )
    )


def get_turning_points(array: np.ndarray) -> np.ndarray:
    """
    Get indices of the turning points of a curve.

    A turning point is a point in a curve where the gradient changes sign (and
    the gradient is zero). A function changes from increasing to decreasing
    (or vice versa) at a turning point. This is also known as a local maximum
    or minimum. For example `y = x^2` has a turning point at `x = 0`.

    The behaviour of this function is undefined for lines of the form `y = c`.
    """
    # Get the sign of first derivative at each point (pos = 1, neg = -1).
    gradient_directions = np.sign(np.gradient(array))
    return np.where(
        # Find where the gradient changes or is zero.
        (np.absolute(np.diff(gradient_directions)) == 2)
        | (gradient_directions[:-1] == 0)  # Ignore the last point.
    )[0]


def get_inflection_points(array: np.ndarray) -> np.ndarray:
    """
    Get indices of the inflection points of a curve.

    An inflection point is a point on a curve where the curvature changes sign.
    A function changes from being concave to convex (or vice versa) at an
    inflection point. For example `y = x^3` has an inflection point at `x = 0`.

    The behaviour of this function is undefined for lines of the form `y = c`.
    """
    # Get the sign of the second derivative at each point (pos = 1, neg = -1).
    sign_of_curvature = np.sign(np.gradient(np.gradient(array)))
    return np.where(
        # Find where the curvature changes sign or is zero.
        (np.absolute(np.diff(sign_of_curvature)) == 2)
        | (sign_of_curvature[:-1] == 0)  # Ignore the last point.
    )[0]
