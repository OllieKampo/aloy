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

def arg_first_where(condition: Callable[[np.ndarray], np.ndarray], array: np.ndarray, axis: int, invalid_val: np.number):
    """
    Get the index of the first element in the array that satisfies the condition.
    
    The condition is a function that takes an array and returns a boolean array.
    """
    mask = condition(array)
    if mask.ndim == 1:
        if axis != 0:
            raise ValueError(f"Axis must be 0 for 1-dimensional arrays. Got; {axis=}.")
        return mask.argmax() if mask.any() else invalid_val
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def get_turning_points(array: np.ndarray) -> np.ndarray:
    """Get indices of the stationary points in the array."""
    gradient_directions = np.sign(np.gradient(array))
    return np.where((np.absolute(np.diff(gradient_directions)) == 2) | (gradient_directions[:-1] == 0))[0]

def get_inflection_points(array: np.ndarray) -> np.ndarray:
    """Get indices of the inflection points in the array."""
    return np.nonzero(np.diff(np.sign(np.gradient(np.gradient(array)))))[0]