###########################################################################
###########################################################################
## Module defining additional item getter functions.                     ##
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

"""Module defining additional item getter functions."""

from typing import Iterable, Sequence, TypeVar, overload

from aloy.auxiliary.typingutils import SupportsRichComparison
from aloy.auxiliary.moreitertools import filter_not_none

__all__ = (
    "default_get",
    "multi_default_get",
    "default_max",
    "get_first_not_none"
)


VT = TypeVar("VT")
VT_C = TypeVar("VT_C", bound=SupportsRichComparison)


@overload
def default_get(
    arg: VT | None, /,
    default: VT
) -> VT:
    """Return the argument if it is not None, else return the default."""
    ...


@overload
def default_get(
    arg: tuple[VT | None, ...], /,
    default: tuple[VT, ...],
    elem_default: VT
) -> tuple[VT, ...]:
    """
    Return the argument if it is not None, else return the default.

    If the argument is a tuple, return a new tuple containing with
    all None items replaced with the element-wise default.

    To instead obtain an iterator over not None elements without
    replacing them with a default use `filter_not_none`.
    """
    ...


def default_get(
    arg: VT | None | tuple[VT | None, ...], /,
    default: VT | tuple[VT, ...],
    elem_default: VT | None = None
) -> VT | tuple[VT]:
    """Return the argument if it is not None, else return the default."""
    # If the element-wise default is given and not None, and the argument is a
    # tuple, then fill elements of the tuple that are None with the default.
    if elem_default is not None and isinstance(arg, tuple):
        return tuple((elem_default if elem is None else elem) for elem in arg)
    # Otherwise return argument if it is not None else the default.
    return default if arg is None else arg


def multi_default_get(arg: VT | None, defaults: Sequence[VT | None]) -> VT:
    """
    Return the argument if it is not None, else return the first default that
    is not None.
    """
    if arg is not None:
        return arg
    result = get_first_not_none(defaults)
    if result is None:
        raise ValueError("All defaults are None.")
    return result


def default_max(
    args: Iterable[VT_C | None],
    default: VT_C | None = None
) -> VT_C:
    """
    Return the maximum argument which is not None, else if the default is
    given and not None, return default if all arguments are None or the
    iterable is empty.
    """
    max_ = max(filter_not_none(args), default=default)
    if max_ is None:
        raise ValueError("All arguments are None.")
    return max_


def default_min(
    args: Iterable[VT_C | None],
    default: VT_C | None = None
) -> VT_C:
    """
    Return the minimum argument which is not None, else if the default is
    given and not None, return default if all arguments are None or the
    iterable is empty.
    """
    min_ = min(filter_not_none(args), default=default)
    if min_ is None:
        raise ValueError("All arguments are None.")
    return min_


def get_first_not_none(
    sequence: Sequence[VT],
    default: VT | None = None
) -> VT:
    """
    Get the first element of the sequence that is not None, if all elements
    are None and the default is given and not None, return default instead.
    """
    for element in sequence:
        if element is not None:
            return element
    if default is None:
        raise ValueError("All elements are None.")
    return default
