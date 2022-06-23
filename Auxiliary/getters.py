###########################################################################
###########################################################################
## Additional item getter functions.                                     ##
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

"""Additional item getter functions."""

from typing import Iterable, Optional, Sequence, TypeVar, overload

from auxiliary.typing import SupportsRichComparison
from auxiliary.moreitertools import filter_not_none

VT = TypeVar("VT")
VT_C = TypeVar("VT_C", bound=SupportsRichComparison)

@overload
def default_get(arg: Optional[VT],
                default: VT
                ) -> VT:
    """Return the argument if it is not None, else return the default."""
    ...

@overload
def default_get(arg: tuple[Optional[VT], ...],
                default: tuple[VT, ...],
                element_default: VT
                ) -> tuple[VT, ...]:
    """
    Return the argument if it is not None, else return the default.
    
    If the argument is a tuple, return a new tuple containing with
    all None items replaced with the element-wise default.
    
    To instead obtain an iterator over not None elements without
    replacing them with a default use `filter_not_none`.
    """
    ...

def default_get(arg: Optional[VT] | tuple[Optional[VT], ...],
                default: VT | tuple[VT, ...],
                element_default: Optional[VT] = None
                ) -> VT | tuple[VT]:
    """Return the argument if it is not None, else return the default."""
    ## If the element-wise default is given and not None, and the argument is a tuple,
    ## then fill elements of the tuple that are None with the default.
    if element_default is not None and isinstance(arg, tuple):
        return tuple((element_default if element is None else element) for element in arg)
    ## Otherwise return argument if it is not None else the default.
    return default if arg is None else arg

def multi_default_get(arg: Optional[VT], defaults: Sequence[Optional[VT]]) -> VT:
    """Return the argument if it is not None, else return the first default that is not None."""
    return default_get(arg, get_first_not_none(defaults))

def default_max(args: Iterable[Optional[VT]], default: Optional[VT_C] = None) -> VT_C:
    """Return the maximum argument which is not None, else return the default if all arguments are None or the iterable is empty."""
    return max(filter_not_none(args), default=default)

def get_first_not_none(sequence: Sequence[VT], default: Optional[VT] = None) -> VT:
    """Get the first element of the sequence that is not None, if all elements are None then return default instead."""
    for element in sequence:
        if element is not None:
            return element
    return default
