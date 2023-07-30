###########################################################################
###########################################################################
## Module defining helpers to type checking and annotations.             ##
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

"""Module defining additional functions for operating on iterables."""

## Useful links for typing:
##      - Type hints: https://docs.python.org/3/library/typing.html
##          - Type Aliasing: https://peps.python.org/pep-0613/
##      - Abstract Base Classes for Containers: https://docs.python.org/3/library/collections.abc.html
##      - Nominal versus structural typing: https://docs.python.org/3/library/typing.html#nominal-vs-structural-subtyping
##          - Protocols: https://docs.python.org/3/library/typing.html#typing.Protocol,
# https://docs.python.org/3/library/typing.html#protocols,
# https://peps.python.org/pep-0544/

from typing import Any, Generic, Hashable, Protocol, TypeVar, runtime_checkable

_KT = TypeVar("_KT")
_KT_co = TypeVar("_KT_co", covariant=True)
_KT_contra = TypeVar("_KT_contra", contravariant=True)
_VT = TypeVar("_VT")
_VT_co = TypeVar("_VT_co", covariant=True)
_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
_T_contra = TypeVar("_T_contra", contravariant=True)

@runtime_checkable
class SupportsLenAndGetitem(Protocol[_T_co], Generic[_T_co]):
    """Protocol for type hinting and checking support for `__len__` and `__getitem__` (i.e. it is sized and subscriptable)."""
    
    def __len__(self) -> int:
        # noqa: D105
        ...
    def __getitem__(self, __key: int) -> _T_co:
        # noqa: D105
        ...

@runtime_checkable
class SupportsRichComparison(Protocol):
    """Protocol for type hinting and checking support for fundamental rich comparison magic methods `__lt__` and `__gt__`."""
    
    def __lt__(self, __other: Any) -> bool:
        # noqa: D105
        ...
    def __gt__(self, __other: Any) -> bool:
        # noqa: D105
        ...

@runtime_checkable
class HashableSupportsRichComparison(Hashable, SupportsRichComparison, Protocol):
    """Protocol for type hinting and checking support for fundamental rich comparison magic methods `__lt__` and `__gt__` and `__hash__`."""
    
    ...
