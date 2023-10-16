###############################################################################
# Copyright (C) 2023 Oliver Michael Kamperis
# Email: olliekampo@gmail.com
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

"""Module defining additional functions for operating on iterables."""

# Useful links for typing:
#      - Type hints: https://docs.python.org/3/library/typing.html
#          - Type Aliasing: https://peps.python.org/pep-0613/
#      - Abstract Base Classes for Containers: https://docs.python.org/3/library/collections.abc.html
#      - Nominal versus structural typing: https://docs.python.org/3/library/typing.html#nominal-vs-structural-subtyping
#          - Protocols: https://docs.python.org/3/library/typing.html#typing.Protocol,
# https://docs.python.org/3/library/typing.html#protocols,
# https://peps.python.org/pep-0544/

from typing import Any, Hashable, Iterator, Protocol, TypeVar, overload, runtime_checkable

_KT = TypeVar("_KT")
_KT_co = TypeVar("_KT_co", covariant=True)
_KT_contra = TypeVar("_KT_contra", contravariant=True)
_VT = TypeVar("_VT")
_VT_co = TypeVar("_VT_co", covariant=True)
_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
_T_contra = TypeVar("_T_contra", contravariant=True)


@runtime_checkable
class SupportsLenAndGetitem(Protocol[_T_co]):
    """
    Protocol for type hinting and checking support for `__len__`,
    `__getitem__`, and `__iter__` (i.e. it is sized, subscriptable, and
    therefore iterable).
    """

    def __len__(self) -> int:
        ...

    @overload
    def __getitem__(
        self,
        __key: int, /
    ) -> _T_co:
        ...

    @overload
    def __getitem__(
        self,
        __key: slice, /
    ) -> "SupportsLenAndGetitem[_T_co]":
        ...

    def __getitem__(
        self,
        __key: int | slice, /
    ) -> _T_co | "SupportsLenAndGetitem[_T_co]":
        ...

    def __iter__(self) -> Iterator[_T_co]:
        ...


@runtime_checkable
class SupportsRichComparison(Protocol):
    """
    Protocol for type hinting and checking support for fundamental rich
    comparison magic methods `__lt__` and `__gt__`.
    """

    def __lt__(self, __other: Any) -> bool:
        ...

    def __gt__(self, __other: Any) -> bool:
        ...


@runtime_checkable
class HashableSupportsRichComparison(
        Hashable, SupportsRichComparison, Protocol):
    """
    Protocol for type hinting and checking support for fundamental rich
    comparison magic methods `__lt__` and `__gt__` and `__hash__`.
    """
    ...
