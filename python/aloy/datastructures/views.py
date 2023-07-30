###########################################################################
###########################################################################
## Module containing immutable views on various data structures.         ##
##                                                                       ##
## Copyright (C) 2022 Oliver Michael Kamperis                            ##
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

"""Module containing mapping and dictionary structures."""

import collections
from typing import (Generic, Hashable, Iterator, Mapping, TypeVar, final,
                    overload)


LT = TypeVar("LT")


@final
class ListView(collections.abc.Sequence, Generic[LT]):
    """Class defining an immutable view of a list."""

    __slots__ = {
        "__list": "The list being viewed."
    }

    def __init__(self, list_: list[LT], /) -> None:
        """Create a new list view."""
        self.__list = list_

    def __repr__(self) -> str:
        """Get an instantiable string representation of the list view."""
        return f"ListView({self.__list})"

    @overload
    def __getitem__(self, index: int, /) -> LT:
        """Get the item at the given index."""
        ...

    @overload
    def __getitem__(self, index: slice, /) -> list[LT]:
        """Get the slice at the given index."""
        ...

    def __getitem__(self, index: int | slice, /) -> LT | list[LT]:
        """Get the item or slice at the given index."""
        return self.__list[index]

    def __iter__(self) -> Iterator[LT]:
        """Iterate over the items in the list."""
        return iter(self.__list)

    def __len__(self) -> int:
        """Get the number of items in the list."""
        return len(self.__list)

    def __hash__(self) -> int:
        """Get the hash of the list view."""
        return hash(tuple(self.__list))


ST = TypeVar("ST", bound=Hashable)


@final
class SetView(collections.abc.Set, Generic[ST]):
    """Class defining a set view type."""

    __slots__ = {
        "__set": "The set being viewed."
    }

    def __init__(self, set_: set[ST], /) -> None:
        """Create a new set view."""
        self.__set = set_

    def __repr__(self) -> str:
        """Get an instantiable string representation of the set view."""
        return f"SetView({self.__set!r})"

    def __contains__(self, item: object, /) -> bool:
        """Check if an item is in the set view."""
        return item in self.__set

    def __iter__(self) -> Iterator[ST]:
        """Iterate over the items in the set view."""
        return iter(self.__set)

    def __len__(self) -> int:
        """Get the number of items in the set view."""
        return len(self.__set)


KT = TypeVar("KT", bound=Hashable)
VT_co = TypeVar("VT_co", bound=Hashable, covariant=True)


@final
class ListValuedMappingView(collections.abc.Mapping, Generic[KT, VT_co]):
    """Class defining a list-valued mapping view type."""

    __slots__ = {
        "__list_valued_mapping": "The list-valued mapping being viewed."
    }

    def __init__(self, mapping: Mapping[KT, list[VT_co]], /) -> None:
        """Create a new list-valued mapping view."""
        self.__list_valued_mapping: Mapping[KT, list[VT_co]] = mapping

    def __repr__(self) -> str:
        """
        Get an instantiable string representation of the list-valued mapping
        view.
        """
        return f"ListValuedMappingView({self.__list_valued_mapping!r})"

    def __contains__(self, key: object, /) -> bool:
        """Check if a key is in the list-valued mapping view."""
        return key in self.__list_valued_mapping

    def __getitem__(self, key: KT, /) -> ListView[VT_co]:
        """Get the list of items in the list-valued mapping view."""
        return ListView(self.__list_valued_mapping[key])

    def __iter__(self) -> Iterator[KT]:
        """Iterate over the items in the list-valued mapping view."""
        return iter(self.__list_valued_mapping)

    def __len__(self) -> int:
        """Get the number of items in the list-valued mapping view."""
        return len(self.__list_valued_mapping)


@final
class SetValuedMappingView(collections.abc.Mapping, Generic[KT, VT_co]):
    """Class defining a set-valued mapping view type."""

    __slots__ = {
        "__set_valued_mapping": "The set-valued mapping being viewed."
    }

    def __init__(self, mapping: Mapping[KT, set[VT_co]], /) -> None:
        """Create a new set-valued mapping view."""
        self.__set_valued_mapping: Mapping[KT, set[VT_co]] = mapping

    def __repr__(self) -> str:
        """
        Get an instantiable string representation of the set-valued mapping
        view.
        """
        return f"SetValuedMappingView({self.__set_valued_mapping!r})"

    def __contains__(self, key: object, /) -> bool:
        """Check if a key is in the set-valued mapping view."""
        return key in self.__set_valued_mapping

    def __getitem__(self, key: KT, /) -> SetView[VT_co]:
        """Get the set of items in the set-valued mapping view."""
        return SetView(self.__set_valued_mapping[key])

    def __iter__(self) -> Iterator[KT]:
        """Iterate over the items in the set-valued mapping view."""
        return iter(self.__set_valued_mapping)

    def __len__(self) -> int:
        """Get the number of items in the set-valued mapping view."""
        return len(self.__set_valued_mapping)
