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

"""Module containing data structure view types."""

import collections
from typing import (Generic, Hashable, Iterator, Mapping, TypeVar, final,
                    overload)

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"
__version__ = "1.0.0"

__all__ = (
    "ListView",
    "DictView",
    "SetView",
    "ListValuedMappingView",
    "SetValuedMappingView"
)


def __dir__() -> tuple[str, ...]:
    """Get the names of module attributes."""
    return __all__


LT = TypeVar("LT")


@final
class ListView(collections.abc.Sequence, Generic[LT]):
    """
    Class defining a view of a list.

    The list cannot be modified through the view, but the view will reflect
    changes made to the list by its owner.
    """

    __slots__ = {
        "__list": "The list being viewed."
    }

    def __init__(self, list_: list[LT], /) -> None:
        """Create a new list view."""
        self.__list: list[LT] = list_

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


KT = TypeVar("KT", bound=Hashable)
VT_co = TypeVar("VT_co", bound=Hashable, covariant=True)


@final
class DictView(collections.abc.Mapping, Generic[KT, VT_co]):
    """
    Class defining a of a dictionary.

    The dictionary cannot be modified through the view, but the view will
    reflect changes made to the dictionary by its owner.
    """

    __slots__ = {
        "__dict": "The dictionary being viewed."
    }

    def __init__(self, dict_: dict[KT, VT_co], /) -> None:
        """Create a new dictionary view."""
        self.__dict: dict[KT, VT_co] = dict_

    def __repr__(self) -> str:
        """Get an instantiable string representation of the dictionary view."""
        return f"DictView({self.__dict!r})"

    def __contains__(self, key: object, /) -> bool:
        """Check if a key is in the dictionary view."""
        return key in self.__dict

    def __getitem__(self, key: KT, /) -> VT_co:
        """Get the item at the given key."""
        return self.__dict[key]

    def __iter__(self) -> Iterator[KT]:
        """Iterate over the items in the dictionary view."""
        return iter(self.__dict)

    def __len__(self) -> int:
        """Get the number of items in the dictionary view."""
        return len(self.__dict)


ST = TypeVar("ST", bound=Hashable)


@final
class SetView(collections.abc.Set, Generic[ST]):
    """
    Class defining a view of a set.

    The set cannot be modified through the view, but the view will reflect
    changes made to the set by its owner.
    """

    __slots__ = {
        "__set": "The set being viewed."
    }

    def __init__(self, set_: set[ST], /) -> None:
        """Create a new set view."""
        self.__set: set[ST] = set_

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


@final
class ListValuedMappingView(collections.abc.Mapping, Generic[KT, VT_co]):
    """
    Class defining a list-valued mapping view type.

    The mapping cannot be modified through the view, but the view will reflect
    changes made to the mapping by its owner.
    """

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
    """
    Class defining a set-valued mapping view type.

    The mapping cannot be modified through the view, but the view will reflect
    changes made to the mapping by its owner.
    """

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
