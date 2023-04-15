###########################################################################
###########################################################################
## Module containing mapping and dictionary structures.                  ##
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

import collections.abc
import itertools
import types
from typing import (Generic, Hashable, ItemsView, Iterable, Iterator, KeysView, Mapping,
                    Optional, TypeVar, ValuesView, final, overload)

__copyright__ = "Copyright (C) 2022 Oliver Michael Kamperis"
__license__ = "GPL-3.0"

__all__ = (
    "frozendict",
    "ReversableDict",
    "FrozenReversableDict"
)


def __dir__() -> tuple[str]:
    """Get the names of module attributes."""
    return sorted(__all__)


KT = TypeVar("KT", bound=Hashable)
VT = TypeVar("VT", bound=Hashable, covariant=True)


@final
class frozendict(collections.abc.Mapping, Generic[KT, VT]):
    """
    A frozen dictionary type.

    A frozen dictionary is an immutable and hashable version of a standard
    dictionary.
    """

    __slots__ = {
        "__dict": "The dictionary mapping.",
        "__hash": "The hash of the dictionary."
    }

    @overload
    def __init__(self, mapping: Mapping[KT, VT], /) -> None:
        """
        Create a new frozen dictionary from a
        mapping object's (key, value) pairs.

        For example:
        ```
        >>> fdict = frozendict({"one" : 1, "two" : 2})
        >>> fdict
        frozendict({"one" : 1, "two" : 2})
        ```
        """
        ...

    @overload
    def __init__(self, iterable: Iterable[tuple[KT, VT]], /) -> None:
        """
        Create a new frozen dictionary from an iterable
        of tuples defining (key, value) pairs.

        For example:
        ```
        >>> fdict = frozendict([("one", 1), ("two", 2)])
        >>> fdict
        frozendict({"one" : 1, "two" : 2})
        ```
        """
        ...

    @overload
    def __init__(self, **kwargs: VT) -> None:
        """
        Create a new frozen dictionary with the key to
        value pairs given in the keyword argument list.

        For example:
        ```
        >>> fdict = frozendict(one=1, two=2)
        >>> fdict
        frozendict({"one" : 1, "two" : 2})
        ```
        """
        ...

    def __init__(self, *args, **kwargs) -> None:
        """Create a new frozen dictionary."""
        self.__dict: dict[KT, VT] = dict(*args, **kwargs)
        self.__hash: Optional[int] = None

    def __repr__(self) -> str:
        """Get an instantiable string representation of the dictionary."""
        return f"frozendict({self.__dict})"

    def __getitem__(self, key: KT, /) -> VT:
        """Get the value associated with the given key."""
        return self.__dict[key]

    def __iter__(self) -> Iterator[KT]:
        """Iterate over the keys in the dictionary."""
        return iter(self.__dict)

    def __len__(self) -> int:
        """Get the number of items in the dictionary."""
        return len(self.__dict)

    def __hash__(self) -> int:
        """Get the hash of the dictionary."""
        if self.__hash is None:
            hash_: int = 0
            for key, value in self.__dict.items():
                hash_ ^= hash((key, value))
            self.__hash = hash_
        return self.__hash

    def __copy__(self) -> "frozendict[KT, VT]":
        """Create a shallow copy of the dictionary."""
        return frozendict(self)


LT = TypeVar("LT", bound=Hashable)


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


@final
class ReversableDict(collections.abc.MutableMapping, Generic[KT, VT]):
    """
    Class defining a mutable reversable dictionary type.

    Supports all methods of standard dictionaries with the
    added ability to find all keys that map to a specific
    value, this is called a reversed many-to-one mapping.

    The reversable dict keeps a reversed version of itself in memory.
    Therefore, insertions are slower and memory usage is higher
    than a standard dictionary, but reversed lookups are fast.
    This also requires that dictionary values be hashable.

    Example Usage
    -------------
    ```
    >>> from jinx.datastructures.mappings import ReversableDict
    >>> rev_dict = ReversableDict({"a" : 1, "b" : 2, "z" : 2})
    >>> rev_dict
    ReversableDict({"a" : 1, "b" : 2, "z" : 2})

    ## Access the standard and reversed mappings directly.
    >>> rev_dict.standard_mapping
    {"a" : 1, "b" : 2, "z" : 2}
    >>> rev_dict.reversed_mapping
    {1 : ["a"], 2 : ["b", "z"]}

    ## Check what 'a' maps to (as in a standard dictionary).
    >>> rev_dict["a"]
    1

    ## Check what keys map to 2 (a reverse operation to a standard dictionary).
    ## The keys are given as a list where order reflects insertion order.
    >>> rev_dict(2)
    ["b", "z"]

    ## Objects are mutable and updates update standard and reverse mappings.
    >>> del rev_dict["z"]
    >>> rev_dict(2)
    ["b"]
    ```
    """

    __slots__ = {
        "__dict": "The standard dictionary mapping.",
        "__reversed_dict": "The reversed dictionary mapping."
    }

    @overload
    def __init__(self, mapping: Mapping[KT, VT], /) -> None:
        """
        Create a new reversable dictionary initialised
        from a mapping object's (key, value) pairs.

        For example:
        ```
        >>> rev_dict = ReversableDict({"one" : 1, "two" : 2})
        >>> rev_dict
        ReversableDict({"one" : 1, "two" : 2})
        ```
        """
        ...

    @overload
    def __init__(self, iterable: Iterable[tuple[KT, VT]], /) -> None:
        """
        Create a new reversable dictionary initialized from
        an iterable of tuples defining (key, value) pairs.

        For example:
        ```
        >>> rev_dict = ReversableDict([("one", 1), ("two", 2)])
        >>> rev_dict
        ReversableDict({"one" : 1, "two" : 2})
        ```
        """
        ...

    @overload
    def __init__(self, **kwargs: VT) -> None:
        """
        Create a new reversable dictionary initialized with the
        key to value pairs given in the keyword argument list.

        For example:
        ```
        >>> rev_dict = ReversableDict(one=1, two=2)
        >>> rev_dict
        ReversableDict({"one" : 1, "two" : 2})
        ```
        """
        ...

    def __init__(self, *args, **kwargs) -> None:
        """Create a new reversable dictionary."""
        self.__dict: dict[KT, VT] = {}
        self.__reversed_dict: dict[VT, list[KT]] = {}
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def __copy__(self) -> "ReversableDict[KT, VT]":
        """Get a shallow copy of the reversable dictionary."""
        return self.__class__(self.__dict)

    def __repr__(self) -> str:
        """
        Get an instantiable string representation of the reversable
        dictionary.
        """
        return f"{self.__class__.__name__}({self.__dict})"

    def __getitem__(self, key: KT, /) -> VT:
        """Get the value for the given key in the standard mapping."""
        return self.__dict[key]

    def __setitem__(self, key: KT, value: VT, /) -> None:
        """
        Add or update an item, given by a (key, value) pair, from the
        standard and reversed mapping.
        """
        # If the key is already in the standard mapping, the value it
        # currently maps to must be replaced in the reversed mapping.
        if (old_value := self.__dict.get(key)) is not None:
            self.__del_reversed_item(key, old_value)
        self.__dict[key] = value
        self.__reversed_dict.setdefault(value, []).append(key)

    def __delitem__(self, key: KT, /) -> None:
        """
        Delete a item (given by a key) from the standard and reversed
        mapping.
        """
        ## Remove from the standard mapping.
        value: VT = self.__dict[key]
        del self.__dict[key]
        ## Remove from the reversed mapping.
        self.__del_reversed_item(key, value)

    def __del_reversed_item(self, key: KT, value: VT, /) -> None:
        """
        Delete an item (given by a [key, value] pair) from the reversed
        mapping.
        """
        # Remove the reference of this value being mapped to by the given key.
        self.__reversed_dict[value].remove(key)
        # If no keys map to this value anymore then remove the value as well.
        if len(self.__reversed_dict[value]) == 0:
            del self.__reversed_dict[value]

    def __iter__(self) -> Iterator[KT]:
        """Iterate over the keys in the standard dictionary mapping."""
        yield from self.__dict

    def __len__(self) -> int:
        """Get the number of items in the standard dictionary mapping."""
        return len(self.__dict)

    def __call__(self, value: VT, /, max_: int | None = None) -> ListView[KT]:
        """
        Get the list of keys that map to a value in the reversed mapping.

        If `max_` is given and not None, return at most the first `max_` keys.
        """
        if max_ is None:
            return ListView(self.__reversed_dict[value])
        keys: list[KT] = self.__reversed_dict[value]
        return ListView(keys[0:min(max_, len(keys))])

    # @property
    # def standard_mapping(self) -> dict[KT, VT]:
    #     """Get the standard mapping as a normal dictionary."""
    #     return self.__dict

    # @property
    # def reversed_mapping(self) -> dict[VT, list[KT]]:
    #     """Get the reversed mapping as a normal dictionary."""
    #     return self.__reversed_dict

    # def reversed_items(self) -> ItemsView[VT, list[KT]]:
    #     """
    #     Get a reversed dictionary items view.

    #     Such that:
    #         - Its keys are the values of the standard dictionary,
    #         - Its values are lists of keys from the standard
    #           dictionary which map to the respective values.
    #     """
    #     return self.__reversed_dict.items()

    def reversed_get(
        self,
        value: VT,
        default: list[KT] | None = None, /,
        max_: int | None = None
    ) -> ListView[KT] | None:
        """
        Get the list of keys that map to a value in the reversed mapping.

        If the value is not found, return `default` if given; otherwise, raise
        a KeyError.

        If `max_` is given and not None, return at most the first `max_` keys.
        """
        if value not in self.__reversed_dict:
            if default is not None:
                return ListView(default)
            return default
        return self(value, max_=max_)

    def reversed_set(self, value: VT, /, *keys: KT) -> None:
        """
        Set a value to map to a series of keys in the reversed mapping.

        Keys are inserted in the order they are given.
        """
        for key in keys:
            self[key] = value

    def reversed_pop(
        self,
        value: VT,
        default: list[KT] | None = None, /
    ) -> list[KT] | None:
        """
        Remove a value from the reversed mapping and return the list of keys
        that mapped to it.

        If the key is not found, return `default` if given; otherwise, raise
        a KeyError.
        """
        if value not in self.__reversed_dict:
            return default
        keys: list[KT] = self.__reversed_dict[value]
        for key in keys:
            del self[key]
        return keys


@final
class FrozenReversableDict(collections.abc.Mapping, Generic[KT, VT]):
    """
    Class defining a frozen reversable dictionary type.

    A frozen dictionary is an immutable and hashable version of a reversable
    dictionary.
    """

    __slots__ = {
        "__reversable_dict": "The reversable dictionary mapping.",
        "__hash": "The hash of the dictionary."
    }

    @overload
    def __init__(self, mapping: Mapping[KT, VT]) -> None:
        """
        Create a new frozen reversable dictionary from a mapping object's
        (key, value) pairs.

        For example:
        ```
        >>> frev_dict = FrozenReversableDict({"one" : 1, "two" : 2})
        >>> frev_dict
        FrozenReversableDict({"one" : 1, "two" : 2})
        ```
        """
        ...

    @overload
    def __init__(self, iterable: Iterable[tuple[KT, VT]]) -> None:
        """
        Create a new frozen reversable dictionary from an iterable of tuples
        defining (key, value) pairs.

        For example:
        ```
        >>> frev_dict = FrozenReversableDict([("one", 1), ("two", 2)])
        >>> frev_dict
        FrozenReversableDict({"one" : 1, "two" : 2})
        ```
        """
        ...

    @overload
    def __init__(self, **kwargs: VT) -> None:
        """
        Create a new frozen reversable dictionary with the key to value pairs
        given in the keyword argument list.

        For example:
        ```
        >>> frev_dict = FrozenReversableDict(one=1, two=2)
        >>> frev_dict
        FrozenReversableDict({"one" : 1, "two" : 2})
        ```
        """
        ...

    def __init__(self, *args, **kwargs) -> None:
        """Create a new frozen reversable dictionary."""
        self.__reversable_dict = ReversableDict(*args, **kwargs)
        self.__hash: Optional[int] = None

    def __repr__(self) -> str:
        standard_mapping = self.__reversable_dict.standard_mapping
        return f"{self.__class__.__name__}({standard_mapping!r})"
    __repr__.__doc__ = ReversableDict.__repr__.__doc__

    def __getitem__(self, key: KT, /) -> VT:
        return self.__reversable_dict[key]
    __getitem__.__doc__ = ReversableDict.__getitem__.__doc__

    def __iter__(self) -> Iterator[VT]:
        return iter(self.__reversable_dict)
    __iter__.__doc__ = ReversableDict.__iter__.__doc__

    def __len__(self) -> int:
        return len(self.__reversable_dict)
    __len__.__doc__ = ReversableDict.__len__.__doc__

    def __call__(self, value: VT, /, *, max_: Optional[int] = None) -> list[KT]:
        return self.__reversable_dict.reversed_get(value, max_=max_)
    __call__.__doc__ = ReversableDict.reversed_get.__doc__

    def __hash__(self) -> int:
        """Get the hash of the dictionary."""
        if self.__hash is None:
            hash_: int = 0
            for item in self.__reversable_dict.items():
                hash_ ^= hash(item)
            self.__hash = hash_
        return self.__hash

    # @property
    # def standard_mapping(self) -> dict[KT, VT]:
    #     return self.__reversable_dict.standard_mapping
    # standard_mapping.__doc__ = ReversableDict.standard_mapping.__doc__

    # @property
    # def reversed_mapping(self) -> dict[VT, list[KT]]:
    #     return self.__reversable_dict.reversed_mapping
    # reversed_mapping.__doc__ = ReversableDict.reversed_mapping.__doc__

    # def reversed_items(self) -> ItemsView[VT, list[KT]]:
    #     return self.__reversable_dict.reversed_items()
    # reversed_items.__doc__ = ReversableDict.reversed_items.__doc__

    def reversed_get(
        self,
        value: VT,
        default: Optional[list[KT]] = None, /, *,
        max_: Optional[int] = None
    ) -> Optional[list[KT]]:
        return self.__reversable_dict.reversed_get(value, default, max_)
    reversed_get.__doc__ = ReversableDict.reversed_get.__doc__


MT = TypeVar("MT", bound=Hashable)


@final
class SetView(collections.abc.Set, Generic[MT]):
    """Class defining a set view type."""

    __slots__ = {
        "__set": "The set being viewed."
    }

    def __init__(self, set_: set[MT], /) -> None:
        """Create a new set view."""
        self.__set = set_

    def __repr__(self) -> str:
        """Get an instantiable string representation of the set view."""
        return f"SetView({self.__set!r})"

    def __contains__(self, item: object, /) -> bool:
        """Check if an item is in the set view."""
        return item in self.__set

    def __iter__(self) -> Iterator[MT]:
        """Iterate over the items in the set view."""
        return iter(self.__set)

    def __len__(self) -> int:
        """Get the number of items in the set view."""
        return len(self.__set)


@final
class SetValuedMappingView(collections.abc.Mapping, Generic[KT, VT]):
    """Class defining a set-valued mapping view type."""

    __slots__ = {
        "__set_valued_mapping": "The set-valued mapping being viewed."
    }

    def __init__(self, mapping: Mapping[KT, set[VT]], /) -> None:
        """Create a new set-valued mapping view."""
        self.__set_valued_mapping: Mapping[KT, set[VT]] = mapping

    def __repr__(self) -> str:
        """
        Get an instantiable string representation of the set-valued mapping
        view.
        """
        return f"SetValuedDictView({self.__set_valued_mapping!r})"

    def __contains__(self, key: object, /) -> bool:
        """Check if a key is in the set-valued mapping view."""
        return key in self.__set_valued_mapping

    def __getitem__(self, key: KT, /) -> SetView[VT]:
        """Get the set of items in the set-valued mapping view."""
        return SetView(self.__set_valued_mapping[key])

    def __iter__(self) -> Iterator[KT]:
        """Iterate over the items in the set-valued mapping view."""
        return iter(self.__set_valued_mapping)

    def __len__(self) -> int:
        """Get the number of items in the set-valued mapping view."""
        return len(self.__set_valued_mapping)


@final
class TwoWayMap(collections.abc.MutableMapping, Generic[MT]):
    """
    Class defining a two-way mapping type.

    A two-way mapping consists of a forwards and backwards mapping, where each
    key from the forwards mapping can map to multiple keys in the backwards
    mapping, and keys of the backwards mapping automatically map back to all
    keys of the forwards that map to them.

    Subscripting the two-way mapping accesses both the forwards and backwards
    mappings. Subscripting the two-way mapping with a key from the forwards
    mapping will return a list of keys from the backwards mapping that map
    to that key in the forwards mapping. Subscripting the two-way mapping with
    a key from the backwards mapping will return a list of keys from the
    forwards mapping that map to the key in the backwards mapping.

    This structure is very useful for specifying many-to-many mappings such as
    input-output connections or parent-child relationships. It is similar to
    an undirected graph-like structure, except keys in the same side of the
    mapping cannot map to each other. The same key cannot exist in both the
    forwards and backwards mappings.

    For example:
    ```
    ## A two-way mapping of parent-child relationships.
    ## parent_1 has two children, with two different partners;
    ## parent_2 and parent_3.
    >>> twm = TwoWayMapping({"parent_1": ["child_1", "child_2"],
                             "parent_2": ["child_1"],
                             "parent_3": ["child_2"]})
    >>> twm
    TwoWayMapping({"parent_1": ["child_1", "child_2"],
                   "parent_2": ["child_1"],
                   "parent_3": ["child_2"]})

    ## Determine who the children of parent_1 are.
    >>> twm["parent_1"]
    SetView({"child_1", "child_2"})

    ## Determine who the parents of child_1 are.
    >>> twm["child_1"]
    SetView({"parent_1", "parent_2"})

    ## Check if parent_3 is a parent of child_1.
    >>> "parent_3" in twm["child_1"]
    False
    ```
    """

    __slots__ = {
        "__forwards": "The forwards mapping.",
        "__backwards": "The backwards mapping."
    }

    @overload
    def __init__(self, initialiser: Mapping[MT, Iterable[MT]]) -> None:
        """
        Create a new two-way mapping from the given mapping.

        For example:
        ```
        >>> twm = TwoWayMapping({"parent_1": ["child_1", "child_2"],
                                 "parent_2": ["child_1"],
                                 "parent_3": ["child_2"]})
        >>> twm
        TwoWayMapping({"parent_1": ["child_1", "child_2"],
                       "parent_2": ["child_1"],
                       "parent_3" : ["child_2"]})
        ```
        """
        ...

    @overload
    def __init__(self, initialiser: Iterable[tuple[MT, MT]]) -> None:
        """
        Create a new two-way mapping from the given iterable of key-value
        pairs.

        For example:
        ```
        >>> twm = TwoWayMapping([("parent_1", "child_1"),
                                 ("parent_1", "child_2"),
                                 ("parent_2", "child_1"),
                                 ("parent_3", "child_2")])
        >>> twm
        TwoWayMapping({"parent_1": ["child_1", "child_2"],
                       "parent_2": ["child_1"],
                       "parent_3" : ["child_2"]})
        ```
        """
        ...

    def __init__(
        self,
        initialiser: Mapping[MT, Iterable[MT]] | Iterable[tuple[MT, MT]]
    ) -> None:
        """Create a new two-way mapping."""
        self.__forwards: dict[MT, set[MT]] = {}
        self.__backwards: dict[MT, set[MT]] = {}
        if isinstance(initialiser, Mapping):
            for key, values in initialiser.items():
                self.__forwards[key] = set(values)
                for value in values:
                    self.__backwards.setdefault(value, set()).add(key)
        else:
            for key, value in initialiser:
                self.__forwards.setdefault(key, set()).add(value)
                self.__backwards.setdefault(value, set()).add(key)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__forwards})"

    def __getitem__(self, key: MT, /) -> SetView[MT]:
        forwards_value = self.__forwards.get(key)
        if forwards_value is not None:
            return SetView(forwards_value)
        backwards_value = self.__backwards.get(key)
        if backwards_value is not None:
            return SetView(backwards_value)
        raise KeyError(key)

    def __iter__(self) -> Iterator[MT]:
        return itertools.chain(self.__forwards, self.__backwards)

    def __len__(self) -> int:
        return len(self.__forwards) + len(self.__backwards)

    def __contains__(self, key: object, /) -> bool:
        return key in self.__forwards or key in self.__backwards

    @property
    def forwards(self) -> types.MappingProxyType[MT, SetView[MT]]:
        """
        Get the forwards mapping as a mapping proxy to a set-valued
        mapping view.
        """
        return types.MappingProxyType(SetValuedMappingView(self.__forwards))

    @property
    def backwards(self) -> types.MappingProxyType[MT, SetView[MT]]:
        """
        Get the backwards mapping as a mapping proxy to a set-valued
        mapping view.
        """
        return types.MappingProxyType(SetValuedMappingView(self.__backwards))

    def maps_to(self, forwards_key: MT, backwards_key: MT, /) -> bool:
        """Check if the forwards key maps to the backwards key."""
        return (((backwards := self.__forwards.get(forwards_key)) is not None)
                and backwards_key in backwards)

    def __setitem__(self, forwards_key: MT, backwards_key: MT, /) -> None:
        """Add a new key pair to the mapping."""
        if forwards_key in self.__backwards:
            raise KeyError(f"Key {forwards_key} already exists in the "
                           "backwards mapping.")
        if backwards_key in self.__forwards:
            raise KeyError(f"Key {backwards_key} already exists in the "
                           "forwards mapping.")
        self.__forwards.setdefault(forwards_key, set()).add(backwards_key)
        self.__backwards.setdefault(backwards_key, set()).add(forwards_key)

    def add(self, forwards_key: MT, backwards_key: MT, /) -> None:
        """Add a new key pair to the mapping."""
        self[forwards_key] = backwards_key

    def __delitem__(self, key_or_keys: MT | tuple[MT, MT], /) -> None:
        """Remove a key or key pair from the two-way mapping."""
        if key_or_keys in self:
            key: MT = key_or_keys  # type: ignore
            if key in self.__forwards:
                side = self.__forwards
                other_side = self.__backwards
            else:
                side = self.__backwards
                other_side = self.__forwards
            other_keys = side[key]
            for other_key in other_keys:
                other_side[other_key].remove(key)
                if not other_side[other_key]:
                    del other_side[other_key]
            del side[key]
        elif isinstance(key_or_keys, tuple):
            forwards_key, backwards_key = key_or_keys
            self.remove(forwards_key, backwards_key)
        else:
            raise KeyError(f"Key or keys {key_or_keys} do not exist in the "
                           "two-way mapping.")

    def remove(self, forwards_key: MT, backwards_key: MT, /) -> None:
        """Remove a key pair from the mapping."""
        if forwards_key not in self.__forwards:
            raise KeyError(f"Key {forwards_key} does not exist in the "
                           "forwards mapping.")
        if backwards_key not in self.__backwards:
            raise KeyError(f"Key {backwards_key} does not exist in the "
                           "backwards mapping.")
        self.__forwards[forwards_key].remove(backwards_key)
        self.__backwards[backwards_key].remove(forwards_key)
        if not self.__forwards[forwards_key]:
            del self.__forwards[forwards_key]
        if not self.__backwards[backwards_key]:
            del self.__backwards[backwards_key]


@final
class LayerMap(collections.abc.Mapping):
    pass


if __name__ == "__main__":
    twm = TwoWayMap({"parent_1": ["child_1", "child_2"],
                        "parent_2": ["child_1"],
                        "parent_3": ["child_2"]})
    # print(twm)
    # print(twm["parent_1"])
    # print(twm["child_1"])
    # print("parent_3" in twm["child_1"])
    # twm.add("parent_4", "child_1")
    # print(twm)
    # twm.remove("parent_4", "child_1")
    # print(twm)
    # print(list(twm.keys()))
    # print(list(twm.values()))
    # print(list(twm.items()))
    forwards = twm.forwards
    print(forwards)
    print(forwards["parent_1"])
    twm.add("parent_4", "child_1")
    print(forwards)
    # print(twm)
    