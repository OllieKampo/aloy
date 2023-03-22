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
from typing import (Generic, Hashable, ItemsView, Iterable, Iterator, Mapping,
                    Optional, TypeVar, final, overload)

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

    def __iter__(self) -> Iterator[VT]:
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
        ## If the key is already in the standard mapping, the value it
        ## currently maps to must be replaced in the reversed mapping.
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
        ## Remove the reference of this value being mapped to by the given key.
        self.__reversed_dict[value].remove(key)
        ## If no keys map to this value anymore then remove the value as well.
        if len(self.__reversed_dict[value]) == 0:
            del self.__reversed_dict[value]

    def __iter__(self) -> Iterator[KT]:
        """Iterate over the keys in the standard dictionary mapping."""
        yield from self.__dict

    def __len__(self) -> int:
        """Get the number of items in the standard dictionary mapping."""
        return len(self.__dict)

    def __call__(self, value: VT, /, max_: int | None = None) -> list[KT]:
        """
        Get the list of keys that map to a value in the reversed mapping.

        If `max_` is given and not None, return at most the first `max_` keys.
        """
        if max_ is None:
            return self.__reversed_dict[value].copy()
        keys: list[KT] = self.__reversed_dict[value]
        return keys[0:min(max_, len(keys))]

    def __copy__(self) -> "ReversableDict[KT, VT]":
        """Get a shallow copy of the reversable dictionary."""
        return self.__class__(self.__dict)

    @property
    def standard_mapping(self) -> dict[KT, VT]:
        """Get the standard mapping as a normal dictionary."""
        return self.__dict

    @property
    def reversed_mapping(self) -> dict[VT, list[KT]]:
        """Get the reversed mapping as a normal dictionary."""
        return self.__reversed_dict

    def reversed_items(self) -> ItemsView[VT, list[KT]]:
        """
        Get a reversed dictionary items view.

        Such that:
            - Its keys are the values of the standard dictionary,
            - Its values are lists of keys from the standard
              dictionary which map to the respective values.
        """
        return self.__reversed_dict.items()

    def reversed_get(
        self,
        value: VT,
        default: list[KT] | None = None, /,
        max_: int | None = None
    ) -> Optional[list[KT]]:
        """
        Get the list of keys that map to a value in the reversed mapping.

        If the value is not found, return `default` if given; otherwise, raise
        a KeyError.

        If `max_` is given and not None, return at most the first `max_` keys.
        """
        if value not in self.__reversed_dict:
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
    ) -> list[KT]:
        """
        Remove a value from the reversed mapping and return the list of keys
        that mapped to it.

        If the key is not found, return `default` if given; otherwise, raise
        a KeyError.
        """
        if value not in self.__reversed_dict:
            return default
        keys: list[KT] = self(value)
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
        self.__reversable_dict: ReversableDict[KT, VT] = dict(*args, **kwargs)
        self.__hash: Optional[int] = None

    def __repr__(self) -> str:  # noqa: D105
        standard_mapping = self.__reversable_dict.standard_mapping
        return f"{self.__class__.__name__}({standard_mapping!r})"
    __repr__.__doc__ = ReversableDict.__repr__.__doc__

    def __getitem__(self, key: KT, /) -> VT:  # noqa: D105
        return self.__reversable_dict[key]
    __getitem__.__doc__ = ReversableDict.__getitem__.__doc__

    def __iter__(self) -> Iterator[VT]:  # noqa: D105
        return iter(self.__reversable_dict)
    __iter__.__doc__ = ReversableDict.__iter__.__doc__

    def __len__(self) -> int:  # noqa: D105
        return len(self.__reversable_dict)
    __len__.__doc__ = ReversableDict.__len__.__doc__

    def __call__(self, value: VT, /, *, max_: Optional[int] = None) -> list[KT]:  # noqa: D102
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

    @property
    def standard_mapping(self) -> dict[KT, VT]:  # noqa: D102
        return self.__reversable_dict.standard_mapping
    standard_mapping.__doc__ = ReversableDict.standard_mapping.__doc__

    @property
    def reversed_mapping(self) -> dict[VT, list[KT]]:  # noqa: D102
        return self.__reversable_dict.reversed_mapping
    reversed_mapping.__doc__ = ReversableDict.reversed_mapping.__doc__

    def reversed_items(self) -> ItemsView[VT, list[KT]]:  # noqa: D102
        return self.__reversable_dict.reversed_items()
    reversed_items.__doc__ = ReversableDict.reversed_items.__doc__

    def reversed_get(
        self,
        value: VT,
        default: Optional[list[KT]] = None, /, *,
        max_: Optional[int] = None
    ) -> Optional[list[KT]]:  # noqa: D102
        return self.__reversable_dict.reversed_get(value, default, max_)
    reversed_get.__doc__ = ReversableDict.reversed_get.__doc__


MT = TypeVar("MT", bound=Hashable)


@final
class TwoWayMap(collections.abc.Mapping, Generic[MT]):
    """
    Class defining a two-way mapping type.

    A two-way mapping consists of a forwards and backwards mapping, where each
    key from the forwards mapping can map to multiple keys in the backwards
    mapping, and keys of the backwards mapping automatically map back those of
    the forwards that map to them.

    This structure is very useful for specifying one-to-many mappings such as
    input-output connections or parent-child relationships. It is similar to
    an undirected graph-like structure, except keys in the same side of the
    mapping cannot map to each other.

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
    {"child_1", "child_2"}

    ## Determine who the parents of child_1 are.
    >>> twm["child_1"]
    {"parent_1", "parent_2"}

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

    def __getitem__(self, key: KT, /) -> set[MT]:
        return self.__forwards[key]

    def __iter__(self) -> Iterator[MT]:
        return iter(self.__forwards)

    def __len__(self) -> int:
        return len(self.__forwards)

    def __call__(self, value: MT, /) -> set[MT]:
        return self.__backwards[value]

    @property
    def forwards(self) -> dict[MT, set[MT]]:
        """Get the forwards mapping."""
        return self.__forwards

    @property
    def backwards(self) -> dict[MT, set[MT]]:
        """Get the backwards mapping."""
        return self.__backwards

    def backwards_items(self) -> ItemsView[MT, set[MT]]:
        """Get the items of the backwards mapping."""
        return self.__backwards.items()

    def backwards_get(
            self,
            value: MT,
            default: set[MT] | None = None, /
    ) -> set[MT] | None:
        """Get the keys that map to a value in the backwards mapping."""
        return self.__backwards.get(value, default)

    def add(self, forwards_key: MT, backwards_key: MT, /) -> None:
        """Add a new key pair to the mapping."""
        self.__forwards.setdefault(forwards_key, set()).add(backwards_key)
        self.__backwards.setdefault(backwards_key, set()).add(forwards_key)

    def remove(self, forwards_key: MT, backwards_key: MT, /) -> None:
        """Remove a key pair from the mapping."""
        self.__forwards[forwards_key].remove(backwards_key)
        self.__backwards[backwards_key].remove(forwards_key)
        if not self.__forwards[forwards_key]:
            del self.__forwards[forwards_key]
        if not self.__backwards[backwards_key]:
            del self.__backwards[backwards_key]


@final
class LayerMap(collections.abc.Mapping):
    pass
