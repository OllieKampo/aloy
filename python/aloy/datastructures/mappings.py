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
from typing import (Generic, Hashable, Iterable, Iterator, Mapping, Optional,
                    TypeAlias, TypeVar, final, overload)

from aloy.datastructures.views import (ListValuedMappingView, ListView,
                                       SetValuedMappingView, SetView)

__copyright__ = "Copyright (C) 2022 Oliver Michael Kamperis"
__license__ = "GPL-3.0"

__all__ = (
    "frozendict",
    "ReversableDict",
    "FrozenReversableDict",
    "TwoWayMap",
    "LayerMap"
)


def __dir__() -> tuple[str, ...]:
    """Get the names of module attributes."""
    return __all__


_KT = TypeVar("_KT", bound=Hashable)
_VT = TypeVar("_VT", bound=Hashable)
_T = TypeVar("_T")


@final
class frozendict(  # pylint: disable=invalid-name
    collections.abc.Mapping,
    Generic[_KT, _VT]
):
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
    def __init__(self, mapping: Mapping[_KT, _VT], /) -> None:
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
    def __init__(self, iterable: Iterable[tuple[_KT, _VT]], /) -> None:
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
    def __init__(self, **kwargs: _VT) -> None:
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
        self.__dict: dict[_KT, _VT] = dict(*args, **kwargs)
        self.__hash: Optional[int] = None

    def __repr__(self) -> str:
        """Get an instantiable string representation of the dictionary."""
        return f"frozendict({self.__dict})"

    def __getitem__(self, key: _KT, /) -> _VT:
        """Get the value associated with the given key."""
        return self.__dict[key]

    def __iter__(self) -> Iterator[_KT]:
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

    def __copy__(self) -> "frozendict[_KT, _VT]":
        """Create a shallow copy of the dictionary."""
        return frozendict(self)


@final
class ReversableDict(collections.abc.MutableMapping, Generic[_KT, _VT]):
    """
    Class defining a mutable reversable dictionary type.

    Supports all methods of standard dictionaries with the added ability to
    find all keys that map to a specific value, this is a one-to-many mapping
    called the reversed mapping (unlike the standard dictionary which is
    either one-to-one or many-to-one).

    The reversable dict keeps the reversed version of itself in memory.
    Therefore, insertions are slower and memory usage is higher
    than a standard dictionary, but reversed lookups are fast.
    This also requires that dictionary values be hashable.

    Example Usage
    -------------
    ```
    >>> from aloy.datastructures.mappings import ReversableDict
    >>> rev_dict = ReversableDict({"a": 1, "b": 2, "z": 2})
    >>> rev_dict
    ReversableDict({"a": 1, "b": 2, "z": 2})

    # Access the standard and reversed mappings directly.
    >>> rev_dict.standard_mapping
    {"a": 1, "b": 2, "z": 2}
    >>> rev_dict.reversed_mapping
    {1 : ["a"], 2 : ["b", "z"]}

    # Check what 'a' maps to (as in a standard dictionary).
    >>> rev_dict["a"]
    1

    # Check what keys map to 2 (a reverse operation to a standard dictionary).
    # The keys are given as a list where order reflects insertion order.
    >>> rev_dict(2)
    ["b", "z"]

    # Objects are mutable and updates update standard and reverse mappings.
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
    def __init__(self, mapping: Mapping[_KT, _VT], /) -> None:
        """
        Create a new reversable dictionary initialised
        from a mapping object's (key, value) pairs.

        For example:
        ```
        >>> rev_dict = ReversableDict({"one": 1, "two": 2})
        >>> rev_dict
        ReversableDict({"one": 1, "two": 2})
        ```
        """
        ...

    @overload
    def __init__(self, iterable: Iterable[tuple[_KT, _VT]], /) -> None:
        """
        Create a new reversable dictionary initialized from
        an iterable of tuples defining (key, value) pairs.

        For example:
        ```
        >>> rev_dict = ReversableDict([("one", 1), ("two", 2)])
        >>> rev_dict
        ReversableDict({"one": 1, "two": 2})
        ```
        """
        ...

    @overload
    def __init__(self, **kwargs: _VT) -> None:
        """
        Create a new reversable dictionary initialized with the
        key to value pairs given in the keyword argument list.

        For example:
        ```
        >>> rev_dict = ReversableDict(one=1, two=2)
        >>> rev_dict
        ReversableDict({"one": 1, "two": 2})
        ```
        """
        ...

    def __init__(self, *args, **kwargs) -> None:
        """Create a new reversable dictionary."""
        self.__dict: dict[_KT, _VT] = {}
        self.__reversed_dict: dict[_VT, list[_KT]] = {}
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def __copy__(self) -> "ReversableDict[_KT, _VT]":
        """Get a shallow copy of the reversable dictionary."""
        return self.__class__(self.__dict)

    def __repr__(self) -> str:
        """
        Get an instantiable string representation of the reversable
        dictionary.
        """
        return f"{self.__class__.__name__}({self.__dict})"

    def __getitem__(self, key: _KT, /) -> _VT:
        """Get the value for the given key in the standard mapping."""
        return self.__dict[key]

    def __setitem__(self, key: _KT, value: _VT, /) -> None:
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

    def __delitem__(self, key: _KT, /) -> None:
        """
        Delete a item (given by a key) from the standard and reversed
        mapping.
        """
        # Remove from the standard mapping.
        value: _VT = self.__dict[key]
        del self.__dict[key]
        # Remove from the reversed mapping.
        self.__del_reversed_item(key, value)

    def __del_reversed_item(
        self,
        key: _KT,
        value: _VT, /
    ) -> None:
        """
        Delete an item (given by a [key, value] pair) from the reversed
        mapping.
        """
        # Remove the reference of this value being mapped to by the given key.
        self.__reversed_dict[value].remove(key)
        # If no keys map to this value anymore then remove the value as well.
        if len(self.__reversed_dict[value]) == 0:
            del self.__reversed_dict[value]

    def __iter__(self) -> Iterator[_KT]:
        """Iterate over the keys in the standard dictionary mapping."""
        yield from self.__dict

    def __len__(self) -> int:
        """Get the number of items in the standard dictionary mapping."""
        return len(self.__dict)

    def __call__(
        self,
        value: _VT, /,
        max_: int | None = None
    ) -> ListView[_KT]:
        """
        Get the list of keys that map to a value in the reversed mapping.

        If `max_` is given and not None, return at most the first `max_` keys.
        """
        if max_ is None:
            return ListView(self.__reversed_dict[value])
        keys: list[_KT] = self.__reversed_dict[value]
        return ListView(keys[0:min(max_, len(keys))])

    @property
    def standard_mapping(self) -> types.MappingProxyType[_KT, _VT]:
        """Get the standard mapping as a normal dictionary."""
        return types.MappingProxyType(self.__dict)

    @property
    def reversed_mapping(self) -> types.MappingProxyType[_VT, ListView[_KT]]:
        """
        Get the reversed mapping as a normal dictionary.

        Such that:
            - Its keys are the values of the standard dictionary,
            - Its values are lists of keys from the standard
              dictionary which map to the respective values.
        """
        return types.MappingProxyType(
            ListValuedMappingView(self.__reversed_dict)
        )

    @overload
    def reversed_get(
        self,
        value: _VT, /, *,
        max_: int | None = None
    ) -> ListView[_KT] | None:
        ...

    @overload
    def reversed_get(
        self,
        value: _VT,
        default: list[_KT], /,
        max_: int | None = None
    ) -> ListView[_KT]:
        ...

    @overload
    def reversed_get(
        self,
        value: _VT,
        default: _T, /, *,
        max_: int | None = None
    ) -> ListView[_KT] | _T:
        ...

    def reversed_get(
        self,
        value: _VT,
        default: list[_KT] | _T | None = None, /,
        max_: int | None = None
    ) -> ListView[_KT] | _T | None:
        """
        Get the list of keys that map to a value in the reversed mapping.

        If the value is not found, return `default` if given; otherwise, raise
        a KeyError.

        If `max_` is given and not None, return at most the first `max_` keys.
        """
        if value not in self.__reversed_dict:
            if default is not None and isinstance(default, list):
                return ListView(default)
            return default
        return self(value, max_=max_)

    def reversed_set(self, value: _VT, /, *keys: _KT) -> None:
        """
        Set a value to map to a series of keys in the reversed mapping.

        Keys are inserted in the order they are given.
        """
        for key in keys:
            self[key] = value

    @overload
    def reversed_pop(
        self,
        value: _VT, /
    ) -> list[_KT] | None:
        ...

    @overload
    def reversed_pop(
        self,
        value: _VT,
        default: list[_KT], /
    ) -> list[_KT]:
        ...

    @overload
    def reversed_pop(
        self,
        value: _VT,
        default: _T, /
    ) -> list[_KT] | _T:
        ...

    def reversed_pop(
        self,
        value: _VT,
        default: list[_KT] | _T | None = None, /
    ) -> list[_KT] | _T | None:
        """
        Remove a value from the reversed mapping and return the list of keys
        that mapped to it.

        If the key is not found, return `default` if given; otherwise, raise
        a KeyError.
        """
        if value not in self.__reversed_dict:
            return default
        keys: list[_KT] = self.__reversed_dict[value]
        for key in keys:
            del self[key]
        return keys


@final
class FrozenReversableDict(collections.abc.Mapping, Generic[_KT, _VT]):
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
    def __init__(self, mapping: Mapping[_KT, _VT]) -> None:
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
    def __init__(self, iterable: Iterable[tuple[_KT, _VT]]) -> None:
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
    def __init__(self, **kwargs: _VT) -> None:
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
        self.__reversable_dict = ReversableDict[_KT, _VT](*args, **kwargs)
        self.__hash: Optional[int] = None

    def __repr__(self) -> str:
        """
        Get an instantiable string representation of the frozen reversable
        dictionary.
        """
        standard_mapping = self.__reversable_dict.standard_mapping
        return f"{self.__class__.__name__}({standard_mapping!r})"

    # pylint: disable=missing-function-docstring
    def __getitem__(self, key: _KT, /) -> _VT:
        return self.__reversable_dict[key]
    __getitem__.__doc__ = ReversableDict.__getitem__.__doc__

    def __iter__(self) -> Iterator[_KT]:
        return iter(self.__reversable_dict)
    __iter__.__doc__ = ReversableDict.__iter__.__doc__

    def __len__(self) -> int:
        return len(self.__reversable_dict)
    __len__.__doc__ = ReversableDict.__len__.__doc__

    def __call__(
        self,
        value: _VT, /, *,
        max_: Optional[int] = None
    ) -> ListView[_KT]:
        return self.__reversable_dict(value, max_=max_)
    __call__.__doc__ = ReversableDict.__call__.__doc__

    def __hash__(self) -> int:
        """Get the hash of the dictionary."""
        if self.__hash is None:
            hash_: int = 0
            for item in self.__reversable_dict.items():
                hash_ ^= hash(item)
            self.__hash = hash_
        return self.__hash

    @property
    def standard_mapping(
        self
    ) -> types.MappingProxyType[_KT, _VT]:
        return self.__reversable_dict.standard_mapping
    standard_mapping.__doc__ = ReversableDict.standard_mapping.__doc__

    @property
    def reversed_mapping(
        self
    ) -> types.MappingProxyType[_VT, ListView[_KT]]:
        return self.__reversable_dict.reversed_mapping
    reversed_mapping.__doc__ = ReversableDict.reversed_mapping.__doc__

    def reversed_get(
        self,
        value: _VT,
        default: list[_KT] | None = None, /, *,
        max_: int | None = None
    ) -> ListView[_KT] | None:
        return self.__reversable_dict.reversed_get(value, default, max_=max_)
    reversed_get.__doc__ = ReversableDict.reversed_get.__doc__
    # pylint: enable=missing-function-docstring


# TODO: InvertableUniqueKeysDict/BijectiveMap, MultiKeyDict, MultiValueDict,
# MultiItemDict.


FK = TypeVar("FK", bound=Hashable)
BK = TypeVar("BK", bound=Hashable)
DK = TypeVar("DK")
TwoWayMapInit: TypeAlias = Mapping[FK, Iterable[BK]] | Iterable[tuple[FK, BK]]


# TODO: Add support for setting;
# kind: Literal["one-to-one", "one-to-many", "many-to-one", "many-to-many"]
@final
class TwoWayMap(collections.abc.MutableMapping, Generic[FK, BK]):
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
    def __init__(self) -> None:
        """
        Create a new empty two-way mapping.

        For example:
        ```
        >>> twm = TwoWayMapping()
        >>> twm
        TwoWayMapping({})
        ```
        """
        ...

    @overload
    def __init__(self, mapping: Mapping[FK, Iterable[BK]], /) -> None:
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
    def __init__(self, iterable: Iterable[tuple[FK, BK]], /) -> None:
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

    def __init__(  # type: ignore
        self,
        init: TwoWayMapInit | None = None, /
    ) -> None:
        """Create a new two-way mapping."""
        self.__forwards: dict[FK, set[BK]] = {}
        self.__backwards: dict[BK, set[FK]] = {}
        if init is not None:
            if isinstance(init, Mapping):
                for key, values in init.items():
                    self.__forwards[key] = set(values)
                    for value in values:
                        self.__backwards.setdefault(value, set()).add(key)
            else:
                for key, value in init:
                    self.__forwards.setdefault(key, set()).add(value)
                    self.__backwards.setdefault(value, set()).add(key)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__forwards})"

    @overload
    def __getitem__(self, key: FK, /) -> SetView[BK]:
        """
        Get the set of backwards keys that map to the given key in the forwards
        mapping.

        For example:
        ```
        >>> twm = TwoWayMapping({"parent_1": ["child_1", "child_2"],
                                 "parent_2": ["child_1"],
                                 "parent_3": ["child_2"]})
        >>> twm["parent_1"]
        SetView({"child_1", "child_2"})
        ```
        """
        ...

    @overload
    def __getitem__(self, key: BK, /) -> SetView[FK]:
        """
        Get the set of forwards keys that map to the given key in the backwards
        mapping.

        For example:
        ```
        >>> twm = TwoWayMapping({"parent_1": ["child_1", "child_2"],
                                 "parent_2": ["child_1"],
                                 "parent_3": ["child_2"]})
        >>> twm["child_1"]
        SetView({"parent_1", "parent_2"})
        ```
        """
        ...

    def __getitem__(self, key: FK | BK, /) -> SetView[FK] | SetView[BK]:
        forwards_value = self.__forwards.get(key)  # type: ignore
        if forwards_value is not None:
            return SetView(forwards_value)
        backwards_value = self.__backwards.get(key)  # type: ignore
        if backwards_value is not None:
            return SetView(backwards_value)
        raise KeyError(key)

    def __iter__(self) -> Iterator[FK | BK]:
        """Iterate over all the keys in both sides of the mapping."""
        return itertools.chain(self.__forwards, self.__backwards)

    def __len__(self) -> int:
        """Get the number of keys in both sides of the mapping."""
        return len(self.__forwards) + len(self.__backwards)

    def __contains__(self, key: object, /) -> bool:
        """Check if the given key is in either side of the mapping."""
        return key in self.__forwards or key in self.__backwards

    @property
    def forwards(self) -> types.MappingProxyType[FK, SetView[BK]]:
        """
        Get the forwards mapping as a mapping proxy to a set-valued
        mapping view.
        """
        return types.MappingProxyType(SetValuedMappingView(self.__forwards))

    @property
    def backwards(self) -> types.MappingProxyType[BK, SetView[FK]]:
        """
        Get the backwards mapping as a mapping proxy to a set-valued
        mapping view.
        """
        return types.MappingProxyType(SetValuedMappingView(self.__backwards))

    def forwards_get(
        self,
        key: FK,
        default: Iterable[BK] | None = None, /
    ) -> SetView[BK] | None:
        """
        Get the set of backwards keys that map to the given key in the forwards
        mapping, or the default value if the key is not present.

        For example:
        ```
        >>> twm = TwoWayMapping({"parent_1": ["child_1", "child_2"],
                                 "parent_2": ["child_1"],
                                 "parent_3": ["child_2"]})
        >>> twm["parent_1"]
        SetView({"child_1", "child_2"})
        ```
        """
        backwards = self.__forwards.get(key)
        if backwards is not None:
            return SetView(backwards)
        return SetView(set(default)) if default is not None else None

    def backwards_get(
        self,
        key: BK,
        default: DK | None = None, /
    ) -> SetView[FK] | DK | None:
        """
        Get the set of forwards keys that map to the given key in the backwards
        mapping, or the default value if the key is not present.

        For example:
        ```
        >>> twm = TwoWayMapping({"parent_1": ["child_1", "child_2"],
                                 "parent_2": ["child_1"],
                                 "parent_3": ["child_2"]})
        >>> twm["child_1"]
        SetView({"parent_1", "parent_2"})
        ```
        """
        forwards = self.__backwards.get(key)
        if forwards is not None:
            return SetView(forwards)
        return default

    def maps_to(self, forwards_key: FK, backwards_key: BK, /) -> bool:
        """Check if the forwards key maps to the backwards key."""
        return (((backwards := self.__forwards.get(forwards_key)) is not None)
                and backwards_key in backwards)

    def __setitem__(self, forwards_key: FK, backwards_key: BK, /) -> None:
        """
        Add a new key pair to the mapping, from the forwards key to the
        backwards key.

        If the forwards key already exists in the backwards mapping, or the
        backwards key already exists in the forwards mapping, a KeyError is
        raised.

        For example:
        ```
        >>> twm = TwoWayMapping()
        >>> twm["parent_1"] = "child_1"
        >>> twm["parent_2"] = "child_1"
        >>> twm["parent_1"]
        SetView({"child_1"})
        >>> twm["child_1"]
        SetView({"parent_1", "parent_2"})
        ```
        """
        if forwards_key in self.__backwards:
            raise KeyError(f"Key {forwards_key} already exists in the "
                           "backwards mapping.")
        if backwards_key in self.__forwards:
            raise KeyError(f"Key {backwards_key} already exists in the "
                           "forwards mapping.")
        self.__forwards.setdefault(forwards_key, set()).add(backwards_key)
        self.__backwards.setdefault(backwards_key, set()).add(forwards_key)

    # pylint: disable=missing-function-docstring
    def add(self, forwards_key: FK, backwards_key: BK, /) -> None:
        self[forwards_key] = backwards_key
    add.__doc__ = __setitem__.__doc__
    # pylint: enable=missing-function-docstring

    def add_many(
        self,
        forwards_key: FK,
        backwards_keys: Iterable[BK], /
    ) -> None:
        """
        Add new key pairs to the mapping, from the forwards key to the
        backwards key.

        If the forwards key already exists in the backwards mapping, or any of
        the backwards keys already exist in the forwards mapping, a KeyError is
        raised.

        For example:
        ```
        >>> twm = TwoWayMapping()
        >>> twm.add_many("parent_1", ["child_1", "child_2"])
        >>> twm["parent_1"]
        SetView({"child_1", "child_2"})
        >>> twm["child_1"]
        SetView({"parent_1"})
        ```
        """
        if forwards_key in self.__backwards:
            raise KeyError(f"Key {forwards_key} already exists in the "
                           "backwards mapping.")
        backwards_keys = set(backwards_keys)
        if conflicts := backwards_keys & self.__forwards.keys():
            raise KeyError(f"Keys {conflicts} already exist in the forwards "
                           "mapping.")
        self.__forwards.setdefault(forwards_key, set()).update(backwards_keys)
        for backwards_key in backwards_keys:
            self.__backwards.setdefault(backwards_key, set()).add(forwards_key)

    def __delitem__(self, key_or_keys: FK | BK | tuple[FK, BK], /) -> None:
        """
        Remove a key or key pair from the two-way mapping.

        If the key is not present, a KeyError is raised.
        """
        # The key type might be a tuple, so check if key is present
        # first, if it is, then it is a single key, otherwise,
        # assume it is a key pair.
        if key_or_keys in self:
            key: FK | BK = key_or_keys  # type: ignore
            if key in self.__forwards:
                side = self.__forwards
                other_side = self.__backwards
            else:
                side = self.__backwards  # type: ignore
                other_side = self.__forwards  # type: ignore
            other_keys = side[key]  # type: ignore
            for other_key in other_keys:
                other_side[other_key].remove(key)  # type: ignore
                if not other_side[other_key]:
                    del other_side[other_key]
            del side[key]  # type: ignore
        elif isinstance(key_or_keys, tuple):
            if len(key_or_keys) != 2:
                raise ValueError("Key pair must be a tuple of length 2."
                                 f"Got; {len(key_or_keys)} instead."
                                 "If your mapping is tuple-keyed, then "
                                 "the key does not exist.")
            forwards_key, backwards_key = key_or_keys
            try:
                self.remove(forwards_key, backwards_key)
            except KeyError as error:
                raise KeyError(
                    f"Key or keys {key_or_keys} do not exist in "
                    "the two-way mapping. If your mapping is "
                    "tuple-keyed, then the key does not exist. "
                    "Otherwise, one of the keys does not exist."
                ) from error
        else:
            raise KeyError(f"Key or keys {key_or_keys} do not exist in the "
                           "two-way mapping.")

    def forwards_remove(self, key: FK, /) -> None:
        """Remove a key from the forwards mapping."""
        if key not in self.__forwards:
            raise KeyError(f"Key {key} does not exist in the forwards "
                           "mapping.")
        for other_key in self.__forwards[key]:
            self.__backwards[other_key].remove(key)
            if not self.__backwards[other_key]:
                del self.__backwards[other_key]
        del self.__forwards[key]

    def backwards_remove(self, key: BK, /) -> None:
        """Remove a key from the backwards mapping."""
        if key not in self.__backwards:
            raise KeyError(f"Key {key} does not exist in the backwards "
                           "mapping.")
        for other_key in self.__backwards[key]:
            self.__forwards[other_key].remove(key)
            if not self.__forwards[other_key]:
                del self.__forwards[other_key]
        del self.__backwards[key]

    def remove(self, forwards_key: FK, backwards_key: BK, /) -> None:
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

    def remove_many(
        self,
        forwards_key: FK,
        backwards_keys: Iterable[BK], /
    ) -> None:
        """Remove key pairs from the mapping."""
        if forwards_key not in self.__forwards:
            raise KeyError(f"Key {forwards_key} does not exist in the "
                           "forwards mapping.")
        self.__forwards[forwards_key].difference_update(backwards_keys)
        for backwards_key in backwards_keys:
            self.__backwards[backwards_key].remove(forwards_key)
            if not self.__backwards[backwards_key]:
                del self.__backwards[backwards_key]
        if not self.__forwards[forwards_key]:
            del self.__forwards[forwards_key]


MT = TypeVar("MT", bound=Hashable)


class LayerMapValue(collections.abc.Sequence, Generic[MT]):
    """Class defining a layered mapping value type."""

    __slots__ = ("__list",)

    def __init__(self, list_: list[MT | None]) -> None:
        """Create a new layered mapping value."""
        self.__list: list[MT | None] = list_

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__list!r})"

    @overload
    def __getitem__(self, index: int, /) -> MT | None:
        ...

    @overload
    def __getitem__(self, index: slice, /) -> list[MT | None]:
        ...

    def __getitem__(
        self,
        index: int | slice, /
    ) -> MT | None | list[MT | None]:
        return self.__list[index]

    def __len__(self) -> int:
        return 2

    @property
    def left(self) -> MT | None:
        """Get the left value."""
        return self.__list[0]

    @property
    def right(self) -> MT | None:
        """Get the right value."""
        return self.__list[1]


@final
class LayerMap(collections.abc.Mapping, Generic[MT]):
    """
    Class defining a layered mapping type.

    A layered mapping is a mapping that contains multiple layers
    arranged in a 'horizontal' sequence. Each layer is a mapping
    that maps to layers to the 'left' and 'right' of it. The same
    key cannot exist in multiple different layers. This is similar
    to a graph-like structure, but keys in the same layer cannot
    map to themselves (they are not directly connected by an arc).
    """

    __slots__ = {
        "__layers": "The layers of the mapping.",
        "__key_to_layer": "The mapping of keys to layers.",
        "__limits": "The limits of the mapping, (min, max)."
    }

    def __init__(self, base_layer: Iterable[MT]) -> None:
        """Create a new layered mapping."""
        self.__layers: dict[int, dict[MT, list[MT | None]]] = {
            0: {key: [None, None] for key in base_layer}
        }
        self.__key_to_layer: dict[MT, int] = {
            key: 0 for key in base_layer
        }
        self.__limits: list[int] = [0, 0]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__layers})"

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.__layers})"

    def __getitem__(self, key: MT, /) -> LayerMapValue[MT]:
        """Get the value of the given key."""
        layer = self.__key_to_layer[key]
        return LayerMapValue(self.__layers[layer][key])

    def __iter__(self) -> Iterator[MT]:
        """Iterate over the keys of the mapping."""
        return iter(self.__key_to_layer)

    def __len__(self) -> int:
        """Get the number of keys in the mapping."""
        return len(self.__key_to_layer)

    @property
    def first_layer(self) -> int:
        """Get the first layer."""
        return self.__limits[0]

    @property
    def last_layer(self) -> int:
        """Get the last layer."""
        return self.__limits[1]

    def layer_of(self, key: MT, /) -> int:
        """Get the layer of the given key."""
        return self.__key_to_layer[key]

    def layer_size(self, layer: int, /) -> int:
        """Get the size of the given layer."""
        return len(self.__layers[layer])

    def get_layer(self, layer: int, /) -> dict[MT, LayerMapValue[MT]]:
        """Get the given layer."""
        layer_ = self.__layers[layer]
        return {key: LayerMapValue(layer_[key]) for key in layer_}

    def get_first_layer(self) -> dict[MT, LayerMapValue[MT]]:
        """Get the first layer."""
        return self.get_layer(self.first_layer)

    def get_last_layer(self) -> dict[MT, LayerMapValue[MT]]:
        """Get the last layer."""
        return self.get_layer(self.last_layer)

    def get_left(self, key: MT, /) -> MT:
        """Get the key to the left of the given key."""
        layer = self.__key_to_layer[key]
        left = self.__layers[layer][key][0]
        if left is None:
            raise KeyError(f"Key {key} does not have a key to the left.")
        return left

    def get_right(self, key: MT, /) -> MT:
        """Get the key to the right of the given key."""
        layer = self.__key_to_layer[key]
        right = self.__layers[layer][key][1]
        if right is None:
            raise KeyError(f"Key {key} does not have a key to the right.")
        return right

    def set_left(self, key: MT, left_key: MT, /) -> None:
        """Set the key to the left of the given key."""
        layer = self.__key_to_layer[key]
        if layer == self.__limits[0]:
            if left_key in self.__key_to_layer:
                raise KeyError(f"Key {left_key} already exists in "
                               f"layer {self.__key_to_layer[left_key]}")
            self.__limits[0] = layer - 1
            self.__key_to_layer[key] = layer - 1
            self.__layers[layer - 1] = {left_key: [None, key]}
        elif left_key in self.__key_to_layer:
            if self.__key_to_layer[left_key] != layer - 1:
                raise KeyError(f"Key {left_key} already exists in "
                               f"layer {self.__key_to_layer[left_key]}")
            self.__layers[layer][key][0] = left_key
            self.__layers[layer - 1][left_key][1] = key
        else:
            self.__key_to_layer[key] = layer - 1
            self.__layers[layer - 1] = {left_key: [None, key]}
        self.__layers[layer][key][0] = left_key

    def set_right(self, key: MT, right_key: MT, /) -> None:
        """Set the key to the right of the given key."""
        layer = self.__key_to_layer[key]
        if layer == self.__limits[1]:
            if right_key in self.__key_to_layer:
                raise KeyError(f"Key {right_key} already exists in "
                               f"layer {self.__key_to_layer[right_key]}")
            self.__limits[1] = layer + 1
            self.__key_to_layer[key] = layer + 1
            self.__layers[layer + 1] = {right_key: [key, None]}
        elif right_key in self.__key_to_layer:
            if self.__key_to_layer[right_key] != layer + 1:
                raise KeyError(f"Key {right_key} already exists in "
                               f"layer {self.__key_to_layer[right_key]}")
            self.__layers[layer][key][1] = right_key
            self.__layers[layer + 1][right_key][0] = key
        else:
            self.__key_to_layer[key] = layer + 1
            self.__layers[layer + 1] = {right_key: [key, None]}
        self.__layers[layer][key][1] = right_key

    def add_key(self, key: MT, layer: int) -> None:
        """Add a new key to the mapping."""
        if key in self.__key_to_layer:
            raise ValueError("Key already exists in the mapping.")
        if layer < self.__limits[0] - 1 or layer > self.__limits[1] + 1:
            raise ValueError("Layer is not in or adjacent to any other layer.")
        self.__key_to_layer[key] = layer
        self.__layers[layer] = {key: [None, None]}
        if layer < self.__limits[0]:
            self.__limits[0] = layer
        elif layer > self.__limits[1]:
            self.__limits[1] = layer

    def remove_key(self, key: MT, /) -> None:
        """Remove a key from the mapping."""
        if key not in self.__key_to_layer:
            raise KeyError(f"Key {key} does not exist in the mapping.")
        layer = self.__key_to_layer[key]
        if (len(self.__layers[layer]) == 1
                and layer != self.__limits[0]
                and layer != self.__limits[1]):
            raise ValueError("Cannot remove key from layer with only one key "
                             "unless it is the first or last layer.")
        left = self.__layers[layer][key][0]
        right = self.__layers[layer][key][1]
        if left is not None:
            self.__layers[layer][left][1] = None
        if right is not None:
            self.__layers[layer][right][0] = None
        del self.__layers[layer][key]
        del self.__key_to_layer[key]
        if not self.__layers[layer]:
            del self.__layers[layer]
            if layer == self.__limits[0]:
                self.__limits[0] = layer + 1
            elif layer == self.__limits[1]:
                self.__limits[1] = layer - 1

    def connect(self, key_1: MT, key_2: MT, /) -> None:
        """Connect two keys."""
        if key_1 not in self.__key_to_layer:
            raise KeyError(f"Key {key_1} does not exist in the mapping.")
        if key_2 not in self.__key_to_layer:
            raise KeyError(f"Key {key_2} does not exist in the mapping.")
        layer_1 = self.__key_to_layer[key_1]
        layer_2 = self.__key_to_layer[key_2]
        if abs(layer_1 - layer_2) != 1:
            raise ValueError("Keys are not adjacent to each other.")
        if layer_1 < layer_2:
            self.__layers[layer_1][key_1][1] = key_2
            self.__layers[layer_2][key_2][0] = key_1
        else:
            self.__layers[layer_1][key_1][0] = key_2
            self.__layers[layer_2][key_2][1] = key_1


def __main() -> None:
    """Execute the main routine."""
    twm = TwoWayMap({"parent_1": ["child_1", "child_2"],
                     "parent_2": ["child_1"],
                     "parent_3": ["child_2"]})
    print(twm)
    print(twm["parent_1"])
    print(twm["child_1"])
    print("parent_3" in twm["child_1"])
    twm.add("parent_4", "child_1")
    print(twm)
    twm.remove("parent_4", "child_1")
    print(twm)
    print(list(twm.keys()))
    print(list(twm.values()))
    print(list(twm.items()))
    forwards = twm.forwards
    print(forwards)
    print(forwards["parent_1"])
    twm.add("parent_4", "child_1")
    print(forwards)
    print(twm)


if __name__ == "__main__":
    __main()
