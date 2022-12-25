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

__copyright__ = "Copyright (C) 2022 Oliver Michael Kamperis"
__license__ = "GPL-3.0"

__all__ = ("frozendict",
           "ReversableDict",
           "FrozenReversableDict")

def __dir__() -> tuple[str]:
    """Get the names of module attributes."""
    return sorted(__all__)

def __getattr__(name: str) -> object:
    """Get an attributes from the module."""
    if name in __all__:
        return globals()[name]
    raise AttributeError(f"Module {__name__!r} has no attribute {name!r}.")

import collections.abc
from typing import (Generic, Hashable, ItemsView, Iterable, Iterator, Mapping,
                    Optional, TypeVar, final, overload)

KT = TypeVar("KT", bound=Hashable)
VT = TypeVar("VT", bound=Hashable, covariant=True)

@final
class frozendict(collections.abc.Mapping, Generic[KT, VT]):
    """
    A frozen dictionary type.
    
    A frozen dictionary is an immutable and hashable version of a standard dictionary.
    """
    
    __slots__ = {"__dict" : "The dictionary mapping.",
                 "__hash" : "The hash of the dictionary."}
    
    @overload
    def __init__(self, mapping: Mapping[KT, VT], /) -> None:
        """
        Create a new frozen dictionary from a mapping object's (key, value) pairs.
        
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
        Create a new frozen dictionary from an iterable of tuples defining (key, value) pairs.
        
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
        Create a new frozen dictionary with the key to value pairs given in the keyword argument list.
        
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
    A mutable reversable dictionary type.
    
    Supports all methods of standard dictionaries with the
    added ability to find all keys that map to a specific value.
    
    The reversable dict maintains the reversed version of itself in memory.
    Therefore, insertions are slower and memory usage is higher
    than a standard dictionary, but reversed lookups are fast.
    This also requires that dictionary values be hashable.
    
    Example Usage
    -------------
    
    ```
    >>> from jinx.datastructures.mappings import ReversableDict
    >>> rev_dict: ReversableDict[str, int] = ReversableDict({"a" : 1, "b" : 2, "z" : 2})
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
    ## The keys are given as a list where return order reflects insertion order.
    >>> rev_dict(2)
    ["b", "z"]
    
    ## Objects are mutable, and updates are handled on both the standard and reverse mappings.
    >>> del rev_dict["z"]
    >>> rev_dict(2)
    ["b"]
    ```
    """
    
    __slots__ = {"__dict" : "The standard dictionary mapping.",
                 "__reversed_dict" : "The reversed dictionary mapping."}
    
    @overload
    def __init__(self, mapping: Mapping[KT, VT], /) -> None:
        """
        Create a new reversable dictionary initialised from a mapping object's (key, value) pairs.
        
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
        Create a new reversable dictionary initialized from an iterable of tuples defining (key, value) pairs.
        
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
        Create a new reversable dictionary initialized with the key to value pairs given in the keyword argument list.
        
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
        """Get an instantiable string representation of the reversable dictionary."""
        return f"ReversableDict({repr(self.__dict)})"
    
    def __getitem__(self, key: KT, /) -> VT:
        """Get the value for the given key in the standard mapping."""
        return self.__dict[key]
    
    def __setitem__(self, key: KT, value: VT, /) -> None:
        """Add or update an item, given by a (key, value) pair, from the standard and reversed mapping."""
        ## If the key is already in the standard mapping, the value it
        ## currently maps to must be replaced in the reversed mapping.
        if (old_value := self.__dict.get(key)) is not None:
            self.__del_reversed_item(key, old_value)
        self.__dict[key] = value
        self.__reversed_dict.setdefault(value, []).append(key)
    
    def __delitem__(self, key: KT, /) -> None:
        """Delete a item (given by a key) from the standard and reversed mapping."""
        ## Remove from the standard mapping.
        value: VT = self.__dict[key]
        del self.__dict[key]
        ## Remove from the reversed mapping.
        self.__del_reversed_item(key, value)
    
    def __del_reversed_item(self, key: KT, value: VT, /) -> None:
        """Delete an item (given by a [key, value] pair) from the reversed mapping."""
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
    
    def __call__(self, value: VT, /, *, max_: Optional[int] = None) -> list[KT]:
        """
        Get the list of keys that map to the given value in the reversed mapping.
        
        If `max_` is given and not None, return at most the first `max_` keys.
        """
        if max_ is None:
            return self.__reversed_dict[value].copy()
        keys: list[KT] = self.__reversed_dict[value]
        return keys[0:min(max_, len(keys))]
    
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
            - Its values are lists of keys from the standard dictionary which map to the respective values.
        """
        return self.__reversed_dict.items()
    
    def reversed_get(self, value: VT, default: Optional[list[KT]] = None, /, *, max_: Optional[int] = None) -> Optional[list[KT]]:
        """
        Get the list of keys that map to a given value in the reversed mapping.
        
        If the value is not found, return `default` if given; otherwise, raise a KeyError.
        
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
    
    def reversed_pop(self, value: VT, default: Optional[list[KT]] = None, /) -> list[KT]:
        """
        Remove a value from the reversed mapping and return the list of keys that mapped to it.
        
        If the key is not found, return `default` if given; otherwise, raise a KeyError.
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
    A frozen reversable dictionary.
    
    A frozen dictionary is an immutable and hashable version of a reversable dictionary.
    """
    
    __slots__ = {"__reversable_dict" : "The reversable dictionary mapping.",
                 "__hash" : "The hash of the dictionary."}
    
    @overload
    def __init__(self, mapping: Mapping[KT, VT]) -> None:
        """
        Create a new frozen reversable dictionary from a mapping object's (key, value) pairs.
        
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
        Create a new frozen reversable dictionary from an iterable of tuples defining (key, value) pairs.
        
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
        Create a new frozen reversable dictionary with the key to value pairs given in the keyword argument list.
        
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
    
    def __repr__(self) -> str:
        """Get an instantiable string representation of the dictionary."""
        return f"ReversableDict({self.__reversable_dict.standard_mapping})"
    
    def __getitem__(self, key: KT, /) -> VT:
        """Get the value associated with the given key."""
        return self.__reversable_dict[key]
    
    def __iter__(self) -> Iterator[VT]:
        """Iterate over the keys in the dictionary."""
        return iter(self.__reversable_dict)
    
    def __len__(self) -> int:
        """Get the number of items in the dictionary."""
        return len(self.__reversable_dict)
    
    def __call__(self, value: VT, /, *, max_: Optional[int] = None) -> list[KT]:
        """
        Get the list of keys that map to the given value in the reversed mapping.
        
        If `max_` is given and not None, return at most the first `max_` keys.
        """
        return self.__reversable_dict.reversed_get(value, max_=max_)
    
    @property
    def standard_mapping(self) -> dict[KT, VT]:
        """Get the standard mapping as a normal dictionary."""
        return self.__reversable_dict.standard_mapping
    
    @property
    def reversed_mapping(self) -> dict[VT, list[KT]]:
        """Get the reversed mapping as a normal dictionary."""
        return self.__reversable_dict.reversed_mapping
    
    def reversed_items(self) -> ItemsView[VT, list[KT]]:
        """
        Get a reversed dictionary items view.
        
        Such that:
            - Its keys are the values of the standard dictionary,
            - Its values are lists of keys from the standard dictionary which map to the respective values.
        """
        return self.__reversable_dict.reversed_items()
    
    def reversed_get(self, value: VT, default: Optional[list[KT]] = None, /, *, max_: Optional[int] = None) -> Optional[list[KT]]:
        """
        Get the list of keys that map to a given value in the reversed mapping.
        
        If the value is not found, return `default` if given; otherwise, raise a KeyError.
        
        If `max_` is given and not None, return at most the first `max_` keys.
        """
        return self.__reversable_dict.reversed_get(value, default=default)
    
    def __hash__(self) -> int:
        """Get the hash of the dictionary."""
        if self.__hash is None:
            hash_: int = 0
            for item in self.__reversable_dict.items():
                hash_ ^= hash(item)
            self.__hash = hash_
        return self.__hash
    
    def __copy__(self) -> "frozendict[KT, VT]":
        """Create a shallow copy of the dictionary."""
        return frozendict(self)
