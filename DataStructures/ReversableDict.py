from typing import Generic, Hashable, ItemsView, Iterable, Iterator, Mapping, Optional, TypeVar, overload
import _collections_abc

KT = TypeVar("KT", bound=Hashable)
VT = TypeVar("VT", bound=Hashable, covariant=True)

class ReversableDict(_collections_abc.MutableMapping, Generic[KT, VT]):
    """
    A mutable reversable dictionary type.
    Supports all methods of standard dictionaries,
    with the additional capability of finding all keys that map to a specific value.
    
    The reversable dict maintains a reversed version of itself in memory.
    Therefore, insertions are slower, and memory usage ish igher, but lookups are fast.
    This also requires that the values be hashable.
    
    Example Usage
    -------------
    
    >>> from Jinx.DataStructures import ReversableDict as RevDict
    >>> dict_: RevDict[str, int] = RevDict({"A" : 2, "B" : 3, "C" : 2})
    
    Check what 'A' maps to (as in a standard dictionary).
    >>> dict_["A"]
    2
    
    Check what keys map to 2 (a reverse operation to a standard dictionary).
    The keys are given as a list, where the order reflects the insertion order
    (the order the keys which map to this value was added to the dictionary).
    >>> dict_(2)
    ["A", "C"]
    
    Objects are mutable, and updates are handled on both the standard and reverse mappings.
    >>> del dict_["A"]
    >>> dict_(2)
    ["C"]
    """
    
    __slots__ = ("__dict",
                 "__reversed_dict")
    
    @overload
    def __init__(self, mapping: Mapping[KT, VT]) -> None:
        """
        A new reversable dictionary initialised from a mapping object's (key, value) pairs.
        For example:
        ```
        >>> rev_dict = ReversableDict({"one" : 1, "two" : 2})
        >>> rev_dict
        {"one" : 1, "two" : 2}
        ```
        """
        ...
    
    @overload
    def __init__(self, iterable: Iterable[tuple[KT, VT]]) -> None:
        """
        A new dictionary initialized from an iterable of tuples defining (key, value) pairs as if via:
        ```
        >>> iterable = [("one", 1), ("two", 2)]
        >>> rev_dict = ReversableDict()
        >>> for key, value in iterable:
        ...     rev_dict[key] = value
        >>> rev_dict
        {"one" : 1, "two" : 2}
        ```
        """
        ...
    
    @overload
    def __init__(self, **kwargs: VT) -> None:
        """
        A new dictionary initialized with the name=value pairs given in the keyword argument list.
        For example:
        ```
        >>> rev_dict = ReversableDict(one=1, two=2)
        >>> rev_dict
        {"one" : 1, "two" : 2}
        ```
        """
        ...
    
    def __init__(self, *args, **kwargs) -> None:
        self.__dict: dict[KT, VT] = {}
        self.__reversed_dict: dict[VT, list[KT]] = {}
        for key, value in dict(*args, **kwargs).items():
            self[key] = value
    
    def __str__(self) -> str:
        return str(self.__dict)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.__dict)})"
    
    def __iter__(self) -> Iterator[KT]:
        yield from self.__dict
    
    def __len__(self) -> int:
        return len(self.__dict)
    
    @property
    def standard_mapping(self) -> dict[KT, VT]:
        "The standard mapping as a normal dictionary."
        return self.__dict
    
    @property
    def reversed_mapping(self) -> dict[VT, list[KT]]:
        "The reversed mapping as a normal dictionary."
        return self.__reversed_dict
    
    def __getitem__(self, key: KT) -> VT:
        "Get the value for the given key in the standard mapping."
        return self.__dict[key]
    
    def __call__(self, value: VT, max_: Optional[int] = None) -> list[KT]:
        """
        Get the list of keys that map to the given value in the reversed mapping.
        If `max_` is given and not None, return at most the first `max_` keys.
        """
        if max_ is None:
            return self.__reversed_dict[value].copy()
        keys: list[KT] = self.__reversed_dict[value]
        return keys[0:min(max_, len(keys))]
    
    def reversed_items(self) -> ItemsView[VT, list[KT]]:
        """
        Get a reversed dictionary items view:
            - Whose keys are the values of the standard dictionary,
            - And whose values are lists of keys from the standard dictionary which map to the respective values.
        """
        return self.__reversed_dict.items()
    
    def reversed_get(self, value: VT, default: Optional[list[KT]] = None) -> Optional[list[KT]]:
        """
        Get the list of keys that map to a given value in the reversed mapping, 
        default is returned of the value is not in the dictionary.
        """
        if value not in self.__reversed_dict:
            return default
        return self(value)
    
    def __setitem__(self, key: KT, value: VT) -> None:
        "Add or update an item, given by a (key, value) pair, from the standard and reversed mapping."
        ## If the key is already in the standard mapping, the value it
        ## currently maps to must be replaced in the reversed mapping.
        if (old_value := self.__dict.get(key)) is not None:
            self.__del_reversed_item(key, old_value)
        self.__dict[key] = value
        self.__reversed_dict.setdefault(value, []).append(key)
    
    def __delitem__(self, key: KT) -> None:
        "Delete a item (given by a key) from the standard and reversed mapping."
        ## Remove from the standard mapping.
        value: VT = self.__dict[key]
        del self.__dict[key]
        ## Remove from the reversed mapping.
        self.__del_reversed_item(key, value)
    
    def __del_reversed_item(self, key: KT, value: VT) -> None:
        "Delete an item (given by a [key, value] pair) from the reversed mapping."
        ## Remove the reference of this value being mapped to by the given key.
        self.__reversed_dict[value].remove(key)
        ## If no keys map to this value anymore then remove the value as well.
        if len(self.__reversed_dict[value]) == 0:
            del self.__reversed_dict[value]
