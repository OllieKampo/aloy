###########################################################################
###########################################################################
## Module containing classes defining mutable atomic objects.            ##
##                                                                       ##
## Copyright (C) 2023 Oliver Michael Kamperis                            ##
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

"""
Module containing classes defining mutable atomic objects.

Atomic objects are thread-safe objects whose updates are atomic.
They are useful for concurrent programming, where multiple threads
may be accessing or updating the same object, but where the updates
may happen over a large function block and not in a single call,
therefore it is important to ensure that the object is not changed
by another thread during the update.

Updates are only allowed within a context manager, accessing an
object (through a function that does not change the object) may
be called outside of a context manager, but will be blocking if
the object is currently being updated by another thread.
"""

import collections.abc
import types
from typing import Generic, Hashable, Iterable, Iterator, Mapping, Sequence, TypeVar, final
from concurrency.synchronization import OwnedRLock

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"

__all__ = (
    "AtomicNumber",
    "AtomicList",
    "AtomicDict",
    "AtomicSet"
)


def __dir__() -> tuple[str]:
    """Get the names of module attributes."""
    return __all__


NT = TypeVar("NT", int, float, complex)


## TODO: Add async support.
@final
class AtomicNumber(Generic[NT]):
    """
    A thread-safe number whose updates are atomic.

    Updates to the number are only allowed within a context manager.
    """

    __slots__ = ("__value", "__lock")

    def __init__(self, value: NT = 0) -> None:
        """
        Create a new atomic number with given initial value.

        The number type can be int, float, or complex.
        """
        self.__value: NT = value
        self.__lock = OwnedRLock()
    
    def __str__(self) -> str:
        return str(self.__value)
    
    def __repr__(self) -> str:
        return f"AtomicNumber({self.__value})"
    
    def __enter__(self) -> None:
        self.__lock.acquire()
    
    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None
    ) -> None:
        self.__lock.release()
    
    def __int__(self) -> int:
        return int(self.__value)
    
    def __float__(self) -> float:
        return float(self.__value)
    
    def __complex__(self) -> complex:
        return complex(self.__value)
    
    @property
    def value(self) -> NT:
        """Returns the current value of the number."""
        return self.__value
    
    @value.setter
    def value(self, value: NT) -> None:
        if not self.__lock.is_locked:
            raise RuntimeError("Cannot update AtomicNumber outside of a context manager.")
        if not self.__lock.is_owner:
            raise RuntimeError("Attempted to update AtomicNumber from a non-owner thread.")
        self.__value = value
    
    def __iadd__(self, value: NT) -> "AtomicNumber[NT]":
        if not self.__lock.is_locked:
            raise RuntimeError("Cannot update AtomicNumber outside of a context manager.")
        if not self.__lock.is_owner:
            raise RuntimeError("Attempted to update AtomicNumber from a non-owner thread.")
        self.__value += value
        return self
    
    def __add__(self, value: NT) -> NT:
        return self.__value + value
    
    def __isub__(self, value: NT) -> "AtomicNumber[NT]":
        if not self.__lock.is_locked:
            raise RuntimeError("Cannot update AtomicNumber outside of a context manager.")
        if not self.__lock.is_owner:
            raise RuntimeError("Attempted to update AtomicNumber from a non-owner thread.")
        self.__value -= value
        return self
    
    def __sub__(self, value: NT) -> NT:
        return self.__value - value
    
    def __imul__(self, value: NT) -> "AtomicNumber[NT]":
        if not self.__lock.is_locked:
            raise RuntimeError("Cannot update AtomicNumber outside of a context manager.")
        if not self.__lock.is_owner:
            raise RuntimeError("Attempted to update AtomicNumber from a non-owner thread.")
        self.__value *= value
        return self
    
    def __mul__(self, value: NT) -> NT:
        return self.__value * value
    
    def __itruediv__(self, value: NT) -> "AtomicNumber[NT]":
        if not self.__lock.is_locked:
            raise RuntimeError("Cannot update AtomicNumber outside of a context manager.")
        if not self.__lock.is_owner:
            raise RuntimeError("Attempted to update AtomicNumber from a non-owner thread.")
        self.__value /= value
        return self
    
    def __truediv__(self, value: NT) -> NT:
        return self.__value / value
    
    def __ifloordiv__(self, value: NT) -> "AtomicNumber[NT]":
        if not self.__lock.is_locked:
            raise RuntimeError("Cannot update AtomicNumber outside of a context manager.")
        if not self.__lock.is_owner:
            raise RuntimeError("Attempted to update AtomicNumber from a non-owner thread.")
        self.__value //= value
        return self
    
    def __floordiv__(self, value: NT) -> NT:
        return self.__value // value


LT = TypeVar("LT")


class AtomicList(collections.abc.MutableSequence, Generic[LT]):
    """
    A thread-safe list whose updates are atomic.

    Updates to the list are only allowed within a context manager.
    """

    __slots__ = ("__list", "__lock")

    def __init__(self, sequence: Sequence[LT]) -> None:
        self.__list: list[LT] = list(sequence)
        self.__lock = OwnedRLock()
    
    def __enter__(self) -> None:
        self.__lock.acquire()
    
    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None
    ) -> None:
        self.__lock.release()
    
    def __iadd__(self, value: list[LT]) -> "AtomicList[LT]":
        if not self.__lock.is_locked:
            raise RuntimeError("Cannot update AtomicList outside of a context manager.")
        if not self.__lock.is_owner:
            raise RuntimeError("Attempted to update AtomicList from a non-owner thread.")
        self.__list += value
        return self
    
    def __add__(self, value: list[LT]) -> list[LT]:
        return self.__list + value
    
    def __imul__(self, value: int) -> "AtomicList[LT]":
        if not self.__lock.is_locked:
            raise RuntimeError("Cannot update AtomicList outside of a context manager.")
        if not self.__lock.is_owner:
            raise RuntimeError("Attempted to update AtomicList from a non-owner thread.")
        self.__list *= value
        return self
    
    def __mul__(self, value: int) -> list[LT]:
        return self.__list * value
    
    def __len__(self) -> int:
        return len(self.__list)
    
    def __getitem__(self, key: int) -> LT:
        return self.__list[key]
    
    def __setitem__(self, key: int, value: object) -> None:
        if not self.__lock.is_locked:
            raise RuntimeError("Cannot update AtomicList outside of a context manager.")
        if not self.__lock.is_owner:
            raise RuntimeError("Attempted to update AtomicList from a non-owner thread.")
        self.__list[key] = value
    
    def __delitem__(self, key: int) -> None:
        if not self.__lock.is_locked:
            raise RuntimeError("Cannot update AtomicList outside of a context manager.")
        if not self.__lock.is_owner:
            raise RuntimeError("Attempted to update AtomicList from a non-owner thread.")
        del self.__list[key]
    
    def append(self, value: object) -> None:
        if not self.__lock.is_locked:
            raise RuntimeError("Cannot update AtomicList outside of a context manager.")
        if not self.__lock.is_owner:
            raise RuntimeError("Attempted to update AtomicList from a non-owner thread.")
        self.__list.append(value)
    
    def extend(self, value: list[LT]) -> None:
        if not self.__lock.is_locked:
            raise RuntimeError("Cannot update AtomicList outside of a context manager.")
        if not self.__lock.is_owner:
            raise RuntimeError("Attempted to update AtomicList from a non-owner thread.")
        self.__list.extend(value)
    
    def insert(self, index: int, value: object) -> None:
        if not self.__lock.is_locked:
            raise RuntimeError("Cannot update AtomicList outside of a context manager.")
        if not self.__lock.is_owner:
            raise RuntimeError("Attempted to update AtomicList from a non-owner thread.")
        self.__list.insert(index, value)
    
    def pop(self, index: int = -1) -> LT:
        if not self.__lock.is_locked:
            raise RuntimeError("Cannot update AtomicList outside of a context manager.")
        if not self.__lock.is_owner:
            raise RuntimeError("Attempted to update AtomicList from a non-owner thread.")
        return self.__list.pop(index)
    
    def remove(self, value: object) -> None:
        if not self.__lock.is_locked:
            raise RuntimeError("Cannot update AtomicList outside of a context manager.")
        if not self.__lock.is_owner:
            raise RuntimeError("Attempted to update AtomicList from a non-owner thread.")
        self.__list.remove(value)
    
    def clear(self) -> None:
        if not self.__lock.is_locked:
            raise RuntimeError("Cannot update AtomicList outside of a context manager.")
        if not self.__lock.is_owner:
            raise RuntimeError("Attempted to update AtomicList from a non-owner thread.")
        self.__list.clear()
    
    def reverse(self) -> None:
        if not self.__lock.is_locked:
            raise RuntimeError("Cannot update AtomicList outside of a context manager.")
        if not self.__lock.is_owner:
            raise RuntimeError("Attempted to update AtomicList from a non-owner thread.")
        self.__list.reverse()


KT = TypeVar("KT", bound=Hashable)
VT = TypeVar("VT")


class AtomicDict(collections.abc.MutableMapping, Generic[KT, VT]):
    """
    A thread-safe dictionary whose updates are atomic.

    Updates to the dictionary are only allowed within a context manager.
    """

    __slots__ = ("__dict", "__lock")

    def __init__(self, mapping: Mapping[KT, VT]) -> None:
        self.__dict: dict[KT, VT] = dict(mapping)
        self.__lock = OwnedRLock()
    
    def __enter__(self) -> None:
        self.__lock.acquire()
    
    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None
    ) -> None:
        self.__lock.release()
    
    def __len__(self) -> int:
        return len(self.__dict)
    
    def __getitem__(self, key: KT) -> VT:
        return self.__dict[key]
    
    def __setitem__(self, key: KT, value: VT) -> None:
        if not self.__lock.is_locked:
            raise RuntimeError("Cannot update AtomicDict outside of a context manager.")
        if not self.__lock.is_owner:
            raise RuntimeError("Attempted to update AtomicDict from a non-owner thread.")
        self.__dict[key] = value
    
    def __delitem__(self, key: KT) -> None:
        if not self.__lock.is_locked:
            raise RuntimeError("Cannot update AtomicDict outside of a context manager.")
        if not self.__lock.is_owner:
            raise RuntimeError("Attempted to update AtomicDict from a non-owner thread.")
        del self.__dict[key]
    
    def __iter__(self) -> Iterator[KT]:
        return iter(self.__dict)
    
    def __contains__(self, key: KT) -> bool:
        return key in self.__dict
    
    def __iadd__(self, value: dict[KT, VT]) -> "AtomicDict[KT, VT]":
        if not self.__lock.is_locked:
            raise RuntimeError("Cannot update AtomicDict outside of a context manager.")
        if not self.__lock.is_owner:
            raise RuntimeError("Attempted to update AtomicDict from a non-owner thread.")
        self.__dict += value
        return self
    
    def __add__(self, value: dict[KT, VT]) -> dict[KT, VT]:
        return self.__dict + value
    
    def __imul__(self, value: int) -> "AtomicDict[KT, VT]":
        if not self.__lock.is_locked:
            raise RuntimeError("Cannot update AtomicDict outside of a context manager.")
        if not self.__lock.is_owner:
            raise RuntimeError("Attempted to update AtomicDict from a non-owner thread.")
        self.__dict *= value
        return self
    
    def __mul__(self, value: int) -> dict[KT, VT]:
        return self.__dict * value
    
    def clear(self) -> None:
        if not self.__lock.is_locked:
            raise RuntimeError("Cannot update AtomicDict outside of a context manager.")
        if not self.__lock.is_owner:
            raise RuntimeError("Attempted to update AtomicDict from a non-owner thread.")
        self.__dict.clear()


ET = TypeVar("ET", bound=Hashable)


class AtomicSet(collections.abc.MutableSet, Generic[ET]):
    """
    A thread-safe set whose updates are atomic.

    Updates to the set are only allowed within a context manager.
    """

    __slots__ = ("__set", "__lock")

    def __init__(self, iterable: Iterable[ET]) -> None:
        self.__set: set[ET] = set(iterable)
        self.__lock = OwnedRLock()
    
    def __enter__(self) -> None:
        self.__lock.acquire()
    
    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None
    ) -> None:
        self.__lock.release()
    
    def __len__(self) -> int:
        return len(self.__set)
    
    def __iter__(self) -> Iterator[ET]:
        return iter(self.__set)
    
    def __contains__(self, item: object) -> bool:
        return item in self.__set
    
    def __iadd__(self, value: set[ET]) -> "AtomicSet[ET]":
        if not self.__lock.is_locked:
            raise RuntimeError("Cannot update AtomicSet outside of a context manager.")
        if not self.__lock.is_owner:
            raise RuntimeError("Attempted to update AtomicSet from a non-owner thread.")
        self.__set += value
        return self
    
    def __add__(self, value: set[ET]) -> set[ET]:
        return self.__set + value
    
    def __imul__(self, value: int) -> "AtomicSet[ET]":
        if not self.__lock.is_locked:
            raise RuntimeError("Cannot update AtomicSet outside of a context manager.")
        if not self.__lock.is_owner:
            raise RuntimeError("Attempted to update AtomicSet from a non-owner thread.")
        self.__set *= value
        return self
    
    def __mul__(self, value: int) -> set[ET]:
        return self.__set * value
    
    def add(self, value: ET) -> None:
        if not self.__lock.is_locked:
            raise RuntimeError("Cannot update AtomicSet outside of a context manager.")
        if not self.__lock.is_owner:
            raise RuntimeError("Attempted to update AtomicSet from a non-owner thread.")
        self.__set.add(value)
    
    def discard(self, value: ET) -> None:
        if not self.__lock.is_locked:
            raise RuntimeError("Cannot update AtomicSet outside of a context manager.")
        if not self.__lock.is_owner:
            raise RuntimeError("Attempted to update AtomicSet from a non-owner thread.")
        self.__set.discard(value)
