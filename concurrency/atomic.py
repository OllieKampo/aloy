import collections.abc
import types
from typing import Generic, Sequence, TypeVar, final
from concurrency.synchronization import OwnedRLock


NT = TypeVar("NT", int, float, complex)


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
    
    def __iadd__(self, value: NT) -> None:
        if not self.__lock.is_locked:
            raise RuntimeError("Cannot update AtomicNumber outside of a context manager.")
        if not self.__lock.is_owner:
            raise RuntimeError("Attempted to update AtomicNumber from a non-owner thread.")
        self.__value += value 
    
    def __add__(self, value: NT) -> NT:
        return self.__value + value
    
    def __isub__(self, value: NT) -> None:
        if not self.__lock.is_locked:
            raise RuntimeError("Cannot update AtomicNumber outside of a context manager.")
        if not self.__lock.is_owner:
            raise RuntimeError("Attempted to update AtomicNumber from a non-owner thread.")
        self.__value -= value
    
    def __sub__(self, value: NT) -> NT:
        return self.__value - value
    
    def __imul__(self, value: NT) -> None:
        if not self.__lock.is_locked:
            raise RuntimeError("Cannot update AtomicNumber outside of a context manager.")
        if not self.__lock.is_owner:
            raise RuntimeError("Attempted to update AtomicNumber from a non-owner thread.")
        self.__value *= value
    
    def __mul__(self, value: NT) -> NT:
        return self.__value * value
    
    def __itruediv__(self, value: NT) -> None:
        if not self.__lock.is_locked:
            raise RuntimeError("Cannot update AtomicNumber outside of a context manager.")
        if not self.__lock.is_owner:
            raise RuntimeError("Attempted to update AtomicNumber from a non-owner thread.")
        self.__value /= value
    
    def __truediv__(self, value: NT) -> NT:
        return self.__value / value
    
    def __ifloordiv__(self, value: NT) -> None:
        if not self.__lock.is_locked:
            raise RuntimeError("Cannot update AtomicNumber outside of a context manager.")
        if not self.__lock.is_owner:
            raise RuntimeError("Attempted to update AtomicNumber from a non-owner thread.")
        self.__value //= value
    
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
    
    def __iadd__(self, value: list[LT]) -> None:
        if not self.__lock.is_locked:
            raise RuntimeError("Cannot update AtomicList outside of a context manager.")
        if not self.__lock.is_owner:
            raise RuntimeError("Attempted to update AtomicList from a non-owner thread.")
        self.__list += value
    
    def __add__(self, value: list[LT]) -> list[LT]:
        return self.__list + value
    
    def __imul__(self, value: int) -> None:
        if not self.__lock.is_locked:
            raise RuntimeError("Cannot update AtomicList outside of a context manager.")
        if not self.__lock.is_owner:
            raise RuntimeError("Attempted to update AtomicList from a non-owner thread.")
        self.__list *= value
    
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
