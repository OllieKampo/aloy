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

"""
Module containing classes mutable atomic object types.

Atomic objects are thread-safe objects whose updates are atomic. They are
useful for concurrent programming, where multiple threads may be accessing or
updating the same object, but where the updates may happen over a large
function block and not in a single call, therefore it is important to ensure
that the object is not changed by another thread during the update.

Updates are only allowed within a context manager, accessing an object
(through a function that does not change the object) may be called outside of
a context manager, but will be blocking if the object is currently being
updated by another thread.

Atomic objects that wrap mutable structures such as lists, dictionaries, and
sets, only allow updates to be made to the wrapped object through their
wrapper methods, and their `get_obj` method returns view of the wrapped object
rather than the object itself. This is to ensure that a reference to the
wrapped object is not accidentally kept and changed by a different thread
whilst some other thread is updating the object.
"""

from abc import ABCMeta, abstractmethod
from collections import defaultdict
import collections.abc
import contextlib
import functools
import sys
import threading
import types
from typing import (
    Any, Callable, Concatenate, Generic, Hashable, ItemsView, Iterable,
    Iterator, KeysView, Mapping, ParamSpec, TypeVar, ValuesView, final,
    overload
)
import weakref
from aloy.concurrency.synchronization.primitives import OwnedRLock
from aloy.datastructures.views import DequeView, DictView, ListView, SetView

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"
__version__ = "0.1.2"

__all__ = (
    "AloyAtomicObjectError",
    "AtomicObject",
    "AtomicNumber",
    "AtomicBool",
    "AtomicList",
    "AtomicDict",
    "AtomicSet"
)


def __dir__() -> tuple[str, ...]:
    """Get the names of module attributes."""
    return __all__


__INSTANCE_ATOMIC_UPDATERS: weakref.WeakKeyDictionary[
    type,
    weakref.WeakKeyDictionary[
        object,
        dict[str, threading.RLock]
    ]
] = weakref.WeakKeyDictionary()
__CLASS_ATOMIC_UPDATERS: weakref.WeakKeyDictionary[
    type,
    dict[
        str,
        threading.RLock
    ]
] = weakref.WeakKeyDictionary()
__ARBITRARY_ATOMIC_UPDATERS: dict[
    str,
    threading.RLock
] = defaultdict(threading.RLock)


@contextlib.contextmanager
def atomic_context(
    context_name: str, /,
    cls: type | None = None,
    inst: object | None = None
) -> Iterator[None]:
    """
    Context manager that ensures atomic updates in the context.

    Atomic updates ensure that only one thread can update the context at a
    time. The same thread can enter the same context multiple times.

    Parameters
    ----------
    `context_name: str` - The name of the context.

    `cls: type | None = None` - The class of the context.

    `inst: object | None = None` - The instance of the context.
    If `cls` is not given or None and `inst` is not None,
    `cls` will be set to `inst.__class__`.
    """
    if inst is not None:
        if cls is None:
            cls = inst.__class__
        lock_ = __INSTANCE_ATOMIC_UPDATERS.setdefault(
            cls, weakref.WeakKeyDictionary()) \
            .setdefault(inst, {}).setdefault(
                context_name, threading.RLock())
    elif cls is not None:
        lock_ = __CLASS_ATOMIC_UPDATERS.setdefault(
            cls, {}) \
            .setdefault(context_name, threading.RLock())
    else:
        lock_ = __ARBITRARY_ATOMIC_UPDATERS[context_name]
    lock_.acquire()
    try:
        yield
    finally:
        lock_.release()


CT = TypeVar("CT")
SP = ParamSpec("SP")
ST = TypeVar("ST")


def atomic_update(
    global_lock: str | None = None,
    method: bool = False
) -> Callable[[Callable[SP, ST]], Callable[SP, ST]]:
    """
    Decorate a function to ensure atomic updates in the decorated function.

    Atomic updates ensure that only one thread can update the context at a
    time. The same thread can enter the same context multiple times.

    Parameters
    ----------
    `global_lock: str | None = None` - If given and not None, the name of the
    global lock to use. Whereby, a global lock can be shared between multiple
    functions. If not given or None, the lock will be a lock unique to the
    decorated function.

    `method: bool = False` - Whether the decorated function is treated as a
    method. If True, a unique lock will be used by each instance of a class.
    If False, a single lock will be used by all instances of a class.

    Example Usage
    -------------
    >>> class Foo:
    ...     def __init__(self):
    ...         self.x = 0
    ...     @atomic_update("x")
    ...     def increment_x(self):
    ...         self.x += 1
    ...     @atomic_update("x")
    ...     def decrement_x(self):
    ...         self.x -= 1
    >>> foo = Foo()
    >>> foo.increment_x()
    >>> foo.x
    1
    >>> foo.decrement_x()
    >>> foo.x
    0
    """
    def decorator(func: Callable[SP, ST]) -> Callable[SP, ST]:
        lock_name: str
        if method:
            if global_lock is not None:
                lock_name = f"__method_global__ {global_lock}"
            else:
                lock_name = f"__method_local__ {func.__name__}"

            @functools.wraps(func)
            def wrapper(*args: SP.args, **kwargs: SP.kwargs) -> ST:
                with atomic_context(lock_name, inst=args[0]):
                    return func(*args, **kwargs)
            return wrapper

        else:
            if global_lock is not None:
                lock_name = f"__function_global__ {global_lock}"
            else:
                lock_name = f"__function_local__ {func.__name__}"

            @functools.wraps(func)
            def wrapper(*args: SP.args, **kwargs: SP.kwargs) -> ST:
                with atomic_context(lock_name):
                    return func(*args, **kwargs)

            return wrapper
    return decorator


class AloyAtomicObjectError(RuntimeError):
    """An exception raised when an error occurs in an atomic object."""


_AT = TypeVar("_AT")
_SP = ParamSpec("_SP")
_ST = TypeVar("_ST")


def _atomic_require_lock(
    func: Callable[Concatenate[Any, _SP], _ST]
) -> Callable[Concatenate[Any, _SP], _ST]:
    """
    Decorator that ensures the object is locked by current thread, and
    therefore is not being updated by another thread before calling the
    method.
    """
    def wrapper(self: Any, *args: _SP.args, **kwargs: _SP.kwargs) -> _ST:
        with self:
            return func(self, *args, **kwargs)
    return wrapper


def _atomic_require_context(
    func: Callable[Concatenate[Any, _SP], _ST]
) -> Callable[Concatenate[Any, _SP], _ST]:
    """
    Decorator that ensures the object is locked and being updated within
    a context manager before calling the method.
    """
    def wrapper(self: Any, *args: _SP.args, **kwargs: _SP.kwargs) -> _ST:
        self._check_context()  # pylint: disable=protected-access
        return func(self, *args, **kwargs)
    return wrapper


class _Atomic(Generic[_AT], metaclass=ABCMeta):
    """Base class for atomic objects."""

    __slots__ = {
        "__lock": "The lock used to ensure atomic updates."
    }

    def __init__(self) -> None:
        """Create a new atomic object."""
        self.__lock = OwnedRLock()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.get_obj()!s}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.get_obj()!r})"

    @_atomic_require_lock
    @abstractmethod
    def get_obj(self) -> _AT:
        """Returns the wrapped object."""

    @_atomic_require_context
    @abstractmethod
    def set_obj(self, value: _AT, /) -> None:
        """Sets the wrapped object."""

    def __enter__(self) -> "_Atomic[_AT]":
        self.__lock.acquire()
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None
    ) -> None:
        self.__lock.release()

    def _check_context(self) -> None:  # pylint: disable=unused-private-member
        """Check that the object is currently being updated."""
        if not self.__lock.is_locked:
            raise AloyAtomicObjectError(
                "Cannot update atomic object outside of a context manager."
            )
        if not self.__lock.is_owner:
            raise AloyAtomicObjectError(
                "Attempted to update atomic object from a non-owner thread."
            )


_OT = TypeVar("_OT")


@final
class AtomicObject(_Atomic[_OT]):
    """
    A thread-safe atomic object wrapper.

    Getting the wrapped object is only allowed whilst another thread does not
    have the object locked. The object is locked whilst a thread has entered
    the object's context manager, and is unlocked when the thread exits the
    context manager. Setting the wrapped object is only allowed within a
    context manager. This allows one to ensure that the object is not got
    by another thread whilst it is being updated by multiple operations
    in another thread, therefore making those operations atomic.

    Note that this does not make accesses/updates to the wrapped object itself
    thread-safe, it only ensures that access to the object through the wrapper
    is thread-safe.
    """

    __slots__ = {
        "__object": "The object being wrapped."
    }

    def __init__(self, object_: _OT, /) -> None:
        """Create a new atomic object."""
        super().__init__()
        self.__object: _OT = object_

    @_atomic_require_lock
    def get_obj(self) -> _OT:
        """Returns the wrapped object."""
        return self.__object

    @_atomic_require_context
    def set_obj(self, object_: _OT, /) -> None:
        """Sets the wrapped object."""
        self.__object = object_


_NT = TypeVar("_NT", int, float, complex)
_NTO = TypeVar("_NTO", int, float, complex)
_NTONC = TypeVar("_NTONC", int, float)


@final
class AtomicNumber(_Atomic[_NT]):
    """
    A thread-safe number whose updates are atomic.

    Updates to the number are only allowed within a context manager.
    """

    __slots__ = {
        "__value": "The current value of the number."
    }

    def __init__(self, value: _NT = 0) -> None:
        """
        Create a new atomic number with given initial value.

        The number type can be int, float, or complex.
        """
        super().__init__()
        self.__value: _NT = value

    @_atomic_require_lock
    def get_obj(self) -> _NT:
        """Returns the current value of the number."""
        return self.__value

    @_atomic_require_context
    def set_obj(self, value: _NT, /) -> None:
        """Sets the number to the given value."""
        self.__value = value

    @_atomic_require_lock
    def __int__(self) -> int:
        if isinstance(self.__value, complex):
            raise TypeError("Cannot convert complex to int.")
        return int(self.__value)

    @_atomic_require_lock
    def __float__(self) -> float:
        if isinstance(self.__value, complex):
            raise TypeError("Cannot convert complex to float.")
        return float(self.__value)

    @_atomic_require_lock
    def __complex__(self) -> complex:
        return complex(self.__value)

    @_atomic_require_context
    def __iadd__(self, value: _NT) -> "AtomicNumber[_NT]":
        self.__value = type(self.__value)(self.__value + value)
        return self

    @_atomic_require_lock
    def __add__(self, value: _NTO) -> _NT | _NTO:
        return self.__value + value

    @_atomic_require_context
    def __isub__(self, value: _NT) -> "AtomicNumber[_NT]":
        self.__value = type(self.__value)(self.__value - value)
        return self

    @_atomic_require_lock
    def __sub__(self, value: _NTO) -> _NT | _NTO:
        return self.__value - value

    @_atomic_require_context
    def __ipow__(self, value: _NT) -> "AtomicNumber[_NT]":
        self.__value = type(self.__value)(self.__value ** value)
        return self

    @_atomic_require_lock
    def __pow__(self, value: _NTO) -> _NT | _NTO:
        return self.__value ** value

    @_atomic_require_context
    def __imul__(self, value: _NT) -> "AtomicNumber[_NT]":
        self.__value = type(self.__value)(self.__value * value)
        return self

    @_atomic_require_lock
    def __mul__(self, value: _NTO) -> _NT | _NTO:
        return self.__value * value

    @_atomic_require_context
    def __itruediv__(
        self,
        value: _NT
    ) -> "AtomicNumber[_NT]":
        self.__value = type(self.__value)(self.__value / value)
        return self

    @_atomic_require_lock
    def __truediv__(self, value: _NTO) -> _NT | _NTO | float:
        return self.__value / value

    @_atomic_require_context
    def __ifloordiv__(self, value: _NT) -> "AtomicNumber[_NT]":
        if isinstance(self.__value, complex):
            raise TypeError("Cannot floor divide a complex number.")
        self.__value = type(self.__value)(self.__value // value)
        return self

    @_atomic_require_lock
    def __floordiv__(self, value: _NTONC) -> _NT | _NTONC:
        if isinstance(self.__value, complex):
            raise TypeError("Cannot floor divide a complex number.")
        if isinstance(value, complex):
            raise TypeError("Cannot floor divide by a complex number.")
        return self.__value // value


@final
class AtomicBool(_Atomic[bool]):
    """A thread-safe boolean whose updates are atomic."""

    __slots__ = {
        "__value": "The current value of the boolean."
    }

    def __init__(self, value: bool = False) -> None:
        """Create a new atomic boolean with given initial value."""
        super().__init__()
        self.__value = bool(value)

    @_atomic_require_lock
    def get_obj(self) -> bool:
        """Returns the current value of the boolean."""
        return self.__value

    @_atomic_require_context
    def set_obj(self, value: bool, /) -> None:
        """Sets the boolean to the given value."""
        self.__value = bool(value)

    @_atomic_require_context
    def get_and_set_value(self, value: bool) -> bool:
        """Sets the boolean to the given value and returns the old value."""
        old_value = self.__value
        self.__value = bool(value)
        return old_value

    @_atomic_require_context
    def compare_and_set_value(self, expected: bool, value: bool) -> bool:
        """
        Sets the boolean to the given value if the current value is equal to
        the expected value. Returns whether the value was set.
        """
        if self.__value == expected:
            self.__value = bool(value)
            return True
        return False

    @_atomic_require_lock
    def __bool__(self) -> bool:
        return self.__value

    @_atomic_require_context
    def __iand__(self, value: bool) -> "AtomicBool":
        self.__value &= bool(value)
        return self

    @_atomic_require_lock
    def __and__(self, value: bool) -> bool:
        return self.__value & value

    @_atomic_require_context
    def __ior__(self, value: bool) -> "AtomicBool":
        self.__value |= value
        return self

    @_atomic_require_lock
    def __or__(self, value: bool) -> bool:
        return self.__value | value

    @_atomic_require_context
    def __ixor__(self, value: bool) -> "AtomicBool":
        self.__value ^= value
        return self

    @_atomic_require_lock
    def __xor__(self, value: bool) -> bool:
        return self.__value ^ value


_LT = TypeVar("_LT")


class AtomicList(_Atomic[list[_LT]], collections.abc.MutableSequence):
    """
    A thread-safe list whose updates are atomic.

    Updates to the list are only allowed within a context manager.
    """

    __slots__ = {
        "__list": "The wrapped list."
    }

    @overload
    def __init__(self) -> None:
        """
        Create a new empty atomic list.

        For example:
        ```
        >>> alist = AtomicList()
        >>> alist
        AtomicList([])
        ```
        """

    @overload
    def __init__(self, __iterable: Iterable[_LT], /) -> None:
        """
        Create a new atomic list with given initial value. The initial value
        will be copied into the list.

        For example:
        ```
        >>> alist = AtomicList(["one", "two"])
        >>> alist
        AtomicList(["one", "two"])
        ```
        """

    def __init__(  # type: ignore[misc]
        self,
        __iterable: Iterable[_LT] | None = None, /
    ) -> None:
        super().__init__()
        self.__list: list[_LT]
        if __iterable is not None:
            self.__list = list(__iterable)
        else:
            self.__list = []

    @_atomic_require_lock
    def get_obj(self) -> ListView[_LT]:  # type: ignore[override]
        """Returns a view of the current list."""
        return ListView(self.__list)

    @_atomic_require_context
    def set_obj(self, value: Iterable[_LT], /) -> None:
        """Sets the list to the given value."""
        self.__list = list(value)

    @_atomic_require_lock
    def __len__(self) -> int:
        return len(self.__list)

    @overload
    def __getitem__(self, key: int, /) -> _LT:
        ...

    @overload
    def __getitem__(self, key: slice, /) -> "list[_LT]":
        ...

    @_atomic_require_lock
    def __getitem__(
        self,
        key: int | slice, /
    ) -> _LT | "list[_LT]":
        return self.__list[key]

    @_atomic_require_lock
    def __contains__(self, value: object) -> bool:
        return value in self.__list

    @_atomic_require_lock
    def __iter__(self) -> Iterator[_LT]:
        return iter(self.__list)

    @overload
    def __setitem__(self, key: int, value: _LT, /) -> None:
        ...

    @overload
    def __setitem__(self, key: slice, value: Iterable[_LT], /) -> None:
        ...

    @_atomic_require_context
    def __setitem__(
        self,
        key: int | slice,
        value: _LT | Iterable[_LT], /
    ) -> None:
        self.__list[key] = value  # type: ignore[index,assignment]

    @overload
    def __delitem__(self, key: int, /) -> None:
        ...

    @overload
    def __delitem__(self, key: slice, /) -> None:
        ...

    @_atomic_require_context
    def __delitem__(self, key: int | slice, /) -> None:
        del self.__list[key]

    @_atomic_require_context
    def __iadd__(self, value: Iterable[_LT]) -> "AtomicList[_LT]":
        self.__list += value
        return self

    @_atomic_require_lock
    def __add__(self, value: list[_LT]) -> list[_LT]:
        return self.__list + value

    @_atomic_require_context
    def __imul__(self, value: int) -> "AtomicList[_LT]":
        self.__list *= value
        return self

    @_atomic_require_lock
    def __mul__(self, value: int) -> list[_LT]:
        return self.__list * value

    @_atomic_require_lock
    def index(  # pylint: disable=arguments-differ
        self,
        value: _LT,
        start: int = 0,
        stop: int = sys.maxsize, /
    ) -> int:
        """
        Return first index of value.

        Raises ValueError if the value is not present.
        """
        super().index(value, start, stop)
        return self.__list.index(value, start, stop)

    @_atomic_require_lock
    def count(self, value: _LT) -> int:
        """Return number of occurrences of value."""
        return self.__list.count(value)

    @_atomic_require_context
    def append(self, value: _LT) -> None:
        """Append an element to the end of the list."""
        self.__list.append(value)

    @_atomic_require_context
    def extend(self, values: Iterable[_LT]) -> None:
        """Extend the list by appending elements from the iterable."""
        self.__list.extend(values)

    @_atomic_require_context
    def insert(self, index: int, value: _LT) -> None:
        """Insert an element before the given index."""
        self.__list.insert(index, value)

    @_atomic_require_context
    def pop(self, index: int = -1) -> _LT:
        """Remove and return an element at the given index."""
        return self.__list.pop(index)

    @_atomic_require_context
    def remove(self, value: _LT) -> None:
        """
        Remove the first occurrence of value.

        Raises ValueError if the value is not present.
        """
        self.__list.remove(value)

    @_atomic_require_context
    def clear(self) -> None:
        """Remove all elements from the list."""
        self.__list.clear()

    @_atomic_require_context
    def reverse(self) -> None:
        """Reverse the elements of the list in-place."""
        self.__list.reverse()


class AtomicDeque(
    _Atomic[collections.deque[_LT]],
    collections.abc.MutableSequence
):
    """
    A thread-safe deque whose updates are atomic.

    Updates to the deque are only allowed within a context manager.
    """

    __slots__ = {
        "__deque": "The wrapped deque."
    }

    @overload
    def __init__(self) -> None:
        """
        Create a new empty atomic deque.

        For example:
        ```
        >>> adeque = AtomicDeque()
        >>> adeque
        AtomicDeque(deque([]))
        ```
        """

    @overload
    def __init__(self, __iterable: Iterable[_LT], /) -> None:
        """
        Create a new atomic deque with given initial value. The initial value
        will be copied into the deque.

        For example:
        ```
        >>> adeque = AtomicDeque(["one", "two"])
        >>> adeque
        AtomicDeque(deque(["one", "two"]))
        ```
        """

    def __init__(  # type: ignore[misc]
        self,
        __iterable: Iterable[_LT] | None = None, /
    ) -> None:
        super().__init__()
        self.__deque: collections.deque[_LT]
        if __iterable is not None:
            self.__deque = collections.deque(__iterable)
        else:
            self.__deque = collections.deque()

    @_atomic_require_lock
    def get_obj(self) -> DequeView[_LT]:  # type: ignore[override]
        """Returns a view of the current deque."""
        return DequeView(self.__deque)

    @_atomic_require_context
    def set_obj(self, value: Iterable[_LT], /) -> None:
        """Sets the deque to the given value."""
        self.__deque = collections.deque(value)

    @_atomic_require_lock
    def __len__(self) -> int:
        return len(self.__deque)

    @_atomic_require_lock
    def __getitem__(  # type: ignore[override]
        self,
        key: int, /
    ) -> _LT | "collections.deque[_LT]":
        return self.__deque[key]

    @_atomic_require_lock
    def __contains__(self, value: object) -> bool:
        return value in self.__deque

    @_atomic_require_lock
    def __iter__(self) -> Iterator[_LT]:
        return iter(self.__deque)

    @_atomic_require_context
    def __setitem__(  # type: ignore[override]
        self,
        key: int,
        value: _LT, /
    ) -> None:
        self.__deque[key] = value

    @_atomic_require_context
    def __delitem__(self, key: int, /) -> None:  # type: ignore[override]
        del self.__deque[key]

    @_atomic_require_context
    def __iadd__(self, value: Iterable[_LT]) -> "AtomicDeque[_LT]":
        self.__deque += value
        return self

    @_atomic_require_lock
    def __add__(self, value: collections.deque[_LT]) -> collections.deque[_LT]:
        return self.__deque + value

    @_atomic_require_context
    def __imul__(self, value: int) -> "AtomicDeque[_LT]":
        self.__deque *= value
        return self

    @_atomic_require_lock
    def __mul__(self, value: int) -> collections.deque[_LT]:
        return self.__deque * value

    @_atomic_require_lock
    def index(  # pylint: disable=arguments-differ
        self,
        value: _LT,
        start: int = 0,
        stop: int = sys.maxsize, /
    ) -> int:
        """
        Return first index of value.

        Raises ValueError if the value is not present.
        """
        super().index(value, start, stop)
        return self.__deque.index(value, start, stop)

    @_atomic_require_lock
    def count(self, value: _LT) -> int:
        """Return number of occurrences of value."""
        return self.__deque.count(value)

    @_atomic_require_context
    def append(self, value: _LT) -> None:
        """Append an element to the end of the deque."""
        self.__deque.append(value)

    @_atomic_require_context
    def appendleft(self, value: _LT) -> None:
        """Append an element to the beginning of the deque."""
        self.__deque.appendleft(value)

    @_atomic_require_context
    def extend(self, values: Iterable[_LT]) -> None:
        """Extend the deque by appending elements from the iterable."""
        self.__deque.extend(values)

    @_atomic_require_context
    def extendleft(self, values: Iterable[_LT]) -> None:
        """Extend the deque by appending elements from the iterable."""
        self.__deque.extendleft(values)

    @_atomic_require_context
    def insert(self, index: int, value: _LT) -> None:
        """Insert an element before the given index."""
        self.__deque.insert(index, value)

    # pylint: disable=arguments-differ
    @_atomic_require_context
    def pop(  # type: ignore[override]
        self
    ) -> _LT:
        """
        Remove and return an element from the right side (end) of the deque.
        """
        return self.__deque.pop()
    # pylint: enable=arguments-differ

    @_atomic_require_context
    def popleft(self) -> _LT:
        """
        Remove and return an element from the left side (start) of the deque.
        """
        return self.__deque.popleft()

    @_atomic_require_context
    def remove(self, value: _LT) -> None:
        """
        Remove the first occurrence of value.

        Raises ValueError if the value is not present.
        """
        self.__deque.remove(value)

    @_atomic_require_context
    def clear(self) -> None:
        """Remove all elements from the deque."""
        self.__deque.clear()

    @_atomic_require_context
    def reverse(self) -> None:
        """Reverse the elements of the deque in-place."""
        self.__deque.reverse()


_KT = TypeVar("_KT", bound=Hashable)
_VT = TypeVar("_VT")


class AtomicDict(_Atomic[dict[_KT, _VT]], collections.abc.MutableMapping):
    """
    A thread-safe dictionary whose updates are atomic.

    Updates to the dictionary are only allowed within a context manager.
    """

    __slots__ = {
        "__dict": "The wrapped dictionary."
    }

    @overload
    def __init__(self, mapping: Mapping[_KT, _VT], /) -> None:
        """
        Create a new atomic dictionary from a mapping object's (key, value)
        pairs. The initial value will be copied into a dictionary.

        For example:
        ```
        >>> adict = AtomicDict({"one": 1, "two": 2})
        >>> adict
        AtomicDict({"one": 1, "two": 2})
        ```
        """

    @overload
    def __init__(self, iterable: Iterable[tuple[_KT, _VT]], /) -> None:
        """
        Create a new atomic dictionary from an iterable of tuples defining
        (key, value) pairs.

        For example:
        ```
        >>> adict = AtomicDict([("one", 1), ("two", 2)])
        >>> adict
        AtomicDict({"one": 1, "two": 2})
        ```
        """

    @overload
    def __init__(self, **kwargs: _VT) -> None:
        """
        Create a new atomic dictionary with the key to value pairs given in
        the keyword argument list.

        For example:
        ```
        >>> adict = AtomicDict(one=1, two=2)
        >>> adict
        AtomicDict({"one": 1, "two": 2})
        ```
        """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.__dict: dict[_KT, _VT] = dict(*args, **kwargs)

    @_atomic_require_lock
    def get_obj(self) -> DictView[_KT, _VT]:  # type: ignore[override]
        """Returns a view of the current dictionary."""
        return DictView(self.__dict)

    @_atomic_require_context
    def set_obj(self, value: Mapping[_KT, _VT], /) -> None:
        """Sets the dictionary to the given value."""
        self.__dict = dict(value)

    @_atomic_require_lock
    def keys(self) -> KeysView[_KT]:
        """Return a view of the current dictionary's keys."""
        return super().keys()

    @_atomic_require_lock
    def values(self) -> ValuesView[_VT]:
        """Return a view of the current dictionary's values."""
        return super().values()

    @_atomic_require_lock
    def items(self) -> ItemsView[_KT, _VT]:
        """Return a view of the current dictionary's items."""
        return super().items()

    @_atomic_require_lock
    def __getitem__(self, key: _KT) -> _VT:
        return self.__dict[key]

    @_atomic_require_context
    def __setitem__(self, key: _KT, value: _VT) -> None:
        self.__dict[key] = value

    @_atomic_require_context
    def __delitem__(self, key: _KT) -> None:
        del self.__dict[key]

    @_atomic_require_lock
    def __iter__(self) -> Iterator[_KT]:
        return iter(self.__dict)

    @_atomic_require_lock
    def __len__(self) -> int:
        return len(self.__dict)


_ET = TypeVar("_ET", bound=Hashable)


class AtomicSet(_Atomic[set[_ET]], collections.abc.MutableSet):
    """
    A thread-safe set whose updates are atomic.

    Updates to the set are only allowed within a context manager.
    """

    __slots__ = {
        "__set": "The wrapped set."
    }

    @overload
    def __init__(self) -> None:
        """
        Create a new empty atomic set.

        For example:
        ```
        >>> aset = AtomicSet()
        >>> aset
        AtomicSet(set())
        ```
        """

    @overload
    def __init__(self, __iterable: Iterable[_ET], /) -> None:
        """
        Create a new atomic set from an iterable.

        For example:
        ```
        >>> aset = AtomicSet({"one", "two"})
        >>> aset
        AtomicSet({"one", "two"})
        ```
        """

    def __init__(  # type: ignore[misc]
        self,
        __iterable: Iterable[_ET] | None = None, /
    ) -> None:
        super().__init__()
        self.__set: set[_ET]
        if __iterable is None:
            self.__set = set()
        else:
            self.__set = set(__iterable)

    def __str__(self) -> str:
        return f"AtomicSet: {self.__set!s}"

    def __repr__(self) -> str:
        return f"AtomicSet({self.__set!r})"

    @_atomic_require_lock
    def get_obj(self) -> SetView[_ET]:  # type: ignore[override]
        """Returns a view of the current set."""
        return SetView(self.__set)

    @_atomic_require_context
    def set_obj(self, value: Iterable[_ET], /) -> None:
        """Sets the set to the given value."""
        self.__set = set(value)

    @_atomic_require_lock
    def __contains__(self, item: object) -> bool:
        return item in self.__set

    @_atomic_require_lock
    def __iter__(self) -> Iterator[_ET]:
        return iter(self.__set)

    @_atomic_require_lock
    def __len__(self) -> int:
        return len(self.__set)

    @_atomic_require_context
    def add(self, value: _ET) -> None:
        self.__set.add(value)
    add.__doc__ = set.add.__doc__

    @_atomic_require_context
    def discard(self, value: _ET) -> None:
        self.__set.discard(value)
    discard.__doc__ = set.discard.__doc__
