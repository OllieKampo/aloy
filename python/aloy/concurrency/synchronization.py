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

"""Module containing functions and classes for synchronization."""

from collections import defaultdict
import contextlib
import functools
import threading
import types
import typing
import weakref

from aloy.auxiliary.metaclasses import create_if_not_exists_in_slots
from aloy.auxiliary.introspection import loads_functions
from aloy.datastructures.graph import Graph
from aloy.datastructures.mappings import ReversableDict

__copyright__ = "Copyright (C) 2022 Oliver Michael Kamperis"
__license__ = "GPL-3.0"
__version__ = "0.3.0"

__all__ = (
    "OwnedRLock",
    "atomic_context",
    "atomic_update",
    "sync",
    "SynchronizedMeta"
)


def __dir__() -> tuple[str, ...]:
    """Get the names of module attributes."""
    return __all__


@typing.final
class OwnedRLock(contextlib.AbstractContextManager):
    """
    Class defining a reentrant lock that keeps track of its owner and
    recursion depth.
    """

    __slots__ = (
        "__name",
        "__lock",
        "__owner",
        "__recursion_depth",
        "__waiting_lock",
        "__waiting"
    )

    def __init__(self, lock_name: str | None = None) -> None:
        """
        Create a new owned reentrant lock with an optional name.

        Parameters
        ----------
        `lock_name : str | None = None` - The name of the lock, or None to
        give it no name.
        """
        self.__name: str | None = lock_name
        self.__lock = threading.RLock()
        self.__owner: threading.Thread | None = None
        self.__recursion_depth: int = 0
        self.__waiting_lock = threading.Lock()
        self.__waiting: set[threading.Thread] = set()

    def __str__(self) -> str:
        """Get a simple string representation of the lock."""
        return f"{'locked' if self.is_locked else 'unlocked'} " \
               f"{self.__class__.__name__} {self.__name}, " \
               f"owned by={self.__owner}, depth={self.__recursion_depth!s}"

    def __repr__(self) -> str:
        """Get a verbose string representation of the lock."""
        return f"<{'locked' if self.is_locked else 'unlocked'} " \
               f"{self.__class__.__name__}, name={self.__name}, " \
               f"owned by={self.__owner}, " \
               f"recursion depth={self.__recursion_depth!s}, " \
               f"lock={self.__lock!r}>"

    @property
    def name(self) -> str | None:
        """Get the name of the lock, or None if the lock has no name."""
        return self.__name

    @property
    def is_locked(self) -> bool:
        """Get whether the lock is currently locked."""
        return self.__owner is not None

    @property
    def owner(self) -> threading.Thread | None:
        """
        Get the thread that currently owns the lock, or None if the lock is
        not locked.
        """
        return self.__owner

    @property
    def is_owner(self) -> bool:
        """Get whether the current thread owns the lock."""
        return threading.current_thread() is self.__owner

    @property
    def recursion_depth(self) -> int:
        """
        Get the number of times the lock has been acquired by the current
        thread.
        """
        return self.__recursion_depth

    @property
    def any_threads_waiting(self) -> bool:
        """
        Get whether any threads are currently waiting to acquire the lock.
        """
        return bool(self.__waiting)

    @property
    def num_threads_waiting(self) -> int:
        """
        Get the number of threads that are currently waiting to acquire the
        lock.
        """
        return len(self.__waiting)

    @property
    def threads_waiting(self) -> frozenset[threading.Thread]:
        """
        Get the threads that are currently waiting to acquire the lock.
        """
        return frozenset(self.__waiting)

    @property
    def is_thread_waiting(self) -> bool:
        """
        Get whether the current thread is waiting to acquire the lock.
        """
        return threading.current_thread() in self.__waiting

    @functools.wraps(
        threading._RLock.acquire,  # pylint: disable=protected-access
        assigned=("__doc__",)
    )
    def acquire(self, blocking: bool = True, timeout: float = -1.0) -> bool:
        """
        Acquire the lock, blocking or non-blocking, with an optional timeout.
        """
        current_thread = threading.current_thread()
        if self.__owner is not current_thread:
            with self.__waiting_lock:
                self.__waiting.add(current_thread)
        if result := self.__lock.acquire(blocking, timeout):
            if self.__owner is None:
                with self.__waiting_lock:
                    self.__waiting.remove(current_thread)
                self.__owner = current_thread
            self.__recursion_depth += 1
        if not result:
            with self.__waiting_lock:
                self.__waiting.remove(current_thread)
        return result

    @functools.wraps(
        threading._RLock.release,  # pylint: disable=protected-access
        assigned=("__doc__",)
    )
    def release(self) -> None:
        """Release the lock, if it is locked by the current thread."""
        if self.__owner is threading.current_thread():
            self.__lock.release()
            self.__recursion_depth -= 1
            if self.__recursion_depth == 0:
                self.__owner = None

    __enter__ = acquire

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None
    ) -> None:
        """Release the lock when the context manager exits."""
        self.release()


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
) -> typing.Iterator[None]:
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


CT = typing.TypeVar("CT")
SP = typing.ParamSpec("SP")
ST = typing.TypeVar("ST")


def atomic_update(
    global_lock: str | None = None,
    method: bool = False
) -> typing.Callable[[typing.Callable[SP, ST]], typing.Callable[SP, ST]]:
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
    def decorator(func: typing.Callable[SP, ST]) -> typing.Callable[SP, ST]:
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


@typing.overload
def sync(
) -> typing.Callable[[typing.Callable[SP, ST]], typing.Callable[SP, ST]]:
    ...


@typing.overload
def sync(
    lock: typing.Literal["all", "method"],
    group_name: str | None = None
) -> typing.Callable[[typing.Callable[SP, ST]], typing.Callable[SP, ST]]:
    ...


@typing.overload
def sync(
    *, group_name: str
) -> typing.Callable[[typing.Callable[SP, ST]], typing.Callable[SP, ST]]:
    ...


def sync(
    lock: typing.Literal["all", "method"] | None = None,
    group_name: str | None = None
) -> typing.Callable[[typing.Callable[SP, ST]], typing.Callable[SP, ST]]:
    """
    Decorate a method or property of a synchronized class to synchronize
    access to the decorated method or property. A synchronized method or
    property can only be called by one thread at a time. Dunder methods
    cannot be synchronized.

    Methods can be synchronized with an instance lock or a method lock.

    Whilst an instance-locked method is running, no other instance-locked or
    method-locked methods can run. Whilst a method-locked method is running,
    no instance-locked methods can run, but other method-locked methods can.

    Instance-locked methods can call method-locked methods, but method-locked
    methods cannot call instance-locked methods as this would require the
    whole instance to be locked.

    Method-locked methods can be grouped to use the same lock if they access
    the same resources. Method-locked methods can call other method-locked
    methods, even if they are not in the same group. However, if two
    method-locked methods load or call each other, they are automatically
    added to the same group, called a loop-lock group, to prevent deadlocks.

    Parameters
    ----------
    `lock: "all" | "method" | None = None` - The type of lock to use, "all"
    creates an instance lock, "method" creates a method lock. If not given or
    None, "all" is used if `group_name` is not given or None, otherwise
    "method" is used if `group_name` is given and not None.

    `group_name: str | None = None` - The name of the group to add the method
    to. If not given or None, the method is not added to a group.
    Instance-locked methods cannot be added to a group.

    Raises
    ------
    `TypeError` - If the lock name or group names are given and not strings.

    `ValueError` - If the method is a dunder method, or the lock name is given
    and not "all" or "method", or the lock name is "all" and the group name is
    given and not None.
    """
    if lock is None:
        if group_name is None:
            lock = "all"
        else:
            lock = "method"

    def sync_dec(method: typing.Callable[SP, ST]) -> typing.Callable[SP, ST]:
        """
        Assign the lock name and group to the method's `__sync__` and
        `__group__` attributes.
        """
        if method.__name__.startswith("__") and method.__name__.endswith("__"):
            raise ValueError("Cannot synchronize a dunder method.")
        if not isinstance(lock, str):
            raise TypeError(
                "Lock name must be a string. "
                f"Got; {lock} of type {type(lock).__name__!r}."
            )
        if lock not in ("all", "method"):
            raise ValueError(
                "Lock name must be either 'all' or 'method'. Got; {lock!r}."
            )
        if group_name is not None and not isinstance(group_name, str):
            raise TypeError(
                "Group name must be a string. "
                f"Got; {group_name} of type {type(group_name)}."
            )
        if lock == "all" and group_name is not None:
            raise ValueError(
                "Instance-locked methods cannot be added to a group. "
                f"Got; {lock=}, {group_name=}."
            )
        method.__sync__ = lock         # type: ignore
        method.__group__ = group_name  # type: ignore
        return method

    return sync_dec


# TODO: Problem with this, is that if a method-locked method if executing,
# then executing another method-locked method always takes precedence over
# executing an instance-locked method, even if the instance-locked method
# was called first or many threads are waiting to the instance-locked method,
# because the instance-locked method has to wait for all method-locked
# methods to finish executing, but method-locked methods do not.
# Solution: Keep a count of how many threads are waiting for the
# instance-locked method, and if we reach a certain threshold, then
# stop the method-locked methods from executing until the instance-locked
# method has executed.


def synchronize_method(
    lock: typing.Literal["all", "method"] = "all"
) -> typing.Callable[
    [typing.Callable[typing.Concatenate[CT, SP], ST]],
    typing.Callable[typing.Concatenate[CT, SP], ST]
]:
    """Decorate a method to synchronize it in a synchronized class."""

    def synchronize_method_decorator(
        method: typing.Callable[typing.Concatenate[CT, SP], ST]
    ) -> typing.Callable[typing.Concatenate[CT, SP], ST]:
        """Return a synchronized method wrapper."""
        if lock == "all":
            @functools.wraps(method)
            def synchronize_method_wrapper(
                self,
                *args: SP.args,
                **kwargs: SP.kwargs
            ) -> ST:
                """Return a synchronized method wrapper."""
                # Acquire the instance lock.
                with self.__lock__:
                    # Wait for all method-locked threads to finish executing.
                    self.__event__.wait()
                    # Execute the method.
                    return method(self, *args, **kwargs)

        elif lock == "method":
            @functools.wraps(method)
            def synchronize_method_wrapper(
                self,
                *args: SP.args,
                **kwargs: SP.kwargs
            ) -> ST:
                """Return a synchronized method wrapper."""
                # If no method-locked methods are executing;
                if self.__event__.is_set():
                    # If the current thread owns the instance lock, then
                    # an instance-locked method is calling a method-locked
                    # method, so simply execute the method.
                    if self.__lock__.owner is threading.current_thread():
                        with self.__lock__:
                            return method(self, *args, **kwargs)

                    # Otherwise, wait to acquire the instance lock, to prevent
                    # any instance-locked methods executing beyond this point
                    # until the method-locks are updated.
                    self.__lock__.acquire()

                    # Attempt to acquire the method lock.
                    self.__method_locks__[method.__name__].acquire()

                    # Update the state to reflect that a method-locked method
                    # is executing. A given thread could only ever pass this
                    # point once, as a method-locked method would be executing
                    # on the second call, and we'd go through the else block.
                    # Therefore, we always acquire the semaphore here, as the
                    # recursion depth will always be one.
                    self.__semaphore__.acquire()
                    self.__event__.clear()

                    # Release the instance lock, no instance-locked methods
                    # can be executed, until all method locks are released.
                    self.__lock__.release()

                # If a method-locked method is already executing;
                else:
                    # Simply acquire the method lock.
                    self.__method_locks__[method.__name__].acquire()

                    # If this is the first time the given thread has called
                    # this method, then acquire the semaphore to show that
                    # this method-locked method is executing.
                    if (self.__method_locks__[method.__name__].recursion_depth
                            == 1):
                        self.__semaphore__.acquire()
                        self.__event__.clear()

                # Execute the method.
                result = method(self, *args, **kwargs)

                # Update the state to reflect that the method-locked method
                # has finished executing before releasing the method lock.
                # If the current thread is about to exit its first call to
                # the method, then release the semaphore to show that this
                # method-locked method has finished executing. If all method
                # locks are released, set the event to allow instance-locked
                # methods to execute again. The exception is if another thread
                # is waiting to execute the same method, in which case, we
                # don't want to set the event, as we want to allow the other
                # thread to execute the method-locked method, before we allow
                # instance-locked methods to execute again.
                if self.__method_locks__[method.__name__].recursion_depth == 1:
                    self.__semaphore__.release()
                # pylint: disable=protected-access
                if (not self.__method_locks__[method.__name__]
                        .any_threads_waiting
                        and self.__semaphore__._value == self.__semaphore__
                            ._initial_value):
                    self.__event__.set()
                self.__method_locks__[method.__name__].release()

                # Return the method's result.
                return result

        else:
            raise ValueError("Lock name must be either 'all' or 'method'.")

        return synchronize_method_wrapper
    return synchronize_method_decorator


def _check_for_sync(
    func: typing.Callable,
    all_methods: dict[str, typing.Callable],
    method_locked_methods: set[str],
    method_locked_methods_groups: ReversableDict[str, str],
    instance_locked_methods: set[str]
) -> typing.Callable:
    if func is not None:
        all_methods[func.__name__] = func
        if lock_name := getattr(func, "__sync__", False):
            if lock_name == "method":
                if ((lock_group := getattr(func, "__group__", None))
                        is not None):
                    method_locked_methods_groups[func.__name__] = lock_group
                else:
                    func.__group__ = None
                method_locked_methods.add(func.__name__)
            else:
                instance_locked_methods.add(func.__name__)
            sync_with_lock = synchronize_method(lock_name)
            func = sync_with_lock(func)
        else:
            func.__sync__ = False
            func.__group__ = None
    return func


class SynchronizedMeta(type):
    """Metaclass for synchronizing method calls for a class."""

    def __new__(
        mcs,
        cls_name: str,
        bases: tuple[typing.Type, ...],
        class_dict: dict[str, typing.Any]
    ) -> type:
        """
        Metaclass for synchronized classes.
        """
        # Ensure that if the class declares `__slots__` then it contains the
        # neccessary lock attributes.
        class_dict = create_if_not_exists_in_slots(
            class_dict,
            __lock__="Instance-locked method synchronization lock.",
            __method_locks__="Method-locked method synchronization locks.",
            __semaphore__="Method-locks semaphore (counts locked "
                          "method-locked methods).",
            __event__="Event signalling when all method-locked methods are "
                      "unlocked."
        )

        all_methods = dict[str, types.FunctionType]()
        method_locked_methods = set[str]()
        method_lock_groups = ReversableDict[str, str]()
        instance_locked_methods = set[str]()

        # Check through the class's attributes to find methods and properties
        # to synchronize.
        methods_to_sync: dict[str, types.FunctionType] = {}
        properties_to_sync: dict[
            str, tuple[types.FunctionType | None, ...]] = {}
        for attr_name, attr in class_dict.items():
            if (attr_name.startswith("__")
                    or attr_name.endswith("__")):
                continue
            if isinstance(attr, types.FunctionType):
                methods_to_sync[attr_name] = attr
            elif isinstance(attr, property):
                properties_to_sync[attr_name] = (
                    attr.fget, attr.fset, attr.fdel)

        _check_for_sync_ = functools.partial(
            _check_for_sync,
            all_methods=all_methods,
            method_locked_methods=method_locked_methods,
            method_locked_methods_groups=method_lock_groups,
            instance_locked_methods=instance_locked_methods
        )

        for func_name, func in methods_to_sync.items():
            class_dict[func_name] = _check_for_sync_(func)

        for attr_name, (fget, fset, fdel) in properties_to_sync.items():
            fget = _check_for_sync_(fget)
            fset = _check_for_sync_(fset)
            fdel = _check_for_sync_(fdel)
            class_dict[attr_name] = property(fget, fset, fdel)

        if method_locked_methods:
            # Obtain a graph of which methods load each other.
            load_graph: Graph[str] = Graph(directed=True)
            for method_name, method in all_methods.items():
                load_graph[method_name] = set(
                    loads_functions(
                        method,
                        all_methods.keys()
                    )
                )

            # Check that no instance-locked methods are loaded by
            # method-locked methods.
            for method_name in method_locked_methods:
                instance_locked_methods_intersection = (
                    load_graph[method_name] & instance_locked_methods
                )
                if instance_locked_methods_intersection:
                    if len(instance_locked_methods_intersection) > 1:
                        intersec = "', '".join(
                            instance_locked_methods_intersection
                        )
                        raise ValueError(
                            f"Method-locked method '{method_name}' cannot "
                            "load or call the instance-locked methods: "
                            f"'{intersec}'."
                        )
                    else:
                        intersec = next(iter(
                            instance_locked_methods_intersection
                        ))
                        raise ValueError(
                            f"Method-locked method '{method_name}' cannot "
                            "load or call the instance-locked method: "
                            f"'{intersec}'."
                        )

            # Group methods that are loaded by each other.
            loop_lock_numbers: ReversableDict[str, int] = ReversableDict()
            loop_lock_number_current: int = 0
            frontier: set[str] = method_locked_methods.copy()
            while frontier:
                method_name = frontier.pop()
                path = load_graph.get_path(
                    method_name,
                    method_name,
                    find_cycle=True,
                    raise_=False
                )
                if path is None:
                    continue
                path_set = set(path)
                if path_set:
                    if looped_methods := (method_locked_methods & path_set):
                        loop_lock_numbers.reversed_set(
                            loop_lock_number_current,
                            *looped_methods
                        )
                        while looped_methods:
                            looped_method = looped_methods.pop()
                            if looped_method in method_lock_groups:
                                lock_method_group = set(
                                    method_lock_groups.reversed_pop(
                                        method_lock_groups[looped_method],
                                        ()
                                    )
                                )
                                loop_lock_numbers.reversed_set(
                                    loop_lock_number_current,
                                    *lock_method_group
                                )
                                looped_methods -= lock_method_group
                        loop_lock_number_current += 1
                    frontier -= path_set

        # Wrap the init method to ensure instances get the lock attributes.
        original_init = class_dict["__init__"]

        @functools.wraps(original_init)
        def __init__(self, *args, **kwargs) -> None:
            if not hasattr(self, "__lock__"):
                self.__lock__ = OwnedRLock(lock_name="Instance lock")
                self.__method_locks__ = {}
            loop_locks: dict[int, OwnedRLock] = {}
            group_locks: dict[str, OwnedRLock] = {}
            total_method_locks: int = 0
            for method_name in method_locked_methods:
                if method_name in loop_lock_numbers:
                    loop_lock_number = loop_lock_numbers[method_name]
                    if loop_lock_number not in loop_locks:
                        lock_name = (
                            f"Loop Lock [{loop_lock_number}]: "
                            f"{loop_lock_numbers(loop_lock_number)}"
                        )
                        loop_locks[loop_lock_number] = OwnedRLock(
                            lock_name=lock_name)
                        total_method_locks += 1
                    self.__method_locks__[method_name] = \
                        loop_locks[loop_lock_number]
                elif method_name in method_lock_groups:
                    lock_group = method_lock_groups[method_name]
                    if lock_group not in group_locks:
                        lock_name = (
                            f"Group Lock [{lock_group}]: "
                            f"{method_lock_groups(lock_group)}"
                        )
                        group_locks[lock_group] = OwnedRLock(
                            lock_name=lock_name)
                        total_method_locks += 1
                    self.__method_locks__[method_name] = \
                        group_locks[lock_group]
                else:
                    self.__method_locks__[method_name] = OwnedRLock(
                        lock_name=method_name)
                    total_method_locks += 1
            if hasattr(self, "__semaphore__"):
                self.__semaphore__ = threading.BoundedSemaphore(
                    total_method_locks + self.__semaphore__._initial_value
                )
            else:
                self.__semaphore__ = threading.BoundedSemaphore(
                    total_method_locks)
            if not hasattr(self, "__event__"):
                self.__event__ = threading.Event()
                self.__event__.set()
            original_init(self, *args, **kwargs)

        # Assign the wrapped init method to the class.
        class_dict["__init__"] = __init__

        # Make the instance lock available as a property.
        class_dict["instance_lock"] = property(lambda self: self.__lock__)

        # Ensure that the class has lock status attributes.
        def is_instance_locked(self) -> bool:
            """Return whether the instance is locked."""
            return self.__lock__.is_locked and self.__event__.is_set()
        class_dict["is_instance_locked"] = is_instance_locked

        def is_method_locked(self, method_name: str | None = None) -> bool:
            """
            Return whether the method-locked method with the given name is
            locked.

            If no method name is given, then return whether any method-locked
            method is locked.
            """
            if method_name is None:
                # pylint: disable=protected-access
                return self.__semaphore__._value != \
                    self.__semaphore__._initial_value
                # pylint: enable=protected-access
            if method_name in self.__method_locks__:
                return self.__method_locks__[method_name].is_locked
            raise AttributeError(
                f"Method '{method_name}' is not method-locked.")
        class_dict["is_method_locked"] = is_method_locked
        class_dict["lockable_methods"] = property(
            lambda self: tuple(self.__method_locks__.keys()))

        if method_locked_methods:
            class_dict["loop_locks"] = property(lambda self: loop_lock_numbers)
            class_dict["group_locks"] = property(
                lambda self: method_lock_groups)
        else:
            class_dict["loop_locks"] = property(lambda self: ReversableDict())
            class_dict["group_locks"] = property(lambda self: ReversableDict())

        return super().__new__(mcs, cls_name, bases, class_dict)

    def __dir__(cls) -> typing.Iterable[str]:
        """Return the attributes of the class."""
        return list(super().__dir__()) + [
            "is_instance_locked",
            "is_method_locked",
            "lockable_methods",
            "loop_locks",
            "group_locks"
        ]


class SynchronizedClass(metaclass=SynchronizedMeta):
    """Base class for synchronizing a class via standard inheritance."""

    __slots__ = ()

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the class."""
        super().__init__(*args, **kwargs)

    instance_lock: OwnedRLock
    is_instance_locked: typing.Callable[[], bool]
    is_method_locked: typing.Callable[[str | None], bool]
    lockable_methods: tuple[str, ...]
    loop_locks: ReversableDict[str, int]
    group_locks: ReversableDict[str, str]
