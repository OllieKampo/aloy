###########################################################################
###########################################################################
## Module containing functions and classes for concurrency/parallelism.  ##
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

"""Module containing functions and classes for concurrency and parallelism."""

__copyright__ = "Copyright (C) 2022 Oliver Michael Kamperis"
__license__ = "GPL-3.0"

__all__ = ("atomic_update",
           "OwnedRLock",
           "sync",
           "synchronized_meta")

def __dir__() -> tuple[str]:
    """Get the names of module attributes."""
    return sorted(__all__)

def __getattr__(name: str) -> object:
    """Get an attributes from the module."""
    if name in __all__:
        return globals()[name]
    raise AttributeError(f"Module {__name__!r} has no attribute {name!r}.")

from collections import defaultdict
import contextlib
import functools
import threading
import types
import typing

from auxiliary.metaclasses import create_if_not_exists_in_slots
from auxiliary.introspection import loads_functions
from datastructures.graph import Graph
from datastructures.mappings import ReversableDict

__atomic_updaters: dict[str, threading.RLock] = defaultdict(threading.RLock)
@contextlib.contextmanager
def atomic_update(context_name: str, cls: type | None = None, inst: object | None = None) -> None:
    """Context manager to ensure that only one thread can update the context at a time."""
    context_name = f"Context: {context_name}"
    if inst is not None:
        context_name = f"Instance: {hex(id(inst))}, " + context_name
    if cls is not None:
        context_name = f"Class: {cls.__name__}, " + context_name
    __atomic_updaters[context_name].acquire()
    try:
        yield
    finally:
        __atomic_updaters[context_name].release()

class OwnedRLock(contextlib.AbstractContextManager):
    """Class defining a reentrant lock that keeps track of its owner and recursion depth."""
    
    __slots__ = ("__name",
                 "__lock",
                 "__owner",
                 "__recursion_depth")
    
    def __init__(self, lock_name: str | None = None) -> None:
        """
        Create a new owned reentrant lock with an optional name.
        
        Parameters
        ----------
        `lock_name : str | None = None` - The name of the lock, or None to give it no name.
        """
        self.__name: str | None = lock_name
        self.__lock = threading.RLock()
        self.__owner: threading.Thread | None = None
        self.__recursion_depth: int = 0
    
    def __str__(self) -> str:
        """Get a simple string representation of the lock."""
        return f"{'locked' if self.is_locked else 'unlocked'} {self.__class__.__name__} {self.__name}, \
            owned by={self.__owner}, depth={self.__recursion_depth!s}"
    
    def __repr__(self) -> str:
        """Get a verbose string representation of the lock."""
        return f"<{'locked' if self.is_locked else 'unlocked'} {self.__class__.__name__}, name={self.__name}, \
            owned by={self.__owner}, recursion depth={self.__recursion_depth!s}, lock={self.__lock!r}>"
    
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
        """Get the thread that currently owns the lock, or None if the lock is not locked."""
        return self.__owner
    
    @property
    def recursion_depth(self) -> int:
        """Get the number of times the lock has been acquired by the current thread."""
        return self.__recursion_depth
    
    @functools.wraps(threading._RLock.acquire, assigned=("__doc__",))
    def acquire(self, blocking: bool = True, timeout: float = -1.0) -> bool:
        """Acquire the lock, blocking or non-blocking, with an optional timeout."""
        if result := self.__lock.acquire(blocking, timeout):
            with atomic_update("is_released", OwnedRLock, self):
                if self.__owner is None:
                    self.__owner = threading.current_thread()
                self.__recursion_depth += 1
        return result
    
    @functools.wraps(threading._RLock.release, assigned=("__doc__",))
    def release(self) -> None:
        """Release the lock, if it is currently locked by the current thread."""
        if self.__owner is not None:
            with atomic_update("is_released", OwnedRLock, self):
                self.__lock.release()
                self.__recursion_depth -= 1
                if self.__recursion_depth == 0:
                    self.__owner = None
    
    __enter__ = acquire
    
    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: types.TracebackType | None) -> None:
        """Release the lock when the context manager exits."""
        self.release()

SP = typing.ParamSpec("SP")
ST = typing.TypeVar("ST")

def sync(lock: typing.Literal["all", "method"] = "all", group_name: str | None = None) -> typing.Callable[[typing.Callable[SP, ST]], typing.Callable[SP, ST]]:
    """
    Decorate a method to declare it as synchronized in a synchronized class.
    
    Methods can be synchronized with an instance lock or a method lock.
    
    Whilst an instance-locked method is running, no other instance-locked or method-locked methods can run.
    Whilst a method-locked method is running, no instance-locked methods can run.
    Whilst a method-locked method is running, other method-locked methods can run.
    
    Instance-locked methods can call method-locked methods,
    but method-locked methods cannot call instance-locked methods
    (as this would require the whole instance to be locked).
    Method-locked methods can be grouped to all use the same lock if they access the same resources.
    Method-locked methods can call other method-locked methods, even if they are not in the same group.
    However, if two method-locked method can call each other, they are automatically add to the same group.
    
    Parameters
    ----------
    `lock : {"all", "method"} = "all"` - The type of lock to use, "all" creates an instance lock, "method" creates a method lock.
    
    `group_name : str | None = None` - The name of the group to add the method to, or None to not add the method to a group.
    Instance-locked methods cannot be added to a group.
    """
    def sync_dec(method: typing.Callable[SP, ST]) -> typing.Callable[SP, ST]:
        """Assign the lock name to the method's `__sync__` attribute."""
        if method.__name__.startswith("__") and method.__name__.endswith("__"):
            raise ValueError("Cannot synchronize a dunder method.")
        if not isinstance(lock, str):
            raise TypeError("Lock name must be a string.")
        if lock not in ("all", "method"):
            raise ValueError("Lock name must be either 'all' or 'method'.")
        method.__sync__ = lock
        if lock == "all" and group_name is not None:
            raise ValueError("Instance-locked methods cannot be added to a group.")
        method.__group__ = group_name
        return method
    return sync_dec

def synchronize_method(lock: typing.Literal["all", "method"] = "all") -> typing.Callable[[typing.Callable[SP, ST]], typing.Callable[SP, ST]]:
    """Decorate a method to synchronize it in a synchronized class."""
    def synchronize_method_decorator(method: typing.Callable[SP, ST]) -> typing.Callable[SP, ST]:
        """Return a synchronized method wrapper."""
        if lock == "all":
            @functools.wraps(method)
            def synchronize_method_wrapper(self, *args: SP.args, **kwargs: SP.kwargs) -> ST:
                """Return a synchronized method wrapper."""
                ## Acquire the instance lock.
                with self.__lock__:
                    ## Wait for all method-locked threads to finish executing.
                    self.__event__.wait()
                    ## Execute the method.
                    return method(self, *args, **kwargs)
        
        elif lock == "method":
            @functools.wraps(method)
            def synchronize_method_wrapper(self, *args: SP.args, **kwargs: SP.kwargs) -> ST:
                """Return a synchronized method wrapper."""
                ## If no method-locked methods are executing;
                if self.__event__.is_set():
                    ## If the current thread owns the instance lock, then execute the method.
                    if self.__lock__.owner is threading.current_thread():
                        with self.__lock__:
                            return method(self, *args, **kwargs)
                    
                    ## Otherwise, acquire the instance lock, to prevent any instance-locked
                    ## methods executing beyond this point until the method-locks are updated.
                    self.__lock__.acquire()
                    
                    ## Attempt to acquire the method lock.
                    self.__method_locks__[method.__name__].acquire()
                    
                    ## Update the state to reflect that a method-locked method is executing.
                    self.__semaphore__.acquire()
                    self.__event__.clear()
                    
                    ## Release the instance lock, no instance-locked methods can be executed.
                    self.__lock__.release()
                    
                else:
                    self.__method_locks__[method.__name__].acquire()
                    if self.__method_locks__[method.__name__].recursion_depth == 1:
                        self.__semaphore__.acquire()
                        self.__event__.clear()
                
                ## Execute the method.
                result = method(self, *args, **kwargs)
                
                ## Update the state to reflect that the method-locked method has finished executing before releasing the method lock.
                if self.__method_locks__[method.__name__].recursion_depth == 1:
                    self.__semaphore__.release()
                if self.__semaphore__._value == self.__semaphore__._initial_value:
                    self.__event__.set()
                self.__method_locks__[method.__name__].release()
                
                ## Return the method's result.
                return result
            
        else: raise ValueError("Lock name must be either 'all' or 'method'.")
        
        return synchronize_method_wrapper
    return synchronize_method_decorator

class synchronized_meta(type):
    """Metaclass for synchronizing method calls for a class."""
    
    def __new__(cls, cls_name: str, bases: tuple[str], class_dict: dict) -> type:
        """
        Metaclass for synchronized classes.
        
        Ensures that if the class declares `__slots__` then it contains the neccessary lock attributes.
        """
        class_dict = create_if_not_exists_in_slots(class_dict,
                                                   __lock__="Instance-locked method synchronization lock.",
                                                   __method_locks__="Method-locked method synchronization locks.",
                                                   __semaphore__="Method-locks semaphore (counts number of current locked method-locked methods).",
                                                   __event__="Event signalling when all method-locked methods are unlocked.")
        
        ## Check through the class's attributes for methods that are to be synchronized.
        instance_locked_methods: set[str] = set()
        lock_methods: list[str] = []
        lock_methods_groups: ReversableDict[str, str] = ReversableDict()
        all_methods: dict[str, types.FunctionType] = {}
        for attr_name, attr in class_dict.items():
            if (isinstance(attr, types.FunctionType) # inspect.isfunction(attr)
                and not (attr_name.startswith("__") and attr_name.endswith("__"))):
                all_methods[attr_name] = attr
                if lock_name := getattr(attr, "__sync__", False):
                    if lock_name == "method":
                        if (lock_group := getattr(attr, "__group__", None)) is not None:
                            lock_methods_groups[attr_name] = lock_group
                        else: class_dict[attr_name].__group__ = None
                        lock_methods.append(attr_name)
                    else: instance_locked_methods.add(attr_name)
                    class_dict[attr_name] = synchronize_method(lock_name)(attr)
                else:
                    class_dict[attr_name].__sync__ = False
                    class_dict[attr_name].__group__ = None
        
        if lock_methods:
            ## Obtain a graph of which methods load each other.
            load_graph: Graph[str] = Graph(directed=True)
            for method_name, method in all_methods.items():
                load_graph[method_name] = set(loads_functions(method, all_methods.keys()))
            
            ## Check that no instance-locked methods are loaded by method-locked methods.
            for method_name in lock_methods:
                if instance_locked_methods_intersection := (load_graph[method_name] & instance_locked_methods):
                    if len(instance_locked_methods_intersection) > 1:
                        instance_locked_methods_intersection = "', '".join(instance_locked_methods_intersection)
                        raise ValueError(f"Method-locked method '{method_name}' cannot load or call the instance-locked methods: '{instance_locked_methods_intersection}'.")
                    else:
                        instance_locked_methods_intersection = next(iter(instance_locked_methods_intersection))
                        raise ValueError(f"Method-locked method '{method_name}' cannot load or call the instance-locked method: '{instance_locked_methods_intersection}'.")
            
            ## Group methods that are loaded by each other.
            loop_lock_numbers: ReversableDict[str, int] = ReversableDict()
            loop_lock_number_current: int = 0
            frontier: set[str] = set(lock_methods)
            while frontier:
                method_name = frontier.pop()
                if ((path := load_graph.get_path(method_name, method_name, raise_=False)) is not None
                    and (path := set(path[:-1]))):
                    if looped_methods := (set(lock_methods) & path):
                        loop_lock_numbers.reversed_set(loop_lock_number_current, *looped_methods)
                        while looped_methods:
                            looped_method = looped_methods.pop()
                            if looped_method in lock_methods_groups:
                                lock_method_group = set(lock_methods_groups.reversed_pop(lock_methods_groups[looped_method]))
                                loop_lock_numbers.reversed_set(loop_lock_number_current, *lock_method_group)
                                looped_methods -= lock_method_group
                        loop_lock_number_current += 1
                    frontier -= path
        
        ## Warp the init method to ensure instances get the lock attributes.
        original_init = class_dict["__init__"]
        @functools.wraps(original_init)
        def __init__(self, *args, **kws) -> None:
            self.__lock__ = OwnedRLock(lock_name="Instance lock")
            self.__method_locks__ = {}
            loop_locks: dict[int, OwnedRLock] = {}
            group_locks: dict[str, OwnedRLock] = {}
            total_method_locks: int = 0
            for method_name in lock_methods:
                if method_name in loop_lock_numbers:
                    loop_lock_number = loop_lock_numbers[method_name]
                    if loop_lock_number not in loop_locks:
                        lock_name = f"Loop Lock [{loop_lock_number}]: {loop_lock_numbers(loop_lock_number)}"
                        loop_locks[loop_lock_number] = OwnedRLock(lock_name=lock_name)
                        total_method_locks += 1
                    self.__method_locks__[method_name] = loop_locks[loop_lock_number]
                elif method_name in lock_methods_groups:
                    lock_group = lock_methods_groups[method_name]
                    if lock_group not in group_locks:
                        lock_name = f"Group Lock [{lock_group}]: {lock_methods_groups(lock_group)}"
                        group_locks[lock_group] = OwnedRLock(lock_name=lock_name)
                        total_method_locks += 1
                    self.__method_locks__[method_name] = group_locks[lock_group]
                else:
                    self.__method_locks__[method_name] = OwnedRLock(lock_name=method_name)
                    total_method_locks += 1
            self.__semaphore__ = threading.BoundedSemaphore(total_method_locks)
            self.__event__ = threading.Event()
            original_init(self, *args, **kws)
        class_dict["__init__"] = __init__
        
        ## Ensure that the class has lock status attributes.
        def is_instance_locked(self) -> bool:
            """Return whether the instance is locked."""
            return self.__lock__.is_locked and self.__event__.is_set()
        class_dict["is_instance_locked"] = is_instance_locked
        class_dict["lockable_methods"] = property(lambda self: tuple(self.__method_locks__.keys()))
        def is_method_locked(self, method_name: str | None = None) -> bool:
            """
            Return whether the method-locked method with the given name is locked.
            
            If no method name is given, then return whether any method-locked method is locked.
            """
            if method_name is None:
                return self.__semaphore__._value != self.__semaphore__._initial_value
            if method_name in self.__method_locks__:
                return self.__method_locks__[method_name].is_locked
            else: raise AttributeError(f"Method '{method_name}' is not method-locked.")
        class_dict["is_method_locked"] = is_method_locked
        
        return super().__new__(cls, cls_name, bases, class_dict)
    
    def __dir__(self) -> typing.Iterable[str]:
        """Return the attributes of the class."""
        return super().__dir__() + ["is_instance_locked", "lockable_methods", "is_method_locked"]