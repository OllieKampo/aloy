from abc import abstractmethod, ABCMeta
import functools
from typing import Callable
import _collections_abc
import threading
import typing
from auxiliary.metaclasses import create_if_not_exists_in_slots

from concurrency.clocks import ClockThread

class Observable(metaclass=ABCMeta):
    """
    Mixin for creating observable classes.
    Observables are thread safe and all methods are synchronised.
    """
    __slots__ = {"__observers" : "The observers of this observable.",
                 "__changed" : "The observers that have changed since the last update.",
                 "__clock" : "The clock that updates the observers.",
                 "__lock" : "Lock that ensure atomic updates to the observable."}
    
    def __init__(self, tick_rate: int = 10) -> None:
        self.__observers: set["Observer"] = set()
        self.__changed: set["Observer"] = set()
        self.__lock = threading.RLock()
        self.__clock: ClockThread = ClockThread(self.__update_observers, tick_rate=tick_rate)
        self.__clock.start()
    
    @property
    def observers(self) -> set["Observer"]:
        """Return the observers of this observable."""
        return self.__observers
    
    def assign_observers(self, *observers: "Observer") -> None:
        """Assign observers to this observable."""
        with self.__lock:
            for observer in observers:
                if isinstance(observer, Observer):
                    self.__observers.add(observer)
                    observer._add_observable(self)
                else:
                    raise TypeError(f"Observer must be of type Observer. Got; {observer} of type {type(observer)}.")
    
    def remove_observers(self, *observers: "Observer") -> None:
        """Remove observers from this observable."""
        with self.__lock:
            remaining_observers = self.__observers.difference(observers)
            self.__observers = remaining_observers
            for observer in remaining_observers:
                observer._remove_observable(self)
    
    @property
    def tick_rate(self) -> int:
        """Return the tick rate of the internal update clock."""
        return self.__clock.tick_rate
    
    @tick_rate.setter
    def tick_rate(self, value: int) -> None:
        """Set the tick rate of the internal update clock."""
        self.__clock.tick_rate = value
    
    def notify(self, *observers: "Observer", raise_: bool = False) -> None:
        """Notify given observers that this observable has changed."""
        with self.__lock:
            if not raise_:
                self.__changed.update(self.__observers.intersection(observers))
            else:
                for observer in observers:
                    if observer in self.__observers:
                        self.__changed.add(observer)
                    else:
                        raise ValueError(f"Observer {observer} is not observing this observable.")
    
    def notify_all(self) -> None:
        """Notify all observers that this observable has changed."""
        with self.__lock:
            self.__changed.update(self.__observers)
    
    def __update_observers(self) -> None:
        """Update all observers that have been notified."""
        observers = self.__changed.copy()
        for observer in observers:
            observer._update(self)
        self.__changed.clear()
    
    def enable_updates(self) -> None:
        """Enable updates to observers."""
        self.__clock.start()
    
    def disable_updates(self) -> None:
        """Disable updates to observers."""
        self.__clock.stop()

SP = typing.ParamSpec("SP")
ST = typing.TypeVar("ST")

def notifies_observers(function: Callable[SP, ST]) -> Callable[SP, ST]:
    """
    Decorator for methods of an observable class.
    Schedules all observers to be updated after the decorated method has returned.
    """
    @functools.wraps(function)
    def wrapper(self, *args: SP.args, **kwargs: SP.kwargs) -> ST:
        return_value = function(self, *args, **kwargs)
        self.notify_all()
        return return_value
    return wrapper

def notifies_observables(function: Callable[SP, ST]) -> Callable[SP, ST]:
    """Decorator function that schedules the observable to be updated after the decorated method has returned."""
    @functools.wraps(function)
    def wrapper(self, *args: SP.args, **kwargs: SP.kwargs) -> ST:
        return_value = function(self, *args, **kwargs)
        with self.__lock__:
            for observable in self.__observables__:
                observable.notify_all()
        return return_value
    return wrapper

class _ObserverMeta(type):
    def __new__(cls, cls_name: str, bases: tuple[str], class_dict: dict) -> type:
        """Metaclass for observer class."""
        class_dict = create_if_not_exists_in_slots(class_dict,
                                                   __observables__="The observables being observed or updated.",
                                                   __lock__="Lock that ensures atomic updates to the observer or updater.")
        
        original_init = class_dict["__init__"]
        @functools.wraps(original_init)
        def __init__(self, *args, **kwargs) -> None:
            self.__observables__: list[Observable] = []
            original_init(self, *args, **kwargs)
        class_dict["__init__"] = __init__
        
        class_dict["__lock__"] = threading.Lock()

        return super().__new__(cls, cls_name, bases, class_dict)

class Observer(_collections_abc.Hashable, metaclass=_ObserverMeta):
    """
    Mixin for creating observer classes.
    
    Observer classes must implement the `update(observable: Observable) -> None` method and be hashable.
    """

    @abstractmethod
    def update(self, observable: Observable) -> None:
        """Update the observer."""
        raise NotImplementedError
    
    def _update(self, observable: Observable) -> None:
        """Update the observer."""
        with self.__lock__:
            self.update(observable)
    
    @property
    def observables(self) -> list[Observable]:
        """Return the observable being observed."""
        return self.__observables__
    
    def _add_observable(self, observable: Observable) -> None:
        """Add an observable to the observer."""
        with self.__lock__:
            self.__observables__.append(observable)
    
    def _remove_observable(self, observable: Observable) -> None:
        """Remove an observable from the observer."""
        with self.__lock__:
            self.__observables__.remove(observable)

class Updater(metaclass=_ObserverMeta):
    """Mixin for creating updater classes."""

    @property
    def observables(self) -> list[Observable]:
        """Return the observables being updated."""
        return self.__observables__
    
    def add_observables(self, *observables: Observable) -> None:
        """Add observables to the updater."""
        with self.__lock__:
            self.__observables__.extend(observables)
    
    def remove_observables(self, *observables: Observable) -> None:
        """Remove observables from the updater."""
        with self.__lock__:
            for observable in observables:
                self.__observables__.remove(observable)
