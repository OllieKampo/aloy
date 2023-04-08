###########################################################################
###########################################################################
## Module defining obversable-observer (publisher-subscriber) interface. ##
##                                                                       ##
## Copyright (C)  2023  Oliver Michael Kamperis                          ##
## Email: o.m.kamperis@gmail.com                                         ##
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

"""Module defining obversable-observer (publisher-subscriber) interface."""

from abc import abstractmethod, ABCMeta
import functools
import logging
from typing import Callable, ParamSpec, TypeVar, final
import threading
from PyQt6.QtCore import QTimer

from concurrency.clocks import ClockThread

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"

__all__ = (
    "Observable",
    "Observer",
    "notifies_observers",
    "notifies_observables"
)


def __dir__() -> tuple[str]:
    """Get the names of module attributes."""
    return __all__


class Observable(metaclass=ABCMeta):
    """
    An abstract class defining an observable object.

    Observables are thread safe and all methods are synchronised.
    """

    __OBSERVABLE_LOGGER = logging.getLogger("Observable")

    __slots__ = {
        "__name": "The name of the observable.",
        "__observers": "The observers of this observable.",
        "__changed": "The observers that have changed since the last update.",
        "__clock": "The clock that updates the observers.",
        "__lock": "Lock that ensure atomic updates to the observable.",
        "__debug": "Whether to log debug messages."
    }

    def __init__(
        self,
        name: str | None = None, /,
        clock: ClockThread | QTimer | None = None, *,
        tick_rate: int = 10,
        start_clock: bool = True,
        debug: bool = False
    ) -> None:
        """
        Create a new observable.

        Sub-classes must always call this method in their `__init__` method.
        For Qt6 GUI applications, `clock` should always be an existing QTimer
        and `start_clock` should be False (it should be started externally
        after construction of the sub-class instance has completed).

        Parameters
        ----------
        `name: str | None = None` - The name of the observable. If None, the
        class name and id of the object are used.

        `clock: ClockThread | QTimer | None = None` - The clock that updates
        the observers. If not given or None, a new ClockThread is created with
        the given tick rate. Otherwise, updates from this observable are
        scheduled on the given clock.

        `tick_rate: int = 10` - The tick rate of the clock if a new clock is
        created. Ignored if an existing clock is given.

        `start_clock: bool = True` - Whether to start the clock if an existing
        clock is given. Ignored if a new clock is created (the clock is always
        started in this case).

        `debug: bool = False` - Whether to log debug messages.
        """
        self.__debug: bool = debug
        if self.__debug:
            self.__OBSERVABLE_LOGGER.debug(
                "Creating new observable with: "
                "name=%s clock=%s, tick_rate=%s, start_clock=%s, debug=%s",
                name, clock, tick_rate, start_clock, debug
            )
        self.__name: str
        if name is None:
            self.__name = f"{type(self).__name__}@{id(self):#x}"
        else:
            self.__name = name
        self.__observers: set["Observer"] = set()
        self.__changed: set["Observer"] = set()
        self.__lock = threading.RLock()
        self.__clock: ClockThread | QTimer
        if clock is None:
            if self.__debug:
                self.__OBSERVABLE_LOGGER.debug(
                    "%s: Creating new ClockThread with tick rate %s.",
                    self.__name, tick_rate
                )
            self.__clock = ClockThread(
                self.__update_observers,
                tick_rate=tick_rate
            )
            self.__clock.start()
        else:
            if self.__debug:
                self.__OBSERVABLE_LOGGER.debug(
                    "%s: Using existing clock of type %s.",
                    self.__name, type(clock)
                )
            self.__clock = clock
            if isinstance(clock, ClockThread):
                self.__clock.schedule(self.__update_observers)
                if start_clock:
                    self.__clock.start()
            elif isinstance(clock, QTimer):
                self.__clock.timeout.connect(self.__update_observers)
                if start_clock:
                    self.__clock.start()
            else:
                raise TypeError("Clock must be of type ClockThread or QTimer."
                                f"Got; {clock} of {type(clock)}.")

    @final
    @property
    def observable_name(self) -> str:
        """Return the name of this observable."""
        return self.__name

    @final
    @property
    def observers(self) -> set["Observer"]:
        """Return the observers of this observable."""
        return self.__observers

    @final
    def assign_observers(self, *observers: "Observer") -> None:
        """Assign observers to this observable."""
        with self.__lock:
            if self.__debug:
                self.__OBSERVABLE_LOGGER.debug(
                    "%s: Assigning observers: %s",
                    self.__name, observers
                )
            for observer in observers:
                if isinstance(observer, Observer):
                    self.__observers.add(observer)
                    observer._add_observable(self)
                else:
                    raise TypeError("Observer must be of type Observer. "
                                    f"Got; {observer} of {type(observer)}.")

    @final
    def remove_observers(self, *observers: "Observer") -> None:
        """Remove observers from this observable."""
        with self.__lock:
            if self.__debug:
                self.__OBSERVABLE_LOGGER.debug(
                    "%s: Removing observers: %s",
                    self.__name, observers
                )
            remaining_observers = self.__observers.difference(observers)
            removed_observers = self.__observers.intersection(observers)
            self.__observers = remaining_observers
            for observer in removed_observers:
                observer._remove_observable(self)

    @final
    def clear_observers(self) -> None:
        """Clear all observers from this observable."""
        with self.__lock:
            if self.__debug:
                self.__OBSERVABLE_LOGGER.debug(
                    "%s: Clearing all observers.",
                    self.__name
                )
            for observer in self.__observers:
                observer._remove_observable(self)
            self.__observers.clear()

    @final
    @property
    def tick_rate(self) -> int:
        """Return the tick rate of the internal update clock."""
        return self.__clock.tick_rate

    @final
    @tick_rate.setter
    def tick_rate(self, value: int) -> None:
        """Set the tick rate of the internal update clock."""
        if self.__debug:
            self.__OBSERVABLE_LOGGER.debug(
                "%s: Setting tick rate to %s.",
                self.__name, value
            )
        self.__clock.tick_rate = value

    @final
    def notify(self, *observers: "Observer", raise_: bool = False) -> None:
        """
        Notify given observers that this observable has changed.

        If `raise_` is True, raise a ValueError if any of the given observers
        are not observing this observable.
        """
        with self.__lock:
            if self.__debug:
                self.__OBSERVABLE_LOGGER.debug(
                    "%s: Notifying observers: %s",
                    self.__name, observers
                )
            if not raise_:
                self.__changed.update(self.__observers.intersection(observers))
            else:
                for observer in observers:
                    if observer in self.__observers:
                        self.__changed.add(observer)
                    else:
                        raise ValueError(f"Observer {observer} is not "
                                         "observing this observable.")

    @final
    def notify_by_name(self, *names: str, raise_: bool = False) -> None:
        """
        Notify observers by name that this observable has changed.

        If `raise_` is True, raise a ValueError if any of the given names
        do not correspond to an observer of this observable.
        """
        with self.__lock:
            if self.__debug:
                self.__OBSERVABLE_LOGGER.debug(
                    "%s: Notifying observers by name: %s",
                    self.__name, names
                )
            if not raise_:
                self.__changed.update(
                    observer
                    for observer in self.__observers
                    if observer.observer_name in names
                )
            else:
                for name in names:
                    for observer in self.__observers:
                        if observer.observer_name == name:
                            self.__changed.add(observer)
                            break
                    else:
                        raise ValueError(f"Observer with name {name} is not "
                                         "observing this observable.")

    @final
    def notify_all(self) -> None:
        """Notify all observers that this observable has changed."""
        with self.__lock:
            if self.__debug:
                self.__OBSERVABLE_LOGGER.debug(
                    "%s: Notifying all observers.",
                    self.__name
                )
            self.__changed.update(self.__observers)

    @final
    def __update_observers(self) -> None:
        """Update all observers that have been notified."""
        with self.__lock:
            observers = self.__changed.copy()
            self.__changed.clear()
        for observer in observers:
            observer._sync_update_observer(self)

    @final
    def enable_updates(self) -> None:
        """Enable updates to observers."""
        if self.__debug:
            self.__OBSERVABLE_LOGGER.debug(
                "%s: Enabling updates.",
                self.__name
            )
        self.__clock.start()

    @final
    def disable_updates(self) -> None:
        """Disable updates to observers."""
        if self.__debug:
            self.__OBSERVABLE_LOGGER.debug(
                "%s: Disabling updates.",
                self.__name
            )
        self.__clock.stop()


class Observer(metaclass=ABCMeta):
    """
    Mixin for creating observer classes.

    Observer classes must implement an `update` method and be hashable.
    """

    __OBSERVER_LOGGER = logging.getLogger("Observer")

    __slots__ = {
        "__name": "Name of the observer.",
        "__observables": "The observables being observed.",
        "__lock": "Lock that ensure atomic updates to the observer.",
        "__debug": "Whether to log debug messages."
    }

    def __init__(
        self,
        name: str | None = None, *,
        debug: bool = False
    ) -> None:
        """Create an observer."""
        self.__name: str
        if name is None:
            self.__name = f"{type(self).__name__}@{hex(id(self))}"
        else:
            self.__name = name
        self.__observables: list[Observable] = []
        self.__lock = threading.RLock()
        self.__debug: bool = debug

    @final
    @property
    def observer_name(self) -> str:
        """Return the name of the observer."""
        return self.__name

    @final
    @property
    def debug(self) -> bool:
        """Return whether to log debug messages."""
        return self.__debug

    @abstractmethod
    def update_observer(self, observable: Observable) -> None:
        """Update the observer."""
        raise NotImplementedError

    @final
    def _sync_update_observer(self, observable: Observable) -> None:
        """Update the observer synchronously."""
        with self.__lock:
            if self.__debug:
                self.__OBSERVER_LOGGER.debug(
                    "Updating observer %s from observable %s.",
                    self.__name, observable.observable_name
                )
            self.update_observer(observable)

    @final
    @property
    def observables(self) -> list[Observable]:
        """Return the observable being observed."""
        return self.__observables

    @final
    def _add_observable(self, observable: Observable) -> None:
        """Add an observable to the observer."""
        with self.__lock:
            self.__observables.append(observable)

    @final
    def _remove_observable(self, observable: Observable) -> None:
        """Remove an observable from the observer."""
        with self.__lock:
            self.__observables.remove(observable)


SP = ParamSpec("SP")
ST = TypeVar("ST")


def notifies_observers(
    *names: str,
    raise_: bool = False
) -> Callable[[Callable[SP, ST]], Callable[SP, ST]]:
    """
    Decorate methods of an observable class.

    If `names` is not given, then schedules all observers to be updated after
    the method has returned. Otherwise, if `names` is given, then schedules
    observers with the given names to be updated after the method has returned.
    If `raise_` is True, raise a ValueError if any of the given names do not
    correspond to an observer of the observable.
    """
    if not names:
        def inner(function: Callable[SP, ST]) -> Callable[SP, ST]:
            @functools.wraps(function)
            def wrapper(self, *args: SP.args, **kwargs: SP.kwargs) -> ST:
                return_value = function(self, *args, **kwargs)
                self.notify_all()
                return return_value
            return wrapper
        return inner
    else:
        def inner(function: Callable[SP, ST]) -> Callable[SP, ST]:
            @functools.wraps(function)
            def wrapper(self, *args: SP.args, **kwargs: SP.kwargs) -> ST:
                return_value = function(self, *args, **kwargs)
                self.notify_by_name(*names, raise_=raise_)
                return return_value
            return wrapper
        return inner


def notifies_observables(function: Callable[SP, ST]) -> Callable[SP, ST]:
    """
    Decorate methods of an observer or updater class.

    Schedules the observable to be updated after the method has returned.
    """
    @functools.wraps(function)
    def wrapper(self, *args: SP.args, **kwargs: SP.kwargs) -> ST:
        return_value = function(self, *args, **kwargs)
        with self.__lock:
            for observable in self.__observables:
                observable.notify_all()
        return return_value
    return wrapper
