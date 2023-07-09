# Copyright (C) 2023 Oliver Michael Kamperis
# Email: o.m.kamperis@gmail.com
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
Module defining obversable-observer pattern interface.

The observable-observer pattern is a single-producer multi-consumer
communication design pattern in which an object, called the observable,
maintains a list of its dependents, called observers, and updates them
automatically when its state changes. The pattern is effective technique for
simple GUI based applications, because it is easy to understand and implement,
with minimal boilerplate code needed.

The observable-observer pattern is unlike the publisher-subscriber pattern in
that observers do not subscribe to specific events or topics. Instead, they
observe the whole observable, are notified when the observable changes in any
way, and it is up to the observer to decide how to update itself accordingly.

The main downside of the observable-observer pattern, is that it scales quite
poorly to large applications. For applictions with many components that need
to be updated very often, the pattern is typically very inefficient. There are
three main reasons for this:
- Firstly, observers are notified that the observable has changed, but not of
  what has changed. This means that observers either need to update themselves
  completely, or keep track of and check what has changed themselves.
- Secondly, although it is possible to only notify a sub-set of observers when
  changing the observable, typically, we don't want anything that updates the
  observable to have to know about the observers. This means that we usually
  notify all observers for any and all state changes. Therefore, every
  component of the application is updated, no matter how small the change is.
- Thirdly, observers are updated through a single method call, which takes
  the observable as an argument. This means that all observables appear to be
  the same to the observer. This is problematic, because we typically want to
  make custom observable sub-classes, which have different methods and
  properties. If we want to observe multiple observables, then the observer
  needs to check the type of the observable, and then call the appropriate
  method. This is not ideal, because it couples the observer to the observable.
"""

from abc import abstractmethod, ABCMeta
import functools
import logging
from typing import Callable, Concatenate, ParamSpec, TypeVar, final
import threading
from PySide6.QtCore import QTimer  # pylint: disable=E0611

from concurrency.clocks import ClockThread

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"

__all__ = (
    "Observable",
    "Observer",
    "notifies_observers",
    "notifies_observables"
)


def __dir__() -> tuple[str, ...]:
    """Get the names of module attributes."""
    return __all__


class Observable(metaclass=ABCMeta):
    """
    An abstract class defining a thread safe observable object.

    An observable can be observed by other objects called observers. Observers
    can be assigned to an observable and are 'notified' when the observable
    changes. When an observer is notified, it is scheduled to be 'updated'
    automatically by the observable, on a seperate thread. An observer or any
    other object (called an updater), can then change the observable's state,
    causing the other observers to be updated in a thread-safe manner.

    The central idea of this pattern, is that observers or updaters only need
    to know about the observable, and do not need to know about each other.
    They only care about how they change the observable and how they are
    updated by it. The observable therefore acts like a central database,
    which is shared by all observers.

    It is not typically the role of an observer or updater to notify other
    observers. One should specify setter methods of the observable that should
    notify observers. This way, the observable can control when it notifies
    other observers, and keeps the observers decoupled from each other.
    """

    __OBSERVABLE_LOGGER = logging.getLogger("Observable")

    __slots__ = {
        "__name": "The name of the observable.",
        "__observers": "The observers of this observable.",
        "__changed": "The observers that have been notified the the state of "
                     "the observable has changed since the last update.",
        "__chained": "The observables that this observable is chained to.",
        "__lock": "Lock ensuring atomic updates to the observable.",
        "__clock": "The clock or timer that updates the observers.",
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
        scheduled on the given clock. See `jinx.concurrency.clocks.ClockThread`
        for details on clock threads.

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

        # All observable objects must have a name.
        self.__name: str
        if name is None:
            self.__name = f"{type(self).__name__}@{id(self):#x}"
        else:
            self.__name = name

        # Ineternal data structures.
        self.__observers: set["Observer"] = set()
        self.__changed: set["Observer"] = set()
        self.__chained: dict[str, "Observable"] = {}

        # The lock and clock used when updateding the observers.
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
                    observer._add_observable(self)  # pylint: disable=W0212
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
                observer._remove_observable(self)  # pylint: disable=W0212

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
                observer._remove_observable(self)  # pylint: disable=W0212
            self.__observers.clear()

    @final
    def chain_notifies_to(self, observable_: "Observable") -> None:
        """Chain notifications from this observable to another observable."""
        with self.__lock:
            if self.__debug:
                self.__OBSERVABLE_LOGGER.debug(
                    "%s: Chaining notifies to %s.",
                    self.__name, observable_
                )
            self.__chained[observable_.observable_name] = observable_

    @final
    def get_chained_observables(self) -> dict[str, "Observable"]:
        """Return the observables that this observable is chained to."""
        return self.__chained

    @property
    def tick_rate(self) -> int:
        """Return the tick rate of the internal update clock."""
        if isinstance(self.__clock, ClockThread):
            return self.__clock.tick_rate
        return int(1.0 / (self.__clock.interval() * 1000))

    @tick_rate.setter
    def tick_rate(self, value: int) -> None:
        """
        Set the tick rate of the internal update clock.

        If the clock is a QTimer, this will have no effect.
        """
        if isinstance(self.__clock, ClockThread):
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

        Typically, an observer or updater should not need to specify
        particular observers to notify, since the desire is for observers
        and updaters to only know about the observable, and be agnostic to all
        other observers. Instead, all observers should be notified of a change
        to the observable, and either the observers should decide whether they
        should update or not, or a separate system should handle assigning and
        removing observers to the observable as needed.
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
            for observable in self.__chained.values():
                observable.notify(*observers, raise_=raise_)

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
            for observable_ in self.__chained.values():
                observable_.notify_by_name(*names, raise_=raise_)

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
            for observable_ in self.__chained.values():
                observable_.notify_all()

    @final
    def __update_observers(self) -> None:
        """Update all observers that have been notified."""
        with self.__lock:
            observers = self.__changed.copy()
            self.__changed.clear()
        for observer in observers:
            observer._sync_update_observer(self)  # pylint: disable=W0212

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
        """
        Create an observer.

        Parameters
        ----------
        `name: str | None = None` - The name of the observable. If None, the
        class name and id of the object are used.

        `debug: bool = False` - Whether to log debug messages.
        """
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
    def update_observer(self, observable_: Observable) -> None:
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
        """
        Return the observables being observed by the observer.

        This field updated automatically when the observer is added to or
        removed from an observable.
        """
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
) -> Callable[
    [Callable[Concatenate[Observable, SP], ST]],
    Callable[Concatenate[Observable, SP], ST]
]:
    """
    Decorate methods of an observable class to set the method to automatically
    notify observers after the method has returned.

    If `names` is not given, then notify all observers to be updated after
    the method has returned. Otherwise, if `names` is given, then notify
    observers with the given names to be updated after the method has returned.
    If `raise_` is True, raise a ValueError if any of the given names do not
    correspond to an observer of the observable.
    """
    if not names:
        def inner(
            function: Callable[Concatenate[Observable, SP], ST]
        ) -> Callable[Concatenate[Observable, SP], ST]:
            @functools.wraps(function)
            def wrapper(
                self: Observable,
                *args: SP.args,
                **kwargs: SP.kwargs
            ) -> ST:
                return_value = function(self, *args, **kwargs)
                self.notify_all()
                return return_value
            return wrapper
        return inner
    else:  # pylint: disable=no-else-return
        def inner(
            function: Callable[Concatenate[Observable, SP], ST]
        ) -> Callable[Concatenate[Observable, SP], ST]:
            @functools.wraps(function)
            def wrapper(
                self: Observable,
                *args: SP.args,
                **kwargs: SP.kwargs
            ) -> ST:
                return_value = function(self, *args, **kwargs)
                self.notify_by_name(*names, raise_=raise_)
                return return_value
            return wrapper
        return inner


def notifies_observables(
    function: Callable[Concatenate[Observer, SP], ST]
) -> Callable[Concatenate[Observer, SP], ST]:
    """
    Decorate methods of an observer class.

    Schedules the observable to be updated after the method has returned.
    """
    @functools.wraps(function)
    def wrapper(
        self: Observer,
        *args: SP.args,
        **kwargs: SP.kwargs
    ) -> ST:
        return_value = function(self, *args, **kwargs)
        with self.__lock:  # pylint: disable=W0212
            for observable in self.__observables:  # pylint: disable=W0212
                observable.notify_all()
        return return_value
    return wrapper
