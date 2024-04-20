###############################################################################
# Copyright (C) 2024 Oliver Michael Kamperis
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

"""Module containing Aloy's external state manager."""

import functools
import threading
import time
from typing import Any, Callable, Generic, TypeVar
from aloy.concurrency.clocks import AsyncClockThread, AsyncCallStats

__copyright__ = "Copyright (C) 2024 Oliver Michael Kamperis"
__license__ = "GPL-3.0"
__version__ = "0.1.0"

__all__ = (
)


def __dir__() -> tuple[str, ...]:
    """Get the names of module attributes."""
    return __all__


VT = TypeVar("VT")


class ExternalState(Generic[VT]):
    """Class to manage the external state of the system."""

    __slots__ = {
        "state": "The external state.",
        "time_obtained": "The time this state value was obtained.",
        "delta_time": "The time since the state was last obtained.",
        "total_updates": "The total number of updates to this state.",
        "tick_rate": "The average rate at which the state is updated."
    }

    def __init__(
        self,
        state: VT,
        time_obtained: float,
        delta_time: float | None,
        total_updates: int,
        tick_rate: float
    ) -> None:
        """Initialize the external state."""
        self.state: VT = state
        self.time_obtained: float = time_obtained
        self.delta_time: float | None = delta_time
        self.total_updates: int = total_updates
        self.tick_rate: float = tick_rate


RT = TypeVar("RT")


def declare_state(
    name: str,
    interval: float
) -> Callable[
    [Callable[["ExternalStateManager"], RT]],
    Callable[["ExternalStateManager", float, float | None], RT]
]:
    def decorator(
        getter: Callable[["ExternalStateManager"], RT]
    ) -> Callable[["ExternalStateManager", float, float | None], RT]:
        @functools.wraps(getter)
        def wrapper(
            manager: "ExternalStateManager",
            threshold: float = 0.0,
            timeout: float | None = None
        ) -> RT:
            return manager.get_state(
                name=name,
                timethreshold=threshold,
                timeout=timeout
            )
        wrapper.__state_name__ = name
        wrapper.__state_interval__ = interval
        return wrapper
    return decorator


class ExternalStateMeta(type):
    """Metaclass for external state classes."""

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any]
    ) -> type:
        """Create a new external state class."""
        cls = super().__new__(mcs, name, bases, namespace)
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if callable(attr):
                if hasattr(attr, "__state_name__"):
                    cls.__states__[attr.__state_name__] = attr.__state_interval__
        return cls


class ExternalStateManager(metaclass=ExternalStateMeta):
    """Class to manage regular polling and storage of external state."""

    __slots__ = {
        "__clock": "The clock thread that polls the external state getters.",
        "__states": "The external states.",
        "__conditions": "The conditions for notifying state changes.",
        "__callbacks": "The callbacks for state changes."
    }

    def __init__(self, max_workers: int | None = None) -> None:
        """Initialize the external state manager."""
        self.__clock = AsyncClockThread(max_workers)
        self.__states: dict[str, ExternalState] = {}
        self.__conditions: dict[str, threading.Condition] = {}
        self.__callbacks: dict[
            str, list[Callable[[str, ExternalState], None]]
        ] = {}

    def add_state(
        self,
        name: str,
        interval: float,
        getter: Callable[[], Any]
    ) -> None:
        """Declare a setter for external state."""
        if name in self.__conditions:
            raise ValueError(
                f"External state '{name}' has already been declared."
            )
        self.__conditions[name] = threading.Condition()
        getter_ = self.__make_getter(getter)
        setter_ = self.__make_setter(name)
        self.__clock.schedule(interval, getter_, return_callback=setter_)

    def __make_getter(
        self,
        getter: Callable[[], Any]
    ) -> Callable[[AsyncCallStats], Any]:
        """Make a getter for external state."""
        def getter_callback(
            call_stats: AsyncCallStats  # pylint: disable=unused-argument
        ) -> Any:
            return getter()
        return getter_callback

    def __make_setter(
        self,
        name: str
    ) -> Callable[[AsyncCallStats, Any], None]:
        """Make a setter callback for external state."""
        def setter_callback(call_stats: AsyncCallStats, state: Any) -> None:
            self.__set_state(
                name=name,
                state=ExternalState(
                    state=state,
                    time_obtained=call_stats.current_time,
                    delta_time=call_stats.delta_time,
                    total_updates=call_stats.total_calls,
                    tick_rate=call_stats.tick_rate
                )
            )
        return setter_callback

    def __set_state(self, name: str, state: ExternalState) -> None:
        """Set the external state."""
        with self.__conditions[name]:
            self.__states[name] = state
            self.__conditions[name].notify_all()
        if name in self.__callbacks:
            for callback in self.__callbacks[name]:
                callback(name, state)

    def add_callback(
        self,
        name: str,
        setter: Callable[[str, ExternalState], None]
    ) -> None:
        """Declare a setter for external state."""
        if name not in self.__conditions:
            raise ValueError(f"External state '{name}' has not been declared.")
        self.__callbacks.setdefault(name, []).append(setter)

    def get_state(
        self,
        name: str,
        timethreshold: float,
        timeout: float | None = None
    ) -> Any:
        """Get the external state."""
        if name not in self.__conditions:
            raise ValueError(f"External state '{name}' has not been declared.")
        with self.__conditions[name]:
            if (name not in self.__states
                    or (self.__states[name].time_obtained
                        < (time.monotonic() - timethreshold))):
                notified = self.__conditions[name].wait(timeout)
                if not notified:
                    raise TimeoutError(
                        f"Timeout waiting for external state '{name}'."
                    )
            return self.__states[name].state
