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
        delta_time: float,
        total_updates: int,
        tick_rate: float
    ) -> None:
        """Initialize the external state."""
        self.state = state
        self.time_obtained = time_obtained
        self.delta_time = delta_time
        self.total_updates = total_updates
        self.tick_rate = tick_rate


class ExternalStateManager:
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
        self.__callbacks: dict[str, list[Callable[[ExternalState], None]]] = {}

    def declare_state(self, name: str, interval: float, getter: Callable[[], Any]) -> None:
        """Declare a setter for external state."""
        if name in self.__conditions:
            raise ValueError(f"External state '{name}' has already been declared.")
        self.__conditions[name] = threading.Condition()
        getter_ = self.__make_getter(getter)
        setter_ = self.__make_setter_callback(name)
        self.__clock.schedule(interval, getter_, return_callback=setter_)

    def __make_getter(self, getter: Callable[[], Any]) -> Callable[[AsyncCallStats], Any]:
        """Make a getter for external state."""
        def getter_callback(call_stats: AsyncCallStats) -> Any:
            return getter()
        return getter_callback

    def __make_setter_callback(self, name: str) -> Callable[[AsyncCallStats, Any], None]:
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
                callback(state)

    def declare_state_callback(self, name: str, setter: Callable[[str, Any], None]) -> None:
        """Declare a setter for external state."""
        if name not in self.__conditions:
            raise ValueError(f"External state '{name}' has not been declared.")
        self.__callbacks.setdefault(name, []).append(setter)

    def get_external_state(self, name: str, timethreshold: float, timeout: float = 1.0) -> Any:
        """Get the external state."""
        if name not in self.__conditions:
            raise ValueError(f"External state '{name}' has not been declared.")
        with self.__conditions[name]:
            if name not in self.__states or self.__states[name].time_obtained < (time.monotonic() - timethreshold):
                notified = self.__conditions[name].wait(timeout)
                if not notified:
                    raise TimeoutError(f"Timeout waiting for external state '{name}'.")
            return self.__states[name].state
