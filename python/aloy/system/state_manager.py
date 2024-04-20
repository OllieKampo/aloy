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
import types
from typing import Any, Callable, Generic, NamedTuple, TypeVar, overload
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


_WRAPPER_ASSIGNMENTS = ("__module__", "__name__", "__qualname__", "__doc__")


def declare_state(
    name: str,
    interval: float
) -> Callable[
    [Callable[["ExternalStateManager"], RT]],
    Callable[["ExternalStateManager", float, float | None], RT]
]:
    """
    Decorator to declare a method of an external state manager as an external
    state getter.

    The getter must take no arguments and return the external state.
    It is converted into a method that takes a threshold and a timeout
    argument and returns the external state.

    Parameters
    ----------
    `name: str` - The name of the external state.

    `interval: float` - The interval at which the external state is updated
    in seconds.
    """
    def decorator(
        getter: Callable[["ExternalStateManager"], RT]
    ) -> Callable[["ExternalStateManager", float, float | None], RT]:
        """Decorator converts the getter method."""

        def wrapper(
            manager: "ExternalStateManager",
            threshold: float = 0.0,
            timeout: float | None = None
        ) -> RT:
            """Wrapper simply gets the state from the manager."""
            return manager.get_state(
                name=name,
                timethreshold=threshold,
                timeout=timeout
            )

        wrapper.__state_name__ = name  # type: ignore[attr-defined]
        wrapper.__state_interval__ = interval  # type: ignore[attr-defined]
        wrapper.__state_getter__ = getter  # type: ignore[attr-defined]

        # Copy the module, names, and docstring of the wrapped function
        # to the wrapper.
        for item_name in _WRAPPER_ASSIGNMENTS:
            item = getattr(getter, item_name, None)
            if item is not None:
                setattr(wrapper, item_name, item)

        # Merge the dictionaries of the getter and the wrapper.
        getattr(wrapper, "__dict__", {}).update(
            getattr(getter, "__dict__", {}))

        # Update the annotations to add the threshold and timeout
        # parameters.
        __annotations__ = getattr(getter, "__annotations__", {})
        __annotations__["threshold"] = float
        __annotations__["timeout"] = float | None
        wrapper.__annotations__ = __annotations__

        return wrapper
    return decorator


class _StateGetter(NamedTuple):
    """Named tuple to store the interval and getter of an external state."""

    interval: float
    getter: Callable[[], Any]


def _make_state_getter(
    getter: Callable[["ExternalStateManager"], Any]
) -> _StateGetter:
    """Make a state getter from a method."""
    return _StateGetter(
        interval=getter.__state_interval__,  # type: ignore[attr-defined]
        getter=getter.__state_getter__  # type: ignore[attr-defined]
    )


class ExternalStateManagerMeta(type):
    """Metaclass for external state manager classes."""

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any]
    ) -> type:
        """Create a new external state class."""
        cls = super().__new__(mcs, name, bases, namespace)
        for attr_name, attr in namespace.items():
            if (attr_name.startswith("__")
                    or attr_name.endswith("__")):
                continue
            if isinstance(attr, types.FunctionType):
                if hasattr(attr, "__state_name__"):
                    cls.__states__[  # type: ignore[attr-defined]
                        attr.__state_name__  # type: ignore[attr-defined]
                    ] = _make_state_getter(attr)
        return cls


class ExternalStateManager(metaclass=ExternalStateManagerMeta):
    """Class to manage regular polling and storage of external state."""

    __slots__ = {
        "__clock": "The clock thread that polls the external state getters.",
        "__states": "The external states.",
        "__conditions": "The conditions for notifying state changes.",
        "__callbacks": "The callbacks for state changes."
    }

    __states__: dict[str, _StateGetter] = {}

    def __init__(self, max_workers: int | None = None) -> None:
        """Initialize the external state manager."""
        self.__clock = AsyncClockThread(max_workers)
        self.__states: dict[str, ExternalState] = {}
        self.__conditions: dict[str, threading.Condition] = {}
        self.__callbacks: dict[
            str, list[Callable[[str, ExternalState], None]]
        ] = {}

        for state_name, state_getter in self.__states__.items():
            self.add_state(
                name=state_name,
                interval=state_getter.interval,
                getter=functools.partial(state_getter.getter, self)
            )

    @overload
    def add_state(
        self,
        name: str,
        interval: float
    ) -> Callable[
        [Callable[[], Any]],
        None
    ]:
        ...

    @overload
    def add_state(
        self,
        name: str,
        interval: float,
        getter: Callable[[], Any]
    ) -> None:
        ...

    def add_state(
        self,
        name: str,
        interval: float,
        getter: Callable[[], Any] | None = None
    ) -> Callable[[Callable[[], Any]], None] | None:
        """Declare a setter for external state."""
        if getter is None:
            return self.__add_state_decorator(name, interval)
        self.__add_state(name, interval, getter)
        return None

    def __add_state(
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
        self.__clock.schedule(
            interval,
            getter_,  # type: ignore[arg-type]
            return_callback=setter_
        )

    def __add_state_decorator(
        self,
        name: str,
        interval: float
    ) -> Callable[[Callable[[], Any]], None]:
        """Declare a setter for external state."""
        def add_state_decorator(getter: Callable[[], Any]) -> None:
            self.__add_state(name, interval, getter)
        return add_state_decorator

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
