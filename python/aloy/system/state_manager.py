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
from typing import (Any, Callable, Generic, NamedTuple, Protocol, TypeVar,
                    overload)

from aloy.concurrency.clocks import AsyncCallStats, AsyncClockThread

__copyright__ = "Copyright (C) 2024 Oliver Michael Kamperis"
__license__ = "GPL-3.0"
__version__ = "0.2.0"

__all__ = (
    "ExternalState",
    "ExternalStateManager",
    "declare_state"
)


def __dir__() -> tuple[str, ...]:
    """Get the names of module attributes."""
    return __all__


VT = TypeVar("VT")


class ExternalState(Generic[VT]):
    """Class to manage the external state of the system."""

    __slots__ = {
        "__name": "The name of the external state.",
        "__state": "The external state.",
        "__time_obtained": "The time this state value was obtained.",
        "_delta_time": "The time since the state was last obtained.",
        "__total_updates": "The total number of updates to this state.",
        "__tick_rate": "The average rate at which the state is updated."
    }

    def __init__(
        self,
        name: str,
        state: VT,
        time_obtained: float,
        delta_time: float | None,
        total_updates: int,
        tick_rate: float
    ) -> None:
        """Initialize the external state."""
        self.__name: str = name
        self.__state: VT = state
        self.__time_obtained: float = time_obtained
        self._delta_time: float | None = delta_time
        self.__total_updates: int = total_updates
        self.__tick_rate: float = tick_rate

    def __str__(self) -> str:
        """Return the string representation of the external state."""
        return (
            "ExternalState("
            f"name={self.__name!r}, "
            f"state={self.__state}, "
            f"time_obtained={self.__time_obtained:.3f}, "
            f"delta_time={self._delta_time:.3f}, "
            f"total_updates={self.__total_updates:d}, "
            f"tick_rate={self.__tick_rate:.3f}"
            ")"
        )

    @property
    def name(self) -> str:
        """Get the name of the external state."""
        return self.__name

    @property
    def state(self) -> VT:
        """Get the state of the external state."""
        return self.__state

    @property
    def time_obtained(self) -> float:
        """Get the time the state was obtained."""
        return self.__time_obtained

    @property
    def delta_time(self) -> float | None:
        """Get the time since the state was last obtained."""
        return self._delta_time

    @property
    def total_updates(self) -> int:
        """Get the total number of updates to the state."""
        return self.__total_updates

    @property
    def tick_rate(self) -> float:
        """Get the average rate at which the state is updated."""
        return self.__tick_rate


RT = TypeVar("RT")
ESM_contra = TypeVar(
    "ESM_contra",
    bound="ExternalStateManager",
    contravariant=True
)


class GetStateHint(Protocol[RT]):
    """Protocol hint for the get state method."""

    def __call__(
        self,
        threshold: float,
        timeout: float | None = None
    ) -> ExternalState[RT]:
        ...


class GetState(Protocol[ESM_contra, RT]):
    """Protocol for the get state method."""

    def __get__(
        self,
        instance: ESM_contra,
        owner: type[ESM_contra]
    ) -> GetStateHint[RT]:
        ...

    def __call__(  # pylint: disable=no-self-argument
        _self,
        self: ESM_contra,
        threshold: float,
        timeout: float | None = None
    ) -> ExternalState[RT]:
        ...


_WRAPPER_ASSIGNMENTS = ("__module__", "__name__", "__qualname__", "__doc__")


def declare_state(
    name: str,
    interval: float
) -> Callable[
    [Callable[[ESM_contra], RT]],
    GetState[ESM_contra, RT]
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
        getter: Callable[[ESM_contra], RT]
    ) -> GetState[ESM_contra, RT]:
        """Decorator converts the getter method."""

        def wrapper(
            self: ESM_contra,
            threshold: float = 0.0,
            timeout: float | None = None
        ) -> ExternalState[RT]:
            """Wrapper simply gets the state from the manager."""
            return self.get_state(
                name=name,
                threshold=threshold,
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

    def start(self) -> None:
        """Start the external state manager."""
        self.__clock.start()

    def stop(self) -> None:
        """Stop the external state manager."""
        self.__clock.stop()

    def shutdown(self) -> None:
        """Shutdown the external state manager."""
        self.__clock.shutdown()

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
            state = ExternalState(
                name=state,
                state=state,
                time_obtained=call_stats.current_time,
                delta_time=0.0,
                total_updates=call_stats.total_calls,
                tick_rate=call_stats.tick_rate
            )
            with self.__conditions[name]:
                self.__states[name] = state
                self.__conditions[name].notify_all()
            if name in self.__callbacks:
                for callback in self.__callbacks[name]:
                    callback(name, state)
        return setter_callback

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
        threshold: float,
        timeout: float | None = None
    ) -> ExternalState[Any]:
        """Get the external state."""
        if name not in self.__conditions:
            raise ValueError(f"External state '{name}' has not been declared.")
        with self.__conditions[name]:
            if (name not in self.__states
                    or (self.__states[name].time_obtained
                        < (time.monotonic() - threshold))):
                notified = self.__conditions[name].wait(timeout)
                if not notified:
                    raise TimeoutError(
                        f"Timeout waiting for external state '{name}'."
                    )
            state = self.__states[name]
            state._delta_time = (  # pylint: disable=protected-access
                time.monotonic() - state.time_obtained)
            return state


def __main() -> None:
    """Main function for testing the module."""
    import time  # pylint: disable=import-outside-toplevel

    import requests  # pylint: disable=import-outside-toplevel

    class TestExternalStateManager(ExternalStateManager):
        """Test class for the external state manager."""

        @declare_state("test_state", 1.0)
        def get_test_state(self) -> int:
            """Get the test state."""
            return 42

        @declare_state("test_request", 2.0)
        def get_test_request(self) -> int:
            """Get the test request."""
            return len(requests.get("https://www.google.com").text)

    manager = TestExternalStateManager()

    manager.add_state("test_add_state", 3.0, lambda: 99)

    @manager.add_state("test_add_state_decorator", 4.0)
    def test_add_state_decorator() -> str:
        """Test the add state decorator."""
        return "So long, and thanks for all the fish!"

    manager.start()
    time.sleep(0.2)
    print(manager.get_test_state(threshold=1.0))
    print(manager.get_state("test_add_state", threshold=1.0))
    print(manager.get_state("test_add_state_decorator", threshold=1.0))
    try:
        while True:
            # print(manager.get_test_state(threshold=10.0))
            print(manager.get_test_request(threshold=1.0))
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop()
        manager.shutdown()


if __name__ == "__main__":
    __main()
