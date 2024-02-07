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

"""Module containing functions and classes for thread clocks."""

import math
import threading
import time
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from typing import (Callable, Final, Generic, ParamSpec, Protocol, TypeVar,
                    final, runtime_checkable)

from aloy.datahandling.runningstats import MovingAverage

__copyright__ = "Copyright (C) 2024 Oliver Michael Kamperis"
__license__ = "GPL-3.0"
__version__ = "0.4.0"

__all__ = (
    "Tickable",
    "SimpleClockThread",
    "ClockThread",
    "RequesterClockThread"
)


def __dir__() -> tuple[str, ...]:
    """Get the names of module attributes."""
    return __all__


class _ClockBase:
    """Base class for clocks."""

    __DELAYED_TICKS_WARNING: Final[int] = 100
    __DELAYED_TICKS_RESET: Final[int] = 10

    __slots__ = {
        "_sleep_time": "The time to sleep between ticks.",
        "_atomic_update_lock": "A lock making start and stop calls atomic.",
        "__thread": "The thread that runs the clock.",
        "__running": "Event handling whether the clock is running.",
        "__stopped": "Event handling whether the clock should stop."
    }

    def __init__(self) -> None:
        """Create a new clock."""
        # Sleep time between ticks.
        self._sleep_time: float = 1.0

        # Lock for atomic start and stop calls.
        self._atomic_update_lock = threading.Lock()

        # Variables for the clock thread.
        self.__thread = threading.Thread(target=self.__run)
        self.__thread.daemon = True
        self.__running = threading.Event()
        self.__stopped = threading.Event()
        self.__thread.start()

    def _call_items(self) -> None:
        """Call the items."""
        raise NotImplementedError

    def start(self) -> None:
        """Start the clock."""
        with self._atomic_update_lock:
            if not self.__running.is_set():
                self.__stopped.clear()
                self.__running.set()

    def stop(self) -> None:
        """Stop the clock."""
        with self._atomic_update_lock:
            if self.__running.is_set():
                self.__stopped.set()
                self.__running.clear()

    def __run(self) -> None:
        """Run the clock."""
        while True:
            self.__running.wait()

            sleep_time: float = self._sleep_time
            start_sleep_time: float = time.perf_counter()
            delayed_ticks: int = 0
            ticks_since_last_delayed_tick: int = 0

            while not self.__stopped.wait(sleep_time):
                actual_sleep_time = time.perf_counter() - start_sleep_time
                if actual_sleep_time > (sleep_time * 1.05):
                    delayed_ticks += 1
                    if (delayed_ticks % self.__DELAYED_TICKS_WARNING) == 0:
                        warnings.warn(
                            f"[{self!s}] Unable to reach tick rate "
                            f"for {delayed_ticks} ticks."
                        )
                elif (delayed_ticks > 0
                        and (ticks_since_last_delayed_tick
                             < self.__DELAYED_TICKS_RESET)):
                    ticks_since_last_delayed_tick += 1
                elif (ticks_since_last_delayed_tick
                      == self.__DELAYED_TICKS_RESET):
                    delayed_ticks = 0
                    ticks_since_last_delayed_tick = 0

                start_update_time = time.perf_counter()
                with self._atomic_update_lock:
                    self._call_items()
                update_time = time.perf_counter() - start_update_time

                if update_time > actual_sleep_time:
                    sleep_time = 0.0
                    warnings.warn(
                        f"[{self!s}] Tick rate too high for scheduled items. "
                        "Update time longer than sleep time. Actual sleep "
                        f"time = {actual_sleep_time:.3f} seconds, items "
                        f"update time = {update_time:.3f} seconds. Setting "
                        "sleep time to 0.0 seconds."
                    )
                else:
                    # Adjust sleep time to account for update time.
                    sleep_time = ((sleep_time + self._sleep_time)
                                  - (actual_sleep_time + update_time))

                start_sleep_time = time.perf_counter()


PS = ParamSpec("PS")
TV_co = TypeVar("TV_co", covariant=True)  # pylint: disable=invalid-name


@runtime_checkable
class Tickable(Protocol[PS, TV_co]):
    """A protocol for tickable objects."""

    def tick(self, *args: PS.args, **kwargs: PS.kwargs) -> TV_co:
        """Tick the object."""


@final
class _SimpleClockItem(Generic[PS, TV_co]):
    """Class defining a simple clock item."""

    __slots__ = {
        "__func": "The function to call.",
        "__args": "The arguments to pass to the function.",
        "__kwargs": "The keyword arguments to pass to the function.",
        "__return_callback": "The callback to call with the return value."
    }

    def __init__(
        self,
        func: Callable[PS, TV_co],
        *args: PS.args,
        return_callback: Callable[[TV_co], None] | None = None,
        **kwargs: PS.kwargs
    ) -> None:
        """Create a new simple clock item."""
        self.__func = func
        self.__args = args
        self.__kwargs = kwargs
        self.__return_callback = return_callback

    def __call__(self) -> None:
        """Call the function."""
        return_: TV_co = self.__func(*self.__args, **self.__kwargs)
        if self.__return_callback is not None:
            self.__return_callback(return_)

    def __eq__(self, __value: object) -> bool:
        """Return whether the value is equal to the item."""
        if isinstance(__value, _SimpleClockItem):
            return self.__func == __value.__func  # pylint: disable=W0212
        if callable(__value):
            return self.__func == __value
        return NotImplemented


@final
class SimpleClockThread(_ClockBase):
    """
    Class defining a thread running a clock that regularly calls a set of
    functions at a global tick rate.
    """

    __slots__ = {
        "__items": "The items to tick."
    }

    def __init__(
        self,
        tick_rate: int = 20
    ) -> None:
        """
        Create a new clock thread with the given tickable items.

        Parameters
        ----------
        `tick_rate: int = 10` - The tick rate of the clock (ticks/second).
        This is approximate, the actual tick rate may vary, the only
        guarantee is that the tick rate will not exceed the given value.
        """
        super().__init__()

        # Scheduled items.
        self.__items: list[_SimpleClockItem] = []

        # Clock tick rate, sets the sleep time.
        self.tick_rate = tick_rate

    def __str__(self) -> str:
        """Return a string representation of the clock thread."""
        return (f"ClockThread: with {len(self.__items)} items "
                f"at tick rate {self.tick_rate} ticks/second.")

    @property
    def tick_rate(self) -> int:
        """Return the tick rate of the clock."""
        return int(1.0 / self._sleep_time)

    @tick_rate.setter
    def tick_rate(self, value: int) -> None:
        """Set the tick rate of the clock."""
        if value <= 0:
            raise ValueError("Tick rate must be greater than 0. "
                             f"Got; {value}.")
        self._sleep_time = 1.0 / value

    def schedule(
        self,
        func: Tickable[PS, TV_co] | Callable[PS, TV_co],
        *args: PS.args,
        return_callback: Callable[[TV_co], None] | None = None,
        **kwargs: PS.kwargs
    ) -> None:
        """
        Schedule an item to be ticked by the clock.

        Parameters
        ----------
        `func: Tickable[PS, TV_co] | Callable[PS, TV_co]` - The function to
        call.

        `*args: Any` - The positional arguments to pass to the function.

        `return_callback: Callable[[TV_co], None] | None` - A callback to
        call with the return value of the function every time it is called.

        `**kwargs: Any` - The keyword arguments to pass to the function.
        """
        with self._atomic_update_lock:
            if isinstance(func, Tickable):
                func = func.tick
            elif not callable(func):
                raise TypeError(f"Item {func!r} of type {type(func)} is "
                                "not tickable or callable.")
            self.__items.append(
                _SimpleClockItem(
                    func,
                    *args,
                    return_callback=return_callback,
                    **kwargs
                )
            )

    def unschedule(
        self,
        func: Tickable[PS, TV_co] | Callable[PS, TV_co]
    ) -> None:
        """Unschedule an item from being ticked by the clock."""
        with self._atomic_update_lock:
            if isinstance(func, Tickable):
                func = func.tick
            elif not callable(func):
                raise TypeError(f"Item {func!r} of type {type(func)} is "
                                "not tickable or callable.")
            self.__items.remove(func)  # type: ignore[arg-type]

    def _call_items(self) -> None:
        """Call the items."""
        for item in self.__items:
            item()


@final
class _TimedClockItem(Generic[PS, TV_co]):
    """Class defining a timed clock item."""

    __slots__ = {
        "__func": "The function to call.",
        "__args": "The arguments to pass to the function.",
        "__kwargs": "The keyword arguments to pass to the function.",
        "__return_callback": "The callback to call with the return value.",
        "interval": "The interval between calls to the function.",
        "last_time": "The last time the function was called.",
        "next_time": "The next time to call the function."
    }

    def __init__(
        self,
        interval: float,
        last_time: float,
        next_time: float,
        func: Callable[PS, TV_co],
        *args: PS.args,
        return_callback: Callable[[TV_co], None] | None = None,
        **kwargs: PS.kwargs
    ) -> None:
        """Create a new timed clock item."""
        # Function to call.
        self.__func = func
        self.__args = args
        self.__kwargs = kwargs
        self.__return_callback = return_callback

        # Timing variables.
        self.interval = interval
        self.last_time = last_time
        self.next_time = next_time

    def __call__(self) -> None:
        """Call the function."""
        return_: TV_co = self.__func(*self.__args, **self.__kwargs)
        if self.__return_callback is not None:
            self.__return_callback(return_)

    def __eq__(self, __value: object) -> bool:
        """Return whether the value is equal to the item."""
        if isinstance(__value, _TimedClockItem):
            return self.__func == __value.__func  # pylint: disable=W0212
        if callable(__value):
            return self.__func == __value
        return NotImplemented


@final
class ClockThread(_ClockBase):
    """
    Class defining a thread running a clock that regularly calls a set of
    functions, each with a unique time interval.
    """

    def __init__(self) -> None:
        """Create a new clock thread."""
        super().__init__()

        # Scheduled items.
        self.__items: list[_TimedClockItem] = []

        # Variables for timing.
        self.__last_time: float | None = None
        self.__frequency = MovingAverage(60)

    @property
    def frequency(self) -> float:
        """Return the frequency of the clock."""
        return 1.0 / self.__frequency.average

    def schedule(
        self,
        interval: float,
        func: Tickable[PS, TV_co] | Callable[PS, TV_co],
        *args: PS.args,
        return_callback: Callable[[TV_co], None] | None = None,
        **kwargs: PS.kwargs
    ) -> None:
        """
        Schedule an item to be ticked by the clock.

        Parameters
        ----------
        `interval: float` - The interval between calls to the function.

        `func: Tickable[PS, TV_co] | Callable[PS, TV_co]` - The function to
        call.

        `*args: Any` - The positional arguments to pass to the function.

        `return_callback: Callable[[TV_co], None] | None` - A callback to
        call with the return value of the function every time it is called.

        `**kwargs: Any` - The keyword arguments to pass to the function.
        """
        with self._atomic_update_lock:
            if isinstance(func, Tickable):
                func = func.tick
            elif not callable(func):
                raise TypeError(f"Item {func!r} of type {type(func)} is "
                                "not tickable or callable.")

            last_time: float
            next_time: float
            if self.__last_time is None:
                last_time = time.perf_counter()
                next_time = last_time + interval
            else:
                last_time = self.__last_time
                next_time = last_time + interval

            self.__items.append(
                _TimedClockItem(
                    interval,
                    last_time,
                    next_time,
                    func,
                    *args,
                    return_callback=return_callback,
                    **kwargs
                )
            )

            self._sleep_time = math.gcd(
                *(
                    int(item.interval * 100)
                    for item in self.__items
                )
            ) / 100

    def unschedule(
        self,
        func: Tickable[PS, TV_co] | Callable[PS, TV_co]
    ) -> None:
        """Unschedule an item from being ticked by the clock."""
        with self._atomic_update_lock:
            if isinstance(func, Tickable):
                func = func.tick
            elif not callable(func):
                raise TypeError(f"Item {func!r} of type {type(func)} is "
                                "not tickable or callable.")

            update_sleep_time: bool = False
            for item in self.__items:
                if item == func:
                    if item.interval <= self._sleep_time:
                        update_sleep_time = True
                    self.__items.remove(item)
                    break

            if update_sleep_time:
                self._sleep_time = math.gcd(
                    *(
                        int(item.interval * 100)
                        for item in self.__items
                    )
                ) / 100

    def _call_items(self) -> None:
        """Call the items."""
        current_time: float = time.perf_counter()
        delta_time: float = 0.0
        if self.__last_time is not None:
            delta_time = current_time - self.__last_time
            self.__frequency.append(delta_time)
        self.__last_time = current_time

        for item in self.__items:
            if item.next_time <= current_time:
                item()
                next_time = item.last_time + item.interval
                if next_time <= current_time:
                    next_time = current_time + item.interval
                item.next_time = next_time
                item.last_time = current_time


class _TimedClockFutureItem(Generic[PS, TV_co]):
    """Class defining a timed clock future item."""

    __slots__ = {
        "__func": "The function to call.",
        "__args": "The arguments to pass to the function.",
        "__kwargs": "The keyword arguments to pass to the function.",
        "__return_callback": "The callback to call with the return value.",
        "interval": "The interval between calls to the function.",
        "timeout": "The timeout for the function.",
        "last_time": "The last time the function was called.",
        "next_time": "The next time to call the function.",
        "next_timeout": "The next timeout for the function."
    }

    def __init__(
        self,
        interval: float,
        timeout: float,
        last_time: float,
        next_time: float,
        next_timeout: float,
        func: Callable[PS, TV_co],
        *args: PS.args,
        return_callback: Callable[[TV_co], None] | None = None,
        **kwargs: PS.kwargs
    ) -> None:
        """Create a new timed clock future item."""
        # Function to call.
        self.__func = func
        self.__args = args
        self.__kwargs = kwargs
        self.__return_callback = return_callback

        # Timing variables.
        self.interval = interval
        self.timeout = timeout
        self.last_time = last_time
        self.next_time = next_time
        self.next_timeout = next_timeout

    def __call__(self) -> None:
        """Call the function."""
        return_: TV_co = self.__func(*self.__args, **self.__kwargs)
        if self.__return_callback is not None:
            self.__return_callback(return_)

    def __eq__(self, __value: object) -> bool:
        """Return whether the value is equal to the item."""
        if isinstance(__value, _TimedClockFutureItem):
            return self.__func == __value.__func  # pylint: disable=W0212
        if callable(__value):
            return self.__func == __value
        return NotImplemented

    def __hash__(self) -> int:
        """Return the hash of the item."""
        return hash(self.__func)


@final
class RequesterClockThread(_ClockBase):
    """
    Class defining a thread used to run multiple clocks for
    regularly calling functions at given intervals.
    """

    def __init__(self, max_workers: int | None = None) -> None:
        """Create a new requester clock thread."""
        super().__init__()

        # Scheduled items.
        self.__items: dict[_TimedClockFutureItem, Future | None] = {}

        # Threadpool for running requests.
        self.__threadpool = ThreadPoolExecutor(
            max_workers=max_workers
        )

        # Variables for timing.
        self.__last_time: float | None = None
        self.__frequency = MovingAverage(60)

    @property
    def frequency(self) -> float:
        """Return the frequency of the clock."""
        return 1.0 / self.__frequency.average

    def schedule(
        self,
        interval: float,
        timeout: float,
        func: Tickable[PS, TV_co] | Callable[PS, TV_co],
        *args: PS.args,
        return_callback: Callable[[TV_co], None] | None = None,
        **kwargs: PS.kwargs
    ) -> None:
        """
        Schedule an item to be ticked by the clock.

        Parameters
        ----------
        `interval: float` - The interval between calls to the function.

        `timeout: float` - The timeout for the function.

        `func: Tickable[PS, TV_co] | Callable[PS, TV_co]` - The function to
        call.

        `*args: Any` - The positional arguments to pass to the function.

        `return_callback: Callable[[TV_co], None] | None` - A callback to
        call with the return value of the function every time it is called.

        `**kwargs: Any` - The keyword arguments to pass to the function.
        """
        with self._atomic_update_lock:
            if isinstance(func, Tickable):
                func = func.tick
            elif not callable(func):
                raise TypeError(f"Item {func!r} of type {type(func)} is "
                                "not tickable or callable.")

            last_time: float
            next_time: float
            next_timeout: float
            if self.__last_time is None:
                last_time = time.perf_counter()
                next_time = last_time + interval
                next_timeout = last_time + timeout
            else:
                last_time = self.__last_time
                next_time = last_time + interval
                next_timeout = last_time + timeout

            self.__items[_TimedClockFutureItem(
                interval,
                timeout,
                last_time,
                next_time,
                next_timeout,
                func,
                *args,
                return_callback=return_callback,
                **kwargs
            )] = None

            self._sleep_time = math.gcd(
                *(
                    int(item.interval * 100)
                    for item in self.__items
                )
            ) / 100

    def unschedule(
        self,
        func: Tickable[PS, TV_co] | Callable[PS, TV_co]
    ) -> None:
        """Unschedule an item from being ticked by the clock."""
        with self._atomic_update_lock:
            if isinstance(func, Tickable):
                func = func.tick
            elif not callable(func):
                raise TypeError(f"Item {func!r} of type {type(func)} is "
                                "not tickable or callable.")

            update_sleep_time: bool = False
            for item in self.__items:
                if item == func:
                    if item.interval <= self._sleep_time:
                        update_sleep_time = True
                    self.__items[item] = None
                    break

            if update_sleep_time:
                self._sleep_time = math.gcd(
                    *(
                        int(item.interval * 100)
                        for item in self.__items
                    )
                ) / 100

    def _call_items(self) -> None:
        """Call the items."""
        current_time: float = time.perf_counter()
        delta_time: float = 0.0
        if self.__last_time is not None:
            delta_time = current_time - self.__last_time
            self.__frequency.append(delta_time)
        self.__last_time = current_time

        for item, future in self.__items.items():
            if future is None:
                if item.next_time <= current_time:
                    self._submit_item(current_time, item)
            elif future.done():
                self.__items[item] = None
                future.result()
            elif item.next_timeout <= current_time:
                if future.running():
                    future.cancel()
                self._submit_item(current_time, item)

    def _submit_item(
        self,
        current_time: float,
        item: _TimedClockFutureItem
    ) -> None:
        """Submit an item to the threadpool."""
        self.__items[item] = self.__threadpool.submit(item)
        next_time = item.last_time + item.interval
        next_timeout = item.last_time + item.timeout
        if next_time <= current_time:
            next_time = current_time + item.interval
            next_timeout = current_time + item.timeout
        item.next_time = next_time
        item.next_timeout = next_timeout
        item.last_time = current_time


def __main() -> None:
    simple_clock = SimpleClockThread()
    simple_clock.schedule(print, "Hello, world!")
    simple_clock.start()
    time.sleep(2)
    simple_clock.stop()

    count1 = 0
    count2 = 0
    count3 = 0
    def print1(text: str) -> None:
        """Print text."""
        nonlocal count1
        count1 += 1
        print(f"{count1}: {text}")
    def print2(text: str) -> None:
        """Print text."""
        nonlocal count2
        count2 += 1
        print(f"{count2}: {text}")
    def print3(text: str) -> None:
        """Print text."""
        nonlocal count3
        count3 += 1
        print(f"{count3}: {text}")

    clock = ClockThread()
    clock.schedule(0.2, print1, "Hello, world!")
    clock.schedule(0.4, print2, "Goodbye, world!")
    clock.schedule(1.0, print3, "See you later!")
    clock.start()
    time.sleep(10)
    clock.stop()

    import random
    def mock_request() -> None:
        """Mock a request."""
        time_ = random.random() * 0.8
        time.sleep(time_)
        if time_ > 0.5:
            print("Request took too long.")
        else:
            print("Request complete.")
    requester_clock = RequesterClockThread()
    requester_clock.schedule(0.2, 0.5, mock_request)
    requester_clock.start()
    time.sleep(10)
    requester_clock.stop()


if __name__ == "__main__":
    __main()
