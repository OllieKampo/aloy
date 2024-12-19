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

"""Module containing functions and classes for thread clocks."""

import math
import threading
import time
import warnings
from abc import ABCMeta, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from typing import (Callable, Concatenate, Final, Generic, ParamSpec, Protocol,
                    TypeVar, final, runtime_checkable)

from aloy.datahandling.runningstats import MovingAverage

__copyright__ = "Copyright (C) 2024 Oliver Michael Kamperis"
__license__ = "GPL-3.0"
__version__ = "0.7.0"

__all__ = (
    "Tickable",
    "CallStats",
    "AsyncCallStats",
    "SimpleClockThread",
    "ClockThread",
    "AsyncClockThread"
)


def __dir__() -> tuple[str, ...]:
    """Get the names of module attributes."""
    return __all__


PS = ParamSpec("PS")
TV_co = TypeVar("TV_co", covariant=True)  # pylint: disable=invalid-name


@runtime_checkable
class Tickable(Protocol[PS, TV_co]):
    """A protocol for tickable objects."""

    def tick(self, *args: PS.args, **kwargs: PS.kwargs) -> TV_co:
        """Tick the object."""


class CallStats:
    """
    Class for call statistics of functions submitted to a
    SimpleClockThread or ClockThread.
    """

    __slots__ = {
        "current_time": "The current time.",
        "time_submitted": "The time the call was submitted.",
        "time_since_submitted": "The time since the call was submitted.",
        "delta_time": "The time since the last call.",
        "total_calls": "The total number of calls.",
        "tick_rate": "The average tick rate of calls."
    }

    def __init__(
        self,
        current_time: float,
        time_submitted: float,
        time_since_submitted: float,
        delta_time: float | None,
        total_calls: int,
        tick_rate: float
    ) -> None:
        """Create a new call statistics object."""
        self.current_time: float = current_time
        self.time_submitted: float = time_submitted
        self.time_since_submitted: float = time_since_submitted
        self.delta_time: float | None = delta_time
        self.total_calls: int = total_calls
        self.tick_rate: float = tick_rate

    def __str__(self) -> str:
        """Return a string representation of the call statistics."""
        return (
            f"CallStats [total time={self.time_since_submitted:.3f}s, "
            f"delta time={self.delta_time:.3f}s, "
            f"total calls={self.total_calls}, "
            f"tick rate={self.tick_rate:.3f}Hz]"
        )


class AsyncCallStats(CallStats):
    """
    Class for call statistics of functions submitted to a
    RequesterClockThread.
    """

    __slots__ = {
        "total_lag_calls": "The total number of lagged calls."
    }

    def __init__(
        self,
        current_time: float,
        time_submitted: float,
        time_since_submitted: float,
        delta_time: float | None,
        total_calls: int,
        tick_rate: float,
        total_lag_calls: int
    ) -> None:
        """Create a new call statistics object."""
        super().__init__(
            current_time,
            time_submitted,
            time_since_submitted,
            delta_time,
            total_calls,
            tick_rate
        )
        self.total_lag_calls: int = total_lag_calls


class _SimpleClockItem(Generic[PS, TV_co]):
    """Class defining a simple clock item."""

    __slots__ = {
        "__func": "The function to call.",
        "__args": "The arguments to pass to the function.",
        "__kwargs": "The keyword arguments to pass to the function.",
        "__return_callback": "The callback to call with the return value.",
        "__except_callback": "The callback to call with any exception.",
        "__dual_callback": "The callback to call with the return value and "
                           "exception.",
        "__time_submitted": "The time the item was submitted.",
        "__total_calls": "The total number of calls.",
        "__tick_rate": "The average tick rate of calls."
    }

    def __init__(
        self,
        func: Callable[Concatenate[CallStats, PS], TV_co],
        *args: PS.args,
        return_callback: Callable[[CallStats, TV_co], None] | None = None,
        except_callback: Callable[[CallStats, Exception], None] | None = None,
        dual_callback: Callable[
            [CallStats, TV_co | None, Exception | None], None] | None = None,
        **kwargs: PS.kwargs
    ) -> None:
        """Create a new simple clock item."""
        # Function to call.
        self.__func = func
        self.__args = args
        self.__kwargs = kwargs

        # Callbacks.
        self.__return_callback = return_callback
        self.__except_callback = except_callback
        self.__dual_callback = dual_callback

        # Call statistics.
        self.__time_submitted = time.monotonic()
        self.__total_calls = 0
        self.__tick_rate = MovingAverage(
            window=10,
            initial=0.0,
            under_full_initial=False
        )

    @property
    def tick_rate(self) -> float:
        """Return the tick rate of the item."""
        if self.__tick_rate.average:
            return 1.0 / self.__tick_rate.average
        return 0.0

    def _get_call_stats(
        self,
        delta_time: float | None,
        current_time: float
    ) -> CallStats:
        """Return the call statistics of the item."""
        return CallStats(
            current_time=current_time,
            time_submitted=self.__time_submitted,
            time_since_submitted=current_time - self.__time_submitted,
            delta_time=delta_time,
            total_calls=self.__total_calls,
            tick_rate=self.tick_rate
        )

    def __repr__(self) -> str:
        """Return a string representation of the item."""
        return f"{self.__class__.__name__} for {self.__func.__name__}"

    def __call__(self, delta_time: float | None, current_time: float) -> None:
        """Call the function."""
        self.__total_calls += 1
        if delta_time is not None:
            self.__tick_rate.append(delta_time)
        call_stats = self._get_call_stats(delta_time, current_time)
        try:
            return_: TV_co = self.__func(
                call_stats,
                *self.__args,
                **self.__kwargs
            )
        except Exception as exc_1:  # pylint: disable=W0703
            if self.__except_callback is not None:
                try:
                    self.__except_callback(call_stats, exc_1)
                    if self.__dual_callback is not None:
                        self.__dual_callback(call_stats, None, exc_1)
                except Exception as exc_2:  # pylint: disable=W0703
                    warnings.warn(
                        f"Exception callback {self.__except_callback} "
                        f"raised an exception: {exc_2!r}."
                    )
        else:
            try:
                if self.__return_callback is not None:
                    self.__return_callback(call_stats, return_)
                if self.__dual_callback is not None:
                    self.__dual_callback(call_stats, return_, None)
            except Exception as exc_3:  # pylint: disable=W0703
                warnings.warn(
                    f"Return callback {self.__return_callback} raised "
                    f"an exception: {exc_3!r}."
                )

    def __eq__(self, __value: object) -> bool:
        """Return whether the value is equal to the item."""
        if isinstance(__value, _SimpleClockItem):
            return self.__func == __value.__func  # pylint: disable=W0212
        if callable(__value):
            return self.__func == __value
        return NotImplemented

    def __hash__(self) -> int:
        """Return the hash of the item."""
        return hash(self.__func)


class _TimedClockItem(_SimpleClockItem[PS, TV_co]):
    """Class defining a timed clock item."""

    __slots__ = {
        "interval": "The interval between calls to the function.",
        "last_time": "The last time the function was called.",
        "next_time": "The next time to call the function."
    }

    def __init__(
        self,
        interval: float,
        last_time: float,
        next_time: float,
        func: Callable[Concatenate[CallStats, PS], TV_co],
        *args: PS.args,
        return_callback: Callable[[CallStats, TV_co], None] | None = None,
        except_callback: Callable[[CallStats, Exception], None] | None = None,
        dual_callback: Callable[
            [CallStats, TV_co | None, Exception | None], None] | None = None,
        **kwargs: PS.kwargs
    ) -> None:
        """Create a new timed clock item."""
        super().__init__(
            func,
            *args,
            return_callback=return_callback,
            except_callback=except_callback,
            dual_callback=dual_callback,
            **kwargs
        )

        # Timing variables.
        self.interval = interval
        self.last_time = last_time
        self.next_time = next_time


class _TimedClockFutureItem(_TimedClockItem[PS, TV_co]):
    """Class defining a timed clock future item."""

    __slots__ = {
        "lag_callback": "The callback to call when the function takes too "
                        "long to return.",
        "lag_interval": "The interval between calls to the lag callback.",
        "next_lag_time": "The next time to call the lag callback.",
        "__total_lag_calls": "The total number of lagged calls."
    }

    def __init__(
        self,
        interval: float,
        last_time: float,
        next_time: float,
        func: Callable[Concatenate[AsyncCallStats, PS], TV_co],
        *args: PS.args,
        return_callback: Callable[
            [AsyncCallStats, TV_co], None] | None = None,
        except_callback: Callable[
            [AsyncCallStats, Exception], None] | None = None,
        dual_callback: Callable[
            [AsyncCallStats, TV_co | None, Exception | None], None
        ] | None = None,
        lag_interval: float | None = None,
        next_lag_time: float | None = None,
        lag_callback: Callable[[AsyncCallStats], None] | None = None,
        **kwargs: PS.kwargs
    ) -> None:
        """Create a new timed clock future item."""
        super().__init__(
            interval,
            last_time,
            next_time,
            func,  # type: ignore[arg-type]
            *args,
            return_callback=return_callback,  # type: ignore[arg-type]
            except_callback=except_callback,  # type: ignore[arg-type]
            dual_callback=dual_callback,  # type: ignore[arg-type]
            **kwargs
        )

        # Request lag variables.
        self.lag_interval = lag_interval
        self.next_lag_time = next_lag_time

        # Additional callbacks.
        self.lag_callback = lag_callback

        # Call statistics.
        self.__total_lag_calls = 0

    def _get_call_stats(
        self,
        delta_time: float | None,
        current_time: float
    ) -> AsyncCallStats:
        """Return the call statistics of the item."""
        stats = super()._get_call_stats(delta_time, current_time)
        return AsyncCallStats(
            current_time=current_time,
            time_submitted=stats.time_submitted,
            time_since_submitted=stats.time_since_submitted,
            delta_time=delta_time,
            total_calls=stats.total_calls,
            tick_rate=self.tick_rate,
            total_lag_calls=self.__total_lag_calls
        )

    def call_lag_callback(
        self,
        delta_time: float | None,
        current_time: float
    ) -> None:
        """Call the lag callback."""
        self.__total_lag_calls += 1
        if self.lag_callback is not None:
            call_stats = self._get_call_stats(delta_time, current_time)
            try:
                self.lag_callback(call_stats)
            except Exception as exc_1:  # pylint: disable=W0703
                warnings.warn(
                    f"Lag callback {self.lag_callback} raised an exception: "
                    f"{exc_1!r}."
                )


CI = TypeVar("CI", _SimpleClockItem, _TimedClockItem, _TimedClockFutureItem)


class _ClockBase(Generic[CI], metaclass=ABCMeta):
    """Base class for clocks."""

    __DELAYED_TICKS_WARNING: Final[int] = 100
    __DELAYED_TICKS_RESET: Final[int] = 10

    __slots__ = {
        "_sleep_time": "The time to sleep between ticks.",
        "_last_time": "The last time the clock ticked.",
        "_items": "The items to call.",
        "_atomic_update_lock": "A lock making start and stop calls atomic.",
        "__thread": "The thread that runs the clock.",
        "__running": "Event handling whether the clock is running.",
        "__stopped": "Event handling whether the clock should stop.",
        "__shutdown": "Event handling whether the clock has been shutdown.",
        "__tick_rate": "The moving average of the clock's tick rate."
    }

    def __init__(self, init_items: list[CI] | dict[CI, Future]) -> None:
        """Create a new clock."""
        # Sleep time between ticks.
        self._sleep_time: float = 1.0
        self._last_time: float | None = None

        # Scheduled items.
        self._items: list[CI] | dict[CI, Future] = init_items

        # Lock for atomic updates.
        self._atomic_update_lock = threading.RLock()

        # Variables for the clock thread.
        self.__thread = threading.Thread(target=self.__run)
        self.__thread.daemon = True
        self.__running = threading.Event()
        self.__stopped = threading.Event()
        self.__shutdown = threading.Event()
        self.__thread.start()

        # Track the tick rate of the clock.
        self.__tick_rate = MovingAverage(
            window=60,
            initial=0.0,
            under_full_initial=False
        )

    @final
    @property
    def tick_rate(self) -> float:
        """Return the tick rate of the clock."""
        if self.__tick_rate.average:
            return 1.0 / self.__tick_rate.average
        return 0.0

    @abstractmethod
    def _call_items(self) -> None:
        """Call the clock's items."""
        raise NotImplementedError

    @final
    def start(self) -> None:
        """Start the clock."""
        with self._atomic_update_lock:
            if self.__shutdown.is_set():
                raise RuntimeError(
                    f"The clock {self} has been shutdown and cannot be "
                    "restarted."
                )
            if not self.__running.is_set():
                self.__stopped.clear()
                self.__running.set()

    @final
    def stop(self) -> None:
        """Stop the clock."""
        with self._atomic_update_lock:
            if self.__running.is_set():
                self.__running.clear()
                self.__stopped.set()

    def shutdown(self) -> None:
        """
        Shutdown the clock, killing its thread.

        Once a clock has been shutdown, it cannot be restarted.
        """
        with self._atomic_update_lock:
            if not self.__shutdown.is_set():
                self.__shutdown.set()
                if not self.__running.is_set():
                    self.__stopped.set()
                    self.__running.set()
                self.__thread.join()

    def unschedule(
        self,
        func: Tickable[Concatenate[CallStats, PS], TV_co] | Callable[
            Concatenate[CallStats, PS], TV_co]
    ) -> None:
        """Unschedule an item from being ticked by the clock."""
        with self._atomic_update_lock:
            if isinstance(func, Tickable):
                func = func.tick
            elif not callable(func):
                raise TypeError(f"Item {func!r} of type {type(func)} is "
                                "not tickable or callable.")
            if isinstance(self._items, list):
                self._items.remove(func)  # type: ignore[arg-type]
            else:
                self._items.pop(
                    func  # type: ignore[arg-type,call-overload]
                )

    def __run(self) -> None:
        """Run the clock."""
        while not self.__shutdown.is_set():
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
                time_since_last_tick: float = 0.0

                if self._last_time is not None:
                    time_since_last_tick = start_update_time - self._last_time
                    self.__tick_rate.append(time_since_last_tick)

                with self._atomic_update_lock:
                    self._call_items()

                update_time = time.perf_counter() - start_update_time
                self._last_time = start_update_time

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


@final
class SimpleClockThread(_ClockBase[_SimpleClockItem]):
    """
    Class defining a thread running a clock that regularly calls a set of
    functions at a global tick rate.
    """

    __slots__ = {}

    def __init__(
        self,
        tick_rate: int = 20
    ) -> None:
        """
        Create a new clock thread with the given tickable items.

        Parameters
        ----------
        `tick_rate: int = 10` - The desired tick rate of the clock
        (ticks/second). This is approximate, the actual tick rate may vary,
        the only guarantee is that the tick rate will not exceed the given
        value.
        """
        super().__init__(init_items=[])

        # Clock tick rate, sets the sleep time.
        self.desired_tick_rate = tick_rate

    def __str__(self) -> str:
        """Return a string representation of the clock thread."""
        return (f"SimpleClockThread: with {len(self._items)} items "
                f"at tick rate {self.desired_tick_rate} ticks/second.")

    @property
    def desired_tick_rate(self) -> int:
        """Return the tick rate of the clock."""
        return int(1.0 / self._sleep_time)

    @desired_tick_rate.setter
    def desired_tick_rate(self, value: int) -> None:
        """Set the desired tick rate of the clock."""
        if value <= 0:
            raise ValueError("Tick rate must be greater than 0. "
                             f"Got; {value}.")
        self._sleep_time = 1.0 / value  # pylint: disable=assigning-non-slot

    def schedule(
        self,
        func: Tickable[Concatenate[CallStats, PS], TV_co] | Callable[
            Concatenate[CallStats, PS], TV_co],
        *args: PS.args,
        return_callback: Callable[[CallStats, TV_co], None] | None = None,
        except_callback: Callable[[CallStats, Exception], None] | None = None,
        dual_callback: Callable[
            [CallStats, TV_co | None, Exception | None], None] | None = None,
        **kwargs: PS.kwargs
    ) -> None:
        """
        Schedule an item to be ticked by the clock.

        Parameters
        ----------
        `func: Tickable[(CallStats, PS), TV_co] | ((CallStats, PS) -> TV_co)`
        - The function to call. Must take a CallStats object as the first
        argument.

        `*args: Any` - The positional arguments to pass to the function.

        `return_callback: ((CallStats, TV_co) -> None) | None` - A callback to
        call with the return value of the function every time it is called if
        the function does not raise an exception. Must take a CallStats object
        as the first argument.

        `except_callback: ((CallStats, Exception) -> None) | None` - A callback
        to call if any exception raised by the function when it is called. Must
        take a CallStats object as the first argument.

        `dual_callback: ((CallStats, TV_co | None, Exception | None) -> None)
        | None` - A callback to call with the return value and exception of the
        function every time it is called. Must take a CallStats object as the
        first argument.

        `**kwargs: Any` - The keyword arguments to pass to the function.
        """
        with self._atomic_update_lock:
            if isinstance(func, Tickable):
                func = func.tick
            elif not callable(func):
                raise TypeError(f"Item {func!r} of type {type(func)} is "
                                "not tickable or callable.")

            self._items.append(  # type: ignore[union-attr]
                _SimpleClockItem(
                    func,
                    *args,
                    return_callback=return_callback,
                    except_callback=except_callback,
                    dual_callback=dual_callback,
                    **kwargs
                )
            )

    def _call_items(self) -> None:
        """Call the clock's items."""
        current_time: float
        delta_time: float | None = None
        last_time: float | None = self._last_time
        for item in self._items:
            current_time = time.monotonic()
            if last_time is not None:
                delta_time = current_time - last_time
            else:
                delta_time = None
            item(delta_time=delta_time, current_time=current_time)


def _less_than_or_close(a: float, b: float) -> bool:
    """
    Return whether a is less than or close to b.

    Where a is close to b if the absolute difference between a and b is less
    than 5e-3, this is equivalent to 5 milliseconds.
    """
    return (a < b) or math.isclose(a, b, abs_tol=5e-3)


def _get_sleep_time(
    items: list[_TimedClockItem] | dict[_TimedClockFutureItem, Future]
) -> float:
    """Return the sleep time based on the items."""
    return math.gcd(
        *(
            int(item.interval * 1000)
            for item in items
        )
    ) / 1000


@final
class ClockThread(_ClockBase[_TimedClockItem]):
    """
    Class defining a thread running a clock that regularly calls a set of
    functions, each with a unique time interval.
    """

    __slots__ = {}

    def __init__(self) -> None:  # pylint: disable=useless-parent-delegation
        """Create a new clock thread."""
        super().__init__(init_items=[])

    def schedule(
        self,
        interval: float,
        func: Tickable[Concatenate[CallStats, PS], TV_co] | Callable[
            Concatenate[CallStats, PS], TV_co],
        *args: PS.args,
        return_callback: Callable[[CallStats, TV_co], None] | None = None,
        except_callback: Callable[[CallStats, Exception], None] | None = None,
        dual_callback: Callable[
            [CallStats, TV_co | None, Exception | None], None] | None = None,
        **kwargs: PS.kwargs
    ) -> None:
        """
        Schedule an item to be ticked by the clock.

        Parameters
        ----------
        `interval: float` - The time interval between calls to the function.

        `func: Tickable[(CallStats, PS), TV_co] | ((CallStats, PS) -> TV_co)`
        - The function to call. Must take a CallStats object as the first
        argument.

        `*args: Any` - The positional arguments to pass to the function.

        `return_callback: ((CallStats, TV_co) -> None) | None` - A callback to
        call with the return value of the function every time it is called if
        the function does not raise an exception. Must take a CallStats object
        as the first argument.

        `except_callback: ((CallStats, Exception) -> None) | None` - A callback
        to call if any exception raised by the function when it is called. Must
        take a CallStats object as the first argument.

        `dual_callback: ((CallStats, TV_co | None, Exception | None) -> None)
        | None` - A callback to call with the return value and exception of the
        function every time it is called. Must take a CallStats object as the
        first argument.

        `**kwargs: Any` - The keyword arguments to pass to the function.
        """
        with self._atomic_update_lock:
            if isinstance(func, Tickable):
                func = func.tick
            elif not callable(func):
                raise TypeError(f"Item {func!r} of type {type(func)} is "
                                "not tickable or callable.")

            last_time: float
            if self._last_time is None:
                last_time = time.perf_counter()
            else:
                last_time = self._last_time
            next_time: float = last_time + interval

            self._items.append(  # type: ignore[union-attr]
                _TimedClockItem(
                    interval,
                    last_time,
                    next_time,
                    func,
                    *args,
                    return_callback=return_callback,
                    except_callback=except_callback,
                    dual_callback=dual_callback,
                    **kwargs
                )
            )

            self._sleep_time = (  # pylint: disable=assigning-non-slot
                _get_sleep_time(self._items)  # type: ignore[arg-type]
            )

    def unschedule(
        self,
        func: Tickable[Concatenate[CallStats, PS], TV_co] | Callable[
            Concatenate[CallStats, PS], TV_co]
    ) -> None:
        """Unschedule an item from being ticked by the clock."""
        with self._atomic_update_lock:
            super().unschedule(func)

            self._sleep_time = (  # pylint: disable=assigning-non-slot
                _get_sleep_time(self._items)  # type: ignore[arg-type]
            )

    def _call_items(self) -> None:
        """Call the clock's items."""
        current_time: float
        delta_time: float | None = None
        for item in self._items:
            current_time = time.monotonic()
            if _less_than_or_close(item.next_time, current_time):
                if item.last_time is not None:
                    delta_time = current_time - item.last_time
                else:
                    delta_time = None
                item(delta_time=delta_time, current_time=current_time)
                next_time = item.last_time + (item.interval * 2.0)
                if next_time <= current_time:
                    next_time = current_time + item.interval
                item.next_time = next_time
                item.last_time = current_time


@final
class AsyncClockThread(_ClockBase[_TimedClockFutureItem]):
    """
    Class defining a thread running a clock that regularly calls a set of
    functions asynchronously, each with a unique time interval and lag
    interval. If the function takes longer than the lag interval to complete,
    a lag callback is called.

    This version of the clock thread is intended for use with functions used
    to make regular network requests. It is advisable to schedule your
    network request functions as follows:
    - Set the time interval to how often you want to make the request.
    - Set the lag interval to how long you expect the request to take. This
      should therefore be less than the time interval. The desire is to allow
      the lag callback to be called if the request takes longer than expected.
    - Set a timeout on the request function itself. This should be longer than
      the lag interval, but typically less than the time interval.
    """

    __slots__ = {
        "__threadpool": "The threadpool for running requests."
    }

    def __init__(self, max_workers: int | None = None) -> None:
        """Create a new requester clock thread."""
        super().__init__(init_items={})

        # Threadpool for running requests.
        self.__threadpool = ThreadPoolExecutor(
            max_workers=max_workers
        )

    def shutdown(self) -> None:
        """
        Shutdown the clock, killing its thread and threadpool.

        This cancels all pending requests, but waits for any running requests
        to complete.

        Once a clock has been shutdown, it cannot be restarted.
        """
        self.__threadpool.shutdown(cancel_futures=True)
        super().shutdown()

    def schedule(
        self,
        interval: float,
        func: Tickable[Concatenate[AsyncCallStats, PS], TV_co] | Callable[
            Concatenate[AsyncCallStats, PS], TV_co],
        *args: PS.args,
        return_callback: Callable[
            [AsyncCallStats, TV_co], None
        ] | None = None,
        except_callback: Callable[
            [AsyncCallStats, Exception], None
        ] | None = None,
        dual_callback: Callable[
            [AsyncCallStats, TV_co | None, Exception | None], None
        ] | None = None,
        lag_interval: float | None = None,
        lag_callback: Callable[[AsyncCallStats], None] | None = None,
        **kwargs: PS.kwargs
    ) -> None:
        """
        Schedule an item to be ticked asynchronously by the clock.

        Parameters
        ----------
        `interval: float` - The time interval between calls to the function.

        `func: Tickable[PS, TV_co] | Callable[PS, TV_co]` - The function to
        call.

        `*args: Any` - The positional arguments to pass to the function.

        `return_callback: ((AsyncCallStats, TV_co) -> None) | None` - A
        callback to call with the return value of the function every time it is
        called if the function does not raise an exception. Must take a
        AsyncCallStats object as the first argument.

        `except_callback: ((AsyncCallStats, Exception) -> None) | None` - A
        callback to call if any exception raised by the function when it is
        called. Must take a AsyncCallStats object as the first argument.

        `dual_callback: ((AsyncCallStats, TV_co | None, Exception | None)
        -> None) | None` - A callback to call with the return value and
        exception of the function every time it is called. Must take a
        AsyncCallStats object as the first argument.

        `lag_interval: float | None` - The interval between calls to the lag
        callback. If None, the lag callback will not be called.

        `lag_callback: ((AsyncCallStats) -> None) | None` - A callback to call
        when the function takes longer than the lag interval to return. Must
        take a AsyncCallStats object as the first argument.

        `**kwargs: Any` - The keyword arguments to pass to the function.
        """
        with self._atomic_update_lock:
            if isinstance(func, Tickable):
                func = func.tick
            elif not callable(func):
                raise TypeError(f"Item {func!r} of type {type(func)} is "
                                "not tickable or callable.")

            last_time: float
            if self._last_time is None:
                last_time = time.perf_counter()
            else:
                last_time = self._last_time
            next_time: float = last_time + interval
            next_lag_time: float | None = None
            if lag_interval is not None:
                next_lag_time = last_time + lag_interval

            self._items[_TimedClockFutureItem(  # type: ignore[call-overload]
                interval,
                last_time,
                next_time,
                func,
                *args,
                return_callback=return_callback,
                except_callback=except_callback,
                dual_callback=dual_callback,
                lag_interval=lag_interval,
                next_lag_time=next_lag_time,
                lag_callback=lag_callback,
                **kwargs
            )] = None  # type: ignore[assignment]

            self._sleep_time = (  # pylint: disable=assigning-non-slot
                _get_sleep_time(self._items)  # type: ignore[arg-type]
            )

    def unschedule(
        self,
        func: Tickable[Concatenate[CallStats, PS], TV_co] | Callable[
            Concatenate[CallStats, PS], TV_co]
    ) -> None:
        """Unschedule an item from being ticked by the clock."""
        with self._atomic_update_lock:
            super().unschedule(func)

            self._sleep_time = (  # pylint: disable=assigning-non-slot
                _get_sleep_time(self._items)  # type: ignore[arg-type]
            )

    def _call_items(self) -> None:
        """Call the clock's items."""
        current_time: float
        delta_time: float | None = None
        for item, future in self._items.items():  # type: ignore[union-attr]
            current_time = time.monotonic()
            if item.last_time is not None:
                delta_time = current_time - item.last_time
            else:
                delta_time = None
            if future is None or future.done():
                if _less_than_or_close(item.next_time, current_time):
                    self._submit_item(item, delta_time, current_time)
            elif (item.next_lag_time is not None
                  and _less_than_or_close(item.next_lag_time, current_time)):
                item.call_lag_callback(delta_time, current_time)

    def _submit_item(
        self,
        item: _TimedClockFutureItem,
        delta_time: float | None,
        current_time: float
    ) -> None:
        """Submit an item to the threadpool."""
        self._items[item] = (  # type: ignore[call-overload]
            self.__threadpool.submit(
                item,
                delta_time=delta_time,
                current_time=current_time
            )
        )
        next_time = item.last_time + (item.interval * 2.0)
        if item.lag_interval is not None:
            next_lag_time = item.last_time + item.interval + item.lag_interval
        if next_time <= current_time:
            next_time = current_time + item.interval
            if item.lag_interval is not None:
                next_lag_time = current_time + item.lag_interval
        item.next_time = next_time
        if item.lag_interval is not None:
            item.next_lag_time = next_lag_time
        item.last_time = current_time


def __main() -> None:
    """Run clock tests."""
    import argparse  # pylint: disable=import-outside-toplevel
    parser = argparse.ArgumentParser(description="Run clock tests.")
    parser.add_argument(
        "--test-simple-clock",
        action="store_true",
        help="Run simple clock tests."
    )
    parser.add_argument(
        "--test-clock",
        action="store_true",
        help="Run clock tests."
    )
    parser.add_argument(
        "--test-async-clock",
        action="store_true",
        help="Run async clock tests."
    )
    args = parser.parse_args()

    if args.test_simple_clock:
        simple_clock = SimpleClockThread()
        simple_clock.schedule(print, "Hello, world!")
        simple_clock.start()
        time.sleep(2)
        simple_clock.stop()

    if args.test_clock:
        count1 = 0
        count2 = 0
        count3 = 0

        def print1(stats: CallStats, text: str) -> None:
            """Print text."""
            nonlocal count1
            count1 += 1
            print(f"[{stats!s}] {count1}: {text}")

        def print2(stats: CallStats, text: str) -> None:
            """Print text."""
            nonlocal count2
            count2 += 1
            print(f"[{stats!s}] {count2}: {text}")

        def print3(stats: CallStats, text: str) -> None:
            """Print text."""
            nonlocal count3
            count3 += 1
            print(f"[{stats!s}] {count3}: {text}")

        clock = ClockThread()
        clock.schedule(0.2, print1, "Hello, world!")
        clock.schedule(0.3, print2, "Goodbye, world!")
        clock.schedule(0.4, print3, "See you later!")
        clock.start()
        start_time = time.monotonic()
        while time.monotonic() - start_time < 10:
            time.sleep(1)
            print(f"Frequency: {clock.tick_rate:.3f} Hz")
        clock.stop()

    if args.test_async_clock:
        import random

        def mock_request(stats: AsyncCallStats) -> None:
            """Mock a request."""
            print(f"Requesting: {stats!s}...")
            time_ = random.random() * 0.1
            time.sleep(time_)
            print("Request complete.")
        def lag_callback(stats: AsyncCallStats) -> None:
            """Call the lag callback."""
            print(f"Lagging {stats!s}...")

        async_clock = AsyncClockThread()
        async_clock.schedule(
            0.2, mock_request,
            lag_interval=0.3,
            lag_callback=lag_callback
        )
        async_clock.start()
        start_time = time.monotonic()
        while time.monotonic() - start_time < 10:
            time.sleep(1)
            print(f"Frequency: {async_clock.tick_rate:.3f} Hz")
        async_clock.stop()


if __name__ == "__main__":
    __main()
