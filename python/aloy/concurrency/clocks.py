###########################################################################
###########################################################################
## Module containing functions and classes for thread clocks.            ##
##                                                                       ##
## Copyright (C) 2023 Oliver Michael Kamperis                            ##
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

"""Module containing functions and classes for thread clocks."""

import inspect
import threading
import time
from typing import Callable, Final, Protocol, final, runtime_checkable
import warnings

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"

__all__ = (
    "ClockThread",
    "Tickable"
)


def __dir__() -> tuple[str, ...]:
    """Get the names of module attributes."""
    return __all__


@runtime_checkable
class Tickable(Protocol):
    """A protocol for tickable objects."""

    def tick(self) -> None:
        """Tick the object."""
        ...


@final
class ClockThread:
    """
    Class defining threads used to run a clock for
    regularly calling functions at a given tick rate.
    """

    __DELAYED_TICKS_WARNING: Final[int] = 100
    __DELAYED_TICKS_RESET: Final[int] = 10

    __slots__ = {
        "__items": "The items to tick.",
        "__atomic_update_lock": "A lock making start and stop calls atomic.",
        "__sleep_time": "The time to sleep between ticks.",
        "__thread": "The thread that runs the clock.",
        "__running": "Event handling whether the clock is running.",
        "__stopped": "Event handling whether the clock should stop."
    }

    def __init__(
        self,
        *items: Tickable | Callable[[], None],
        tick_rate: int = 10,
        check_items: bool = True
    ) -> None:
        """
        Create a new clock thread with the given tickable items.

        Parameters
        ----------
        `*items: Tickable | Callable[[], None]` - The items to tick.
        Must be either;
            - A tickable object implementing `tick()`, see the protocol
              `aloy.concurrency.clocks.Tickable`,
            - A callable object that takes no parameters.

        `tick_rate: int = 10` - The tick rate of the clock (ticks/second).
        This is approximate, the actual tick rate may vary, the only
        guarantee is that the tick rate will not exceed the given value.

        `check_items: bool = True` - Whether to check that items either
        implement `tick()` or are callable with no parameters. If they
        do not, a `TypeError` will be raised.
        """
        # Schedule items.
        self.__atomic_update_lock = threading.Lock()
        self.__items: list[Callable[[], None]] = []
        self.schedule(*items, check_items=check_items)
        self.tick_rate = tick_rate

        # Variables for the clock thread.
        self.__thread = threading.Thread(target=self.__run)
        self.__thread.daemon = True
        self.__running = threading.Event()
        self.__stopped = threading.Event()
        self.__thread.start()

    def __str__(self) -> str:
        """Return a string representation of the clock thread."""
        return (f"ClockThread: with {len(self.__items)} items "
                f"at tick rate {self.tick_rate} ticks/second.")

    @property
    def items(self) -> list[Callable[[], None]]:
        """Return the items scheduled to be ticked by the clock."""
        return self.__items

    def schedule(
        self,
        *items: Tickable | Callable[[], None],
        check_items: bool = True
    ) -> None:
        """Schedule an item to be ticked by the clock."""
        with self.__atomic_update_lock:
            for item in items:
                if isinstance(item, Tickable):
                    self.__items.append(item.tick)
                elif callable(item):
                    self.__items.append(item)
                else:
                    raise TypeError(f"Item {item!r} of type {type(item)} is "
                                    "not tickable or callable.")
                if check_items:
                    item = self.__items[-1]
                    params = inspect.signature(item).parameters
                    if not (num_params := len(params)) == 0:
                        raise ValueError(f"Function {item!r} must take no "
                                         f"parameters. Got {num_params}.")

    def unschedule(self, *items: Tickable | Callable[[], None]) -> None:
        """Unschedule an item from being ticked by the clock."""
        with self.__atomic_update_lock:
            for item in items:
                if isinstance(item, Tickable):
                    item = item.tick
                self.__items.remove(item)

    @property
    def tick_rate(self) -> int:
        """Return the tick rate of the clock."""
        return int(1.0 / self.__sleep_time)

    @tick_rate.setter
    def tick_rate(self, value: int) -> None:
        """Set the tick rate of the clock."""
        if value <= 0:
            raise ValueError("Tick rate must be greater than 0. "
                             f"Got; {value}.")
        self.__sleep_time = 1.0 / value

    def __run(self) -> None:
        """Run the clock."""
        while True:
            self.__running.wait()

            sleep_time: float = self.__sleep_time
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
                with self.__atomic_update_lock:
                    for item in self.__items:
                        item()
                update_time = time.perf_counter() - start_update_time

                if update_time > actual_sleep_time:
                    sleep_time = 0.0
                    warnings.warn(
                        f"[{self!s}] Tick rate too high for scheduled items. "
                        "Sleep time longer than update time. Actual sleep "
                        f"time = {actual_sleep_time:.3f} seconds, items tick "
                        f"time = {update_time:.3f} seconds. Setting sleep "
                        "time to 0.0 seconds."
                    )
                else:
                    # Adjust sleep time to account for update time.
                    sleep_time = ((sleep_time + self.__sleep_time)
                                  - (actual_sleep_time + update_time))

                start_sleep_time = time.perf_counter()

    def start(self) -> None:
        """Start the clock."""
        with self.__atomic_update_lock:
            if not self.__running.is_set():
                self.__stopped.clear()
                self.__running.set()

    def stop(self) -> None:
        """Stop the clock."""
        with self.__atomic_update_lock:
            if self.__running.is_set():
                self.__stopped.set()
                self.__running.wait()
