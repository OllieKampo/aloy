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

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"

__all__ = ("ClockThread",
           "Tickable")

def __dir__() -> tuple[str]:
    """Get the names of module attributes."""
    return __all__

import inspect
import threading
import time
import typing

@typing.runtime_checkable
class Tickable(typing.Protocol):
    """A protocol for tickable objects."""
    
    def tick(self) -> None:
        """Tick the object."""
        ...

@typing.final
class ClockThread:
    """A thread that can be used to run a clock."""

    __slots__ = {"__items" : "The items to tick.",
                 "__atomic_update_lock" : "A lock to ensure that start and stop calls are atomic.",
                 "__sleep_time" : "The time to sleep between ticks.",
                 "__thread" : "The thread that runs the clock.",
                 "__running" : "Event handling whether the clock is running.",
                 "__stopped" : "Event handling whether the clock should stop."}
    
    def __init__(self,
                 *items: Tickable | typing.Callable[[], None],
                 tick_rate: int = 10,
                 check_items: bool = True
                 ) -> None:
        """
        Create a clock thread.
        
        Parameters
        ----------
        `*items : Tickable | Callable[[], None]` - The items to tick.
        Must be either;
            - A tickable object implementing `tick()`, see `jinx.concurrency.clocks.Tickable`,
            - A callable object that takes no parameters.
        
        `tick_rate : int = 10` - The tick rate of the clock in ticks per second.
        This is approximate, the actual tick rate may vary, the only
        guarantee is that the tick rate will not exceed the given value.

        `check_items : bool = True` - Whether to check that all items implement `tick()`.
        """
        ## Scheduled items.
        self.__items: list[typing.Callable[[], None]] = []
        self.schedule(*items, check_items=check_items)

        ## Parameters for run methods.
        self.__atomic_update_lock = threading.Lock()
        self.tick_rate = tick_rate

        ## Variables for the clock thread.
        self.__thread = threading.Thread(target=self.__run)
        self.__thread.daemon = True
        self.__running = threading.Event()
        self.__stopped = threading.Event()
        self.__thread.start()
    
    @property
    def items(self) -> list[Tickable]:
        """Return the items scheduled to be ticked by the clock."""
        return self.__items
    
    def schedule(self, *items: Tickable | typing.Callable[[], None], check_items: bool = True) -> None:
        """Schedule an item to be ticked by the clock."""
        with self.__atomic_update_lock:
            for item in items:
                if isinstance(item, Tickable):
                    self.__items.append(item.tick)
                elif callable(item):
                    self.__items.append(item)
                else:
                    raise TypeError(f"Item {item!r} of type {type(item)} is not tickable or callable.")
                if check_items:
                    item = self.__items[-1]
                    if not (num_params := len(inspect.signature(item).parameters)) == 0:
                        raise ValueError(f"Function {item!r} must take no parameters. Got {num_params}.")
    
    @property
    def tick_rate(self) -> int:
        """Return the tick rate of the clock."""
        return 1.0 / self.__sleep_time
    
    @tick_rate.setter
    def tick_rate(self, value: int) -> None:
        """Set the tick rate of the clock."""
        if value <= 0:
            raise ValueError(f"Tick rate must be greater than 0. Got; {value}.")
        self.__sleep_time = 1.0 / value
    
    def __run(self) -> None:
        """Run the clock."""
        while True:
            self.__running.wait()
            sleep_time: float = self.__sleep_time
            while not self.__stopped.wait(sleep_time):
                start_time: float = time.perf_counter()
                for item in self.__items:
                    item()
                sleep_time = self.__sleep_time - (time.perf_counter() - start_time)
                if sleep_time < 0.0:
                    sleep_time = 0.0
    
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
