###########################################################################
###########################################################################
## Module defining custom CLI progress bars.                             ##
##                                                                       ##
## Copyright (C)  2022  Oliver Michael Kamperis                          ##
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

"""Module defining custom CLI progress bars."""

from contextlib import contextmanager
import os
import sys
import threading
from time import sleep
from typing import Any, Iterable, Iterator
import psutil
from tqdm import tqdm

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"

__all__ = (
    "ResourceProgressBar",
)


def __dir__() -> tuple[str]:
    """Get the names of module attributes."""
    return __all__


class ResourceProgressBar:
    """
    Class defining a simple tqdm based progress bar which displays current
    memory and CPU usage.

    The usage statistics are updated ten times per second.
    """

    __slots__ = {
        "__process": "Process used for getting resource usage statistics.",
        "__postfix": "Postfix dictionary used for updating the progress bar.",
        "__progress_bar": "The progress bar itself.",
        "__resource_update_interval": "Interval between resource updates.",
        "__running": "A boolean variable used for stopping the update thread.",
        "__cpu_thread": "Thread used for updating resource statistics."
    }

    def __init__(
        self,
        iterable: Iterable[Any] | None = None,
        initial: int | float = 0,
        total: int | float | None = None,
        desc: str | None = None,
        unit: str = "it",
        leave: bool = False,
        ncols: int | None = None,
        dynamic_ncols: bool = False,
        miniters: int = 1,
        colour: str = "cyan",
        resource_update_interval: float = 0.1
    ) -> None:
        """
        Create a resouce usage progress bar.

        See `tqdm.tqdm` for a description of parameters.
        """
        # Process variable used for updating resource usage statistics.
        self.__process = psutil.Process(os.getpid())

        # The progress bar itself.
        self.__postfix = {
            "Mem(Mb)": self.__get_mem(),
            "CPU(%)": self.__get_cpu()
        }
        self.__progress_bar = tqdm(
            iterable=iterable,
            initial=initial,
            total=total,
            desc=desc,
            unit=unit,
            leave=leave,
            ncols=ncols,
            dynamic_ncols=dynamic_ncols,
            miniters=miniters,
            colour=colour,
            postfix=self.__postfix
        )

        # Variables for running an additional thread which
        # updates resource statistics ten times per second.
        self.__resource_update_interval: float = resource_update_interval
        self.__running: bool = True
        self.__cpu_thread = threading.Thread(target=self.__update)
        self.__cpu_thread.daemon = True
        self.__cpu_thread.start()

    def __get_mem(self) -> str:
        """Get current memory usage in megabits."""
        memory = self.__process.memory_info().rss / (1024 ** 2)
        return str(int(memory)).zfill(5)

    def __get_cpu(self) -> str:
        """Get cpu usage in percent."""
        return format(self.__process.cpu_percent(), "0.2f").zfill(6)

    def __update(self) -> None:
        """Target for the update thread."""
        while self.__running:
            self.__postfix["Mem(Mb)"] = self.__get_mem()
            self.__postfix["CPU(%)"] = self.__get_cpu()
            sleep(self.__resource_update_interval)

    def __iter__(self) -> Iterator[Any]:
        """
        Iterate over the progress bar.

        Raises an exception if an iterable was not provided to the upon
        creation of the progress bar.
        """
        for val in self.__progress_bar:
            self.__progress_bar.set_postfix(self.__postfix)
            yield val

    @property
    def n(self) -> int:
        """Get the current progress bar value."""
        return self.__progress_bar.n

    def get_resource_usage(self) -> tuple[float, float]:
        """
        Get the current memory and CPU usage.

        Returns a tuple of the form `(memory: float, cpu: float)`, where
        memory is the current memory usage in megabytes and cpu is the current
        CPU usage in percent.
        """
        return (float(self.__get_mem()), float(self.__get_cpu()))

    def update(
        self,
        n: int = 1, /,
        data: dict[str, str] | None = None
    ) -> None:
        """
        Update the progress bar.

        Parameters
        ----------
        `n: int = 1` - The number of increments ran since the last update.

        `data: {dict[str, str] | None} = None` - An optional dictionary
        of additional statistics to display in the progress bar's postfix,
        given as a mapping of name to value pairs. An empty dictionary will
        clear the postfix, None will leave the postfix unchanged.
        """
        if data is not None:
            self.__progress_bar.set_postfix(data | self.__postfix)
        else:
            self.__progress_bar.set_postfix(self.__postfix)
        self.__progress_bar.update(n)

    def set_description(self, desc: str) -> None:
        """Set the progress bar's description (its prefix)."""
        self.__progress_bar.set_description(desc)

    def write(self, msg: str, /) -> None:
        """
        Write a single message to the console without overlapping the progress
        bar.

        Message will be written above the progress bar.

        This is essentially equal to:
        ```
        bar.clear()
        print("This is a test.")
        bar.refresh()
        ```
        or
        ```
        with bar.external_write_mode():
            print("This is a test.")
        ```
        """
        self.__progress_bar.write(msg)

    @contextmanager
    def external_write_mode(self) -> Iterator[None]:
        """
        Set the progress bar to external write mode.

        This clears the progress bar and allows printing to the console
        without overlapping the progress bar. When the context manager
        exits, the progress bar is re-displayed, below any printed messages.
        """
        try:
            with self.__progress_bar.external_write_mode():
                yield None
        finally:
            pass

    def get_input(self, prompt: str, /) -> str:
        """
        Clear the progress bar, display a prompt, and pause for user input.

        When user input is received, the prompt is cleared and the progress
        bar is re-displayed.
        """
        self.__progress_bar.clear()
        user_input = input(prompt)
        sys.stdout.write("\033[A")
        sys.stdout.flush()
        self.__progress_bar.refresh()
        return user_input

    def reset(self, total: int | float | None = None) -> None:
        """
        Reset the progress bar to 0 iterations for repeated use
        (generally more efficient than creating a new instance).

        Consider combining with `leave=True`.

        If `total` is specified, then use it as the total for the new
        progress bar.
        """
        self.__progress_bar.reset(total)

    def clear(self) -> None:
        """
        Clear the progress bar.

        This will remove the progress bar from the terminal.
        """
        self.__progress_bar.clear()

    def refresh(self) -> None:
        """
        Refresh the progress bar.

        This will update the progress bar's display. If the progress bar
        was previously cleared, this will re-display it.
        """
        self.__progress_bar.refresh()

    def close(self, wait: bool = False) -> None:
        """
        Cleanup and close the progress bar.

        If `wait` is True, block until the progress bar is closed.

        Once closed, the progress bar cannot be re-opened.
        """
        self.__progress_bar.close()
        self.__running = False
        if wait:
            self.__cpu_thread.join()
