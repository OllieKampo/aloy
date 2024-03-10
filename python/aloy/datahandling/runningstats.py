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

"""Module containing classes for calculating running statistics."""

from collections import deque
import functools
from numbers import Number
from typing import Callable, Generic, TypeVar, final

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"
__version__ = "0.1.0"

__all__ = (
    "MovingAverage",
    "ExponentialMovingAverage",
    "RunningStandardDeviation"
)


def __dir__() -> tuple[str, ...]:
    """Get the names of module attributes."""
    return __all__


NT = TypeVar("NT", int, float)


@final
class MovingAverage:
    """A running average of a stream of numbers."""

    __slots__ = {
        "__window": "The current window size.",
        "__total": "The current total value in the window.",
        "__average": "The current average in the window.",
        "__under_full_initial": "Whether to return the initial value if the "
                                "window is not full.",
        "__data": "The data in the window if tracking is enabled.",
        "__stored": "The number of items stored in the window if tracking is "
                    "disabled.",
        "__ratio": "The ratio of the current total value to be kept in the "
                   "when calculating the new total if tracking is disabled.",
        "__append": "The append method to use."
    }

    def __init__(
        self,
        window: int,
        initial: float | None = None,
        track_window: bool = True,
        fast_track: bool = False,
        under_full_initial: bool = True
    ) -> None:
        """
        Create a new moving average.

        The moving average can either; compute an exact moving average by
        tracking all the data in the window, or more rapidly compute an
        approximate moving average by only tracking the total value of the
        window and discarding the data after using it to calculate the average.
        For a very large window size (at least 200) and many appends, the
        approximate method can be many times faster than the exact. However,
        for cases where appended numbers have a large and random variance, the
        approximate method can be extremely inaccurate. In cases where appended
        numbers have small variance, or increase or decrease monotonically,
        the approximate method may be accurate enough for the trade off to
        be worth it.

        Parameters
        ----------
        `window: int` - The size of the window.

        `initial: float | None = None` - The initial value to use for the
        average. If `None`, the average will start at 0.

        `track_window: bool = True` - If `True`, the data will be 'tracked'
        such that it is stored in a deque and can be accessed via the `data`
        property. If `False`, the data will is not tracked, and the average
        is calclulated using the approximate method.

        `fast_track: bool = False` - Whether to use the fast track method.
        If `track_window` is `True`; then if `fast_track` is `True`, instead
        of summing the data in the window, the total oldest value is
        substracted from the currrent running total and the new value is
        added (this is faster, but can become inaccurate over a large number
        of appends), otherwise if `False`, the data in the window will be
        summed each time a new value is appended.

        `under_full_initial: bool = True` - If `True`, the average will return
        the initial value if the window is not full. If `False`, the average
        will return the average of the data in the window.
        """
        if not isinstance(window, int) or window < 1:
            raise ValueError(
                "Window must be a positive integer. "
                f"Got; {window} of {type(window)} instead."
            )
        self.__window: int = window
        if initial is not None and not isinstance(initial, Number):
            raise TypeError(
                "Initial must be an integer, float, or None. "
                f"Got; {initial} of {type(initial)} instead."
            )

        self.__total: float = 0.0 if initial is None else (initial * window)
        self.__average: float = 0.0 if initial is None else initial
        self.__under_full_initial: bool = under_full_initial
        self.__data: deque[float] | None
        self.__stored: int = 0
        self.__ratio: float = (window - 1) / window
        self.__append: Callable[[float], None]

        if track_window:
            self.__data = deque(maxlen=window)
            if fast_track:
                self.__append = self.__append_track_fast
            else:
                self.__append = self.__append_track
        else:
            self.__data = None
            self.__append = self.__append_no_track

    def __str__(self) -> str:
        """Return a string representation of the moving average."""
        return (
            f"{self.__class__.__name__}: total={self.__total}, "
            f"average={self.__average}, length={len(self)}"
        )

    def __repr__(self) -> str:
        """
        Return an instaniable string representation of the moving average.
        """
        return f"{self.__class__.__name__}({self.__window}, {self.__total}, " \
               f"{self.__data is not None}, {self.__under_full_initial})"

    def __len__(self) -> int:
        """Return the number of items in the moving average window."""
        if self.__data is not None:
            return len(self.__data)
        else:
            return self.__stored  # type: ignore[return-value]

    @property
    def window(self) -> int:
        """The current window size."""
        return self.__window

    @property
    def total(self) -> float:
        """The current total of the window."""
        return self.__total

    @property
    def average(self) -> float:
        """The current average of the window."""
        return self.__average

    @property
    def data(self) -> deque[float] | None:
        """The data in the window if tracking is enabled."""
        return self.__data

    def append(self, value: float) -> None:
        """Append a value to the moving average."""
        self.__append(value)

    @functools.wraps(append)
    def __append_track(self, value: float) -> None:
        self.__data.append(value)  # type: ignore[union-attr]
        self.__total = sum(self.__data)  # type: ignore[arg-type]
        if (len(self.__data) == self.__window  # type: ignore[arg-type]
                or not self.__under_full_initial):
            self.__average = \
                self.__total / len(self.__data)  # type: ignore[arg-type]

    @functools.wraps(append)
    def __append_track_fast(self, value: float) -> None:
        if len(self.__data) == self.__window:  # type: ignore[arg-type]
            self.__total -= self.__data[0]  # type: ignore[index]
        self.__data.append(value)  # type: ignore[union-attr]
        self.__total += value
        if (len(self.__data) == self.__window  # type: ignore[arg-type]
                or not self.__under_full_initial):
            self.__average = \
                self.__total / len(self.__data)  # type: ignore[arg-type]

    @functools.wraps(append)
    def __append_no_track(self, value: float) -> None:
        if self.__stored < self.__window:
            self.__total += value
            self.__stored += 1
            if self.__stored == self.__window or not self.__under_full_initial:
                self.__average = self.__total / self.__stored
        else:
            self.__total = ((self.__total * self.__ratio) + value)
            self.__average = self.__total / self.__window


@final
class ExponentialMovingAverage:
    """
    Exponential moving average: smoothing to give progressively lower
    weights to older values.
    """

    __slots__ = {
        "__smoothing": "The smoothing factor.",
        "__total": "The current total of the moving average.",
        "__appends": "The number of appends to the moving average."
    }

    def __init__(
        self,
        initial_value: float = 0.0,
        smoothing: float = 0.3
    ) -> None:
        """
        Create a new exponential moving average.

        Parameters
        ----------
        `initial_value: float = 0.0` - The initial value of the moving average.

        `smoothing: float = 0.3` - Smoothing factor for the moving average.
        Must be in the range [0.0, 1.0]. Higher values give more weight to
        recent values and lower values give more weight to older values.
        """
        if not 0.0 <= smoothing <= 1.0:
            raise ValueError(
                "Smoothing must be in the range [0.0, 1.0]. "
                f"Got; {smoothing} instead."
            )
        self.__smoothing: float = smoothing
        self.__total: float = initial_value
        self.__appends: int = 0

    def __repr__(self) -> str:
        """
        Return an instaniable string representation of the moving average.
        """
        return f"{self.__class__.__name__}({self.average}, {self.__smoothing})"

    def __str__(self) -> str:
        """
        Return a human readable string representation of the moving average.
        """
        return f"{self.__class__.__name__}: total={self.__total}, " \
               f"average={self.average}, appends={self.__appends}"

    @property
    def smoothing(self) -> float:
        """The smoothing factor of the moving average."""
        return self.__smoothing

    @property
    def appends(self) -> int:
        """The number of appends to the moving average."""
        return self.__appends

    @property
    def average(self) -> float:
        """The current value of the moving average."""
        if self.__appends:
            return (
                self.__total
                / (1.0 - ((1.0 - self.__smoothing)
                          ** self.__appends)))
        return self.__total

    def add_value(self, value: float) -> float:
        """
        Add a new value to the moving average and return the new average.

        Parameters
        ----------
        `value: float` - New value to add to the moving average.
        """
        self.__total = (
            (self.__smoothing * value)
            + ((1.0 - self.__smoothing) * self.__total)
        )
        self.__appends += 1
        return self.average


@final
class RunningStandardDeviation(Generic[NT]):
    pass


if __name__ == "__main__":
    import random
    import timeit

    def test_1():
        ma_tw = MovingAverage(500, track_window=True, under_full_initial=False)
        for i in random.sample(range(100000), 10000):
            ma_tw.append(i)
            ma_tw.average

    def test_2():
        ma_nt = MovingAverage(500, track_window=False, under_full_initial=False)
        for i in random.sample(range(100000), 10000):
            ma_nt.append(i)
            ma_nt.average

    print(timeit.timeit(test_1, number=1000))
    print(timeit.timeit(test_2, number=1000))
