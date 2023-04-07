


from collections import deque
import functools
from numbers import Number
from typing import Callable, Generic, TypeVar, final


NT = TypeVar("NT", bound=Number)


@final
class MovingAverage(Generic[NT]):
    """A running average of a stream of numbers."""

    __slots__ = {
        "__window": "The current window size.",
        "__total": "The current total value in the window.",
        "__average": "The current average in the window.",
        "__under_full": "Whether to return 0 if the window is not full.",
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
        initial: NT | None = None,
        track_window: bool = False,
        under_full_zero: bool = True
    ) -> None:
        """
        Create a new moving average.

        The moving average can either; compute an exact moving average by
        tracking all the data in the window, or more rapidly compute an
        approximate moving average by only tracking the total value of the
        window and discarding the data after using it to calculate the
        average. For large a large window size (200+) and many appends, the
        approximate method can be more than 5x faster. However, for cases
        where appended numbers have a large and random variance, the
        approximate method can be extremely inaccurate. In cases where
        append numbers have small variance, or increase or increase
        monotomically, the approximate method is reasonably accurate.

        Parameters
        ----------
        `window: int` - The size of the window.

        `initial: NT | None = None` - The initial value to use for the
        average. If `None`, the average will start at 0.

        `track_window: bool = False` - Whether to track the data in the
        window. If `False`, the data will be discarded after it is used to
        calculate the average. If `True`, the data will be stored in a
        `collections.deque` and can be accessed via the `data` property.

        `under_full_zero: bool = True` - Whether to return 0 if the window
        is not full. If `True`, the average will return 0 if the window is
        not full. If `False`, the average will return the average of the
        data in the window.
        """
        if not isinstance(window, int) or window < 1:
            raise ValueError("Window must be a positive integer. "
                             f"Got; {window} of {type(window)} instead.")
        self.__window: int = window
        if initial is not None and not isinstance(initial, Number):
            raise TypeError("Initial must be a number or None. "
                            f"Got; {initial} of {type(initial)} instead.")
        self.__total: NT = 0 if initial is None else (initial * window)
        self.__average: NT = 0 if initial is None else initial
        self.__under_full: bool = under_full_zero
        self.__data: deque[NT] | None
        self.__stored: int | None
        self.__ratio: float = (window - 1) / window
        self.__append: Callable[[NT], None]

        if track_window:
            self.__data = deque(maxlen=window)
            self.__stored = None
            self.__append = self.__append_track
        else:
            self.__data = None
            self.__stored = 0
            self.__append = self.__append_no_track

    def __str__(self) -> str:
        """Return a string representation of the moving average."""
        return f"{self.__class__.__name__}: total={self.__total}, " \
               f"average={self.__average}, length={len(self)}"

    def __repr__(self) -> str:
        """
        Return an instaniable string representation of the moving
        average.
        """
        return f"{self.__class__.__name__}({self.__window}, {self.__total}, " \
               f"{self.__data is not None}, {self.__under_full})"

    def __len__(self) -> int:
        """Return the number of items in the moving average window."""
        return len(self.__data) if self.__data is not None else self.__stored

    @property
    def window(self) -> int:
        """The current window size."""
        return self.__window

    @property
    def total(self) -> NT:
        """The current total of the window."""
        return self.__total

    @property
    def average(self) -> NT:
        """The current average of the window."""
        return self.__average

    @property
    def data(self) -> deque[NT] | None:
        """The data in the window if tracking is enabled."""
        return self.__data

    def append(self, value: NT) -> None:
        """Append a value to the moving average."""
        self.__append(value)

    @functools.wraps(append)
    def __append_track(self, value: NT) -> None:
        self.__data.append(value)
        self.__total = sum(self.__data)
        if len(self.__data) == self.__window or not self.__under_full:
            self.__average = self.__total / len(self.__data)
        else:
            self.__average = 0

    @functools.wraps(append)
    def __append_no_track(self, value: NT) -> None:
        if self.__stored < self.__window:
            self.__total += value
            self.__stored += 1
            if self.__stored == self.__window or not self.__under_full:
                self.__average = self.__total / self.__stored
            else:
                self.__average = 0
        else:
            self.__total = ((self.__total * self.__ratio) + value)
            self.__average = self.__total / self.__window


@final
class RunningStandardDeviation(Generic[NT]):
    pass


if __name__ == "__main__":
    import random
    import timeit

    def test_1():
        ma_tw = MovingAverage(500, track_window=True, under_full_zero=False)
        for i in random.sample(range(100000), 10000):
            ma_tw.append(i)
            ma_tw.average

    def test_2():
        ma_nt = MovingAverage(500, track_window=False, under_full_zero=False)
        for i in random.sample(range(100000), 10000):
            ma_nt.append(i)
            ma_nt.average

    print(timeit.timeit(test_1, number=1000))
    print(timeit.timeit(test_2, number=1000))
