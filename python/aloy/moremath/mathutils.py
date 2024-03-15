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

"""Module defining various math utility functions."""

import math
from typing import Iterable, TypeVar

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"
__version__ = "0.0.1"

__all__ = (
    "exp_decay_between",
    "normalize_between",
    "normalize_to_sum",
    "closest_integer_factors",
    "closest_squares"
)


def __dir__() -> tuple[str, ...]:
    """Get the names of module attributes."""
    return __all__


_NT = TypeVar("_NT", float, int)


def exp_decay_between(
    value: _NT,
    min_: _NT,
    max_: _NT
) -> float:
    """Exponentially decay a value between a given range (inclusive)."""
    if value < min_:
        return 0.0
    if value > max_:
        return 1.0
    return (1.0 - (math.log(value - (min_ - 1.0)) / math.log(max_)))


def normalize_between(
    iterable: Iterable[_NT],
    lower: _NT,
    upper: _NT
) -> list[float]:
    """Normalize an iterable of numbers between a given range (inclusive)."""
    list_: list[_NT] = (
        list(iterable)
        if not isinstance(iterable, list)
        else iterable
    )
    min_: float = min(list_, default=0.0)
    max_: float = max(list_, default=1.0)
    min_max_range: float = max_ - min_
    lower_upper_range: float = upper - lower
    return [
        lower
        + (lower_upper_range
           * ((item - min_)
               / min_max_range))
        for item in list_
    ]


def normalize_to_sum(
    iterable: Iterable[_NT],
    sum_: _NT
) -> list[float]:
    """Normalize an iterable of numbers to sum to a given value."""
    list_: list[_NT] = (
        list(iterable)
        if not isinstance(iterable, list)
        else iterable
    )
    list_sum: float = sum(list_, 0.0)
    factor: float = sum_ / list_sum
    return [item * factor for item in list_]


def closest_integer_factors(value: int) -> tuple[int, int]:
    """
    Find the closest two integer factors of a given integer. Such that
    `x_factor * y_factor == value` and `x_factor <= y_factor`.

    See:
    https://stackoverflow.com/questions/16266931/input-an-integer-find-the-two-closest-integers-which-when-multiplied-equal-th
    """
    x_factor = math.floor(math.sqrt(value))
    while (value % x_factor) != 0:
        x_factor -= 1
    y_factor = value // x_factor
    return (x_factor, y_factor)


def closest_squares(value: int) -> tuple[int, int]:
    """Find the closest square numbers to a given integer."""
    sqrt = math.sqrt(value)
    if sqrt.is_integer():
        return (int(sqrt),) * 2
    return (math.floor(sqrt) ** 2, math.ceil(sqrt) ** 2)


def truncate(value: float, precision: int = 0) -> float:
    """
    Truncate a number to a given precision.

    This simply removes digits after the decimal point, such that the
    remaining number of digits is equal to the given precision.

    Parameters:
    -----------
    `value: float` - The number to truncate.

    `precision: int` - The number of decimal places to truncate to.

    Returns:
    --------
    `float` - The truncated number.

    Examples:
    ---------
    ```python
    >>> truncate(1.234567)
    1.0
    ```

    ```python
    >>> truncate(1.234567, 2)
    1.23
    ```

    ```python
    >>> truncate(1.234567, 4)
    1.2345
    ```
    """
    factor: float = 10 ** precision
    return int(value * factor) / factor


def round_up(value: float, precision: int = 0) -> float:
    """
    Round a number up to a given precision.

    This removes digits after the decimal point and rounds up the trailing
    digits, such that the remaining number of digits is equal to the given
    precision.

    Parameters:
    -----------
    `value: float` - The number to round up.

    `precision: int` - The number of decimal places to round to.

    Returns:
    --------
    `float` - The rounded number.

    Examples:
    ---------
    ```python
    >>> round_up(1.234567)
    2.0
    ```

    ```python
    >>> round_up(1.234567, 2)
    1.24
    ```

    ```python
    >>> round_up(1.234567, 4)
    1.2346
    ```
    """
    factor: float = 10 ** precision
    return math.ceil(value * factor) / factor


def round_down(value: float, precision: int = 0) -> float:
    """
    Round a number down to a given precision.

    This removes digits after the decimal point and rounds down the trailing
    digits, such that the remaining number of digits is equal to the given
    precision.

    Parameters:
    -----------
    `value: float` - The number to round down.

    `precision: int` - The number of decimal places to round to.

    Returns:
    --------
    `float` - The rounded number.

    Examples:
    ---------
    ```python
    >>> round_down(1.234567)
    1.0
    ```

    ```python
    >>> round_down(1.234567, 2)
    1.23
    ```

    ```python
    >>> round_down(1.234567, 4)
    1.2345
    ```
    """
    factor: float = 10 ** precision
    return math.floor(value * factor) / factor
