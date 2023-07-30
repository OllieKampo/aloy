import math
from numbers import Number
from typing import Iterable, TypeVar


NT = TypeVar("NT", bound=Number)


def exp_decay_between(
    value: NT,
    min_: NT,
    max_: NT
) -> NT:
    """Exponentially decay a value between a given range (inclusive)."""
    if value < min_:
        return 0.0
    if value > max_:
        return 1.0
    return (1.0 - (math.log(value - (min_ - 1.0)) / math.log(max_)))


def normalize_between(
    iterable: Iterable[NT],
    lower: NT,
    upper: NT
) -> list[NT]:
    """Normalize an iterable of numbers between a given range (inclusive)."""
    list_: list[NT] = list(iterable)
    min_ = min(list_)
    max_ = max(list_)
    min_max_range = max_ - min_
    lower_upper_range = upper - lower
    return [lower
            + (lower_upper_range
               * ((item - min_)
                  / min_max_range))
            for item in list_]


def normalize_to_sum(
    iterable: Iterable[NT],
    sum_: NT
) -> list[NT]:
    """Normalize an iterable of numbers to sum to a given value."""
    list_: list[NT] = list(iterable)
    list_sum = sum(list_)
    return [item * (sum_ / list_sum) for item in list_]


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
