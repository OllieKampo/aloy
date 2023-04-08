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
