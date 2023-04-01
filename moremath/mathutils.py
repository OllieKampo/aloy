from numbers import Number
from typing import Iterable, TypeVar


NT = TypeVar("NT", bound=Number)


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
