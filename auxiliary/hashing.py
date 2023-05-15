
"""Module defining functions for hashing objects."""

from functools import lru_cache, reduce
from typing import Hashable


def hash_all(*args: Hashable, **kwargs: Hashable) -> int:
    return reduce(lambda a, b: a ^ hash(b), (*args, *kwargs.values()), 0)


@lru_cache(typed=True)
def cached_hash_all(*args: Hashable, **kwargs: Hashable) -> int:
    return hash_all(lambda a, b: a ^ hash(b), (*args, *kwargs.values()), 0)
