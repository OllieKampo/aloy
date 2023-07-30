###########################################################################
###########################################################################
## Module defining additional functions for operating on iterables.      ##
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

"""Module defining additional functions for operating on iterables."""

__all__ = (
    "getitem_zip",
    "all_equal",
    "iter_to_len",
    "cycle_to",
    "cycle_for",
    "index_sequence",
    "chunk",
    "ichunk_iterable",
    "ichunk_sequence",
    "max_n",
    "min_n",
    "arg_max",
    "arg_min",
    "arg_max_n",
    "arg_min_n",
    "find_first",
    "find_all",
    "filter_not_none",
    "alternate"
)

import collections.abc
import itertools
from fractions import Fraction
import math
from numbers import Real
from typing import (Callable, Generic, Iterable, Iterator, Optional, Sequence,
                    Type, TypeVar, final, overload)

from aloy.auxiliary.typingutils import SupportsLenAndGetitem, SupportsRichComparison

VT = TypeVar("VT")
ST = TypeVar("ST", bound=Sequence)
VT_C = TypeVar("VT_C", bound=SupportsRichComparison)


def __extract_args(*args_or_iterable: VT | Iterable[VT]) -> Iterator[VT]:
    if len(args_or_iterable) == 1:
        return iter(args_or_iterable[0])  # type: ignore
    else:
        return iter(args_or_iterable)     # type: ignore


@final
class getitem_zip(collections.abc.Sequence, Generic[VT]):
    """
    A class for a sequence-like zip construct.

    Supports `__len__` and `__getitem__`.
    """

    __slots__ = {
        "__sequences": "The zipped sequences.",
        "__shortest": "Whether zipping is up to the shortest "
                      "or longest zipped sequence.",
        "__len": "The resulting length of the zip.",
        "__fill": "The fill value for missing items of shorter sequences."
    }

    @overload
    def __init__(
        self,
        *sequences: SupportsLenAndGetitem[VT],
        strict: bool = False
    ) -> None:
        """
        Create a getitem zip.

        The given iterables must support `__len__` and `__getitem__`.

        The length of the zip will be the length of the shortest iterable.

        If `strict` is True, raise an error if the lengths of the iterables
        are not equal.
        """
        ...

    @overload
    def __init__(
        self,
        *sequences: SupportsLenAndGetitem[VT],
        shortest: bool,
        fill: VT | None = None,
        strict: bool = False
    ) -> None:
        """
        Create a getitem zip.

        The given iterables must support `__len__` and `__getitem__`.

        If `shortest` is True, then the length of the zip will be the length
        of the shortest iterable, otherwise it will be the length of the
        longest. If `fill` is not None and `shortest` is False, then the `fill`
        value will be used for missing items of shorter iterables.

        If `strict` is True, raise an error if the lengths of the iterables
        are not equal.
        """
        ...

    def __init__(
        self,
        *sequences: SupportsLenAndGetitem[VT],
        shortest: bool = True,
        fill: VT | None = None,
        strict: bool = False
    ) -> None:
        """
        Create a getitem zip.

        The given iterables must support `__len__` and `__getitem__`.
        """
        if not all(isinstance(sequence, SupportsLenAndGetitem)
                   for sequence in sequences):
            raise TypeError("Sequences must support __len__ and __getitem__.")
        if strict and not all_equal(len(seq) for seq in sequences):
            raise ValueError("All sequences must have the same length.")
        self.__sequences: tuple[SupportsLenAndGetitem[VT], ...] = sequences
        self.__shortest: bool = bool(shortest)
        self.__len: int
        if shortest:
            self.__len = min(len(seq) for seq in sequences)
        else:
            self.__len = max(len(seq) for seq in sequences)
        self.__fill: VT | None = fill

    @property
    def shortest(self) -> bool:
        """
        Get whether zipping is up to the shortest or longest zipped
        iterable.
        """
        return self.__shortest

    @property
    def fill(self) -> VT | None:
        """Get the fill value."""
        return self.__fill

    def __iter__(self) -> Iterator[tuple[VT | None, ...]]:
        """Get an iterator over the zip."""
        if self.__shortest:
            yield from zip(*self.__sequences)
        else:
            yield from itertools.zip_longest(
                *self.__sequences,
                fillvalue=self.__fill
            )

    def __getitem__(
        self,
        index_or_slice: int | slice
    ) -> tuple[VT | None, ...]:
        """
        Get a n-length tuple of items, one from each of the n zipped
        iterables, for the given index.
        """
        if isinstance(index_or_slice, slice):
            raise TypeError("Cannot slice a getitem_zip.")
        if self.__shortest:
            return tuple(sequence[index_or_slice]
                         for sequence in self.__sequences)
        return tuple(sequence[index_or_slice]
                     if index_or_slice < len(sequence) else self.__fill
                     for sequence in self.__sequences)

    def __len__(self) -> int:
        """Get the length of the zip."""
        return self.__len


def all_equal(
    iterable: Iterable[VT],
    hint: bool = False
) -> bool:
    """
    Check if all items in the iterable are equal.

    See: https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
    """
    if hint and isinstance(iterable, Sequence):
        return (not bool(iterable)
                or iterable.count(iterable[0]) == len(iterable))
    groups = itertools.groupby(iterable)
    return next(groups, True) and not next(groups, False)


def first_n(iterable: Iterable[VT], n: int) -> Iterator[VT]:
    """Return the first n elements of an iterable."""
    return itertools.islice(iterable, n)


def iter_to_len(
    iterable: Iterable[VT],
    len_: int, /,
    fill: VT | Sequence[VT] | None = None
) -> Iterator[VT]:
    """
    Yield items from the iterable up to the given length.

    If the iterable is exhausted before the given length, then; if `fill` is
    not a sequence and not None then repeatedly yield the fill value, otherwise
    if `fill` is a sequence then cycle through it up to the given length.

    To yield a string repeatedly as a fill value, use a one-element tuple.
    """
    if isinstance(iterable, Sequence):
        if len(iterable) >= len_:
            yield from itertools.islice(iterable, len_)
        else:
            yield from iterable
            if isinstance(fill, Sequence):
                yield from cycle_to(fill, len_ - len(iterable))
            else:
                yield from itertools.repeat(fill, len_ - len(iterable))
    else:
        index = 0
        for index, element in enumerate(iterable):
            yield element
        if index < len_:
            if isinstance(fill, Sequence):
                yield from cycle_to(fill, len_ - index)
            else:
                yield from itertools.repeat(fill, len_ - index)


def cycle_to(
    sequence: Sequence[VT],
    len_: int, /
) -> Iterator[VT]:
    """Cycle through the given sequence to a given length."""
    if len_ < 0:
        raise ValueError("Length must be a non-negative integer.")
    if len_ > 0:
        yield from (sequence[i % len(sequence)] for i in range(len_))


def cycle_for(
    sequence: Sequence[VT],
    cycles: Real,
    preempt: bool = False
) -> Iterator[VT]:
    """
    Cycle through the given sequence for a given number of cycles.

    The number of cycles can be any real number, including floating points and
    fractions.

    By default the function checks if the number of cycles has been exceeded
    after yielding items from the sequence. Therefore, if the desired number
    of cycles is not divisible by the reciprocal of the length of the sequence
    such that `(cycles % (1.0 / len(sequence))) > 0` then; if `preempt` is
    False the iterator will yield `ceil(len(sequence) * cycles)` number of
    items, and otherwise if `preempt` is True it will yield
    `floor(len(sequence) * cycles)` number of items.
    """
    if cycles < 0:
        raise ValueError("Cycles must be a non-negative real number.")
    cycles = Fraction(int(cycles))
    if cycles > 0:
        cycled = Fraction(0.0)
        cycles_per_item = Fraction(f"1/{len(sequence)}")
        i: int = 0
        while ((not preempt and cycled < cycles)
               or (cycled + cycles_per_item) < cycles):
            i %= len(sequence)
            yield sequence[i]
            i += 1
            cycled += cycles_per_item


def index_sequence(
    sequence: Sequence[VT],
    indices: Iterable[int]
) -> Iterator[VT]:
    """Get an iterator over the sequence at the given indices."""
    yield from (sequence[i] for i in indices)


# TODO: Need:
#   - divide: divide a sequence into n equal parts
#   - split: split a sequence into parts of given sizes


# TODO: Not much difference between chunk and ichunk_sequence.
def chunk(
    sequence: SupportsLenAndGetitem[VT],
    size: int,
    quantity: int | None = None,
    as_type: Type[ST] | None = None
) -> Iterator[Sequence[VT]]:
    """
    Yield an iterator over the size and quantity of chunks of the sequence.

    If `as_type` is None (the default), then the chunks are returned as
    sequence slices of the same type as the argument sequence. If `as_type`
    is a type object, the chunks are cast to the given type before yielding.
    """
    if quantity is not None:
        if size * quantity > len(sequence):
            raise ValueError("Size and quantity are too large. "
                             f"Got; size={size}, quantity={quantity}, "
                             f"sequence length={len(sequence)}.")
    else:
        quantity = len(sequence) // size
    if as_type is not None and type(sequence) is not as_type:
        yield from (as_type(sequence[index:index + size])  # type: ignore
                    for index in range(0, quantity * size, size))
    else:
        yield from (sequence[index:index + size]
                    for index in range(0, quantity * size, size))


@overload
def ichunk_iterable(
    iterable: Iterable[VT],
    size: int, /
) -> Iterator[Iterator[VT]]:
    ...


@overload
def ichunk_iterable(
    iterable: Iterable[VT],
    size: int, /, *,
    quantity: int | None
) -> Iterator[Iterator[VT]]:
    ...


@overload
def ichunk_iterable(
    iterable: Iterable[VT],
    size: int, /, *,
    infinite: bool
) -> Iterator[Iterator[VT]]:
    ...


def ichunk_iterable(
    iterable: Iterable[VT],
    size: int, /, *,
    quantity: int | None = None,
    infinite: bool = False
) -> Iterator[Iterator[VT]]:
    """
    Yield an iterator of chunk of an iterable of a given size and quantity.

    Any iterable can be chunked, even if it does not support `__len__` and
    `__getitem__`.

    Unlike `chunk`, the chunks are returned as "slice" iterators over the
    argument iterable. If `quantity` is not None, then only at most `quantity`
    chunks are yielded. Otherwise, if `quantity` is None, then yield chunks
    until the iterable is exhausted. If `infinite` is True, then yield chunks
    indefinitely, achieving a performance benefit over `quantity=None` for
    infinite iterators. Note that if the iterable is not an infinite iterator,
    then when the iterable is exhausted, empty chunks will be yielded
    indefinitely.

    Example Usage
    -------------
    ```
    >>> from itertools import count
    >>> from more_itertools import ichunk_iterable

    # Chunk an iterable into chunks of size 3, yielding at most 2 chunks.
    >>> for chunk in ichunk_iterable(count(), 3, 2):
    ...     print(list(chunk))
    [0, 1, 2]
    [3, 4, 5]

    # Chunk an iterable into chunks of size 3, yielding chunks indefinitely.
    >>> for chunk in ichunk_iterable(count(), 3, infinite=True):
    ...     print(list(chunk))
    [0, 1, 2]
    [3, 4, 5]
    [6, 7, 8]
    # Continues infinitely...
    ```
    """
    iterator = iter(iterable)
    # Advance the iterator to the next chunk and yield
    # the chunk as an iterator if such a chunk exists.
    if infinite:
        while True:
            yield itertools.islice(iterator, size)
    elif quantity is None:
        while True:
            chunk_ = itertools.islice(iterator, size)
            try:
                first = next(chunk_)
            except StopIteration:
                break
            yield itertools.chain((first,), chunk_)
    else:
        for _ in range(quantity):
            chunk_ = itertools.islice(iterator, size)
            try:
                first = next(chunk_)
            except StopIteration:
                break
            yield itertools.chain((first,), chunk_)


def ichunk_sequence(
    sequence: Sequence[VT],
    size: int,
    quantity: int | None = None
) -> Iterator[Sequence[VT]]:
    """
    Yield an iterator of chunks of a sequence of a given size and quantity.

    Unlike `chunk`, the chunks are returned as "slice" iterators over the
    argument sequence. If `quantity` is not None, then only at most `quantity`
    chunks are yielded. Otherwise, if `quantity` is None, then yield chunks
    until the sequence is exhausted. If the length of the sequence is not
    divisible by `size`, then the last chunk will be shorter than `size`.

    Example Usage
    -------------
    ```
    >>> from more_itertools import ichunk_sequence

    ## Chunk a sequence into chunks of size 3, yielding at most 2 chunks.
    >>> for chunk in ichunk_sequence(range(11), 3, 2):
    ...     print(list(chunk))
    [0, 1, 2]
    [3, 4, 5]

    ## Chunk a sequence into chunks of size 3, yielding chunks until the
    ## sequence is exhausted.
    >>> for chunk in ichunk_sequence(range(11), 3, None):
    ...     print(list(chunk))
    [0, 1, 2]
    [3, 4, 5]
    [6, 7, 8]
    [9, 10]
    ```
    """
    if not isinstance(sequence, Sequence):
        raise TypeError(f"Input must be a sequence. Got; {type(sequence)}.")
    if quantity is not None:
        if size * quantity > len(sequence):
            raise ValueError("Size and quantity are too large. "
                             f"Got; size={size}, quantity={quantity}, "
                             f"sequence length={len(sequence)}.")
    else:
        quantity = math.ceil(len(sequence) / size)
    yield from (sequence[index:index + size]
                for index in range(0, quantity * size, size))


@overload
def max_n(
    iterable: Iterable[VT_C], /, *,
    n: int = 1
) -> list[VT_C]:
    """Return the n largest elements of an iterable."""
    ...


@overload
def max_n(
    iterable: Iterable[VT_C], /, *,
    n: int = 1,
    key: Callable[[VT_C], SupportsRichComparison] | None = None
) -> list[VT_C]:
    """Return the n largest elements of an iterable given a key function."""
    ...


@overload
def max_n(
    *args: VT_C,
    n: int = 1
) -> list[VT_C]:
    """Return the n largest arguments."""
    ...


@overload
def max_n(
    *args: VT_C,
    n: int = 1,
    key: Callable[[VT_C], SupportsRichComparison] | None = None
) -> list[VT_C]:
    """Return the n largest arguments given a key function."""
    ...


def max_n(
    *args_or_iterable: VT_C | Iterable[VT_C],
    n: int = 1,
    key: Callable[[VT_C], SupportsRichComparison] | None = None
) -> list[VT_C]:
    """Return the n largest elements of an iterable."""
    iterable = __extract_args(*args_or_iterable)
    if n < 0:
        raise ValueError("n must be a non-negative integer.")
    if n == 0:
        return []
    if n == 1:
        return [max(iterable, key=key)]
    return sorted(iterable, key=key)[-n:]


@overload
def min_n(
    iterable: Iterable[VT_C], /, *,
    n: int = 1
) -> list[VT_C]:
    """Return the n smallest elements of an iterable."""
    ...


@overload
def min_n(
    iterable: Iterable[VT_C], /, *,
    n: int = 1,
    key: Callable[[VT_C], SupportsRichComparison] | None = None
) -> list[VT_C]:
    """Return the n smallest elements of an iterable given a key function."""
    ...


@overload
def min_n(
    *args: VT_C,
    n: int = 1
) -> list[VT_C]:
    """Return the n smallest arguments."""
    ...


@overload
def min_n(
    *args: VT_C,
    n: int = 1,
    key: Callable[[VT_C], SupportsRichComparison] | None = None
) -> list[VT_C]:
    """Return the n smallest arguments given a key function."""
    ...


def min_n(
    *args_or_iterable: VT_C | Iterable[VT_C],
    n: int = 1,
    key: Callable[[VT_C], SupportsRichComparison] | None = None
) -> list[VT_C]:
    """Return the n smallest elements of an iterable."""
    iterable = __extract_args(*args_or_iterable)
    if n < 0:
        raise ValueError("n must be a non-negative integer.")
    if n == 0:
        return []
    if n == 1:
        return [min(iterable, key=key)]
    return sorted(iterable, key=key)[:n]


def arg_max(
    iterable: Iterable[VT_C], /,
    key: Callable[[VT_C], SupportsRichComparison] | None = None,
    default: int | None = None
) -> int:
    """Return the index of the largest element of an iterable."""
    if key is None:
        return max(
            enumerate(iterable),
            key=lambda i: i[1],
            default=(0, default)
        )[0]
    return max(
        enumerate(iterable),
        key=lambda i: key(i[1]),
        default=(0, default)
    )[0]


def arg_min(
    iterable: Iterable[VT_C], /,
    key: Callable[[VT_C], SupportsRichComparison] | None = None,
    default: int | None = None
) -> int:
    """Return the index of the smallest element of an iterable."""
    if key is None:
        return min(
            enumerate(iterable),
            key=lambda i: i[1],
            default=(0, default)
        )[0]
    return min(
        enumerate(iterable),
        key=lambda i: key(i[1]),
        default=(0, default)
    )[0]


def arg_max_n(
    iterable: Iterable[VT_C], /,
    n: int = 1,
    key: Callable[[VT_C], SupportsRichComparison] | None = None
) -> list[int]:
    """Return the indices of the n largest elements of an iterable."""
    if n < 0:
        raise ValueError("n must be a non-negative integer.")
    if n == 0:
        return []
    if n == 1:
        return [arg_max(iterable, key=key)]
    if key is None:
        return [
            index for index, _ in
            sorted(enumerate(iterable), key=lambda i: i[1])[-n:]
        ]
    return [
        index for index, _ in
        sorted(enumerate(iterable), key=lambda i: key(i[1]))[-n:]
    ]


def arg_min_n(
    iterable: Iterable[VT_C], /,
    n: int = 1,
    key: Callable[[VT_C], SupportsRichComparison] | None = None
) -> list[int]:
    """Return the indices of the n smallest elements of an iterable."""
    if n < 0:
        raise ValueError("n must be a non-negative integer.")
    if n == 0:
        return []
    if n == 1:
        return [arg_min(iterable, key=key)]
    if key is None:
        return [
            index for index, _ in
            sorted(enumerate(iterable), key=lambda i: i[1])[:n]
        ]
    return [
        index for index, _ in
        sorted(enumerate(iterable), key=lambda i: key(i[1]))[:n]
    ]


def find_first(
    iterable: Iterable[VT],
    condition: Callable[[int, VT], bool]
) -> tuple[int, VT]:
    """
    Find the first element of the iterable where the condition holds true.
    """
    for index, element in enumerate(iterable):
        if condition(index, element):
            return (index, element)
    raise ValueError("No element found.")


def find_all(iterable: Iterable[VT],
             condition: Callable[[int, VT], bool],
             limit: Optional[int] = None
             ) -> Iterator[tuple[int, VT]]:
    """
    Return an iterator over all the elements of the iterable where the
    condition holds true.
    """
    for index, element in enumerate(iterable):
        if index == limit:
            break
        if condition(index, element):
            yield (index, element)


def filter_not_none(iterable: Iterable[VT]) -> Iterator[VT]:
    """Return an iterator over the items of the iterable which are not None."""
    return (arg for arg in iterable if arg is not None)


def alternate(*iterables: Iterable[VT]) -> Iterator[VT]:
    """
    Return an iterator which alternates between yielding items from the
    given iterables.
    """
    yield from itertools.chain.from_iterable(itertools.zip_longest(*iterables))
