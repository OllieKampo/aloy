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

__all__ = ("getitem_zip",
           "all_equal",
           "first_n",
           "iter_to_length",
           "cycle_for",
           "chunk",
           "max_n",
           "find_first",
           "find_all",
           "filter_replace",
           "filter_not_none")

import collections.abc
import itertools
from fractions import Fraction
from numbers import Real
from typing import Callable, Generic, Iterable, Iterator, Optional, Sequence, Type, TypeVar, final, overload

from auxiliary.typingutils import SupportsLenAndGetitem, SupportsRichComparison

VT = TypeVar("VT")
ST = TypeVar("ST", bound=Sequence)
VT_C = TypeVar("VT_C", bound=SupportsRichComparison)

def __extract_args(*args_or_iterable: VT | Iterable[VT]) -> Iterator[VT]:
    if len(args_or_iterable) == 1:
        yield from args_or_iterable[0]
    else: yield from args_or_iterable

@final
class getitem_zip(collections.abc.Sequence, Generic[VT]):
    """A class for a sequence-like zip construct supporting `__len__` and `__getitem__`."""
    
    __slots__ = {"__sequences" : "The zipped sequences.",
                 "__shortest" : "Whether zipping is up to the shortest or longest zipped sequence.",
                 "__len" : "The resulting length of the zip.",
                 "__fill" : "The fill value for missing items of shorter sequences."}
    
    @overload
    def __init__(self,
                 *sequences: SupportsLenAndGetitem[VT],
                 strict: bool = False
                 ) -> None:
        """
        Create a getitem zip.
        
        The given iterables must support `__len__` and `__getitem__`.
        
        The length of the zip will be the length of the shortest iterable.
        
        If `strict` is True, raise an error if the lengths of the iterables are not equal.
        """
        ...
    
    @overload
    def __init__(self,
                 *sequences: SupportsLenAndGetitem[VT],
                 shortest: bool,
                 fill: VT | None = None,
                 strict: bool = False
                 ) -> None:
        """
        Create a getitem zip.
        
        The given iterables must support `__len__` and `__getitem__`.
        
        If `shortest` is True,
        then the length of the zip will be the length of the shortest iterable,
        otherwise it will be the length of the longest.
        If `fill` is not None and `shortest` is False,
        then the `fill` value will be used for missing items of shorter iterables.
        
        If `strict` is True, raise an error if the lengths of the iterables are not equal.
        """
        ...
    
    def __init__(self,
                 *sequences: SupportsLenAndGetitem[VT],
                 shortest: bool = True,
                 fill: VT | None = None,
                 strict: bool = False
                 ) -> None:
        """
        Create a getitem zip.
        
        The given iterables must support `__len__` and `__getitem__`.
        """
        if not all(isinstance(sequence, SupportsLenAndGetitem) for sequence in sequences):
                raise TypeError("All sequences must support __len__ and __getitem__.")
        if strict and not all_equal(len(seq) for seq in sequences):
            raise ValueError("All sequences must have the same length.")
        self.__sequences: tuple[SupportsLenAndGetitem[VT]] = sequences
        self.__shortest: bool = bool(shortest)
        if shortest:
            self.__len: int = min(len(seq) for seq in sequences)
        else:
            self.__len: int = max(len(seq) for seq in sequences)
        self.__fill: VT | None = fill
    
    @property
    def shortest(self) -> bool:
        """Get whether zipping is up to the shortest or longest zipped iterable."""
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
            yield from itertools.zip_longest(*self.__sequences, fillvalue=self.__fill)
    
    def __getitem__(self, index: int) -> tuple[VT | None, ...]:
        """Get a n-length tuple of items, one from each of the n zipped iterables, for the given index."""
        if self.__shortest:
            return tuple(sequence[index] for sequence in self.__sequences)
        return tuple(sequence[index] if index < len(sequence) else self.__fill
                     for sequence in self.__sequences)
    
    def __len__(self) -> int:
        """Get the length of the zip."""
        return self.__len

def all_equal(iterable: Iterable[VT], hint: bool = False) -> bool:
    """
    Check if all items in the iterable are equal.
    
    See: https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
    """
    if hint and isinstance(iterable, Sequence):
        return not bool(iterable) or iterable.count(iterable[0]) == len(iterable)
    groups = itertools.groupby(iterable)
    return next(groups, True) and not next(groups, False)

def first_n(iterable: Iterable[VT], n: int) -> Iterator[VT]:
    """Return the first n elements of an iterable."""
    return itertools.islice(iterable, n)

def iter_to_length(iterable: Iterable[Optional[VT]], length: int, fill: VT) -> Iterator[VT]:
    """
    Yield items from the iterable up to the given length.
    
    If the iterable is exhausted before the given length,
    then instead yield the fill value repeatedly.
    """
    if isinstance(iterable, Sequence):
        if len(iterable) >= length:
            yield from itertools.islice(iterable, length)
        else:
            yield from iterable
            yield from itertools.repeat(fill, length - len(iterable))
    else:
        for index, element in enumerate(iterable):
            yield element
        if index < length:
            yield from itertools.repeat(fill, length - index)

def cycle_for(sequence: Sequence[VT],
              cycles: Real,
              preempt: bool = False
              ) -> Iterator[VT]:
    """
    Cycle through the given sequence for a given number of cycles.
    
    The number of cycles can be any real number, including floating points and fractions.
    
    By default the function checks it the number of cycles has been exceeded after yielding items
    from the sequence. Therefore, if the desired number of cycles is not divisible by the reciprocal
    of the length of the sequence such that `(cycles % (1.0 / len(iterable))) > 0` then; if `preempt`
    if False the iterator will yield `ceil(len(iterables) * cycles)` number of items, and otherwise
    if `preempt` is True it will yield `floor(len(iterables) * cycles)` number of items.
    """
    cycles = Fraction(cycles)
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

def chunk(sequence: Sequence[VT],
          size: int,
          quantity: int,
          as_type: Type[ST] | None
          ) -> Iterator[Sequence[VT]]:
    """
    Yield an iterator over the given quantity of chunks of the given size over the given sequence.
    
    The chunks are returned as sequences of the same type as the argument sequence,
    unless `as_type` is a different type, in which case the chunks are cast to the given type.
    """
    if not isinstance(sequence, Sequence):
        raise TypeError(f"Input must be a sequence. Got; {type(sequence)}.")
    if as_type and type(sequence) is not as_type:
        yield from (as_type(sequence[index : index + size])
                    for index in range(0, quantity * size, size))
    else:
        yield from (sequence[index : index + size]
                    for index in range(0, quantity * size, size))

def ichunk(iterable: Iterable[VT],
           size: int,
           quantity: int) -> Iterator[Iterator[VT]]:
    """
    Yield an iterator over the given quantity of chunks of the given size over the given iterable.
    
    Unlike `chunk`, the chunks are returned as "slice" iterators over the argument iterable.
    
    Any iterable can be chunked, even if it does not support `__len__` and `__getitem__`.
    """
    iterator = iter(iterable)
    for _ in range(quantity):
        ## Advance the iterator to the next chunk and yield
        ## the chunk as an iterator if such a chunk exists.
        if chunk_ := itertools.islice(iterator, size):
            yield chunk_
        else: break

@overload
def max_n(iterable: Iterable[VT_C], *, n: int = 1) -> list[VT_C]:
    """Return the n largest elements of an iterable."""
    ...

@overload
def max_n(iterable: Iterable[VT_C], *, n: int = 1, key: Callable[[VT_C], SupportsRichComparison] | None = None) -> list[VT_C]:
    """Return the n largest elements of an iterable given a key function."""
    ...

@overload
def max_n(*args: VT_C, n: int = 1) -> list[VT_C]:
    """Return the n largest arguments."""
    ...

@overload
def max_n(*args: VT_C, n: int = 1, key: Callable[[VT_C], SupportsRichComparison] | None = None) -> list[VT_C]:
    """Return the n largest arguments given a key function."""
    ...

def max_n(*args_or_iterable: VT_C | Iterable[VT_C], n: int = 1, key: Callable[[VT_C], SupportsRichComparison] | None = None) -> list[VT_C]:
    """Return the n largest elements of an iterable."""
    iterable = __extract_args(*args_or_iterable)
    if n == 1:
        return [max(iterable, key=key)]
    return sorted(iterable, key=key)[:n]

def find_first(iterable: Iterable[VT], condition: Callable[[int, VT], bool]) -> tuple[int, VT]:
    """Find the first element of the iterable where the condition holds true."""
    for index, element in enumerate(iterable):
        if condition(index, element):
            return (index, element)

def find_all(iterable: Iterable[VT],
             condition: Callable[[int, VT], bool],
             limit: Optional[int] = None
             ) -> Iterator[tuple[int, VT]]:
    """Return an iterator over all the elements of the iterable where the condition holds true."""
    for index, element in enumerate(iterable):
        if index == limit:
            break
        if condition(index, element):
            yield (index, element)

def filter_not_none(iterable: Iterable[VT]) -> Iterator[VT]:
    """Return an iterator over the items of the iterable which are not None."""
    return (arg for arg in iterable if arg is not None)

def alternate(*iterables: Iterable[VT]) -> Iterator[VT]:
    """Return an iterator which alternates between yielding items from the given iterables."""
    yield from itertools.chain.from_iterable(itertools.zip_longest(*iterables))
