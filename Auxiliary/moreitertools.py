###########################################################################
###########################################################################
## Additional iteration functions and algorithms.                        ##
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

"""Additional iteration functions and algorithms."""

import itertools
from typing import Callable, Iterable, Iterator, Optional, Sequence, TypeVar

from fractions import Fraction
from numbers import Real

from auxiliary.typing import SupportsLenAndGetitem, SupportsRichComparison

VT = TypeVar("VT")
VT_C = TypeVar("VT_C", bound=SupportsRichComparison)

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
    if False the iterator will yield `roundup(len(iterables) * cycles)` number of items, and otherwise
    if `preempt` is True it will yield `rounddown(len(iterables) * cycles)` number of items.
    """
    cycles = Fraction(cycles)
    if cycles > 0:
        cycled = Fraction(0.0)
        cycles_per_item = Fraction(f"1/{len(sequence)}")
        i: int = 0
        while ((not preempt
                and cycled < cycles)
               or (cycled + cycles_per_item) < cycles):
            i %= len(sequence)
            yield sequence[i]
            cycled += cycles_per_item
            i += 1

class getitem_zip(Sequence[VT]):
    """A class for a sequence-line zip construct supporting `__len__` and `__getitem__`."""
    
    __slots__ = ("__sequences",
                 "__len")
    
    def __init__(self, *sequences: SupportsLenAndGetitem[VT]) -> None:
        """Create a getitem zip, this requires the given iterables to support `getitem` and `len`."""
        self.__sequences = sequences
        self.__len = min(map(len, sequences))
    
    def __getitem__(self, index: int) -> tuple[VT, ...]:
        """Get a tuple of items, one from each iterable, for the given index."""
        return (sequence[index] for sequence in self.__sequences)
    
    def __len__(self) -> int:
        """Get the length of the shortest iterable."""
        return self.__len

def get_first_n(iterable: Iterable[VT], n: int) -> Sequence[VT]:
    """Return the first n elements of an iterable."""
    return itertools.islice(iterable, n)

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

def filter_replace(iterable: Iterable[VT],
                   replacement: Callable[[int, VT], VT]
                   ) -> Iterator[VT]:
    """Return an iterator over the items of the iterable, where each item is replaced by the result of the replacement function."""
    yield from (replacement(index, element) for index, element in enumerate(iterable))

def filter_not_none(iterable: Iterable[VT]) -> Iterator[VT]:
    """Return an iterator over the items of the iterable which are not None."""
    return (arg for arg in iterable if arg is not None)

def iterate_to_length(iterable: Iterable[Optional[VT]], length: int, fill: VT) -> Iterator[VT]:
    """
    Yield items from the iterable up to the given length.
    
    If the fill value is given and not None then instead yield the
    fill value if the iterable is exhausted before the given length.
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

def chunk(iterable: Iterable[VT],
          size: int,
          quantity: int,
          as_type: bool = True
          ) -> Iterator[Iterator[VT] | Sequence[VT]]:
    """
    Yield an iterator over the given quantity of chunks of the given size of an iterable.
    
    Any iterable can be chunked, even if it does not support `__len__` and `__getitem__`.
    If the iterable is a sequence, then the chunks are returned as sequences of the same type
    unless `as_type` is False. Otherwise, the chunks are returned as iterators.
    """
    ## If the iterable is a sequence, then chunk by index
    ## and return the chunks as sequences of the same type.
    if (as_type and isinstance(iterable, Sequence)):
        _type = type(iterable)
        yield from (_type(iterable[index : index + size])
                    for index in range(0, quantity * size, size))
    ## Otherwise chunk by iterator.
    else:
        iterator = iter(iterable)
        for _ in range(quantity):
            ## Advance the iterator to the next chunk and yield
            ## the chunk as an iterator if such a chunk exists.
            if chunk_ := itertools.islice(iterator, size):
                yield chunk_
            else: break
