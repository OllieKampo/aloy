###########################################################################
###########################################################################
## Module containing sequence data structures.                           ##
##                                                                       ##
## Copyright (C) 2023 Oliver Michael Kamperis                            ##
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

"""Module containing sorted and priority queue data structures."""

import collections.abc
import heapq
from dataclasses import dataclass
from numbers import Real
from typing import (Callable, Generic, Hashable, Iterable, Iterator, Optional,
                    TypeVar, overload)

from auxiliary.typingutils import HashableSupportsRichComparison

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"

__all__ = (
    "SortedQueue",
    "PriorityQueue"
)


def __dir__() -> tuple[str]:
    """Get the names of module attributes."""
    return sorted(__all__)


ST = TypeVar("ST", bound=HashableSupportsRichComparison)


class SortedList:
    """A self-sorting list structure."""
    pass


class HashList:
    """A list structure with a hash table for fast membership testing and lazy deletion."""
    pass


class FixedLengthList:
    """A list structure with a fixed and pre-allocated length."""
    pass


class FixedLengthHashList:
    """A hash list structure with a fixed and pre-allocated length."""
    pass


class ChunkedList:
    """A list structure built from chunks of fixed length."""
    pass


class ChunkedHashList:
    """A hash list structure built from chunks of fixed length."""
    pass