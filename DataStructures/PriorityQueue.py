###########################################################################
###########################################################################
## A priority queue data structure.                                      ##
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

import collections.abc
from dataclasses import dataclass
from numbers import Real
from typing import Callable, Generic, Hashable, Iterable, Iterator, NamedTuple, Optional, TypeVar, overload
import heapq

PT = TypeVar("PT", bound=Hashable)

@dataclass(frozen=True, order=True)
class QItem(Generic[PT]):
    "Dataclass for storing custom valued priority queue elements."
    value: Real
    element: PT
    
    def __hash__(self) -> int:
        return hash(self.element)

class PriorityQueue(collections.abc.Collection, Generic[PT]):
    """
    A priority queue implementation wrapping Python's built-in heap-queue algorithm.
    
    An additional hash table is used to allow for fast membership tests, at the cost of memory.
    Similarly, a lazy delete list is used to allow to fast member removal without the need to re-heapify the queue, at the cost of memory.
    """
    
    __slots__ = ("__heap",
                 "__get",
                 "__set",
                 "__members",
                 "__delete")
    
    @overload
    def __init__(self,
                 iterable: Iterable[PT]
                 ) -> None:
        """
        Create a priority queue from an iterable.
        The priority of the elements is defined by the
        natural ordering of the elements according to
        their rich comparison methods.
        """
        ...
    
    @overload
    def __init__(self,
                 iterable: Iterable[PT],
                 priority_func: Optional[Callable[[PT], Real]] = None,
                 min_first: bool = True
                 ) -> None:
        """
        Create a priority queue from an iterable and a priority function.
        The priority of the elements is defined by the ordering over the
        respective priority values returned from the function, where the
        ordering can be set to pop either the min of max priority first.
        """
        ...
    
    def __init__(self,
                 iterable: Iterable[PT],
                 priority_func: Optional[Callable[[PT], Real]] = None,
                 min_first: bool = True
                 ) -> None:
        
        ## The queue itself is a heap;
        ##      - The get and set functions convert
        ##        to and from the value-element tuples
        ##        if the priority function is given.
        if priority_func is None:
            self.__heap: list[PT] = list(iterable)
            self.__get: Callable[[PT], PT] = lambda element: element
            self.__set: Callable[[PT], PT] = self.__get
        else:
            if not min_first:
                priority_func = lambda e: -priority_func(e)
            self.__heap: list[QItem[PT]] = [QItem(priority_func(element), element)
                                            for element in iterable]
            self.__get: Callable[[QItem[PT]], PT] = lambda qitem: qitem.element
            self.__set: Callable[[PT], QItem[PT]] = lambda element: QItem(priority_func(element), element)
        
        ## Heapify the heap.
        heapq.heapify(self.__heap)
        
        ## Store a set of members for fast membership checks.
        self.__members: set[PT] = set(iterable)
        
        ## Store a lazy delete "list" for fast element removal.
        self.__delete: set[PT] = set()
    
    def __str__(self) -> str:
        return f"Priority Queue: elements = {len(self)}"
    
    def __repr__(self) -> str:
        if not self.__delete:
            return repr(self.__heap)
        return repr(list(self))
    
    def __contains__(self, element: PT) -> bool:
        return element in self.__members
    
    def __iter__(self) -> Iterator[PT]:
        yield from self.__members
    
    def __len__(self) -> int:
        return len(self.__members)
    
    def __bool__(self) -> bool:
        return bool(self.__members)
    
    def push(self, element: PT) -> None:
        """
        Push an item onto the queue in-place.
        
        Parameters
        ----------
        `element: PT@PriorityQueue` - The item to push.
        """
        if element not in self:
            heapq.heappush(self.__heap, self.__set(element))
            self.__members.add(element)
            if element in self.__delete:
                self.__delete.remove(element)
    
    def pop(self) -> PT:
        """
        Pop an item from the queue.
        
        Raises
        ------
        `IndexError` - If the queue is empty.
        """
        while self:
            element: PT = self.__get(heapq.heappop(self.__heap))
            if element not in self.__delete:
                self.__members.remove(element)
                return element
            self.__delete.remove(element)
        raise IndexError("Pop from empty priority queue.")
    
    def remove(self, element: PT) -> None:
        """
        Remove a given item from the queue.
        
        Parameters
        ----------
        `element: PT@PriorityQueue` - The item to remove.
        
        Raises
        ------
        `KeyError` - If given item is not in the queue.
        """
        if element in self:
            self.__members.remove(element)
            self.__delete.add(element)
        else: raise KeyError(f"The element {element} is not in the priority queue.")