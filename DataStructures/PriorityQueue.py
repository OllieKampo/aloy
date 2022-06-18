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
from typing import Callable, Generic, Iterable, Iterator, Optional, TypeVar, overload
import heapq

from Auxiliary.Typing import HashableSupportsRichComparison

PT = TypeVar("PT", bound=HashableSupportsRichComparison)

@dataclass(frozen=True, order=True)
class QItem(Generic[PT]):
    "Dataclass for storing custom valued priority queue items."
    value: Real
    item: PT
    
    def __hash__(self) -> int:
        return hash(self.item)

class PriorityQueue(collections.abc.Collection, Generic[PT]):
    """
    A priority queue implementation wrapping Python's built-in heap-queue algorithm.
    
    An additional hash table is used to allow for fast membership tests, at the cost of memory.
    Similarly, a lazy delete list is used to allow to fast member removal without the need to re-heapify the queue, at the cost of memory.
    
    Iterating over the queue does not necessary yield items in priority order.
    
    """
    
    __slots__ = ("__heap",
                 "__get",
                 "__set",
                 "__members",
                 "__delete")
    
    @overload
    def __init__(self,
                 *items: PT
                 ) -> None:
        """
        Create a priority queue from a series of items.
        The priority of the items is defined by the
        natural ordering of the items according to
        their rich comparison methods.
        """
        ...
    
    @overload
    def __init__(self,
                 *items: PT,
                 prio_func: Optional[Callable[[PT], Real]] = None,
                 min_first: bool = True
                 ) -> None:
        """
        Create a priority queue from a series of items and a priority function.
        The priority of the items is defined by the ordering over the
        respective priority values returned from the function, where the
        ordering can be set to pop either the min of max priority first.
        """
        ...
    
    @overload
    def __init__(self,
                 iterable: Iterable[PT]
                 ) -> None:
        """
        Create a priority queue from an iterable.
        The priority of the items is defined by the
        natural ordering of the items according to
        their rich comparison methods.
        """
        ...
    
    @overload
    def __init__(self,
                 iterable: Iterable[PT], *,
                 prio_func: Optional[Callable[[PT], Real]] = None,
                 min_first: bool = True
                 ) -> None:
        """
        Create a priority queue from an iterable and a priority function.
        The priority of the items is defined by the ordering over the
        respective priority values returned from the function, where the
        ordering can be set to pop either the min of max priority first.
        """
        ...
    
    def __init__(self,
                 *items: PT | Iterable[PT],
                 prio_func: Optional[Callable[[PT], Real]] = None,
                 min_first: bool = True
                 ) -> None:
        
        if len(items) == 1:
            iterable = iter(items[0])
        else: iterable = iter(items)
        
        ## The queue itself is a heap;
        ##      - The get and set functions convert
        ##        to and from the value-item tuples
        ##        if the priority function is given.
        if prio_func is None:
            self.__heap: list[PT] = list(iterable)
            self.__get: Callable[[PT], PT] = lambda item: item
            self.__set: Callable[[PT], PT] = self.__get
        else:
            if not min_first:
                prio_func = lambda item: -prio_func(item)
            self.__heap: list[QItem[PT]] = [QItem(prio_func(item), item)
                                            for item in iterable]
            self.__get: Callable[[QItem[PT]], PT] = lambda qitem: qitem.item
            self.__set: Callable[[PT], QItem[PT]] = lambda item: QItem(prio_func(item), item)
        
        ## Heapify the heap.
        heapq.heapify(self.__heap)
        
        ## Store a set of members for fast membership checks.
        self.__members: set[PT] = set(self.__heap)
        
        ## Store a lazy delete "list" for fast item removal.
        self.__delete: set[PT] = set()
    
    def __str__(self) -> str:
        return f"Priority Queue: items = {len(self)}"
    
    def __repr__(self) -> str:
        if not self.__delete:
            return repr(self.__heap)
        return repr(list(self))
    
    def __contains__(self, item: PT) -> bool:
        return item in self.__members
    
    def __iter__(self) -> Iterator[PT]:
        yield from self.__members
    
    def __len__(self) -> int:
        return len(self.__members)
    
    def __bool__(self) -> bool:
        return bool(self.__members)
    
    def push(self, item: PT) -> None:
        """
        Push an item onto the queue in-place.
        
        Parameters
        ----------
        `item: PT@PriorityQueue` - The item to push.
        """
        if item not in self:
            self.__members.add(item)
            if item in self.__delete:
                self.__delete.remove(item)
            else: heapq.heappush(self.__heap, self.__set(item))
    
    @overload
    def push_all(self, *items: PT) -> None:
        """
        Push a series of items onto the queue in-place.
        
        Parameters
        ----------
        `*items: PT@PriorityQueue` - The items to push.
        """
        ...
    
    @overload
    def push_all(self, items: Iterable[PT]) -> None:
        """
        Push an iterable of items onto the queue in-place.
        
        Parameters
        ----------
        `items: Iterable[PT@PriorityQueue]` - The items to push.
        """
        ...
    
    def push_all(self, *items: PT | Iterable[PT]) -> None:
        if len(items) == 1:
            items = items[0]
        
        ## If the iterable is a set we can use the hash-based set
        ## operations to speed up the necessary membership testing.
        if isinstance(items, set):
            ## Add all items that are not already members.
            items = items - self.__members
            self.__members |= items
            
            ## Push all items not in the lazy delete list to the heap.
            push_items = items - self.__delete
            for item in push_items:
                heapq.heappush(self.__heap, self.__set(item))
            
            ## Remove lazy deletes for non-members.
            self.__delete -= items
        
        ## Otherwise simply iterate over the items.
        else:
            for item in items:
                self.push(item)
    
    def pop(self) -> PT:
        """
        Pop the lowest priority item from the queue.
        
        Returns
        -------
        `PT@PriorityQueue` - The lowest priority item.
        
        Raises
        ------
        `IndexError` - If the queue is empty.
        """
        while self:
            item: PT = self.__get(heapq.heappop(self.__heap))
            if item not in self.__delete:
                self.__members.remove(item)
                return item
            self.__delete.remove(item)
        raise IndexError("Pop from empty priority queue.")
    
    def iter_pop(self) -> Iterator[PT]:
        """
        Return an iterator over queue items in ascending priority order.
        This lazily pops from the queue, such that items are only removed
        as the iterator is iterated over, and the queue is only emptied
        if the iterator is exhausted.
        
        Returns
        -------
        `Iterator[PT@PriorityQueue}` - The lowest priority item.
        """
        while self:
            item: PT = self.__get(heapq.heappop(self.__heap))
            if item not in self.__delete:
                self.__members.remove(item)
                yield item
            else: self.__delete.remove(item)
    
    def list_pop(self) -> Iterator[PT]:
        """
        Return a list over queue items in ascending priority order.
        The queue is immediately emptied as a result.
        
        Returns
        -------
        `list[PT@PriorityQueue}` - The lowest priority item.
        """
        return [item for item in self.iter_pop()]
    
    def remove(self, item: PT) -> None:
        """
        Remove a given item from the queue.
        
        Parameters
        ----------
        `item: PT@PriorityQueue` - The item to remove.
        
        Raises
        ------
        `KeyError` - If given item is not in the queue.
        """
        if item in self:
            self.__members.remove(item)
            self.__delete.add(item)
        else: raise KeyError(f"The item {item} is not in the priority queue.")