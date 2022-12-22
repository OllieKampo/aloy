###########################################################################
###########################################################################
## Module defining priority queue data structures.                       ##
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

"""Module defining sorted and priority queue data structures."""

__all__ = ("JinxQueue",
           "SortedQueue",
           "PriorityQueue")

from abc import ABCMeta
import collections.abc
import heapq
from dataclasses import dataclass
from numbers import Real
from typing import (Callable, Generic, Hashable, Iterable, Iterator, Optional,
                    TypeVar, overload)

from auxiliary.typingutils import HashableSupportsRichComparison

ST = TypeVar("ST", bound=HashableSupportsRichComparison)

class JinxQueue(collections.abc.Collection, metaclass=ABCMeta):
    """Base class for sorted and priority queues."""
    
    pass

@dataclass(frozen=True, order=True)
class QItem(Generic[ST]):
    """Dataclass for storing custom valued sorted queue items."""
    
    value: Real
    item: ST
    
    def __hash__(self) -> int:
        return hash(self.item)

class SortedQueue(JinxQueue, Generic[ST]):
    """
    A sorted queue implementation wrapping Python's built-in heap-queue algorithm.
    
    A sorted queue is a queue that maintains and pops its items in sorted order.
    Optionally, a key function can be provided to define the order of the items.
    
    A hash table is used to allow for fast membership tests, at the cost of memory.
    Similarly, a lazy delete list is used to allow to fast member removal
    without the need to re-heapify the queue, at the cost of memory.
    
    Iterating over the queue does not necessary yield items in sorted order.
    """
    
    __slots__ = ("__heap",
                 "__get",
                 "__set",
                 "__members",
                 "__delete")
    
    @overload
    def __init__(self,
                 *items: ST
                 ) -> None:
        """
        Create a sorted queue from a series of items.
        
        The order of the items is defined by the
        natural ordering of the items according to
        their rich comparison methods.
        """
        ...
    
    @overload
    def __init__(self,
                 *items: ST,
                 key: Optional[Callable[[ST], Real]] = None,
                 min_first: bool = True
                 ) -> None:
        """
        Create a sorted queue from a series of items and a key function.
        
        The order of the items is defined by the ordering over the
        respective values returned from the key function, where the
        ordering can be set to pop either the min of max value first.
        """
        ...
    
    @overload
    def __init__(self,
                 iterable: Iterable[ST]
                 ) -> None:
        """
        Create a sorted queue from an iterable.
        
        The order of the items is defined by the
        natural ordering of the items according to
        their rich comparison methods.
        """
        ...
    
    @overload
    def __init__(self,
                 iterable: Iterable[ST], *,
                 key: Optional[Callable[[ST], Real]] = None,
                 min_first: bool = True
                 ) -> None:
        """
        Create a sorted queue from an iterable and a key function.
        
        The order of the items is defined by the ordering over the
        respective values returned from the function, where the
        ordering can be set to pop either the min of max value first.
        """
        ...
    
    def __init__(self,
                 *items: ST | Iterable[ST],
                 key: Optional[Callable[[ST], Real]] = None,
                 min_first: bool = True
                 ) -> None:
        """Create a sorted queue of items."""
        if len(items) == 1:
            try:
                iterable = iter(items[0])
                items = items[0]
            except:
                items = [items]
                iterable = iter(items)
        else: iterable = iter(items)
        
        ## The queue itself is a heap;
        ##      - The get and set functions convert
        ##        to and from the value-item tuples
        ##        if the key function is given.
        if key is None:
            self.__heap: list[ST] = list(iterable)
            self.__get: Callable[[ST], ST] = lambda item: item
            self.__set: Callable[[ST], ST] = self.__get
        else:
            if not min_first:
                key = lambda item: -key(item)
            self.__heap: list[QItem[ST]] = [QItem(key(item), item)
                                            for item in iterable]
            self.__get: Callable[[QItem[ST]], ST] = lambda qitem: qitem.item
            self.__set: Callable[[ST], QItem[ST]] = lambda item: QItem(key(item), item)
        
        ## Heapify the heap.
        heapq.heapify(self.__heap)
        
        ## Store a set of members for fast membership checks.
        self.__members: set[ST] = set(items)
        
        ## Store a lazy delete "list" for fast item removal.
        self.__delete: set[ST] = set()
    
    def __str__(self) -> str:
        """Return a string representation of the queue."""
        return f"Priority Queue with {len(self)} items"
    
    def __repr__(self) -> str:
        """Return an instantiable string representation of the queue."""
        if not self.__delete:
            return f"{self.__class__.__name__}({self.__heap!r})"
        return f"{self.__class__.__name__}({[item for item in self.__heap if item not in self.__delete]})"
    
    def __contains__(self, item: ST) -> bool:
        """Return whether an item is in the queue."""
        return item in self.__members
    
    def __iter__(self) -> Iterator[ST]:
        """Return an iterator over the items in the queue."""
        yield from self.__members
    
    def __len__(self) -> int:
        """Return the number of items in the queue."""
        return len(self.__members)
    
    def __bool__(self) -> bool:
        """Return True if the queue is not empty."""
        return bool(self.__members)
    
    def push(self, item: ST) -> None:
        """
        Push an item onto the queue in-place.
        
        Parameters
        ----------
        `item: ST@SortedQueue` - The item to push.
        """
        if item not in self:
            self.__members.add(item)
            if item in self.__delete:
                self.__delete.remove(item)
            else: heapq.heappush(self.__heap, self.__set(item))
    
    @overload
    def push_all(self, *items: ST) -> None:
        """
        Push a series of items onto the queue in-place.
        
        Parameters
        ----------
        `*items: ST@SortedQueue` - The items to push.
        """
        ...
    
    @overload
    def push_all(self, items: Iterable[ST]) -> None:
        """
        Push an iterable of items onto the queue in-place.
        
        Parameters
        ----------
        `items: Iterable[ST@SortedQueue]` - The items to push.
        """
        ...
    
    def push_all(self, *items: ST | Iterable[ST]) -> None:
        """Push a series of items onto the queue in-place."""
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
    
    def pop(self) -> ST:
        """
        Pop the lowest order item from the queue.
        
        Returns
        -------
        `ST@SortedQueue` - The lowest order item.
        
        Raises
        ------
        `IndexError` - If the queue is empty.
        """
        while self:
            item: ST = self.__get(heapq.heappop(self.__heap))
            if item not in self.__delete:
                self.__members.remove(item)
                return item
            self.__delete.remove(item)
        raise IndexError("Pop from empty sorted queue.")
    
    def remove(self, item: ST) -> None:
        """
        Remove a given item from the queue.
        
        Parameters
        ----------
        `item: ST@SortedQueue` - The item to remove.
        
        Raises
        ------
        `KeyError` - If given item is not in the queue.
        """
        if item in self:
            self.__members.remove(item)
            self.__delete.add(item)
        else: raise KeyError(f"The item {item} is not in the sorted queue.")

VT = TypeVar("VT", bound=HashableSupportsRichComparison)
QT = TypeVar("QT", bound=Hashable)

class PriorityQueue(JinxQueue, collections.abc.MutableMapping, Generic[VT, QT]):
    """
    A priority queue implementation wrapping Python's built-in heap-queue algorithm.
    
    A priority queue is a queue where items are popped in priority order.
    The priority of an item must be given when the item is pushed onto the queue.
    
    A hash table is used to allow for fast membership tests, at the cost of memory.
    Similarly, a lazy delete list is used to allow to fast member removal
    without the need to re-heapify the queue, at the cost of memory.
    
    Iterating over the queue does not necessary yield items in priority order.
    """
    
    __slots__ = ("__heap",
                 "__members",
                 "__delete")
    
    @overload
    def __init__(self,
                 *items: tuple[QT, VT]
                 ) -> None:
        """
        Create a priority queue from a series of item-priority tuple pairs.
        
        Items with lower priority values are popped from the queue first.
        """
        ...
    
    @overload
    def __init__(self,
                 iterable: Iterable[tuple[QT, VT]]
                 ) -> None:
        """
        Create a priority queue from an iterable of item-priority tuple pairs.
        
        The iterable itself must not be a tuple.
        
        Items with lower priority values are popped from the queue first.
        """
        ...
    
    def __init__(self,
                 *items: tuple[QT, VT] | Iterable[tuple[QT, VT]]
                 ) -> None:
        """Create a priority queue of item-priority tuple pairs."""
        if len(items) == 1:
            if not isinstance(items[0], tuple):
                items = iter(items[0])
        
        ## Store a set of members for fast membership checks.
        self.__members: dict[QT, VT] = dict(items)
        
        ## The queue itself is a heap.
        self.__heap: list[tuple[VT, QT]] = [(value, item) for item, value in self.__members.items()]
        
        ## Heapify the heap.
        heapq.heapify(self.__heap)
        
        ## Store a lazy delete "list" for fast item removal.
        self.__delete: set[tuple[VT, QT]] = set()
    
    def __str__(self) -> str:
        """Return a string representation of the queue."""
        return f"Priority Queue with {len(self)} items"
    
    def __repr__(self) -> str:
        """Return an instantiable string representation of the queue."""
        if not self.__delete:
            return f"{self.__class__.__name__}({self.__heap!r})"
        return f"{self.__class__.__name__}({[item for item in self.__heap if item not in self.__delete]})"
    
    def __contains__(self, item: QT) -> bool:
        """Return whether an item is in the queue."""
        return item in self.__members
    
    def __getitem__(self, item: QT) -> VT:
        """Return the priority of an item in the queue."""
        return self.__members[item]
    
    def __setitem__(self, item: QT, priority: VT) -> None:
        """Push an item onto the queue with given priority in-place."""
        self.push(item, priority)
    
    def __delitem__(self, item: QT) -> None:
        """Delete an item from the queue."""
        self.remove(item)
    
    def __iter__(self) -> Iterator[QT]:
        """Return an iterator over the items in the queue."""
        yield from self.__members
    
    def __len__(self) -> int:
        """Return the number of items in the queue."""
        return len(self.__members)
    
    def __bool__(self) -> bool:
        """Return True if the queue is not empty."""
        return bool(self.__members)
    
    def push(self, item: QT, priority: VT, /) -> None:
        """
        Push an item onto the queue with given priority in-place.
        
        If the item is aleady present, replace its priority with the given value.
        
        Parameters
        ----------
        `item: QT@PriorityQueue` - The item to push.
        
        `priority: VT@PriorityQueue` - The priority of the item.
        """
        if item not in self:
            self.__members[item] = priority
            if (priority, item) in self.__delete:
                self.__delete.remove((priority, item))
            else: heapq.heappush(self.__heap, (priority, item))
        else:
            self.__delete.add((self.__members[item], item))
            self.__members[item] = priority
            heapq.heappush(self.__heap, (priority, item))
    
    def pop(self) -> QT:
        """
        Pop the lowest priority item from the queue.
        
        Returns
        -------
        `QT@PriorityQueue` - The lowest priority item.
        
        Raises
        ------
        `IndexError` - If the queue is empty.
        """
        while self:
            priority, item = heapq.heappop(self.__heap)
            if (priority, item) not in self.__delete:
                del self.__members[item]
                return item
            self.__delete.remove((priority, item))
        raise IndexError("Pop from empty priority queue.")
    
    def popitem(self) -> tuple[QT, VT]:
        """
        Pop the lowest priority item and its priority from the queue.
        
        Returns
        -------
        `(QT@PriorityQueue, VT@PriorityQueue)` - The lowest priority item and its priority value.
        
        Raises
        ------
        `IndexError` - If the queue is empty.
        """
        while self:
            priority, item = heapq.heappop(self.__heap)
            if (priority, item) not in self.__delete:
                del self.__members[item]
                return (item, priority)
            self.__delete.remove((priority, item))
        raise IndexError("Pop from empty priority queue.")
    
    def remove(self, item: QT) -> None:
        """
        Remove a given item from the queue.
        
        Parameters
        ----------
        `item: QT@PriorityQueue` - The item to remove.
        
        Raises
        ------
        `KeyError` - If given item is not in the queue.
        """
        if item in self:
            self.__delete.add((self.__members[item], item))
            del self.__members[item]
        else: raise KeyError(f"The item {item} is not in the priority queue.")
