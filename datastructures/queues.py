###########################################################################
###########################################################################
## Module containing priority queue data structures.                     ##
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

import queue

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


@dataclass(frozen=True, order=True)
class QItem(Generic[ST]):
    """Dataclass for storing custom valued sorted queue items."""

    value: Real
    item: ST

    def __hash__(self) -> int:
        return hash(self.item)


class SortedQueue(collections.abc.Collection, Generic[ST]):
    """
    A sorted queue implementation.

    A sorted queue is a queue that maintains and pops its items in sorted
    order. A key function can be provided to define the order of the items.

    Wraps Python's built-in heap-queue algorithm for OOP use, and adds
    fast membership tests and deletes via a hash table and lazy delete list.

    Iterating over the queue does not necessary yield items in priority order.
    Unlike a heap-queue, the first iten in the queue is not necessarily the
    item with the lowest priority value, because of the lazy delete list.

    Instances are not thread-safe.
    """

    __slots__ = {
        "__heap": "The heap-queue of item-priority tuple pairs.",
        "__get": "The item getter function, accounts for key function.",
        "__set": "The item setter function, accounts for key function.",
        "__members": "Hash table for fast membership checks.",
        "__delete": "Lazy delete 'list' (a set) for fast item removal."
    }

    @overload
    def __init__(self, *items: ST) -> None:
        """
        Create a sorted queue from a series of items.

        The order of the items is defined by the
        natural ordering of the items according to
        their rich comparison methods.
        """
        ...

    @overload
    def __init__(
        self,
        *items: ST,
        key: Callable[[ST], Real] | None = None,
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
    def __init__(self, iterable: Iterable[ST], /) -> None:
        """
        Create a sorted queue from an iterable.

        The order of the items is defined by the
        natural ordering of the items according to
        their rich comparison methods.
        """
        ...

    @overload
    def __init__(
        self,
        iterable: Iterable[ST], /, *,
        key: Callable[[ST], Real] | None = None,
        min_first: bool = True
    ) -> None:
        """
        Create a sorted queue from an iterable and a key function.

        The order of the items is defined by the ordering over the
        respective values returned from the function, where the
        ordering can be set to pop either the min of max value first.
        """
        ...

    def __init__(
        self,
        *items: ST | Iterable[ST],
        key: Optional[Callable[[ST], Real]] = None,
        min_first: bool = True
    ) -> None:
        """Create a sorted queue of items."""
        if len(items) == 1:
            try:
                # Must not assign items = items[0] first,
                # as this breaks the except.
                iterable = iter(items[0])
                items = items[0]
            except TypeError:
                items = [items]
                iterable = iter(items)
        else:
            iterable = iter(items)

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
                def _key(item: ST) -> Real:
                    return -key(item)
                key = _key
            self.__heap: list[QItem[ST]] = [QItem(key(item), item)
                                            for item in iterable]
            self.__get: Callable[[QItem[ST]], ST]
            self.__get = lambda qitem: qitem.item
            self.__set: Callable[[ST], QItem[ST]]
            self.__set = lambda item: QItem(key(item), item)

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
        heap = [item for item in self.__heap if item not in self.__delete]
        return f"{self.__class__.__name__}({heap})"

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

    def push(self, item: ST, /) -> None:
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
            else:
                heapq.heappush(self.__heap, self.__set(item))

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
    def push_all(self, items: Iterable[ST], /) -> None:
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

    def peek(self) -> ST:
        """
        Peek at the lowest order item in the queue.

        Returns
        -------
        `ST@SortedQueue` - The lowest order item.

        Raises
        ------
        `IndexError` - If the queue is empty.
        """
        while self:
            item: ST = self.__get(self.__heap[0])
            if item not in self.__delete:
                return item
            heapq.heappop(self.__heap)
            self.__delete.remove(item)
        raise IndexError("Peek from empty sorted queue.")

    def remove(self, item: ST, /) -> None:
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
        else:
            raise KeyError(f"The item {item} is not in the sorted queue.")


VT = TypeVar("VT", bound=HashableSupportsRichComparison)
QT = TypeVar("QT", bound=Hashable)


class PriorityQueue(collections.abc.MutableMapping, Generic[VT, QT]):
    """
    Class defining a priority queue implementation.

    A priority queue is a queue where items are popped in priority order.
    The priority of an item is given when the item is pushed onto the queue.

    Wraps Python's built-in heap-queue algorithm for OOP use, and adds
    fast membership tests and delete via a hash table and lazy delete list.

    Iterating over the queue does not necessary yield items in priority order.
    Unlike a heap-queue, the first iten in the queue is not necessarily the
    item with the lowest priority value, because of the lazy delete list.

    Instances are not thread-safe.
    """

    __slots__ = {
        "__heap": "The heap-queue of item-priority tuple pairs.",
        "__members": "Hash table for fast membership checks.",
        "__delete": "Lazy delete 'list' (a set) for fast item removal."
    }

    @overload
    def __init__(self, *items: tuple[QT, VT]) -> None:
        """
        Create a priority queue from a series of item-priority tuple pairs.

        Items with lower priority values are popped from the queue first.
        """
        ...

    @overload
    def __init__(self, iterable: Iterable[tuple[QT, VT]], /) -> None:
        """
        Create a priority queue from an iterable of item-priority tuple pairs.

        The iterable itself must not be a tuple.

        Items with lower priority values are popped from the queue first.
        """
        ...

    def __init__(
        self,
        *items: tuple[QT, VT] | Iterable[tuple[QT, VT]]
    ) -> None:
        """Create a priority queue of item-priority tuple pairs."""
        if len(items) == 1:
            if not isinstance(items[0], tuple):
                items = iter(items[0])

        ## Store a set of members for fast membership checks.
        self.__members: dict[QT, VT] = dict(items)

        ## The queue itself is a heap.
        self.__heap: list[tuple[VT, QT]] = [
            (value, item) for item, value in self.__members.items()
        ]

        ## Heapify the heap.
        heapq.heapify(self.__heap)

        ## Store a lazy delete "list" for fast item removal.
        self.__delete: set[tuple[VT, QT]] = set()

    def copy(self) -> "PriorityQueue[VT, QT]":
        """Return a shallow copy of the queue."""
        heap = self.__class__()
        heap.__members = self.__members.copy()
        heap.__heap = self.__heap.copy()
        heap.__delete = self.__delete.copy()
        return heap

    def __str__(self) -> str:
        """Return a string representation of the queue."""
        return f"Priority Queue with {len(self)} items"

    def __repr__(self) -> str:
        """Return an instantiable string representation of the queue."""
        if not self.__delete:
            heap = [(item, priority) for priority, item in self.__heap]
            return f"{self.__class__.__name__}({heap!r})"
        heap = [(item, priority) for priority, item in self.__heap
                if item not in self.__delete]
        return f"{self.__class__.__name__}({heap!r})"

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

    def iter_ordered(self) -> Iterator[QT]:
        """Iterate over the items in the queue in priority order."""
        if not self:
            return
        indices: list[int] = [0]
        __getitem__ = self.__heap.__getitem__
        len_ = len(self)
        while indices:
            min_index = min(indices, key=__getitem__)
            yield self.__heap[min_index][1]
            indices.remove(min_index)
            index: int = (min_index * 2) + 1
            if index < len_:
                indices.append(index)
            if index + 1 < len_:
                indices.append(index + 1)

    def iter_ordered_prio(self) -> Iterator[tuple[QT, VT]]:
        """
        Iterate over the items and their priorities in the queue in priority
        order.
        """
        yield from ((item, self.__members[item])
                    for item in self.iter_ordered())

    def push(self, item: QT, priority: VT, /) -> None:
        """
        Push an item onto the queue with given priority in-place.

        If the item is aleady present, replace its priority with the given
        priority value.

        Parameters
        ----------
        `item: QT@PriorityQueue` - The item to push.

        `priority: VT@PriorityQueue` - The priority of the item.
        """
        if item not in self:
            self.__members[item] = priority
            priority_item = (priority, item)
            if priority_item in self.__delete:
                self.__delete.remove(priority_item)
            else:
                heapq.heappush(self.__heap, priority_item)
        elif priority != (old_priority := self.__members[item]):
            self.__delete.add((old_priority, item))
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
            priority_item = heapq.heappop(self.__heap)
            if priority_item not in self.__delete:
                del self.__members[priority_item[1]]
                return priority_item[1]
            self.__delete.remove(priority_item)
        raise IndexError("Pop from empty priority queue.")

    def pop_prio(self) -> tuple[QT, VT]:
        """
        Pop the lowest priority item and its priority from the queue.

        Returns
        -------
        `(QT@PriorityQueue, VT@PriorityQueue)` - The lowest priority item and
        its priority value.

        Raises
        ------
        `IndexError` - If the queue is empty.
        """
        while self:
            priority_item = heapq.heappop(self.__heap)
            if priority_item not in self.__delete:
                del self.__members[priority_item[1]]
                return priority_item[1], priority_item[0]
            self.__delete.remove(priority_item)
        raise IndexError("Pop from empty priority queue.")

    def __push_pop(self, item: QT, priority: VT, /) -> tuple[QT, VT]:
        if item not in self:
            self.__members[item] = priority
            priority_item = (priority, item)
            if priority_item in self.__delete:
                self.__delete.remove(priority_item)
                priority_item = heapq.heappop(self.__heap)
            else:
                priority_item = heapq.heappushpop(self.__heap, priority_item)
        elif priority != (old_priority := self.__members[item]):
            self.__delete.add((old_priority, item))
            self.__members[item] = priority
            priority_item = heapq.heappushpop(self.__heap, (priority, item))
        return priority_item

    def push_pop(self, item: QT, priority: VT, /) -> QT:
        """
        Push an item onto the queue and then pop the lowest priority item.

        This is more efficient than pushing and then popping separately.

        If the pushed item is aleady present, replace its priority with the
        given priority value.

        Parameters
        ----------
        `item: QT@PriorityQueue` - The item to push.

        `priority: VT@PriorityQueue` - The priority of the item.

        Returns
        -------
        `QT@PriorityQueue` - The lowest priority item.

        Raises
        ------
        `IndexError` - If the queue is empty.
        """
        priority_item = self.__push_pop(item, priority)

        while self:
            if priority_item not in self.__delete:
                del self.__members[priority_item[1]]
                return priority_item[1]
            self.__delete.remove(priority_item)
            priority_item = heapq.heappop(self.__heap)
        raise IndexError("Pop from empty priority queue.")

    def push_pop_prio(self, item: QT, priority: VT, /) -> tuple[QT, VT]:
        """
        Push an item onto the queue and then pop the lowest priority item and
        its priority.

        This is more efficient than pushing and then popping separately.

        If the pushed item is aleady present, replace its priority with the
        given priority value.

        Parameters
        ----------
        `item: QT@PriorityQueue` - The item to push.

        `priority: VT@PriorityQueue` - The priority of the item.

        Returns
        -------
        `(QT@PriorityQueue, VT@PriorityQueue)` - The lowest priority item and
        its priority value.

        Raises
        ------
        `IndexError` - If the queue is empty.
        """
        priority_item = self.__push_pop(item, priority)

        while self:
            if priority_item not in self.__delete:
                del self.__members[priority_item[1]]
                return priority_item[1], priority_item[0]
            self.__delete.remove(priority_item)
            priority_item = heapq.heappop(self.__heap)
        raise IndexError("Pop from empty priority queue.")

    def peek(self) -> QT:
        """
        Peek at the lowest priority item in the queue.

        Returns
        -------
        `QT@PriorityQueue` - The lowest priority item.

        Raises
        ------
        `IndexError` - If the queue is empty.
        """
        while self:
            priority_item = self.__heap[0]
            if priority_item not in self.__delete:
                return priority_item[1]
            heapq.heappop(self.__heap)
            self.__delete.remove(priority_item)
        raise IndexError("Peek at empty priority queue.")

    def peek_prio(self) -> tuple[QT, VT]:
        """
        Peek at the lowest priority item and its priority in the queue.

        Returns
        -------
        `(QT@PriorityQueue, VT@PriorityQueue)` - The lowest priority item and
        its priority value.

        Raises
        ------
        `IndexError` - If the queue is empty.
        """
        while self:
            priority_item = self.__heap[0]
            if priority_item not in self.__delete:
                return priority_item[1], priority_item[0]
            heapq.heappop(self.__heap)
            self.__delete.remove(priority_item)
        raise IndexError("Peek at empty priority queue.")

    def remove(self, item: QT, /) -> None:
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
        else:
            raise KeyError(f"The item {item} is not in the priority queue.")
