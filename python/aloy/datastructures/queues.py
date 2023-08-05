# Copyright (C) 2023 Oliver Michael Kamperis
# Email: o.m.kamperis@gmail.com
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Module containing sorted and priority queue data structures.

These data structures are for algorithmic use. They are not thread-safe, and
not intended to be used in multi-threaded or multi-process applications. Use
the Python standard library `queue` module instead for such purposes.

The implementations of the queue types contain significant overlap, but do not
inherit from a common base class for performance reasons.
"""

import collections.abc
import heapq
from dataclasses import dataclass
from numbers import Real
from typing import (Callable, Generic, Hashable, ItemsView, Iterable, Iterator,
                    KeysView, TypeVar, ValuesView, overload)

from aloy.auxiliary.typingutils import HashableSupportsRichComparison

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"

__all__ = (
    "SortedQueue",
    "PriorityQueue"
)


def __dir__() -> tuple[str, ...]:
    """Get the names of module attributes."""
    return tuple(sorted(__all__))


ST = TypeVar("ST", bound=HashableSupportsRichComparison)


@dataclass(frozen=True, order=True)
class _SortedQueueItem(Generic[ST]):
    """Dataclass for storing custom valued sorted queue items."""

    value: Real
    item: ST

    def __hash__(self) -> int:
        return hash(self.item)


class SortedQueue(collections.abc.Collection[ST]):
    """
    A sorted queue implementation.

    A sorted queue is a queue that maintains and pops its items in sorted
    order. A key function can be provided to define the order of the items.

    Wraps Python's built-in heap-queue algorithm for OOP use, and adds
    fast membership tests and deletes via a hash table and lazy delete list.

    Iterating over the queue does not necessary yield items in priority order,
    and unlike a heap-queue, the first item in the queue is not necessarily the
    item with the lowest priority value, because of the lazy delete list.

    Instances are not thread-safe.
    """

    __slots__ = {
        "__heap": "The heap-queue of items or value-item tuple pairs.",
        "__get": "The item getter function, accounts for key function.",
        "__set": "The item setter function, accounts for key function.",
        "__members": "Hash table for fast membership checks.",
        "__delete": "Lazy delete 'list' (a set) for fast item removal."
    }

    @overload
    def __init__(self, *items: ST) -> None:
        """
        Create a sorted queue from a series of items.

        The order of the items is defined by the natural ordering of the items
        according to their rich comparison methods.
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

        The order of the items is defined by the ordering over the respective
        values returned from the key function, where the ordering can be set
        to pop either the min of max value first.
        """
        ...

    def __init__(
        self,
        *items: ST,
        key: Callable[[ST], Real] | None = None,
        min_first: bool = True
    ) -> None:
        """Create a sorted queue of items."""
        # The queue itself is a heap.
        self.__heap: list[ST | _SortedQueueItem[ST]]

        # The get and set functions convert to and from the value-item tuples
        # if the key function is given.
        self.__get: Callable[[ST | _SortedQueueItem[ST]], ST]
        self.__set: Callable[[ST], _SortedQueueItem[ST] | ST]

        iterable = iter(items)
        if key is None:
            self.__heap = list(iterable)

            def _get(item):
                return item

            self.__get = _get
            self.__set = _get
        else:
            if not min_first:
                def _key(item: ST) -> Real:
                    return -key(item)  # type: ignore
                key = _key

            self.__heap = [
                _SortedQueueItem(key(item), item)
                for item in iterable
            ]

            def _get(qitem):
                return qitem.item

            def _set(item):
                return _SortedQueueItem(key(item), item)

            self.__get = _get
            self.__set = _set

        # Heapify the heap.
        heapq.heapify(self.__heap)

        # Store a set of members for fast membership checks.
        self.__members: set[ST] = set(items)

        # Store a lazy delete "list" for fast item removal.
        self.__delete: set[ST] = set()

    @overload
    @classmethod
    def from_iterable(cls, iterable: Iterable[ST], /) -> "SortedQueue[ST]":
        """
        Create a sorted queue from an iterable.

        The order of the items is defined by the
        natural ordering of the items according to
        their rich comparison methods.
        """
        ...

    @overload
    @classmethod
    def from_iterable(
        cls,
        iterable: Iterable[ST], /, *,
        key: Callable[[ST], Real] | None = None,
        min_first: bool = True
    ) -> "SortedQueue[ST]":
        """
        Create a sorted queue from an iterable and a key function.

        The order of the items is defined by the ordering over the
        respective values returned from the function, where the
        ordering can be set to pop either the min of max value first.
        """
        ...

    @classmethod
    def from_iterable(
        cls,
        iterable: Iterable[ST],
        key: Callable[[ST], Real] | None = None,
        min_first: bool = True
    ) -> "SortedQueue[ST]":
        """Create a sorted queue from an iterable."""
        return cls(*iterable, key=key, min_first=min_first)

    # pylint: disable=W0212,W0238
    def copy(self) -> "SortedQueue[ST]":
        """Return a shallow copy of the queue."""
        heap: "SortedQueue[ST]" = self.__class__()
        heap.__members = self.__members.copy()
        heap.__get = self.__get
        heap.__set = self.__set
        heap.__heap = self.__heap.copy()
        heap.__delete = self.__delete.copy()
        return heap

    def __str__(self) -> str:
        """Return a string representation of the queue."""
        return f"Sorted Queue with {len(self)} items"

    def __repr__(self) -> str:
        """Return an instantiable string representation of the queue."""
        heap = ", ".join(
            str(item)
            for item in self
        )
        return f"{self.__class__.__name__}({heap})"

    def __contains__(self, item: object) -> bool:
        """Return whether an item is in the queue."""
        return item in self.__members

    def __iter__(self) -> Iterator[ST]:
        """
        Return an iterator over the items in the queue.

        The items are yielded in arbitrary order (not in sorted order).
        """
        yield from self.__members

    def __len__(self) -> int:
        """Return the number of items in the queue."""
        return len(self.__members)

    def __bool__(self) -> bool:
        """Return True if the queue is not empty."""
        return bool(self.__members)

    def iter_ordered(self) -> Iterator[ST]:
        """
        Iterate over the items in the queue in sorted order.

        Note that this method has to clear the lazy delete list, which
        can be expensive if the queue is large and many deletions have
        been made since the last pop.

        Returns
        -------
        `Iterator[ST]` - An iterator over the items in the queue in sorted
        order.
        """
        self.clear_deletes()
        if not self:
            return
        indices: list[int] = [0]
        __getitem__ = self.__heap.__getitem__
        len_ = len(self)
        while indices:
            min_index = min(indices, key=__getitem__)
            yield self.__get(self.__heap[min_index])
            indices.remove(min_index)
            index: int = (min_index * 2) + 1
            if index < len_:
                indices.append(index)
            if index + 1 < len_:
                indices.append(index + 1)

    def clear_deletes(self) -> None:
        """
        Clear the lazy delete list.

        Lazy deletes are usually cleared progressively as items are
        popped from the queue, and as such calling this method is
        usually unnecessary. However, if one wants to store or save
        a large queue that has had many deletions, it may save some
        memory to clear the lazy delete list explicitly.
        """
        if not self:
            self.__delete.clear()
            self.__heap.clear()
        else:
            remove_indices: list[int] = []
            for index, item in enumerate(self.__heap):
                if item in self.__delete:
                    remove_indices.append(index)
            for remove_index in reversed(remove_indices):
                self.__heap[remove_index] = self.__heap[-1]
                self.__heap.pop()
            self.__delete.clear()

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

    def push_all(self, *items: ST) -> None:
        """
        Push a series of items onto the queue in-place.

        Parameters
        ----------
        `*items: ST@SortedQueue` - The items to push.
        """
        for item in items:
            self.push(item)

    def push_from(self, iterable: Iterable[ST], /) -> None:
        """
        Push an iterable of items onto the queue in-place.

        Parameters
        ----------
        `items: Iterable[ST@SortedQueue]` - The items to push.
        """
        # If the iterable is a set we can use the hash-based set
        # operations to speed up the necessary membership testing.
        if isinstance(iterable, set):
            # Add all items that are not already members.
            items = iterable - self.__members
            self.__members |= items

            # Push all items not in the lazy delete list to the heap.
            push_items = items - self.__delete
            for item in push_items:
                heapq.heappush(self.__heap, self.__set(item))

            # Remove lazy deletes for non-members.
            self.__delete -= items

        # Otherwise simply iterate over the items.
        else:
            for item in iterable:
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


QT = TypeVar("QT", bound=Hashable)
VT = TypeVar("VT", bound=HashableSupportsRichComparison)


class PriorityQueue(collections.abc.Mapping[QT, VT]):
    """
    Class defining a priority queue implementation.

    A priority queue is a queue where items are popped in priority order
    (always lowest priority first). The priority of an item must be given when
    the item is pushed onto the queue.

    Wraps Python's built-in heap-queue algorithm for OOP use, and adds
    fast membership tests and deletes via a hash table and lazy delete list.

    Iterating over the queue does not necessary yield items in priority order,
    and unlike a heap-queue, the first item in the queue is not necessarily the
    item with the lowest priority value, because of the lazy delete list.

    Instances are not thread-safe.
    """

    __slots__ = {
        "__heap": "The heap-queue of priority-item tuple pairs.",
        "__members": "Hash table for fast membership checks.",
        "__delete": "Lazy delete 'list' (a set) for fast item removal."
    }

    def __init__(self, *items: tuple[QT, VT]) -> None:
        """
        Create a priority queue from a series of item-priority tuple pairs.

        Items with lower priority values are popped from the queue first.
        """
        # Store a set of members for fast membership checks.
        self.__members: dict[QT, VT] = dict(items)

        # The queue itself is a heap.
        self.__heap: list[tuple[VT, QT]] = [
            (value, item) for item, value in items
        ]

        # Heapify the heap.
        heapq.heapify(self.__heap)

        # Store a lazy delete "list" for fast item removal.
        self.__delete: set[tuple[VT, QT]] = set()

    @classmethod
    def from_iterable(
        cls,
        iterable: Iterable[tuple[QT, VT]], /
    ) -> "PriorityQueue[QT, VT]":
        """
        Create a priority queue from an iterable of item-priority tuple pairs.

        Items with lower priority values are popped from the queue first.
        """
        return cls(*iterable)

    # pylint: disable=W0212,W0238
    def copy(self) -> "PriorityQueue[QT, VT]":
        """Return a shallow copy of the queue."""
        heap: "PriorityQueue[QT, VT]" = self.__class__()
        heap.__members = self.__members.copy()
        heap.__heap = self.__heap.copy()
        heap.__delete = self.__delete.copy()
        return heap

    def __str__(self) -> str:
        """Return a string representation of the queue."""
        return f"Priority Queue with {len(self)} items"

    def __repr__(self) -> str:
        """Return an instantiable string representation of the queue."""
        heap = ", ".join(
            str(item_priority)
            for item_priority in self.items()
        )
        return f"{self.__class__.__name__}({heap})"

    def __contains__(self, item: object) -> bool:
        """Return whether an item is in the queue."""
        return item in self.__members

    def __getitem__(self, item: QT) -> VT:
        """Return the priority of an item in the queue."""
        return self.__members[item]

    def __iter__(self) -> Iterator[QT]:
        """
        Return an iterator over the items in the queue.

        The items are yielded in arbitrary order (not in priority order).
        """
        yield from self.__members

    def __len__(self) -> int:
        """Return the number of items in the queue."""
        return len(self.__members)

    def __bool__(self) -> bool:
        """Return True if the queue is not empty."""
        return bool(self.__members)

    def keys(self) -> KeysView[QT]:
        """Return a view of the items in the queue."""
        return self.__members.keys()

    def values(self) -> ValuesView[VT]:
        """Return a view of the priorities in the queue."""
        return self.__members.values()

    def items(self) -> ItemsView[QT, VT]:
        """Return a view of the items and their priorities in the queue."""
        return self.__members.items()

    def iter_ordered(self) -> Iterator[QT]:
        """
        Iterate over the items in the queue in priority order.

        Note that this method has to clear the lazy delete list, which
        can be expensive if the queue is large and many deletions have
        been made since the last pop.

        Returns
        -------
        `Iterator[QT]` - An iterator over the items in the queue in priority
        order.
        """
        self.clear_deletes()
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

        Note that this method has to clear the lazy delete list, which
        can be expensive if the queue is large and many deletions have
        been made since the last pop.

        Returns
        -------
        `Iterator[tuple[QT, VT]]` - An iterator over the items and their
        priorities in the queue in priority order.
        """
        yield from ((item, self.__members[item])
                    for item in self.iter_ordered())

    def clear_deletes(self) -> None:
        """
        Clear the lazy delete list.

        Lazy deletes are usually cleared progressively as items are
        popped from the queue, and as such calling this method is
        usually unnecessary. However, if one wants to store or save
        a large queue that has had many deletions, it may save some
        memory to clear the lazy delete list explicitly.
        """
        if not self:
            self.__delete.clear()
            self.__heap.clear()
        else:
            remove_indices: list[int] = []
            for index, item_priority in enumerate(self.__heap):
                if item_priority in self.__delete:
                    remove_indices.append(index)
            for remove_index in reversed(remove_indices):
                self.__heap[remove_index] = self.__heap[-1]
                self.__heap.pop()
            self.__delete.clear()

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

    def __push_pop(self, item: QT, priority: VT, /) -> tuple[VT, QT]:
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
