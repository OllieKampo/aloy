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
from typing import Generic, Iterator, Sequence, TypeVar

from aloy.auxiliary.moreitertools import ichunk_sequence
from aloy.auxiliary.typingutils import HashableSupportsRichComparison

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"

__all__ = ()


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


LT = TypeVar("LT")


class ChunkedList(collections.abc.MutableSequence, Generic[LT]):
    """
    A list structure built from chunks of fixed length.

    This structure is useful for building large lists iteratively, as it
    preallocates fixed length chunks, and only allocates new chunks when
    necessary. Appending to the list is typically faster than appending to a
    regular list as it only requies an assignment operation, and resizing only
    happens when the current chunk is full. The downside is that inserting or
    deleting items in the middle or start of the list is slower, as it requires
    shifting all the items in all the chunks to the right or left respectively.
    """

    __slots__ = {
        "__chunks": "The chunks of the list.",
        "__chunk_size": "The size of each chunk.",
        "__length": "The length of the last chunk."
    }

    def __init__(
        self,
        init: Sequence[LT] = [],
        chunk_size: int = 20
    ) -> None:
        """Create a new chunked list."""
        if not isinstance(chunk_size, int):
            raise TypeError("chunk_size must be an integer")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        self.__chunks: list[list[LT]] = [
            list(chunk) for chunk in ichunk_sequence(init, chunk_size)
        ]
        if not self.__chunks:
            self.__chunks.append([])
        self.__chunk_size: int = chunk_size
        self.__length: int = len(self.__chunks[-1])
        if self.__length != chunk_size:
            self.__chunks[-1].extend([None] * (chunk_size - self.__length))

    def __str__(self) -> str:
        return str(list(self))

    def __repr__(self) -> str:
        return f"ChunkedList({self})"

    def __len__(self) -> int:
        """Get the length of the list."""
        return ((len(self.__chunks) - 1) * self.__chunk_size) + self.__length

    def __getitem__(self, index: int | slice) -> LT | list[LT]:
        """Get the item at the given index."""
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")
        chunk_index, chunk_offset = divmod(index, self.__chunk_size)
        return self.__chunks[chunk_index][chunk_offset]

    def __setitem__(self, index: int | slice, value: LT | Sequence[LT]) -> None:
        """Set the item at the given index."""
        if isinstance(index, slice):
            range_ = range(*index.indices(len(self)))
            if isinstance(value, collections.abc.Sequence):
                for i, v in zip(range_, value):
                    chunk_index, chunk_offset = divmod(i, self.__chunk_size)
                    self.__chunks[chunk_index][chunk_offset] = v
                if len(value) > len(range_):
                    for i, v in enumerate(value[range_.stop:],
                                          start=range_.stop):
                        self.insert(i, v)
                if len(value) < len(range_):
                    del self[range_.start + len(value):range_.stop]
            else:
                for i in range_:
                    chunk_index, chunk_offset = divmod(i, self.__chunk_size)
                    self.__chunks[chunk_index][chunk_offset] = value
            return
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")
        chunk_index, chunk_offset = divmod(index, self.__chunk_size)
        self.__chunks[chunk_index][chunk_offset] = value

    def __delitem__(self, index: int | slice) -> None:
        """Delete the item at the given index."""
        if isinstance(index, slice):
            range_ = range(*index.indices(len(self)))
            for i in range_:
                del self[i]
            return
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")
        chunk_index, chunk_offset = divmod(index, self.__chunk_size)
        del self.__chunks[chunk_index][chunk_offset]
        if chunk_index != len(self.__chunks) - 1:
            while chunk_index < (len(self.__chunks) - 2):
                first = self.__chunks[chunk_index + 1].pop(0)
                self.__chunks[chunk_index].append(first)
                chunk_index += 1
            if self.__chunks[-1][0] is not None:
                first = self.__chunks[-1].pop(0)
                self.__chunks[-2].append(first)
                if self.__chunks[-1][0] is None:
                    self.__chunks.pop()
                    self.__length = self.__chunk_size
                else:
                    self.__chunks[-1].append(None)
                    self.__length -= 1
            else:
                self.__chunks[-2].append(None)
                self.__chunks.pop()
                self.__length = self.__chunk_size
        else:
            self.__chunks[chunk_index].append(None)
            self.__length -= 1

    def __iter__(self) -> Iterator[LT]:
        """Iterate over the list."""
        for chunk in self.__chunks[:-1]:
            yield from chunk
        yield from self.__chunks[-1][:self.__length]

    @property
    def chunks(self) -> list[list[LT]]:
        """Iterate over the chunks of the list."""
        return [*self.__chunks[:-1], self.__chunks[-1][:self.__length]]

    @property
    def ichunks(self) -> Iterator[list[LT]]:
        """Iterate over the chunks of the list."""
        yield from self.__chunks[:-1]
        yield self.__chunks[-1][:self.__length]

    def append(self, value: LT) -> None:
        """Append the given value to the list."""
        if self.__length == self.__chunk_size:
            self.__chunks.append([None] * self.__chunk_size)
            self.__length = 0
        self.__chunks[-1][self.__length] = value
        self.__length += 1

    def insert(self, index: int, value: ST) -> None:
        """Insert the given value at the given index."""
        if index < 0:
            index += len(self)
        if index < 0 or index > len(self):
            raise IndexError("Index out of range")
        chunk_index, chunk_offset = divmod(index, self.__chunk_size)
        if chunk_index == len(self.__chunks):
            if chunk_offset != 0:
                raise RuntimeError("Chunk index out of range")
            # Create a new chunk if the offset is 0.
            # This only happens when appending to the list.
            self.__chunks.insert(chunk_index, [None] * self.__chunk_size)
            self.__chunks[chunk_index][chunk_offset] = value
            self.__length = 1
        elif chunk_index == len(self.__chunks) - 1:
            if chunk_offset == self.__length:
                # Append to the last chunk if the offset is the length of
                # the last chunk.
                self.__chunks[chunk_index][chunk_offset] = value
                self.__length += 1
            else:
                # Insert into the last chunk if the offset is less than the
                # length of the last chunk.
                self.__chunks[chunk_index].insert(chunk_offset, value)
                self.__length += 1
                if self.__length > self.__chunk_size:
                    # If the last chunk is over-full, create a new chunk.
                    self.__chunks.append([None] * self.__chunk_size)
                    self.__chunks[-1][0] = self.__chunks[-2].pop()
        else:
            # Insert into a middle chunk, and shift-up the rest of the list.
            self.__chunks[chunk_index].insert(chunk_offset, value)
            while chunk_index < (len(self.__chunks) - 1):
                last = self.__chunks[chunk_index].pop()
                self.__chunks[chunk_index + 1].insert(0, last)
                chunk_index += 1
            # If the last chunk is over-full, create a new chunk.
            if self.__chunks[-1][-1] is not None:
                last = self.__chunks[-1].pop()
                self.__chunks.append([None] * self.__chunk_size)
                self.__chunks[-1][0] = last
                self.__length = self.__chunk_size
            else:
                self.__chunks[-1].pop()
                self.__length += 1

    def clear(self) -> None:
        """Clear the list."""
        self.__chunks = [[]]
        self.__length = 0


class PreallocList(collections.abc.MutableSequence, Generic[LT]):
    """A list that preallocates memory."""

    __slots__ = ("__length", "__list", "__postalloc")

    def __init__(self, prealloc: int = 1000, postalloc: int = 100) -> None:
        self.__length = 0
        self.__list = [None] * prealloc
        self.__postalloc = postalloc
    
    def __len__(self) -> int:
        return self.__length
    
    def __getitem__(self, index: int) -> LT:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")
        return self.__list[index]
    
    def __setitem__(self, index: int, value: LT) -> None:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")
        self.__list[index] = value
    
    def __delitem__(self, index: int) -> None:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")
        del self.__list[index]
        self.__length -= 1
    
    def __iter__(self) -> Iterator[LT]:
        for item in self.__list[:self.__length]:
            yield item
    
    def append(self, value: LT) -> None:
        if self.__length == len(self.__list):
            self.__list.extend([None] * self.__postalloc)
        self.__list[self.__length] = value
        self.__length += 1
    
    def insert(self, index: int, value: LT) -> None:
        if index < 0:
            index += len(self)
        if index < 0 or index > len(self):
            raise IndexError("Index out of range")
        if self.__length == len(self.__list):
            self.__list.extend([None] * self.__postalloc)
        self.__list.insert(index, value)
        self.__length += 1


class ChunkedHashList:
    """A hash list structure built from chunks of fixed length."""
    pass


if __name__ == "__main__":
    cl = ChunkedList([], chunk_size=5)
    cl.append(6)
    print(cl.chunks)
    cl[:2] = [10, 10, 10, 10]
    print(cl.chunks)
    cl.insert(2, 20)
    print(cl.chunks)

    import timeit

    def test_list():
        """Test the list."""
        test_list = []
        for i in range(1000):
            test_list.append(i)
    
    def test_prealloc_list():
        """Test the prealloc list."""
        test_list = PreallocList()
        for i in range(1000):
            test_list.append(i)
    
    print("Standard List: ", timeit.timeit(test_list, number=100))
    print("Prealloc List: ", timeit.timeit(test_prealloc_list, number=100))
