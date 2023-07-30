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

"""Module defining utilities for manipulating strings."""

import collections
import itertools
import textwrap
from typing import Iterable, final, overload
import warnings

from aloy.datastructures.views import ListView

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"

__all__ = (
    "StringBuilder",
    "center_text"
)


def __dir__() -> tuple[str, ...]:
    """Get the names of module attributes."""
    return __all__


@final
class StringBuilder(collections.abc.Sequence):
    """
    Class defining a string builder.

    String builders allow efficient string construction by repeated
    concatenation with linear runtime cost, using either the in-place
    addition operator `+=`, or the `append()` and `extend()` methods.
    Note that the `+` operator creates a new string builder containing
    a copy of the original string builder and the added string.

    In constrast, concatenation on immutable sequences like strings by
    addition (with the `+` or `+=` operators) results in a new object.
    This means that building a string by repeated concatenation will
    always have a quadratic runtime cost in the total string length.

    The alternative to a string builder is to use a list of strings
    and `str.join()` them at the end, with the same linear runtime
    cost as a string builder. However, the string builder provides
    many convenience methods, and keeps track of both the compiled
    string and the strings queued to be concatenated.
    """

    __slots__ = {
        "__compiled": "The compiled string.",
        "__strings": "A list of strings queued to be appended on the next "
                     "compile.",
        "__duplicator_flags": "A list of flags for duplicating strings "
                              "queued to be appended.",
    }

    def __init__(self, string: str = "") -> None:
        """Initialize the string builder, optionally with a string."""
        self.__compiled: str = str(string)
        self.__strings: list[str] = []
        self.__duplicator_flags: dict[str, int] = {}

    def __repr__(self) -> str:
        """
        Return an instansiable string representation of the string
        builder (this compiles the string builder).
        """
        return f"{self.__class__.__name__}({self.compile()!r})"

    def __str__(self) -> str:
        """
        Return a string representation of the string builder (this
        does not compile the string builder).
        """
        return f"{self.__class__.__name__} : Compiled length = {len(self)}, " \
               f"Queued appends = {self.queued_appends}, " \
               f"Queued length = {self.queued_length}"

    @overload
    def __getitem__(self, index: int) -> str:
        """Return the character at the given index."""
        ...

    @overload
    def __getitem__(self, index: slice) -> str:
        """Return a string of characters over the given slice."""
        ...

    def __getitem__(self, index: int | slice) -> str:
        return self.__compiled[index]

    def __len__(self) -> int:
        """Return the current length of the string builder."""
        return len(self.__compiled)

    @property
    def queued_appends(self) -> int:
        """Return the number of strings queued to be appended."""
        return len(self.__strings)

    @property
    def queued_length(self) -> int:
        """Return the total length of the strings queued to be appended."""
        return sum(map(len, self.__strings))

    @overload
    def get_queued(self, index: int) -> str:
        """Return the string queued to be appended at the given index."""
        ...

    @overload
    def get_queued(self, index: slice) -> ListView[str]:
        """
        Return a list of strings queued to be appended over the given slice.
        """
        ...

    def get_queued(  # pylint: disable=missing-function-docstring
        self,
        index: int | slice
    ) -> str | ListView[str]:
        if isinstance(index, int):
            return self.__strings[index]
        return ListView(self.__strings[index])

    def __iadd__(self, other: str) -> "StringBuilder":
        """Concatenate the string builder with another string in-place."""
        self.__strings.append(other)
        return self

    def __add__(self, other: str) -> "StringBuilder":
        """
        Return the concatenation of the string builder and another string as a
        new string builder.
        """
        builder = StringBuilder(self.__compiled)
        builder.extend(self.__strings)
        builder.append(other)
        return builder

    def __radd__(self, other: str) -> "StringBuilder":
        """
        Return the concatenation of the string builder and another string
        as a new string builder.
        """
        builder = StringBuilder(other)
        builder.append(self.__compiled)
        builder.extend(self.__strings)
        return builder

    def copy(self) -> "StringBuilder":
        """Return a copy of the string builder."""
        builder = self.__class__(self.__compiled)
        builder.extend(self.__strings)
        return builder

    def append(self, string: str, /, *, end: str | None = None) -> None:
        """Append a string to the string builder in-place."""
        self.__strings.append(string)
        if end is not None:
            self.__strings.append(end)

    def append_all(
        self,
        *strings: str,
        sep: str | None = None,
        end: str | None = None
    ) -> None:
        """
        Append all of a sequence of strings to the string builder in-place.

        If `sep` is given and not None, it is inserted between each string in
        the sequence. If `end` is given and not None, it is appended to the
        end of the sequence.
        """
        self.extend(strings, sep=sep, end=end)

    def append_many(
        self,
        string: str,
        times: int, /, *,
        sep: str | None = None,
        end: str | None = None
    ) -> None:
        """
        Append a string to the string builder in-place `times` times.

        If `times` is negative, it is treated as zero. If `sep` is given and
        not None, it is inserted between each string in the sequence. If `end`
        is given and not None, it is appended to the end of the sequence.
        """
        if times > 0:
            self.extend(itertools.repeat(string, times), sep=sep, end=end)

    def extend(
        self,
        strings: Iterable[str], /, *,
        sep: str | None = None,
        end: str | None = None
    ) -> None:
        """
        Extend the string builder with a sequence of strings in-place.

        If `sep` is given and not None, it is inserted between each string in
        the sequence. If `end` is given and not None, it is appended to the
        end of the sequence.
        """
        if sep is not None:
            self.__strings.append(sep.join(strings))
        else:
            self.__strings.extend(strings)
        if end is not None:
            self.__strings.append(end)

    @overload
    def duplicate(
        self,
        back: int, /, *,
        times: int = 1
    ) -> None:
        """
        Duplicate in-place the last `back` appends to the string builder
        `times` times.

        If `back` is negative, it is treated as zero. If `back` is greater than
        the number of queued appends, it is treated as the number of queued
        appends. If `times` is negative, it is treated as zero.
        """
        ...

    @overload
    def duplicate(
        self,
        back: int,
        skip: int, /, *,
        times: int = 1
    ) -> None:
        """
        Duplicate in-place the last `back` appends to the string builder
        `times` times ignoring the last `skip` appends.

        If `back` is negative, it is treated as zero. If `back` is greater than
        the number of queued appends, it is treated as the number of queued
        appends. If `skip` is negative, it is treated as zero. If `skip` is
        greater than `back`, it is treated as `back`. If `times` is negative,
        it is treated as zero.
        """
        ...

    def duplicate(  # pylint: disable=missing-function-docstring
        self,
        back: int,
        skip: int | None = None, /, *,
        times: int | None = None
    ) -> None:
        if skip is None:
            skip = 0
        if times is None:
            times = 1
        back = min(len(self.__strings), max(back, 0))
        skip = min(back, max(skip, 0))
        times = max(times, 0)
        if skip == 0:
            self.__strings.extend(self.__strings[-back:] * times)
        else:
            self.__strings.extend(self.__strings[-back:-skip] * times)

    def duplicate_flagged(
        self,
        from_flag: str,
        to_flag: str | None = None, /, *,
        times: int = 1
    ) -> None:
        """
        Duplicate in-place the strings queued to be appended from the flag
        `from_flag` to the flag `to_flag` to the string builder `times` times.

        If `to_flag` is None, the strings are duplicated to the most recent
        append.
        """
        if to_flag is None:
            to_flag = ""
        from_index = self.__duplicator_flags.get(from_flag, 0)
        to_index = self.__duplicator_flags.get(to_flag, self.queued_appends)
        self.__strings.extend(self.__strings[from_index:to_index] * times)

    def set_duplicator_flag(self, flag: str, index: int | None = None) -> None:
        """
        Set a flag for the duplicator.

        If `index` is not given or None, the flag is set for the most recent
        append. If `index` is negative, it is treated as zero. If `index` is
        greater than the number of queued appends, it is treated as the number
        of queued appends.
        """
        if index is None or index > self.queued_appends:
            index = self.queued_appends
        elif index < 0:
            index = 0
        self.__duplicator_flags[flag] = index

    def compile(self) -> str:
        """Compile and return the string builder into a string."""
        if self.__strings:
            self.__compiled += "".join(self.__strings)
            self.__strings.clear()
        return self.__compiled


def center_text(
    text: str,
    wrapping_width: int = 100,
    centering_width: int | float = 1.2,
    framing_width: int | float = 1.1,
    framing_char: str = '=',
    frame_before: bool = True,
    frame_after: bool = True,
    vbar_left: str = '',
    vbar_right: str = '',
    prefix: str = "\n",
    postfix: str = "\n"
) -> str:
    """
    Center a string for pretty printing to the console.

    The string is wrapped to the given `wrapping_width`, then centered to the
    given `centering_width` (padding with spaces), and framed to the given
    `framing_width` with `framing_char`. If `vbar_left` and `vbar_right` are
    given, they are added to the left and right of each wrapped line.
    """
    centered_text = StringBuilder()

    if wrapping_width <= 0:
        raise ValueError("Wrapping width must be greater than 0. "
                         f"Got; {wrapping_width}.")
    if isinstance(centering_width, float):
        centering_width = int(centering_width * wrapping_width)
    centering_width = max(wrapping_width, centering_width)
    if isinstance(framing_width, float):
        framing_width = int(framing_width * wrapping_width)
    framing_width = max(0, min(centering_width, framing_width))

    free_space: int = wrapping_width - (len(vbar_left) + len(vbar_right))
    if free_space <= 0:
        raise ValueError("Wrapping width is too small to accommodate the "
                         "vertical bars.")
    if free_space < wrapping_width * 0.1:
        warnings.warn("Wrapping width is very small compared to the size of "
                      "the vertical bars.")

    line_iter = itertools.chain(
        *[
            textwrap.wrap(
                f"{part:^{free_space}}",
                width=free_space,
                replace_whitespace=False,
                drop_whitespace=False
            )
            for part in text.split("\n")
        ]
    )

    if prefix:
        centered_text += prefix
    if framing_width != 0:
        frame: str = f"\n{(framing_char * framing_width):^{centering_width}}"
        if frame_before:
            centered_text += frame
            centered_text += "\n"
    centered_text.extend(
        (f"{vbar_left}{line}{vbar_right}".center(centering_width)
         for line in line_iter),
        sep="\n"
    )
    if framing_width != 0 and frame_after:
        centered_text += frame
    if postfix:
        centered_text += postfix

    return centered_text.compile()
