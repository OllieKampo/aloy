###########################################################################
###########################################################################
## Module defining utility functions for manipulating strings.           ##
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

"""Module defining utility functions for manipulating strings."""

import collections
import functools
import itertools
import math
import textwrap
from typing import Iterable, final

@final
@functools.total_ordering
class StringBuilder(collections.abc.Sequence):
    """
    Class defining a string builder.
    
    String builders allow efficient string construction by repeated
    concatenation with linear runtime cost, using either the in-place
    addition operator `+=`, or the `append()` and `extend()` methods.
    
    In constrast, concatenation on immutable sequences like strings by
    addition (with the `+` or `+=` operators) results in a new object.
    This means that building a string by repeated concatenation will
    always have a quadratic runtime cost in the total string length.
    """
    
    __slots__ = {"__compiled" : "The compiled string.",
                 "__strings" : "A list of strings queued to be concatenated on the next compile."}
    
    def __init__(self, string: str = "") -> None:
        """Initialize the string builder, optionally with a string."""
        self.__compiled: str = str(string)
        self.__strings: list[str] = []
    
    def __repr__(self) -> str:
        """Return the compiled string."""
        return self.compile()
    
    def __getitem__(self, index: int) -> str:
        """Return the character at the given index."""
        if index < len(self):
            return self.__compiled[index]
        raise IndexError(f"Index {index} out of range.")
    
    def __len__(self) -> int:
        """Return the current length of the compiled string."""
        return len(self.compile())
    
    def __eq__(self, __o: object) -> bool:
        """Return whether the string builder is equal to another object."""
        if isinstance(__o, str):
            return self.compile() == __o
        elif isinstance(__o, StringBuilder):
            return self.compile() == __o.compile()
        return NotImplemented
    
    def __lt__(self, __o: object) -> bool:
        """Return whether the string builder is less than another object."""
        if isinstance(__o, str):
            return self.compile() == __o
        elif isinstance(__o, StringBuilder):
            return self.compile() == __o.compile()
        return NotImplemented
    
    def __iadd__(self, other: str) -> "StringBuilder":
        """Concatenate the string builder with another string in-place."""
        self.append(other)
        return self
    
    def __add__(self, other: str) -> "StringBuilder":
        """Return the concatenation of the string builder and another string as a new string builder."""
        builder = StringBuilder(self.compile())
        builder.append(other)
        return builder
    
    def __radd__(self, other: str) -> "StringBuilder":
        """Return the concatenation of the string builder and another string as a new string builder."""
        builder = StringBuilder(self.compile())
        builder.append(other)
        return builder
    
    def __format__(self, __format_spec: str) -> str:
        """Format the string builder according to the given format specifier."""
        return format(repr(self), __format_spec)
    
    def copy(self) -> "StringBuilder":
        """Return a copy of the string builder."""
        return self.__class__(self.compile())
    
    def append(self, string: str) -> None:
        """Append a string to the string builder in-place."""
        self.__strings.append(string)
    
    def extend(self, strings: Iterable[str], sep: str | None = None) -> None:
        """
        Extend the string builder with a sequence of strings in-place.
        
        If `sep` is given and not None, it is inserted between each string in the sequence.
        """
        if sep is not None:
            self.__strings.append(sep.join(strings))
        else: self.__strings.extend(strings)
    
    def compile(self) -> str:
        """Compile and return the string builder into a string."""
        if self.__strings:
            self.__compiled += "".join(self.__strings)
            self.__strings.clear()
        return self.__compiled

def center_text(text: str,
                centering_width: int = 120,
                prefix: str = "\n",
                postfix: str = "\n",
                framing_width: int | float = 0.8,
                frame_before: bool = True,
                frame_after: bool = True,
                framing_char: str = '=',
                vbar_left: str = '',
                vbar_right: str = ''
                ) -> str:
    """Center a string for pretty printing to the console."""
    centered_text = StringBuilder()
    
    if centering_width <= 0:
        raise ValueError(f"Centering width must be between greater than 0. Got; {centering_width}.")
    
    if isinstance(framing_width, float):
        framing_width = max(0, min(centering_width, int(framing_width * centering_width)))
    elif framing_width > centering_width:
        framing_width = centering_width
    
    free_space: int = framing_width - (len(vbar_left) + len(vbar_right))
    if free_space < 0:
        raise ValueError("Framing width is too small to accommodate the vertical bars.")
    line_iter = itertools.chain(*[textwrap.wrap(f"{vbar_left + (' ' * math.floor((free_space - len(part)) / 2)):>s}{part}{(' ' * math.ceil((free_space - len(part)) / 2)) + vbar_right:<s}",
                                                width=centering_width, replace_whitespace=False, drop_whitespace=False) for part in text.split("\n")])
    
    if prefix:
        centered_text += prefix
    if framing_width != 0 and frame_before:
        centered_text += f"{(framing_char * framing_width).center(centering_width)}\n"
    centered_text.extend((line.center(centering_width) for line in line_iter), sep="\n")
    if framing_width != 0 and frame_after:
        centered_text += f"\n{(framing_char * framing_width).center(centering_width)}"
    if postfix:
        centered_text += postfix
    
    return centered_text.compile()
