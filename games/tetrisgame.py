###########################################################################
###########################################################################
## Module defining a tetris game.                                        ##
##                                                                       ##
## Copyright (C)  2023  Oliver Michael Kamperis                          ##
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

"""Module defining a tetris game."""

import enum

from datastructures.mappings import frozendict

@enum.unique
class PieceShape(enum.Enum):
    """Enumeration of the different tetris piece shapes."""

    ZShape = "Z"
    SShape = "S"
    LineShape = "Line"
    TShape = "T"
    SquareShape = "Square"
    LShape = "L"
    MirroredLShape = "Mirrored L"


@enum.unique
class PieceColor(enum.Enum):
    """Enumeration of the different piece colors."""

    Red = "Red"
    Green = "Green"
    Blue = "Blue"
    Yellow = "Yellow"
    Magenta = "Magenta"
    Cyan = "Cyan"
    Orange = "Orange"
    White = "White"


@enum.unique
class Rotation(enum.Enum):
    """Enumeration of the different Rotations."""

    Left = "Left"
    Right = "Right"
    Up = "Up"
    Down = "Down"


class TetrisGameLogic:
    """
    Internal logic of the tetris game.
    """

    __slots__ = (
        "__weakref__",

        # Actions
        "__direction",

        # Game state
        "__playing",
        "__grid_size",
        "__score",
        "__level",
        "__lines_filled",
        "__pieces_used",
    )

    def __init__(
        self,
        cells_grid_size: tuple[int, int],
        manual_udpate: bool = False
    ) -> None:
        """Initialize the tetris game logic."""
        self.__direction = None
        self.__playing = False
        self.__grid_size = cells_grid_size
        self.__score: int = 0
        self.__level = 1
        self.__lines_filled: dict[str, int] = {
            "Total": 0,
            "Single": 0,
            "Double": 0,
            "Triple": 0,
            "Tetris": 0
        }
        self.__pieces_used: dict[str, int] = {
            "Total": 0
        }
        self.__pieces_used.update({
            piece_shape.value: 0
            for piece_shape in PieceShape
        })


class TetrisPiece:
    """Class representing a tetris piece."""

    PIECE_RELATIVE_POSITIONS = frozendict({
        PieceShape.ZShape: frozendict({
            Rotation.Up: ((0, 0), (0, 1), (1, 1), (1, 2)),
            Rotation.Right: ((0, 1), (1, 1), (1, 0), (2, 0)),
            Rotation.Down: ((0, 0), (0, 1), (1, 1), (1, 2)),
            Rotation.Left: ((0, 1), (1, 1), (1, 0), (2, 0))
        }),
        PieceShape.SShape: frozendict({
            Rotation.Up: ((0, 1), (0, 2), (1, 0), (1, 1)),
            Rotation.Right: ((0, 0), (1, 0), (1, 1), (2, 1)),
            Rotation.Down: ((0, 1), (0, 2), (1, 0), (1, 1)),
            Rotation.Left: ((0, 0), (1, 0), (1, 1), (2, 1))
        }),
        PieceShape.LineShape: frozendict({
            Rotation.Up: ((0, 0), (0, 1), (0, 2), (0, 3)),
            Rotation.Right: ((0, 0), (1, 0), (2, 0), (3, 0)),
            Rotation.Down: ((0, 0), (0, 1), (0, 2), (0, 3)),
            Rotation.Left: ((0, 0), (1, 0), (2, 0), (3, 0))
        }),
        PieceShape.TShape: frozendict({
            Rotation.Up: ((0, 0), (0, 1), (0, 2), (1, 1)),
            Rotation.Right: ((0, 1), (1, 1), (2, 1), (1, 0)),
            Rotation.Down: ((0, 1), (1, 0), (1, 1), (1, 2)),
            Rotation.Left: ((0, 1), (1, 1), (2, 1), (1, 2))
        }),
        PieceShape.SquareShape: frozendict({
            Rotation.Up: ((0, 0), (0, 1), (1, 0), (1, 1)),
            Rotation.Right: ((0, 0), (0, 1), (1, 0), (1, 1)),
            Rotation.Down: ((0, 0), (0, 1), (1, 0), (1, 1)),
            Rotation.Left: ((0, 0), (0, 1), (1, 0), (1, 1))
        }),
        PieceShape.LShape: frozendict({
            Rotation.Up: ((0, 0), (0, 1), (0, 2), (1, 2)),
            Rotation.Right: ((0, 1), (1, 1), (2, 1), (2, 0)),
            Rotation.Down: ((0, 0), (1, 0), (1, 1), (1, 2)),
            Rotation.Left: ((0, 0), (0, 1), (1, 0), (2, 0))
        }),
        PieceShape.MirroredLShape: frozendict({
            Rotation.Up: ((0, 2), (1, 0), (1, 1), (1, 2)),
            Rotation.Right: ((0, 0), (0, 1), (1, 1), (2, 1)),
            Rotation.Down: ((0, 0), (0, 1), (0, 2), (1, 0)),
            Rotation.Left: ((0, 0), (1, 0), (2, 0), (2, 1))
        })
    })

    PIECE_COLOURS = frozendict({
        PieceShape.ZShape: PieceColor.Red,
        PieceShape.SShape: PieceColor.Green,
        PieceShape.LineShape: PieceColor.Blue,
        PieceShape.TShape: PieceColor.Yellow,
        PieceShape.SquareShape: PieceColor.Magenta,
        PieceShape.LShape: PieceColor.Cyan,
        PieceShape.MirroredLShape: PieceColor.Orange
    })

    def __init__(self, shape: PieceShape) -> None:
        """Create a new tetris piece."""
        self.__shape: PieceShape = shape
        self.__colour: PieceColor = self.PIECE_COLOURS[shape]
        self.__rotation: Rotation = Rotation.Up
        self.__relative_position: tuple[tuple[int, int], ...] = \
            self.PIECE_RELATIVE_POSITIONS[shape][self.__rotation]

    @property
    def shape(self) -> str:
        """Get the shape of the piece."""
        return self.__shape

    @property
    def colour(self) -> str:
        """Get the colour of the piece."""
        return self.__colour

    @property
    def rotation(self) -> int:
        """Get the rotation of the piece."""
        return self.__rotation

    def rotate(self) -> None:
        """Rotate the piece."""
        self.__rotation = (self.__rotation + 1) % 4

    def __str__(self) -> str:
        """Get a string representation of the piece."""
        return f"{self.__shape} {self.__colour} {self.__rotation}"
