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

"""Module defining a tetris game."""

import enum
import random
import sys
from typing import overload

from PySide6 import QtCore, QtGui, QtWidgets

from aloy.concurrency.atomic import AtomicObject
from aloy.datastructures.mappings import frozendict
from aloy.guis.gui import AloyGuiWindow, AloySystemData, AloyWidget
from aloy.games._game_errors import AloyGameInternalError


@enum.unique
class Piece(enum.Enum):
    """Enumeration of the different tetris pieces."""

    Z_SHAPE = "Z"
    S_SHAPE = "S"
    LINE_SHAPE = "Line"
    T_SHAPE = "T"
    SQUARE_SHAPE = "Square"
    L_SHAPE = "L"
    MIRRORED_L_SHAPE = "Mirrored L"


@enum.unique
class PieceColor(enum.Enum):
    """Enumeration of the different piece colors."""

    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"
    MAGENTA = "magenta"
    CYAN = "cyan"
    ORANGE = "orange"
    WHITE = "white"


# Mapping of pieces to colors.
PIECE_COLOURS = frozendict({
    Piece.Z_SHAPE: PieceColor.RED,
    Piece.S_SHAPE: PieceColor.GREEN,
    Piece.LINE_SHAPE: PieceColor.BLUE,
    Piece.T_SHAPE: PieceColor.YELLOW,
    Piece.SQUARE_SHAPE: PieceColor.MAGENTA,
    Piece.L_SHAPE: PieceColor.CYAN,
    Piece.MIRRORED_L_SHAPE: PieceColor.ORANGE
})


# Mapping of piece shapes to their shapes at each rotation.
PIECE_SHAPES = frozendict({
    Piece.Z_SHAPE: frozendict({
        0: ((0, 0), (0, 1), (1, 1), (1, 2)),
        1: ((0, 1), (1, 1), (1, 0), (2, 0)),
        2: ((0, 0), (0, 1), (1, 1), (1, 2)),
        3: ((0, 1), (1, 1), (1, 0), (2, 0))
    }),
    Piece.S_SHAPE: frozendict({
        0: ((0, 1), (0, 2), (1, 0), (1, 1)),
        1: ((0, 0), (1, 0), (1, 1), (2, 1)),
        2: ((0, 1), (0, 2), (1, 0), (1, 1)),
        3: ((0, 0), (1, 0), (1, 1), (2, 1))
    }),
    Piece.LINE_SHAPE: frozendict({
        0: ((0, 0), (0, 1), (0, 2), (0, 3)),
        1: ((0, 0), (1, 0), (2, 0), (3, 0)),
        2: ((0, 0), (0, 1), (0, 2), (0, 3)),
        3: ((0, 0), (1, 0), (2, 0), (3, 0))
    }),
    Piece.T_SHAPE: frozendict({
        0: ((0, 0), (0, 1), (0, 2), (1, 1)),
        1: ((0, 1), (1, 1), (2, 1), (1, 0)),
        2: ((0, 1), (1, 0), (1, 1), (1, 2)),
        3: ((0, 1), (1, 1), (2, 1), (1, 2))
    }),
    Piece.SQUARE_SHAPE: frozendict({
        0: ((0, 0), (0, 1), (1, 0), (1, 1)),
        1: ((0, 0), (0, 1), (1, 0), (1, 1)),
        2: ((0, 0), (0, 1), (1, 0), (1, 1)),
        3: ((0, 0), (0, 1), (1, 0), (1, 1))
    }),
    Piece.L_SHAPE: frozendict({
        0: ((0, 0), (0, 1), (0, 2), (1, 2)),
        1: ((0, 1), (1, 1), (2, 1), (2, 0)),
        2: ((0, 0), (1, 0), (1, 1), (1, 2)),
        3: ((0, 0), (0, 1), (1, 0), (2, 0))
    }),
    Piece.MIRRORED_L_SHAPE: frozendict({
        0: ((0, 2), (1, 0), (1, 1), (1, 2)),
        1: ((0, 0), (0, 1), (1, 1), (2, 1)),
        2: ((0, 0), (0, 1), (0, 2), (1, 0)),
        3: ((0, 0), (1, 0), (2, 0), (2, 1))
    })
})


def _rotate_piece_left(
    piece: Piece,
    piece_position: tuple[tuple[int, int], ...],
    piece_rotation: int
) -> tuple[tuple[tuple[int, int], ...], int]:
    """
    Rotate the piece to the left.

    Return the new piece position and rotation.
    """
    new_piece_rotation = (piece_rotation - 1) % 4
    return (
        _rotate_piece(
            piece,
            piece_position,
            piece_rotation,
            new_piece_rotation
        ),
        new_piece_rotation
    )


def _rotate_piece_right(
    piece: Piece,
    piece_position: tuple[tuple[int, int], ...],
    piece_rotation: int
) -> tuple[tuple[tuple[int, int], ...], int]:
    """
    Rotate the piece to the right.

    Return the new piece position and rotation.
    """
    new_piece_rotation = (piece_rotation + 1) % 4
    return (
        _rotate_piece(
            piece,
            piece_position,
            piece_rotation,
            new_piece_rotation
        ),
        new_piece_rotation
    )


def _rotate_piece(
    piece: Piece,
    piece_position: tuple[tuple[int, int], ...],
    piece_rotation: int,
    new_piece_rotation: int
) -> tuple[tuple[int, int], ...]:
    """Rotate the piece to the new rotation."""
    new_piece_position = list(piece_position)
    for index, section in enumerate(piece_position):
        new_piece_position[index] = (
            section[0] - PIECE_SHAPES[piece][piece_rotation][index][0]
            + PIECE_SHAPES[piece][new_piece_rotation][index][0],
            section[1] - PIECE_SHAPES[piece][piece_rotation][index][1]
            + PIECE_SHAPES[piece][new_piece_rotation][index][1]
        )
    return tuple(new_piece_position)


@enum.unique
class Direction(enum.Enum):
    """Enumeration of the possible directions a piece can move."""

    LEFT = enum.auto()
    RIGHT = enum.auto()
    ROTATE_LEFT = enum.auto()
    ROTATE_RIGHT = enum.auto()
    DOWN = enum.auto()
    FAST_DOWN = enum.auto()
    DROP = enum.auto()
    STORE_PIECE = enum.auto()


# Game logic constants
_INITIAL_DIRECTION: Direction = Direction.DOWN
_INITIAL_SCORE: int = 0
_INITIAL_LEVEL: int = 1
_INITIAL_LINES_FILLED: dict[str, int] = {
    "Total": 0,
    "Single": 0,
    "Double": 0,
    "Triple": 0,
    "Tetris": 0
}
_INITIAL_PIECES_USED: dict[Piece | str, int] = {
    "Total": 0
} | {
    piece: 0
    for piece in Piece
}
_INITIAL_SECONDS_PER_MOVE: float = 0.25
_SECONDS_PER_MOVE_DECREASE_PER_LEVEL: float = 0.01
_MIN_SECONDS_PER_MOVE: float = 0.10
_DEFAULT_SPEED: float = 1.0
_MIN_SPEED: float = 0.25
_MAX_SPEED: float = 1.75
_DEFAULT_GHOST_PIECE_ENABLED: bool = True
_DEFAULT_ALLOW_STORE_PIECE: bool = True
_DEFAULT_SHOW_GRID: bool = False
_TICKS_PER_MOVE_DOWN: int = 5
_SCORE_PER_LEVEL: int = 1000
_NUM_LINES_NAME: dict[int, str] = {
    1: "Single",
    2: "Double",
    3: "Triple",
    4: "Tetris"
}
_NUM_LINES_SCORE: dict[int, int] = {
    1: 100,
    2: 300,
    3: 600,
    4: 1000
}


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
        "__board_size",
        "__board_cells",
        "__score",
        "__level",
        "__lines_filled",
        "__pieces_used",
        "__seconds_per_move",
        "__game_over",
        "__paused",
        "__current_piece_position",
        "__current_piece_rotation",
        "__current_piece",
        "__ghost_piece_position",
        "__next_piece",
        "__stored_piece",
        "__ticks_since_last_move_down",
        "__manual_update",

        # Game options
        "speed",
        "ghost_piece_enabled",
        "allow_store_piece",
        "show_grid"
    )

    def __init__(
        self,
        board_size: tuple[int, int],
        manual_udpate: bool = False
    ) -> None:
        """Initialize the tetris game logic."""
        if not isinstance(board_size, tuple):
            raise TypeError("The grid size must be a tuple. "
                            f"Got; {type(board_size)!r}.")
        if len(board_size) != 2:
            raise ValueError("The grid size must be a tuple of length 2. "
                             f"Got; {len(board_size)}.")
        if any(size <= 0 for size in board_size):
            raise ValueError("The grid size must be positive. "
                             f"Got; {board_size}.")

        # Actions
        self.__direction = AtomicObject[Direction](_INITIAL_DIRECTION)

        # Game state
        self.__playing = False
        self.__board_size = board_size
        self.__board_cells: list[list[PieceColor]] = [
            [PieceColor.WHITE for _ in range(board_size[0])]
            for _ in range(board_size[1])
        ]
        self.__score: int = _INITIAL_SCORE
        self.__level: int = _INITIAL_LEVEL
        self.__lines_filled: dict[str, int] = _INITIAL_LINES_FILLED.copy()
        self.__pieces_used: dict[Piece | str, int] \
            = _INITIAL_PIECES_USED.copy()
        self.__seconds_per_move: float = _INITIAL_SECONDS_PER_MOVE
        self.__game_over: bool = False
        self.__paused: bool = False
        self.__current_piece_position: tuple[tuple[int, int], ...] | None \
            = None
        self.__current_piece_rotation: int = 0
        self.__current_piece: Piece | None = None
        self.__ghost_piece_position: tuple[tuple[int, int], ...] | None = None
        self.__next_piece: Piece | None = None
        self.__stored_piece: Piece | None = None
        self.__ticks_since_last_move_down: int = 0
        self.__manual_update: bool = manual_udpate

        # Game options
        self.speed: float = _DEFAULT_SPEED
        self.ghost_piece_enabled: bool = _DEFAULT_GHOST_PIECE_ENABLED
        self.allow_store_piece: bool = _DEFAULT_ALLOW_STORE_PIECE
        self.show_grid: bool = _DEFAULT_SHOW_GRID

    @property
    def direction(self) -> AtomicObject[Direction]:
        """Get the direction of the piece."""
        return self.__direction

    @property
    def playing(self) -> bool:
        """Get whether the game is playing."""
        return self.__playing

    @property
    def board_size(self) -> tuple[int, int]:
        """Get the size of the board."""
        return self.__board_size

    @property
    def board_cells(self) -> list[list[PieceColor]]:
        """Get the cells of the board."""
        return self.__board_cells

    @property
    def score(self) -> int:
        """Get the score of the game."""
        return self.__score

    @property
    def level(self) -> int:
        """Get the level of the game."""
        return self.__level

    @property
    def lines_filled(self) -> dict[str, int]:
        """Get the number of lines filled."""
        return self.__lines_filled

    @property
    def pieces_used(self) -> dict[Piece | str, int]:
        """Get the number of pieces used."""
        return self.__pieces_used

    @property
    def seconds_per_move(self) -> float:
        """Get the number of seconds per move."""
        return self.__seconds_per_move

    @property
    def game_over(self) -> bool:
        """Get whether the game is over."""
        return self.__game_over

    @property
    def paused(self) -> bool:
        """Get whether the game is paused."""
        return self.__paused

    @paused.setter
    def paused(self, paused: bool) -> None:
        self.__paused = paused

    @property
    def current_piece(self) -> Piece | None:
        """Get the current piece."""
        return self.__current_piece

    @property
    def current_piece_position(self) -> tuple[tuple[int, int], ...] | None:
        """Get the position of the current piece."""
        return self.__current_piece_position

    @property
    def current_piece_rotation(self) -> int:
        """Get the rotation of the current piece."""
        return self.__current_piece_rotation

    @property
    def ghost_piece_position(self) -> tuple[tuple[int, int], ...] | None:
        """Get the position of the ghost piece."""
        return self.__ghost_piece_position

    @property
    def next_piece(self) -> Piece | None:
        """Get the next piece."""
        return self.__next_piece

    @property
    def stored_piece(self) -> Piece | None:
        """Get the stored piece."""
        return self.__stored_piece

    @property
    def ticks_since_last_move_down(self) -> int:
        """Get the number of ticks since the last move down."""
        return self.__ticks_since_last_move_down

    def move_piece(self) -> None:
        """Move the current piece."""
        if not self.__playing:
            self.__playing = True

        if self.__game_over or self.__paused:
            return

        direction = self.__direction.get_obj()
        with self.__direction:
            self.__direction.set_obj(Direction.DOWN)

        if self.__current_piece is None:
            raise AloyGameInternalError(
                "Current piece was None when trying to move piece.")

        if self.__current_piece_position is None:
            raise AloyGameInternalError(
                "Current piece position was None when trying to move piece.")

        # Get theoretical new piece position
        new_piece_position = self.__current_piece_position
        new_piece_rotation = self.__current_piece_rotation
        if direction == Direction.LEFT:
            new_piece_position = tuple(
                (x_pos - 1, y_pos)
                for x_pos, y_pos in new_piece_position
            )
        if direction == Direction.RIGHT:
            new_piece_position = tuple(
                (x_pos + 1, y_pos)
                for x_pos, y_pos in new_piece_position
            )
        if direction == Direction.ROTATE_LEFT:
            new_piece_position, new_piece_rotation = _rotate_piece_left(
                self.__current_piece,
                new_piece_position,
                new_piece_rotation
            )
        if direction == Direction.ROTATE_RIGHT:
            new_piece_position, new_piece_rotation = _rotate_piece_right(
                self.__current_piece,
                new_piece_position,
                new_piece_rotation
            )
        if direction == Direction.FAST_DOWN:
            new_piece_position = tuple(
                (x_pos, y_pos + 1)
                for x_pos, y_pos in new_piece_position
            )
        if direction == Direction.DROP:
            while self.__can_move_piece_down(new_piece_position):
                new_piece_position = tuple(
                    (x_pos, y_pos + 1)
                    for x_pos, y_pos in new_piece_position
                )

        # Check if the piece was forced down
        forced_down: bool = False
        if direction not in (Direction.FAST_DOWN, Direction.DROP):
            if self.__ticks_since_last_move_down < _TICKS_PER_MOVE_DOWN:
                self.__ticks_since_last_move_down += 1
            else:
                new_piece_position = tuple(
                    (x_pos, y_pos + 1)
                    for x_pos, y_pos in new_piece_position
                )
                self.__ticks_since_last_move_down = 0
                forced_down = True
        else:
            self.__ticks_since_last_move_down = 0

        # Check if the new piece position is valid
        if (direction == Direction.DROP
                or self.__can_move_piece(new_piece_position)):
            self.__current_piece_position = new_piece_position
            self.__current_piece_rotation = new_piece_rotation
        elif direction in (Direction.ROTATE_LEFT, Direction.ROTATE_RIGHT):
            # Try moving the piece left
            new_piece_position = tuple(
                (x_pos - 1, y_pos)
                for x_pos, y_pos in new_piece_position
            )
            if self.__can_move_piece(new_piece_position):
                self.__current_piece_position = new_piece_position
                self.__current_piece_rotation = new_piece_rotation

        # Check if the piece has landed
        if (direction == Direction.DROP
            or ((direction == Direction.FAST_DOWN
                 or forced_down)
                and not self.__can_move_piece_down(
                    self.__current_piece_position))):
            self.__land_piece()

        # Set the ghost piece position
        self.__set_ghost_piece_position()

    def store_piece(self) -> None:
        """Store the current piece."""
        if (not self.allow_store_piece
                or not self.__playing
                or self.__game_over
                or self.__paused):
            return

        if self.__stored_piece is None:
            self.__stored_piece = self.__current_piece
            self.__current_piece = self.__next_piece
            self.__next_piece = self.__get_random_piece()
        else:
            self.__stored_piece, self.__current_piece \
                = self.__current_piece, self.__stored_piece

        self.__set_initial_piece_position_and_rotation()

    def __can_move_piece(
        self,
        piece_position: tuple[tuple[int, int], ...]
    ) -> bool:
        """Check if the piece can move to the given position."""
        for x_pos, y_pos in piece_position:
            if (x_pos < 0
                    or x_pos >= self.__board_size[0]
                    or y_pos < 0
                    or y_pos >= self.__board_size[1]):
                return False
            if self.__board_cells[y_pos][x_pos] != PieceColor.WHITE:
                return False
        return True

    def __can_move_piece_down(
        self,
        piece_position: tuple[tuple[int, int], ...]
    ) -> bool:
        """Check if the piece can move down."""
        return self.__can_move_piece(tuple(
            (x_pos, y_pos + 1)
            for x_pos, y_pos in piece_position
        ))

    def __land_piece(self) -> None:
        """Land the current piece."""
        if self.__current_piece is None:
            raise AloyGameInternalError(
                "Current piece was None when trying to land piece.")

        if self.__current_piece_position is None:
            raise AloyGameInternalError(
                "Current piece position was None when trying to land piece.")

        # Fix the current piece position on the board.
        for x_pos, y_pos in self.__current_piece_position:
            color = PIECE_COLOURS[self.__current_piece]
            self.__board_cells[y_pos][x_pos] = color

        # Set the current piece to the next piece and get a new next piece.
        self.__current_piece = self.__next_piece
        self.__set_initial_piece_position_and_rotation()
        self.__next_piece = self.__get_random_piece()

        # Clear any lines that have been filled.
        self.__clear_lines()

        # Check if the game is over, i.e. the new piece cannot be placed.
        if not self.__can_move_piece(self.__current_piece_position):
            self.__game_over = True

    def __clear_lines(self) -> None:
        """
        Clear any lines that have been filled.

        Update the score, level, lines filled, and pieces used.
        """
        if self.__current_piece is None:
            raise AloyGameInternalError(
                "Current piece was None when trying to land piece.")

        if self.__current_piece_position is None:
            raise AloyGameInternalError(
                "Current piece position was None when trying to land piece.")

        lines_filled = 0
        for index, row in enumerate(self.__board_cells):
            if all(cell != PieceColor.WHITE for cell in row):
                lines_filled += 1
                self.__board_cells.pop(index)
                self.__board_cells.insert(
                    0,
                    [PieceColor.WHITE] * self.__board_size[0]
                )

        if lines_filled > 0:
            self.__lines_filled["Total"] += lines_filled
            self.__lines_filled[_NUM_LINES_NAME[lines_filled]] += 1
            self.__score += _NUM_LINES_SCORE[lines_filled] * self.__level
            self.__level = (self.__score // _SCORE_PER_LEVEL) + _INITIAL_LEVEL
            self.__seconds_per_move = max(
                self.__seconds_per_move - _SECONDS_PER_MOVE_DECREASE_PER_LEVEL,
                _MIN_SECONDS_PER_MOVE
            )

        self.__pieces_used[self.__current_piece] += 1

    def __set_initial_piece_position_and_rotation(self) -> None:
        """Set the initial position and rotation of the current piece."""
        if self.__current_piece is None:
            raise AloyGameInternalError(
                "Current piece was None when trying to set initial piece "
                "position and rotation.")
        self.__current_piece_position = tuple(
            (x_pos + (self.__board_size[0] // 2) - 1, y_pos)
            for x_pos, y_pos in PIECE_SHAPES[self.__current_piece][0]
        )
        self.__current_piece_rotation = 0

    def __set_ghost_piece_position(self) -> None:
        """Set the ghost piece position."""
        if self.ghost_piece_enabled:
            if self.__current_piece_position is None:
                raise AloyGameInternalError(
                    "Current piece position was None when trying to set ghost "
                    "piece position.")
            ghost_piece_position = self.__current_piece_position
            while self.__can_move_piece_down(ghost_piece_position):
                ghost_piece_position = tuple(
                    (x_pos, y_pos + 1)
                    for x_pos, y_pos in ghost_piece_position
                )
            self.__ghost_piece_position = ghost_piece_position
        elif self.__ghost_piece_position is not None:
            self.__ghost_piece_position = None

    def __get_random_piece(self) -> Piece:
        """Get a random piece."""
        return random.choice(list(Piece))

    def _reset_game_state(self) -> None:
        """Reset the game state."""
        with self.__direction:
            self.__direction.set_obj(Direction.DOWN)
        self.__board_cells = [
            [PieceColor.WHITE] * self.__board_size[0]
            for _ in range(self.__board_size[1])
        ]
        self.__score = _INITIAL_SCORE
        self.__level = _INITIAL_LEVEL
        self.__lines_filled = _INITIAL_LINES_FILLED.copy()
        self.__pieces_used = _INITIAL_PIECES_USED.copy()
        self.__seconds_per_move = _INITIAL_SECONDS_PER_MOVE
        self.__game_over = False
        self.__paused = False
        self.__current_piece = self.__get_random_piece()
        self.__set_initial_piece_position_and_rotation()
        self.__set_ghost_piece_position()
        self.__next_piece = self.__get_random_piece()
        self.__stored_piece = None
        self.__ticks_since_last_move_down = 0

    def restart(self) -> None:
        """Restart the game."""
        self._reset_game_state()


class TetrisGameAloyWidget(AloyWidget):
    """
    A class to represent the tetris game on a Aloy widget.
    """

    @overload
    def __init__(
        self,
        parent: QtWidgets.QWidget,
        widget_size: tuple[int, int],
        board_size: tuple[int, int], *,
        manual_update: bool = False,
        debug: bool = False
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        parent: QtWidgets.QWidget,
        widget_size: tuple[int, int],
        tetris_game_logic: TetrisGameLogic, *,
        manual_update: bool = False,
        debug: bool = False
    ) -> None:
        ...

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        widget_size: tuple[int, int],
        board_size: tuple[int, int] | None = None,
        tetris_game_logic: TetrisGameLogic | None = None, *,
        manual_update: bool = False,
        debug: bool = False
    ) -> None:
        """Create a new tetris game widget."""
        super().__init__(
            parent,
            name="Tetris Game",
            size=widget_size,
            set_size="hint",
            size_policy=(
                QtWidgets.QSizePolicy.Policy.Preferred,
                QtWidgets.QSizePolicy.Policy.Preferred
            ),
            debug=debug
        )

        # Set up the game logic
        self._logic: TetrisGameLogic
        if tetris_game_logic is None:
            if board_size is None:
                raise ValueError("Board size must be given if game logic is "
                                 "not given.")
            self._logic = TetrisGameLogic(board_size)
        else:
            self._logic = tetris_game_logic
            if board_size is not None and board_size != self._logic.board_size:
                raise ValueError("Board size must be the same as the game "
                                 "logic board size.")
            else:
                board_size = self._logic.board_size

        # Check the widget size is valid
        if (widget_size[0] < board_size[0]
                or widget_size[1] < board_size[1]):
            raise ValueError("Widget size must be greater than board size")
        if (widget_size[0] % board_size[0] != 0
                or widget_size[1] % board_size[1] != 0):
            raise ValueError("Widget size must be divisible by board size")

        # Size of an individual cell
        _spacing: int = 5
        _width_size_ratio: float = (1.0 - (3 / (board_size[0] + 3)))
        _width: int = int((widget_size[0] - _spacing) * _width_size_ratio)
        _height: int = widget_size[1]
        self.__cell_size: tuple[int, int] = (
            _width // board_size[0],
            _height // board_size[1]
        )

        # Set up the timer to update the game
        self.__manual_update: bool = manual_update
        self.__timer = QtCore.QTimer()
        if not manual_update:
            self.__timer.setInterval(100)
            self.__timer.timeout.connect(self.__update_game)

        # Widget and layout
        self.__layout = QtWidgets.QGridLayout()
        self.__layout.setContentsMargins(0, 0, 0, 0)
        self.__layout.setSpacing(_spacing)
        self.qwidget.setLayout(self.__layout)
        self.qwidget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.qwidget.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        side_bar_width: int = int(
            (widget_size[0] - _spacing)
            * (1.0 - _width_size_ratio)
        )

        # Next and stored piece widgets
        self.__next_piece_widget = TetrisPieceWidget(
            "Next Piece",
            self.__cell_size
        )
        self.__stored_piece_widget = TetrisPieceWidget(
            "Stored Piece",
            self.__cell_size
        )
        self.__next_piece_widget.setFixedWidth(side_bar_width)
        self.__stored_piece_widget.setFixedWidth(side_bar_width)
        self.__next_piece_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.__stored_piece_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.__layout.addWidget(self.__next_piece_widget, 0, 0, 1, 1)
        self.__layout.addWidget(self.__stored_piece_widget, 1, 0, 1, 1)

        # Statistics widgets
        self.__statistics_group_box = QtWidgets.QGroupBox("Statistics")
        self.__statistics_group_box.setFixedWidth(side_bar_width)
        self.__statistics_group_box.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.__statistics_layout = QtWidgets.QFormLayout()
        self.__statistics_layout.setContentsMargins(0, 0, 0, 0)
        self.__statistics_layout.setHorizontalSpacing(20)
        self.__statistics_layout.setVerticalSpacing(0)
        self.__statistics_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        self.__statistics_layout.setRowWrapPolicy(
            QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows
        )
        self.__statistics_layout.setLabelAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft
        )
        self.__statistics_layout.setFormAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft
        )
        self.__score_value_label = QtWidgets.QLabel("0")
        self.__statistics_layout.addRow("Score:", self.__score_value_label)
        self.__level_value_label = QtWidgets.QLabel("1")
        self.__statistics_layout.addRow("Level:", self.__level_value_label)
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.__statistics_layout.addRow(separator)
        self.__statistics_layout.addRow(QtWidgets.QLabel("Lines Cleared:"))
        self.__lines_cleared_value_labels: dict[
            str, QtWidgets.QLabel] = {}
        for lines_type, num_lines in self._logic.lines_filled.items():
            lines_type_value_label = QtWidgets.QLabel(str(num_lines))
            self.__lines_cleared_value_labels[lines_type] = (
                lines_type_value_label)
            self.__statistics_layout.addRow(
                f"{lines_type}:",
                lines_type_value_label
            )
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.__statistics_layout.addRow(separator)
        self.__statistics_layout.addRow(QtWidgets.QLabel("Pieces Used:"))
        self.__pieces_used_value_labels: dict[
            str, QtWidgets.QLabel] = {}
        for piece_type, num_pieces in self._logic.pieces_used.items():
            if isinstance(piece_type, Piece):
                piece_type = piece_type.value
            piece_type_value_label = QtWidgets.QLabel(str(num_pieces))
            self.__pieces_used_value_labels[piece_type] = (
                piece_type_value_label)
            self.__statistics_layout.addRow(
                f"{piece_type} shape:" if piece_type != "Total" else "Total:",
                piece_type_value_label
            )
        self.__statistics_group_box.setLayout(self.__statistics_layout)
        self.__layout.addWidget(self.__statistics_group_box, 2, 0, 1, 1)

        # Controls widget
        self.__controls_group_box = QtWidgets.QGroupBox("Controls")
        self.__controls_group_box.setFixedWidth(side_bar_width)
        self.__controls_group_box.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.__controls_layout = QtWidgets.QFormLayout()
        self.__controls_layout.setContentsMargins(0, 0, 0, 0)
        self.__controls_layout.setVerticalSpacing(0)
        self.__controls_layout.setHorizontalSpacing(45)
        self.__controls_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        self.__controls_layout.setRowWrapPolicy(
            QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows
        )
        self.__controls_layout.setLabelAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft
        )
        self.__controls_layout.setFormAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft
        )
        self.__controls_layout.addRow(
            QtWidgets.QLabel("Move Left:"),
            QtWidgets.QLabel("A")
        )
        self.__controls_layout.addRow(
            QtWidgets.QLabel("Move Right:"),
            QtWidgets.QLabel("D")
        )
        self.__controls_layout.addRow(
            QtWidgets.QLabel("Fast Down:"),
            QtWidgets.QLabel("S")
        )
        self.__controls_layout.addRow(
            QtWidgets.QLabel("Rotate Left:"),
            QtWidgets.QLabel("Q")
        )
        self.__controls_layout.addRow(
            QtWidgets.QLabel("Rotate Right:"),
            QtWidgets.QLabel("E")
        )
        self.__controls_layout.addRow(
            QtWidgets.QLabel("Drop Piece:"),
            QtWidgets.QLabel("Space")
        )
        self.__controls_layout.addRow(
            QtWidgets.QLabel("Store Piece:"),
            QtWidgets.QLabel("Shift")
        )
        self.__controls_layout.addRow(
            QtWidgets.QLabel("Pause:"),
            QtWidgets.QLabel("P")
        )
        self.__controls_layout.addRow(
            QtWidgets.QLabel("Restart:"),
            QtWidgets.QLabel("R")
        )
        self.__controls_group_box.setLayout(self.__controls_layout)
        self.__layout.addWidget(self.__controls_group_box, 3, 0, 1, 1)

        # Options widget
        self.__options_widget = QtWidgets.QWidget()
        self.__options_widget.setFixedWidth(side_bar_width)
        self.__options_layout = QtWidgets.QFormLayout()
        self.__options_layout.setContentsMargins(10, 10, 10, 10)
        self.__options_layout.setSpacing(10)
        self.__options_widget.setLayout(self.__options_layout)
        self.__options_layout.addRow(
            QtWidgets.QLabel("Game Options")
        )

        # Add speed slider
        self.__speed_slider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal)
        self.__speed_slider.setRange(0, 10)
        self.__speed_slider.setValue(5)
        self.__speed_slider.setTickPosition(
            QtWidgets.QSlider.TickPosition.TicksBelow
        )
        self.__speed_slider.setTickInterval(1)
        def speed_slider_changed(value: int):
            self._logic.speed = (
                _DEFAULT_SPEED
                + (_MAX_SPEED - _MIN_SPEED)
                * ((value - 1) / 10)
            )
        self.__speed_slider.valueChanged.connect(speed_slider_changed)
        self.__options_layout.addRow("Speed:", self.__speed_slider)

        # Add ghost piece toggle
        self.__ghost_toggle = QtWidgets.QCheckBox("Enable Ghost Piece")
        self.__ghost_toggle.setChecked(_DEFAULT_GHOST_PIECE_ENABLED)
        def ghost_toggle_changed(checked: bool):
            self._logic.ghost_piece_enabled = checked
        self.__ghost_toggle.stateChanged.connect(ghost_toggle_changed)
        self.__options_layout.addRow(self.__ghost_toggle)

        # Add store piece toggle
        self.__store_toggle = QtWidgets.QCheckBox("Allow Store Piece")
        self.__store_toggle.setChecked(_DEFAULT_ALLOW_STORE_PIECE)
        def store_toggle_changed(checked: bool):
            self._logic.allow_store_piece = checked
        self.__store_toggle.stateChanged.connect(store_toggle_changed)
        self.__options_layout.addRow(self.__store_toggle)

        # Add grid toggle
        self.__grid_toggle = QtWidgets.QCheckBox("Show Grid")
        self.__grid_toggle.setChecked(_DEFAULT_SHOW_GRID)
        def grid_toggle_changed(checked: bool):
            self._logic.show_grid = checked
        self.__grid_toggle.stateChanged.connect(grid_toggle_changed)
        self.__options_layout.addRow(self.__grid_toggle)

        # Add options widget to main layout
        self.__layout.addWidget(self.__options_widget, 4, 0, 1, 1)

        # Tetris grid display widget
        display_widget, graphics_scene = self.__create_display(
            (self.size.width - _spacing)  # type: ignore[union-attr]
            * _width_size_ratio,
            self.size.height  # type: ignore[union-attr]
        )
        self.__scene = graphics_scene
        self.__layout.addWidget(display_widget, 0, 1, 5, 1)

        # Set up the key press event
        event = self.__key_press_event
        self.qwidget.keyPressEvent = event  # type: ignore[assignment]

        self._logic.restart()
        if not manual_update:
            self.__timer.start()

    def __create_display(
        self,
        scene_width: int,
        scene_height: int
    ) -> tuple[QtWidgets.QWidget, QtWidgets.QGraphicsScene]:
        display_widget = QtWidgets.QWidget()
        display_widget.setStyleSheet("background-color: black;")
        display_widget.setFixedSize(scene_width, scene_height)
        display_layout = QtWidgets.QGridLayout()
        display_layout.setContentsMargins(0, 0, 0, 0)
        display_layout.setSpacing(0)
        display_widget.setLayout(display_layout)
        scene = QtWidgets.QGraphicsScene(0, 0, scene_width, scene_height)
        view = QtWidgets.QGraphicsView(scene)
        view.setStyleSheet("background-color: white;")
        view.setFixedSize(scene_width, scene_height)
        view.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        display_layout.addWidget(view, 0, 0, 1, 1)
        return display_widget, scene

    def update_observer(self, observable_: AloySystemData) -> None:
        """Update the obaerver."""
        pass

    def __key_press_event(self, event: QtGui.QKeyEvent) -> None:
        """Handle a key press event."""
        key = event.key()
        with self._logic.direction:
            if key == QtCore.Qt.Key.Key_A:
                self._logic.direction.set_obj(Direction.LEFT)
            elif key == QtCore.Qt.Key.Key_D:
                self._logic.direction.set_obj(Direction.RIGHT)
            elif key == QtCore.Qt.Key.Key_E:
                self._logic.direction.set_obj(Direction.ROTATE_RIGHT)
            elif key == QtCore.Qt.Key.Key_Q:
                self._logic.direction.set_obj(Direction.ROTATE_LEFT)
            elif key == QtCore.Qt.Key.Key_S:
                self._logic.direction.set_obj(Direction.FAST_DOWN)
            elif key == QtCore.Qt.Key.Key_Space:
                self._logic.direction.set_obj(Direction.DROP)
            elif key == QtCore.Qt.Key.Key_Shift:
                self._logic.store_piece()
            elif key == QtCore.Qt.Key.Key_P:
                self._logic.paused = not self._logic.paused
            elif key == QtCore.Qt.Key.Key_R:
                self._logic.restart()
            else:
                return
        self.__update_game(forced=True)

    def manual_update_game(self) -> None:
        pass

    def __update_game(self, forced: bool = False) -> None:
        """Update the game."""
        self._logic.move_piece()
        if forced:
            self.__timer.stop()
            self.__update_timer()
            self.__timer.start()
        else:
            self.__update_timer()
        self.__draw_all()
        self.__update_statistics()

    def __update_timer(self) -> None:
        """Update the timer."""
        self.__timer.setInterval(
            int((1000 * self._logic.seconds_per_move) / self._logic.speed))

    def __draw_all(self) -> None:
        """Draw all the pieces."""
        self.__scene.clear()
        self.__draw_grid()
        self.__draw_ghost_piece()
        self.__draw_piece()
        self.__next_piece_widget.draw_piece(
            self._logic.next_piece)  # type: ignore[arg-type]
        self.__stored_piece_widget.draw_piece(
            self._logic.stored_piece)  # type: ignore[arg-type]
        if self.debug:
            self.__draw_debug()
        if self._logic.game_over:
            self.__draw_game_over()
        if self._logic.paused:
            self.__draw_paused()
        self.qwidget.update()

    def __draw_grid(self) -> None:
        """Draw the grid."""
        width, height = self._logic.board_size
        cells = self._logic.board_cells
        if self._logic.show_grid:
            pen = QtGui.QPen(QtGui.QColor("black"))
        else:
            pen = QtGui.QPen(QtGui.QColor("white"))
        for x_pos in range(width):
            for y_pos in range(height):
                cell = cells[y_pos][x_pos]
                self.__scene.addRect(
                    x_pos * self.__cell_size[0],
                    y_pos * self.__cell_size[1],
                    self.__cell_size[0],
                    self.__cell_size[1],
                    pen,
                    QtGui.QBrush(QtGui.QColor(cell.value))
                )

    def __draw_ghost_piece(self) -> None:
        """Draw the ghost piece."""
        piece = self._logic.current_piece
        piece_cells = self._logic.ghost_piece_position
        if piece_cells is not None:
            for x_pos, y_pos in piece_cells:
                self.__scene.addRect(
                    x_pos * self.__cell_size[0],
                    y_pos * self.__cell_size[1],
                    self.__cell_size[0],
                    self.__cell_size[1],
                    QtGui.QPen(QtGui.QColor("black")),
                    QtGui.QBrush(
                        QtGui.QColor(
                            PIECE_COLOURS[piece].value  # type: ignore[index]
                        ).lighter(150)
                    )
                )

    def __draw_piece(self) -> None:
        """Draw the current piece."""
        piece = self._logic.current_piece
        piece_cells = self._logic.current_piece_position
        if piece_cells is not None:
            for x_pos, y_pos in piece_cells:
                self.__scene.addRect(
                    x_pos * self.__cell_size[0],
                    y_pos * self.__cell_size[1],
                    self.__cell_size[0],
                    self.__cell_size[1],
                    QtGui.QPen(QtGui.QColor("black")),
                    QtGui.QBrush(
                        QtGui.QColor(
                            PIECE_COLOURS[piece].value  # type: ignore[index]
                        )
                    )
                )

    def __draw_debug(self) -> None:
        """Draw debug information."""
        self.__scene.addText(
            f"Direction: {self._logic.direction.get_obj()}",
            QtGui.QFont("Arial", 12)
        ).setPos(0, 0)
        self.__scene.addText(
            f"Current piece: {self._logic.current_piece}",
            QtGui.QFont("Arial", 12)
        ).setPos(0, 20)
        self.__scene.addText(
            f"Next piece: {self._logic.next_piece}",
            QtGui.QFont("Arial", 12)
        ).setPos(0, 40)
        self.__scene.addText(
            f"Stored piece: {self._logic.stored_piece}",
            QtGui.QFont("Arial", 12)
        ).setPos(0, 60)
        self.__scene.addText(
            f"Current piece position: {self._logic.current_piece_position}",
            QtGui.QFont("Arial", 12)
        ).setPos(0, 80)
        self.__scene.addText(
            f"Ghost piece position: {self._logic.ghost_piece_position}",
            QtGui.QFont("Arial", 12)
        ).setPos(0, 100)
        self.__scene.addText(
            f"Current piece rotation: {self._logic.current_piece_rotation}",
            QtGui.QFont("Arial", 12)
        ).setPos(0, 120)
        self.__scene.addText(
            f"Score: {self._logic.score}",
            QtGui.QFont("Arial", 12)
        ).setPos(0, 140)
        self.__scene.addText(
            f"Level: {self._logic.level}",
            QtGui.QFont("Arial", 12)
        ).setPos(0, 160)

    def __draw_game_over(self) -> None:
        """Draw the game over text."""
        text: QtWidgets.QGraphicsTextItem = self.__scene.addText(
            "Game Over",
            QtGui.QFont("Arial", 72)
        )
        text_size: QtCore.QSizeF = text.boundingRect().size()
        text.setPos(
            ((self._logic.board_size[0] * self.__cell_size[0]) // 2)
            - (text_size.width() // 2),
            ((self._logic.board_size[1] * self.__cell_size[1]) // 2)
            - (text_size.height() // 2)
        )

    def __draw_paused(self) -> None:
        """Draw the paused text."""
        text: QtWidgets.QGraphicsTextItem = self.__scene.addText(
            "Paused",
            QtGui.QFont("Arial", 72)
        )
        text_size: QtCore.QSizeF = text.boundingRect().size()
        text.setPos(
            ((self._logic.board_size[0] * self.__cell_size[0]) // 2)
            - (text_size.width() // 2),
            ((self._logic.board_size[1] * self.__cell_size[1]) // 2)
            - (text_size.height() // 2)
        )

    def __update_statistics(self) -> None:
        """Update the statistics."""
        self.__score_value_label.setText(str(self._logic.score))
        self.__level_value_label.setText(str(self._logic.level))
        for lines_type, num_lines in self._logic.lines_filled.items():
            self.__lines_cleared_value_labels[lines_type].setText(
                str(num_lines))
        for piece_type, num_pieces in self._logic.pieces_used.items():
            if isinstance(piece_type, Piece):
                piece_type = piece_type.value
            self.__pieces_used_value_labels[piece_type].setText(
                str(num_pieces))


class TetrisPieceWidget(QtWidgets.QWidget):
    """Widget to display a tetris piece."""

    def __init__(self, name: str, cell_size: tuple[int, int]) -> None:
        """Initialise the widget."""
        super().__init__()
        self.__layout = QtWidgets.QGridLayout()
        self.__layout.setContentsMargins(0, 0, 0, 0)
        self.__layout.setSpacing(0)
        self.setLayout(self.__layout)

        # Create the label to display the piece name
        self.__widget_name: str = name
        self.__name_label = QtWidgets.QLabel(name)
        self.__layout.addWidget(self.__name_label, 0, 0, 1, 1)

        # Draw piece on a 4x2 grid
        self.__cell_size = cell_size
        width, height = cell_size[0] * 4, cell_size[1] * 2

        # Create the scene and view to draw the piece on
        self.__scene = QtWidgets.QGraphicsScene(0, 0, width, height)
        self.__view = QtWidgets.QGraphicsView(self.__scene)
        self.__view.setStyleSheet("background-color: white;")
        self.__view.setFixedSize(width, height)
        self.__view.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.__view.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.__layout.addWidget(self.__view, 1, 0, 1, 1)

    def draw_piece(self, piece: Piece) -> None:
        """Draw the piece."""
        self.__scene.clear()
        if piece is None:
            self.update()
            return
        self.__name_label.setText(f"{self.__widget_name} ({piece.value})")
        piece_colour = PIECE_COLOURS[piece]
        piece_shape = PIECE_SHAPES[piece][1]
        cell_size = self.__cell_size
        for x_pos, y_pos in piece_shape:
            self.__scene.addRect(
                x_pos * cell_size[0],
                y_pos * cell_size[1],
                cell_size[0],
                cell_size[1],
                QtGui.QPen(QtGui.QColor("black")),
                QtGui.QBrush(QtGui.QColor(piece_colour.value))
            )
        self.update()


def play_tetris_game(
    width: int,
    height: int,
    ghost_piece_enabled: bool = _DEFAULT_GHOST_PIECE_ENABLED,
    allow_store_piece: bool = _DEFAULT_ALLOW_STORE_PIECE,
    debug: bool = False
) -> None:
    """Play the tetris game."""
    size = (width, height)

    qapp = QtWidgets.QApplication([])
    qtimer = QtCore.QTimer()

    aloy_data = AloySystemData(
        name="Tetris GUI Data",
        clock=qtimer,
        debug=debug
    )
    aloy_gui = AloyGuiWindow(
        qapp=qapp,
        data=aloy_data,
        name="Tetris GUI Window",
        size=size,
        kind=None,
        debug=debug
    )

    tetris_qwidget = QtWidgets.QWidget()
    tetris_game_logic = TetrisGameLogic(
        board_size=(10, 20)
    )
    tetris_game_logic.ghost_piece_enabled = ghost_piece_enabled
    tetris_game_logic.allow_store_piece = allow_store_piece

    tetris_game_aloy_widget = TetrisGameAloyWidget(
        parent=tetris_qwidget,
        widget_size=size,
        board_size=(10, 20),
        tetris_game_logic=tetris_game_logic,
        debug=debug
    )

    aloy_gui.add_view(tetris_game_aloy_widget)

    aloy_gui.qwindow.show()
    sys.exit(qapp.exec())
