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
from typing import TypeAlias
from concurrency.atomic import AtomicObject

from PySide6 import QtCore, QtGui, QtWidgets

from datastructures.mappings import frozendict
from guis.gui import JinxWidget

@enum.unique
class Piece(enum.Enum):
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

    Red = "red"
    Green = "green"
    Blue = "blue"
    Yellow = "yellow"
    Magenta = "magenta"
    Cyan = "cyan"
    Orange = "orange"
    White = "white"


PIECE_SHAPES = frozendict({
    Piece.ZShape: frozendict({
        0: ((0, 0), (0, 1), (1, 1), (1, 2)),
        1: ((0, 1), (1, 1), (1, 0), (2, 0)),
        2: ((0, 0), (0, 1), (1, 1), (1, 2)),
        3: ((0, 1), (1, 1), (1, 0), (2, 0))
    }),
    Piece.SShape: frozendict({
        0: ((0, 1), (0, 2), (1, 0), (1, 1)),
        1: ((0, 0), (1, 0), (1, 1), (2, 1)),
        2: ((0, 1), (0, 2), (1, 0), (1, 1)),
        3: ((0, 0), (1, 0), (1, 1), (2, 1))
    }),
    Piece.LineShape: frozendict({
        0: ((0, 0), (0, 1), (0, 2), (0, 3)),
        1: ((0, 0), (1, 0), (2, 0), (3, 0)),
        2: ((0, 0), (0, 1), (0, 2), (0, 3)),
        3: ((0, 0), (1, 0), (2, 0), (3, 0))
    }),
    Piece.TShape: frozendict({
        0: ((0, 0), (0, 1), (0, 2), (1, 1)),
        1: ((0, 1), (1, 1), (2, 1), (1, 0)),
        2: ((0, 1), (1, 0), (1, 1), (1, 2)),
        3: ((0, 1), (1, 1), (2, 1), (1, 2))
    }),
    Piece.SquareShape: frozendict({
        0: ((0, 0), (0, 1), (1, 0), (1, 1)),
        1: ((0, 0), (0, 1), (1, 0), (1, 1)),
        2: ((0, 0), (0, 1), (1, 0), (1, 1)),
        3: ((0, 0), (0, 1), (1, 0), (1, 1))
    }),
    Piece.LShape: frozendict({
        0: ((0, 0), (0, 1), (0, 2), (1, 2)),
        1: ((0, 1), (1, 1), (2, 1), (2, 0)),
        2: ((0, 0), (1, 0), (1, 1), (1, 2)),
        3: ((0, 0), (0, 1), (1, 0), (2, 0))
    }),
    Piece.MirroredLShape: frozendict({
        0: ((0, 2), (1, 0), (1, 1), (1, 2)),
        1: ((0, 0), (0, 1), (1, 1), (2, 1)),
        2: ((0, 0), (0, 1), (0, 2), (1, 0)),
        3: ((0, 0), (1, 0), (2, 0), (2, 1))
    })
})


PIECE_COLOURS = frozendict({
    Piece.ZShape: PieceColor.Red,
    Piece.SShape: PieceColor.Green,
    Piece.LineShape: PieceColor.Blue,
    Piece.TShape: PieceColor.Yellow,
    Piece.SquareShape: PieceColor.Magenta,
    Piece.LShape: PieceColor.Cyan,
    Piece.MirroredLShape: PieceColor.Orange
})


def _rotate_piece_left(
    piece_position: tuple[tuple[int, int], ...],
    piece: Piece,
    rotation: int
) -> tuple[tuple[tuple[int, int], ...], int]:
    """
    Rotate the piece to the left.

    Return the new piece position and rotation.
    """
    new_rotation = (rotation - 1) % 4
    return (
        _rotate_piece(piece_position, piece, rotation, new_rotation),
        new_rotation
    )


def _rotate_piece_right(
    piece_position: tuple[tuple[int, int], ...],
    piece: Piece,
    rotation: int
) -> tuple[tuple[tuple[int, int], ...], int]:
    """
    Rotate the piece to the right.

    Return the new piece position and rotation.
    """
    new_rotation = (rotation + 1) % 4
    return (
        _rotate_piece(piece_position, piece, rotation, new_rotation),
        new_rotation
    )


def _rotate_piece(
    piece_position: tuple[tuple[int, int], ...],
    piece: Piece,
    rotation: int,
    new_rotation: int
) -> tuple[tuple[int, int], ...]:
    """Rotate the piece to the new rotation."""
    new_piece_position = list(piece_position)
    for index, section in enumerate(piece_position):
        new_piece_position[index] = (
            section[0] - PIECE_SHAPES[piece][rotation][index][0]
            + PIECE_SHAPES[piece][new_rotation][index][0],
            section[1] - PIECE_SHAPES[piece][rotation][index][1]
            + PIECE_SHAPES[piece][new_rotation][index][1]
        )
    return tuple(new_piece_position)


@enum.unique
class Direction(enum.Enum):
    """Direction of the piece."""

    Left = enum.auto()
    Right = enum.auto()
    RotateLeft = enum.auto()
    RotateRight = enum.auto()
    Down = enum.auto()
    FastDown = enum.auto()
    Drop = enum.auto()
    StorePiece = enum.auto()


_INITIAL_DIRECTION: Direction = Direction.Down

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
        "__game_over",
        "__paused",
        "__current_piece_position",
        "__current_piece_rotation",
        "__current_piece",
        "__next_piece",
        "__stored_piece",
        "__ticks_since_last_move_down",
        "__manual_update",

        # Game options
        "speed",
        "ghost_piece",
        "allow_store_piece",
        "show_next_piece",
        "show_stored_piece",
        "show_grid"
    )

    def __init__(
        self,
        cells_grid_size: tuple[int, int],
        manual_udpate: bool = False
    ) -> None:
        """Initialize the tetris game logic."""
        if not isinstance(cells_grid_size, tuple):
            raise TypeError("The grid size must be a tuple. "
                            f"Got; {type(cells_grid_size)!r}.")
        if len(cells_grid_size) != 2:
            raise ValueError("The grid size must be a tuple of length 2. "
                             f"Got; {len(cells_grid_size)}.")
        if any(size <= 0 for size in cells_grid_size):
            raise ValueError("The grid size must be positive. "
                             f"Got; {cells_grid_size}.")

        # Actions
        self.__direction: AtomicObject[Direction]
        self.__direction = AtomicObject(_INITIAL_DIRECTION)

        # Game state
        self.__playing = False
        self.__board_size = cells_grid_size
        self.__board_cells: list[list[PieceColor]] = [
            [PieceColor.White for _ in range(cells_grid_size[0])]
            for _ in range(cells_grid_size[1])
        ]
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
            for piece_shape in Piece
        })
        self.__game_over: bool = False
        self.__paused: bool = False
        self.__current_piece_position: tuple[tuple[int, int], ...] | None \
            = None
        self.__current_piece_rotation: int | None = None
        self.__current_piece: Piece | None = None
        self.__next_piece: Piece | None = None
        self.__stored_piece: Piece | None = None
        self.__ticks_since_last_move_down: int = 0
        self.__manual_update: bool = manual_udpate

        # Game options
        self.speed: int = 1
        self.ghost_piece: bool = True
        self.allow_store_piece: bool = True
        self.show_next_piece: bool = True
        self.show_stored_piece: bool = True
        self.show_grid: bool = True

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
    def pieces_used(self) -> dict[str, int]:
        """Get the number of pieces used."""
        return self.__pieces_used

    def move_piece(self) -> None:
        """Move the current piece."""
        if not self.__playing:
            self.__playing = True

        if self.__game_over or self.__paused:
            return

        direction = self.__direction.get_obj()

        # Get theoretical new piece position
        new_piece_position = self.__current_piece_position
        new_piece_rotation = self.__current_piece_rotation
        if direction == Direction.Left:
            new_piece_position = tuple(
                (x - 1, y)
                for x, y in self.__current_piece_position
            )
        elif direction == Direction.Right:
            new_piece_position = tuple(
                (x + 1, y)
                for x, y in self.__current_piece_position
            )
        elif direction == Direction.RotateLeft:
            new_piece_position, new_piece_rotation = _rotate_piece_left(
                self.__current_piece,
                self.__current_piece_position,
                self.__current_piece_rotation
            )
        elif direction == Direction.RotateRight:
            new_piece_position, new_piece_rotation = _rotate_piece_right(
                self.__current_piece,
                self.__current_piece_position,
                self.__current_piece_rotation
            )
        elif direction == Direction.Down:
            if self.__ticks_since_last_move_down < _TICKS_PER_MOVE_DOWN:
                self.__ticks_since_last_move_down += 1
                return
            new_piece_position = tuple(
                (x, y + 1)
                for x, y in self.__current_piece_position
            )
            self.__ticks_since_last_move_down = 0
        elif direction == Direction.FastDown:
            new_piece_position = tuple(
                (x, y + 1)
                for x, y in self.__current_piece_position
            )
            self.__ticks_since_last_move_down = 0
        elif direction == Direction.Drop:
            new_piece_position = self.__current_piece_position
            while self.__can_move_piece_down(new_piece_position):
                new_piece_position = tuple(
                    (x, y + 1)
                    for x, y in new_piece_position
                )
            self.__ticks_since_last_move_down = 0
        else:
            raise ValueError(f"Invalid direction: {direction}")

        # Check if the new piece position is valid
        if self.__can_move_piece(new_piece_position):
            self.__current_piece_position = new_piece_position
            self.__current_piece_rotation = new_piece_rotation
            with self.__direction:
                self.__direction.set_obj(Direction.Down)

        # Check if the piece has landed
        if (direction in (Direction.Down, Direction.FastDown, Direction.Drop)
                and not self.__can_move_piece_down(self.__current_piece_position)):
            self.__land_piece()

    def store_piece(self) -> None:
        """Store the current piece."""
        if not self.allow_store_piece:
            return

        if self.__stored_piece is None:
            self.__stored_piece = self.__current_piece
            self.__current_piece = self.__next_piece
            self.__next_piece = self.__get_random_piece()
        else:
            self.__stored_piece, self.__current_piece \
                = self.__current_piece, self.__stored_piece

        self.__current_piece_position = self.__get_initial_piece_position()
        self.__current_piece_rotation = 0

    def __can_move_piece(self, piece_position: tuple[tuple[int, int], ...]) -> bool:
        """Check if the piece can move to the given position."""
        for x, y in piece_position:
            if x < 0 or x >= self.__board_size[0] or y < 0 or y >= self.__board_size[1]:
                return False
            if self.__board_cells[y][x] != PieceColor.White:
                return False
        return True

    def __can_move_piece_down(self, piece_position: tuple[tuple[int, int], ...]) -> bool:
        """Check if the piece can move down."""
        return self.__can_move_piece(tuple(
            (x, y + 1)
            for x, y in piece_position
        ))

    def __land_piece(self) -> None:
        """Land the current piece."""
        for x_pos, y_pos in self.__current_piece_position:
            self.__board_cells[y_pos][x_pos] = PIECE_COLOURS[self.__current_piece]

        self.__clear_lines()

        if not self.__can_move_piece(self.__current_piece_position):
            self.__game_over = True

        self.__current_piece = self.__next_piece
        self.__next_piece = self.__get_random_piece()

    def __clear_lines(self) -> None:
        """
        Clear any lines that have been filled.

        Updat the score, level, lines filled, and pieces used.
        """
        lines_filled = 0
        for index, row in enumerate(self.__board_cells):
            if all(cell != PieceColor.White for cell in row):
                lines_filled += 1
                self.__board_cells.pop(index)
                self.__board_cells.insert(
                    0,
                    [PieceColor.White] * self.__board_size[0]
                )

        self.__lines_filled["Total"] += lines_filled
        self.__lines_filled[_NUM_LINES_NAME[lines_filled]] += 1

        self.__score += _NUM_LINES_SCORE[lines_filled] * self.__level
        self.__level = self.__score // _SCORE_PER_LEVEL

        self.__pieces_used[self.__current_piece] += 1

    def __get_initial_piece_position(self) -> tuple[tuple[int, int], ...]:
        """Get the initial position of the current piece."""
        return tuple(
            (x + (self.__board_size[0] // 2) - 1, y)
            for x, y in PIECE_SHAPES[self.__current_piece][0]
        )

    def __get_random_piece(self) -> Piece:
        """Get a random piece."""
        return random.choice(list(Piece))


class TetrisGameJinxWidget(JinxWidget):
    """
    A class to represent the tetris game on a Jinx widget.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        widget_size: tuple[int, int],
        board_size: tuple[int, int],
        tetris_game_logic: TetrisGameLogic | None = None,
        manual_update: bool = False,
        debug: bool = False
    ) -> None:
        """Create a new tetris game widget."""
        super().__init__(
            parent,
            name="Tetris Game",
            size=widget_size,
            debug=debug
        )
        if (board_size[0] <= 0
                or board_size[1] <= 0):
            raise ValueError("Board size must be positive")
        if (widget_size[0] < board_size[0]
                or widget_size[1] < board_size[1]):
            raise ValueError("Widget size must be greater than board size")
        if (widget_size[0] % board_size[0] != 0
                or widget_size[1] % board_size[1] != 0):
            raise ValueError("Widget size must be divisible by board size")

        # Size of an individual cell
        self.__cell_size: tuple[int, int] = (
            widget_size[0] // board_size[0],
            widget_size[1] // board_size[1]
        )

        # Set up the game logic
        self._logic: TetrisGameLogic
        if tetris_game_logic is None:
            self._logic = TetrisGameLogic(board_size)
        else:
            self._logic = tetris_game_logic
            self._logic.board_size = board_size

        # Set up the timer to update the game
        self.__manual_update: bool = manual_update
        self.__timer = QtCore.QTimer()
        if not manual_update:
            self.__timer.setInterval(100)
            self.__timer.timeout.connect(self.__update_game)

        # Widget and layout
        self.__layout = QtWidgets.QGridLayout()
        self.__layout.setContentsMargins(0, 0, 0, 0)
        self.__layout.setSpacing(0)
        self.qwidget.setLayout(self.__layout)
        self.qwidget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.qwidget.setStyleSheet("background-color: black;")
        self.qwidget.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        # Next and stored piece widgets
        self.__next_piece_widget = TetrisPieceWidget(self.__cell_size)
        self.__stored_piece_widget = TetrisPieceWidget(self.__cell_size)
        self.__next_piece_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.__stored_piece_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.__layout.addWidget(self.__next_piece_widget, 0, 0, 1, 1)
        self.__layout.addWidget(self.__stored_piece_widget, 1, 0, 1, 1)

        # Control widget
        self.__controls_widget = QtWidgets.QWidget()
        self.__controls_group_box = QtWidgets.QGroupBox("Controls")
        self.__controls_layout = QtWidgets.QFormLayout()
        self.__controls_layout.setContentsMargins(0, 0, 0, 0)
        self.__controls_layout.setSpacing(0)
        self.__controls_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        self.__controls_layout.setRowWrapPolicy(
            QtWidgets.QFormLayout.RowWrapPolicy.WrapAllRows
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
            QtWidgets.QLabel("Move Down:"),
            QtWidgets.QLabel("W")
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
        self.__controls_group_box.setLayout(self.__controls_layout)
        self.__controls_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.__layout.addWidget(self.__controls_group_box, 2, 0, 2, 1)

        # Tetris grid widget
        width, height = self.size
        self.__display_widget = QtWidgets.QWidget()
        self.__display_widget.setStyleSheet("background-color: black;")
        self.__display_widget.setFixedSize(width, height)
        self.__display_layout = QtWidgets.QGridLayout()
        self.__display_layout.setContentsMargins(0, 0, 0, 0)
        self.__display_layout.setSpacing(0)
        self.__display_widget.setLayout(self.__display_layout)
        self.__scene = QtWidgets.QGraphicsScene(0, 0, width, height)
        self.__view = QtWidgets.QGraphicsView(self.__scene)
        self.__view.setStyleSheet("background-color: white;")
        self.__view.setFixedSize(width, height)
        self.__view.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.__view.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.__display_layout.addWidget(self.__view, 0, 0, 1, 1)
        self.__layout.addWidget(self.__display_widget, 0, 1, 3, 1)

        # Statistics widgets
        self.__statistics_widget = TetrisStatisticsWidget(
            self._logic,
            debug=debug
        )

        # Set up the key press event
        self.qwidget.keyPressEvent = self.__key_press_event

        self._logic.restart()
        if not manual_update:
            self.__timer.start()

    def __key_press_event(self, event: QtGui.QKeyEvent) -> None:
        """Handle a key press event."""
        key = event.key()
        with self._logic.direction:
            if key == QtCore.Qt.Key.Key_A:
                self._logic.direction.set_obj(Direction.Left)
            if key == QtCore.Qt.Key.Key_D:
                self._logic.direction.set_obj(Direction.Right)
            if key == QtCore.Qt.Key.Key_W:
                self._logic.direction.set_obj(Direction.Down)
            if key == QtCore.Qt.Key.Key_S:
                self._logic.direction.set_obj(Direction.FastDown)
            if key == QtCore.Qt.Key.Key_E:
                self._logic.direction.set_obj(Direction.RotateRight)
            if key == QtCore.Qt.Key.Key_Q:
                self._logic.direction.set_obj(Direction.RotateLeft)
            if key == QtCore.Qt.Key.Key_Space:
                self._logic.direction.set_obj(Direction.Drop)

        if key == QtCore.Qt.Key.Key_Shift:
            self._logic.store_piece()
        if key == QtCore.Qt.Key.Key_P:
            self._logic.paused = not self._logic.paused

    def manual_update_game(self) -> None:
        pass

    def __update_game(self) -> None:
        """Update the game."""
        self._logic.move_piece()
        self.__update_timer()
        self.__draw_all()

    def __update_timer(self) -> None:
        """Update the timer."""
        self.__timer.setInterval(
            int((1000 * self._logic.seconds_per_move) / self._logic.speed))

    def __draw_all(self) -> None:
        """Draw all the pieces."""
        self.__scene.clear()
        self.__draw_grid()
        self.__draw_piece()
        self.__draw_ghost_piece()
        self.__next_piece_widget.draw_piece()
        self.__stored_piece_widget.draw_piece()
        self.__statistics_widget.draw_statistics()
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
        for x in range(width):
            for y in range(height):
                cell = cells[x][y]
                self.__scene.addRect(
                    x * _CELL_SIZE,
                    y * _CELL_SIZE,
                    _CELL_SIZE,
                    _CELL_SIZE,
                    QtGui.QPen(QtGui.QColor("black")),
                    QtGui.QBrush(QtGui.QColor(cell.value))
                )

    def __draw_piece(self) -> None:
        """Draw the current piece."""
        piece = self._logic.current_piece
        piece_cells = self._logic.current_piece_position
        if piece_cells is not None:
            for x, y in piece_cells:
                self.__scene.addRect(
                    x * _CELL_SIZE,
                    y * _CELL_SIZE,
                    _CELL_SIZE,
                    _CELL_SIZE,
                    QtGui.QPen(QtGui.QColor("black")),
                    QtGui.QBrush(QtGui.QColor(PIECE_COLOURS[piece]))
                )

    def __draw_ghost_piece(self) -> None:
        """Draw the ghost piece."""
        piece = self._logic.current_piece
        piece_cells = self._logic.ghost_piece_position
        if piece_cells is not None:
            for x, y in piece_cells:
                self.__scene.addRect(
                    x * _CELL_SIZE,
                    y * _CELL_SIZE,
                    _CELL_SIZE,
                    _CELL_SIZE,
                    QtGui.QPen(QtGui.QColor("black")),
                    QtGui.QBrush(QtGui.QColor(PIECE_COLOURS[piece]).lighter(150))
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


class TetrisPieceWidget(QtWidgets.QWidget):
    """Widget to display a tetris piece."""

    def __init__(self, cell_size: tuple[int, int]) -> None:
        """Initialise the widget."""
        super().__init__()
        self.__layout = QtWidgets.QGridLayout()
        self.__layout.setContentsMargins(0, 0, 0, 0)
        self.__layout.setSpacing(0)
        self.setLayout(self.__layout)

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
        self.__layout.addWidget(self.__view, 0, 0, 1, 1)

    def draw_piece(self, piece: Piece) -> None:
        """Draw the piece."""
        piece_colour = PIECE_COLOURS[piece]
        piece_shape = PIECE_SHAPES[piece]
        cell_size = self.__cell_size
        for x, y in piece_shape:
            self.__scene.addRect(
                x * cell_size[0],
                y * cell_size[1],
                cell_size[0],
                cell_size[1],
                QtGui.QPen(QtGui.QColor("black")),
                QtGui.QBrush(QtGui.QColor(piece_colour))
            )
        self.update()
