###########################################################################
###########################################################################
## Module defining a snake game.                                         ##
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

"""Module defining a snake game."""

import random
from itertools import count
from math import copysign
from PyQt6 import QtWidgets
from PyQt6 import QtCore
from PyQt6 import QtGui
from concurrency.atomic import AtomicObject
from guis.gui import JinxGuiData, JinxGuiWindow, JinxObserverWidget

from moremath.vectors import (vector_add, vector_distance, vector_modulo,
                              vector_multiply)

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"

__all__ = (
    "SnakeGameLogic",
    "SnakeGameJinxWidget",
    "SnakeGameOptionsJinxWidget"
)


def __dir__() -> tuple[str, ...]:
    """Get the names of module attributes."""
    return __all__


# Game logic constants
_INITIAL_DIRECTION: tuple[int, int] = (1, 0)
_INITIAL_SCORE: int = 0
_INITIAL_SECONDS_PER_MOVE: float = 0.40
_DEFAULT_DIFFICULTY: str = "medium"
_DEFAULT_SHOW_PATH: bool = False
_DEFAULT_WALLS: bool = False
_DEFAULT_SPEED: float = 1.0
_DEFAULT_FOOD_PER_SNAKE_GROWTH: int = 1
_DEFAULT_INITIAL_SNAKE_LENGTH: int = 4


class SnakeGameLogic:
    """
    The internal logic of the snake game.
    """

    __slots__ = (
        "__weakref__",

        # Actions
        "_direction",

        # Game state
        "_grid_size",
        "_score",
        "_seconds_per_move",
        "_game_over",
        "_paused",
        "_snake",
        "_food",
        "_obstacles",

        # Game options
        "_difficulty",
        "_show_path",
        "_walls",
        "_speed",
        "_food_per_snake_growth",
        # "_food_time_limit",
        # "_food_per_level",
        "_initial_snake_length",
    )

    def __init__(
        self,
        cells_grid_size: tuple[int, int]
    ) -> None:
        """
        Create a new snake game logic object.

        Parameters
        ----------
        `cells_grid_size: tuple[int, int]` - The size of the grid in cells.
        This is the number of cells in the x and y directions.
        """
        if not isinstance(cells_grid_size, tuple):
            raise TypeError("The grid size must be a tuple. "
                            f"Got; {type(cells_grid_size)!r}.")
        if len(cells_grid_size) != 2:
            raise ValueError("The grid size must be a tuple of length 2. "
                             f"Got; {len(cells_grid_size)}.")
        if any(size <= 0 for size in cells_grid_size):
            raise ValueError("The grid size must be positive. "
                             f"Got; {cells_grid_size}.")

        ## Actions
        self._direction: AtomicObject[tuple[int, int]]
        self._direction = AtomicObject(_INITIAL_DIRECTION)

        ## Game state
        self._grid_size: tuple[int, int] = cells_grid_size
        self._score: int = _INITIAL_SCORE
        self._seconds_per_move: float = _INITIAL_SECONDS_PER_MOVE
        self._game_over: bool = False
        self._paused: bool = False
        self._snake: list[tuple[int, int]] = []
        self._food: tuple[int, int] | None = None
        self._obstacles: list[tuple[int, int]] = []

        ## Game options
        self._difficulty: str = _DEFAULT_DIFFICULTY
        self._show_path: bool = _DEFAULT_SHOW_PATH
        self._walls: bool = _DEFAULT_WALLS
        self._speed: float = _DEFAULT_SPEED
        # self._food_time_limit: float = 10.0
        # self._food_per_level: int = 1
        self._food_per_snake_growth: int = _DEFAULT_FOOD_PER_SNAKE_GROWTH
        # self._seconds_per_move_reduction_per_snake_growth: float = 0.01
        # self._min_seconds_per_move: float = 0.10
        self._initial_snake_length: int = _DEFAULT_INITIAL_SNAKE_LENGTH

    def _move_snake(self) -> None:
        """Move the snake in the current direction."""
        if self._game_over or self._paused:
            return

        with self._direction:
            direction = self._direction.get_object()

            ## Get the current head of the snake
            x, y = self._snake[0]

            ## Get the new head of the snake
            new_x = (x + direction[0]) % self._grid_size[0]
            new_y = (y + direction[1]) % self._grid_size[1]
            new_head = (new_x, new_y)

            ## Check if the snake has hit itself or an obstacle
            if new_head in self._snake or new_head in self._obstacles:
                self._game_over = True
                return

            ## Check if the snake has eaten the food
            if new_head == self._food:
                self._score += 1
                self._random_food()
                self._random_obstacles()
                if self._score % self._food_per_snake_growth == 0:
                    self._seconds_per_move = max(
                        0.1, self._seconds_per_move - 0.005
                    )
                else:
                    self._snake.pop()
            else:
                self._snake.pop()

            ## Add the new head to the snake
            self._snake.insert(0, new_head)

    def _random_start(self) -> None:
        """Start the snake at a random location."""
        valid: bool = False
        while not valid:
            ## Place the snake's head at a random location
            x = random.randint(0, self._grid_size[0] - 1)
            y = random.randint(0, self._grid_size[1] - 1)
            if (x, y) in self._obstacles or (x, y) == self._food:
                continue
            self._snake = [(x, y)]
            ## Randomly add segments to the snake's tail
            for _ in range(self._initial_snake_length - 1):
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                random.shuffle(directions)
                for direction in directions:
                    x += direction[0]
                    y += direction[1]
                    x = x % self._grid_size[0]
                    y = y % self._grid_size[1]
                    position = (x, y)
                    ## Do not put segments on top of each other
                    if (position not in self._snake
                            and position not in self._obstacles
                            and position != self._food):
                        self._snake.append(position)
                        break
                    else:
                        x, y = self._snake[-1]
            if len(self._snake) != self._initial_snake_length:
                continue
            ## Stop the snake from hitting itself immediately
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            random.shuffle(directions)
            for direction in directions:
                next_cell = tuple(vector_add(self._snake[0], direction))
                if next_cell not in self._snake:
                    with self._direction:
                        self._direction.set_object(direction)
                    valid = True
                    break

    def _random_food(self) -> None:
        """Place food at a random location."""
        limit: int = ((self._grid_size[0] * self._grid_size[1])
                      // _CELL_SIZE) ** 1.5
        for i in count(start=1):
            food = (
                random.randint(0, self._grid_size[0] - 1),
                random.randint(0, self._grid_size[1] - 1),
            )
            ## Food must not be in the snake,
            if (self._difficulty == "easy"
                    and food not in self._snake):
                self._food = food
                break
            ## and must not be in the snake or obstacles,
            elif (self._difficulty == "medium"
                    and food not in self._snake
                    and food not in self._obstacles):
                self._food = food
                break
            ## and must be at least 10 cells away from the snake's head.
            elif (self._difficulty == "hard"
                    and food not in self._snake
                    and food not in self._obstacles
                    and (not self._snake
                         or vector_distance(
                             self._snake[0], food, manhattan=True
                         ) > 10)):
                self._food = food
                break
            if i > limit:
                raise RuntimeError("Unable to find valid food location "
                                   f"after {limit} attempts.")

    def _random_obstacles(self) -> None:
        """Place obstacles at random locations."""
        self._obstacles = []
        if self._walls:
            self._obstacles.extend(
                [(x, 0) for x in range(self._grid_size[0])]
            )
            self._obstacles.extend(
                [(x, self._grid_size[1] - 1)
                 for x in range(self._grid_size[0])]
            )
            self._obstacles.extend(
                [(0, y) for y in range(self._grid_size[1])]
            )
            self._obstacles.extend(
                [(self._grid_size[0] - 1, y)
                 for y in range(self._grid_size[1])]
            )
        if self._difficulty == "easy":
            return
        elif self._difficulty == "medium":
            upper = self._score // 20
            lower = 0
        elif self._difficulty == "hard":
            upper = self._score // 10
            lower = max(0, upper // 2)
        if upper == 0:
            return
        range_ = range(lower, upper + 1)
        total_obstacles = random.choices(
            range_, range_, k=1
        )[0]
        for _ in range(total_obstacles):
            while True:
                obstacle = (
                    random.randint(0, self._grid_size[0] - 1),
                    random.randint(0, self._grid_size[1] - 1),
                )
                ## Make sure the obstacle is not in the snake, adjacent to
                ## the head of the snake, within three blocks infront of the
                ## head of the snake, or on top of the food.
                if (obstacle not in self._snake
                        and obstacle not in self.__adjacent(self._snake[0])
                        and obstacle not in self.__infront(self._snake[0], 4)
                        and obstacle != self._food):
                    self._obstacles.append(obstacle)
                    break

    def __adjacent(self, point: tuple[int, int]) -> list[tuple[int, int]]:
        """Get the adjacent points to a given point."""
        x, y = point
        return [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ]

    def __infront(
        self,
        head: tuple[int, int],
        distance: int
    ) -> list[tuple[int, int]]:
        """Get the cells infront of the snake's head."""
        with self._direction:
            direction = self._direction.get_object()
        return [
            tuple(  # type: ignore
                vector_modulo(
                    vector_add(
                        head,
                        vector_multiply(
                            direction,
                            _distance
                        )
                    ),
                    self._grid_size
                )
            )
            for _distance in range(1, distance + 1)
        ]

    def _reset_game_state(self) -> None:
        """Reset the game state."""
        with self._direction as direction:
            direction.set_object(_INITIAL_DIRECTION)
        self._score = _INITIAL_SCORE
        self._seconds_per_move = _INITIAL_SECONDS_PER_MOVE
        self._game_over = False
        self._paused = False
        self._snake = []
        self._food = None
        self._obstacles = []

    def _restart(self) -> None:
        """Reset the game."""
        self._reset_game_state()
        self._random_obstacles()
        self._random_food()
        self._random_start()


_CELL_SIZE: int = 20


class SnakeGameJinxWidget(JinxObserverWidget):
    """
    A class to represent the snake game on a Jinx widget.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget, /,
        width: int,
        height: int, *,
        snake_game_logic: SnakeGameLogic | None = None,
        manual_update: bool = False,
        debug: bool = False
    ) -> None:
        """Create a new snake game widget."""
        super().__init__(parent, "Snake Game", debug=debug)
        if width % _CELL_SIZE != 0 or height % _CELL_SIZE != 0:
            raise ValueError(
                f"The width and height must be divisible by {_CELL_SIZE}."
                f"Got; {width=} and {height=}, respectively."
            )
        self.__grid_size: tuple[int, int] = (
            width // _CELL_SIZE,
            height // _CELL_SIZE
        )

        ## Set up the timer to update the game
        self.__manual_update: bool = manual_update
        self.__timer = QtCore.QTimer()
        if not manual_update:
            self.__timer.setInterval(100)
            self.__timer.timeout.connect(self.__update_game)

        ## Set up the game logic
        self._logic: SnakeGameLogic
        if snake_game_logic is None:
            self._logic = SnakeGameLogic(self.__grid_size)
        else:
            self._logic = snake_game_logic
            self._logic._grid_size = self.__grid_size

        ## Widget and layout
        self.__layout = QtWidgets.QGridLayout()
        self.__layout.setContentsMargins(0, 0, 0, 0)
        self.__layout.setSpacing(0)
        self.widget.setLayout(self.__layout)
        self.widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.widget.setStyleSheet("background-color: black;")
        self.widget.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        ## Score label
        self.__score_label = QtWidgets.QLabel("Score: 0")
        self.__score_label.setStyleSheet("color: white;")
        self.__layout.addWidget(self.__score_label, 0, 0)

        ## Restart button (reset score, snake, and food)
        self.__restart_button = QtWidgets.QPushButton("Restart")
        self.__restart_button.setStyleSheet("color: white;")
        self.__restart_button.clicked.connect(self._logic._restart)
        self.__layout.addWidget(self.__restart_button, 0, 1)

        ## Add are scene to draw the snake and food on
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
        self.__display_layout.addWidget(self.__view, 0, 0, 1, 2)
        self.__layout.addWidget(self.__display_widget, 1, 0, 1, 2)

        ## Set up the key press event
        if not manual_update:
            self.widget.keyPressEvent = self.__key_press_event

        ## Start the game
        self._logic._restart()
        if not manual_update:
            self.__timer.start()

    def update_observer(self, data: JinxGuiData) -> None:
        """Update the observer."""
        self._logic._difficulty = data.get_data("difficulty", "medium")
        self._logic._show_path = data.get_data("show_path", False)
        self._logic._walls = data.get_data("walls", False)
        # self.__food_time_limit = data.get_data("food_time_limit", 0.0)
        # self.__food_per_level = data.get_data("food_per_level", 1)
        self._logic._food_per_snake_growth = data.get_data(
            "food_per_snake_growth", 1)
        # self.__seconds_per_move_reduction_per_snake_growth = data.get_data(
        #     "seconds_per_move_reduction_per_snake_growth", 0.01
        # )
        # self.__min_seconds_per_move = data.get_data("min_seconds_per_move", 0.10)
        self._logic._initial_snake_length = data.get_data(
            "initial_snake_length", 4)
        self._logic._speed = data.get_data("speed", 1.0)

    def __key_press_event(self, event: QtCore.QEvent) -> None:
        """
        Handle key press events.

        This handles all the actions that can be taken by the user in this
        game.
        """
        with self._logic._direction:
            if event.key() == QtCore.Qt.Key.Key_W:  # type: ignore
                self._logic._direction.set_object((0, -1))
            elif event.key() == QtCore.Qt.Key.Key_S:  # type: ignore
                self._logic._direction.set_object((0, 1))
            elif event.key() == QtCore.Qt.Key.Key_A:  # type: ignore
                self._logic._direction.set_object((-1, 0))
            elif event.key() == QtCore.Qt.Key.Key_D:  # type: ignore
                self._logic._direction.set_object((1, 0))
            elif event.key() == QtCore.Qt.Key.Key_Space:  # type: ignore
                self._logic._paused = not self._logic._paused

    def manual_update_game(self, action: str | int) -> None:
        """Update the game manually."""
        if not self.__manual_update:
            raise RuntimeError("Cannot manually update the game.")
        with self._logic._direction:
            match action:
                case "up" | "w" | 0:
                    self._logic._direction.set_object((0, -1))
                case "down" | "s" | 1:
                    self._logic._direction.set_object((0, 1))
                case "left" | "a" | 2:
                    self._logic._direction.set_object((-1, 0))
                case "right" | "d" | 3:
                    self._logic._direction.set_object((1, 0))
                case "restart" | "r":
                    self._logic._restart()
                case "pause" | "p":
                    self._logic._paused = not self._logic._paused
                case _:
                    raise ValueError(f"Invalid action: {action}")
        self._logic._move_snake()
        self.__draw_all()

    def __update_game(self) -> None:
        """
        Update the game.

        This method is called by the widget's internal timer to update the
        game when it is being played interactively by a human.
        """
        self._logic._move_snake()
        self.__update_timer()
        self.__draw_all()

    def __update_timer(self) -> None:
        """Update the timer."""
        self.__timer.setInterval(
            int((1000 * self._logic._seconds_per_move) / self._logic._speed))

    def __draw_all(self) -> None:
        """Draw the snake and food on the grid."""
        self.__scene.clear()
        self.__draw_snake()
        self.__draw_food()
        self.__draw_obstacles()
        self.__update_score()
        if self.debug:
            self.__draw_debug()
        # if self.automated:
        #     self.__draw_automated()
        if self._logic._game_over:
            self.__draw_game_over()
        elif self._logic._paused:
            self.__draw_paused()
        self.widget.update()

    def __draw_snake(self) -> None:
        """Draw the snake on the grid."""
        if self._logic._show_path:
            ## Draw the manhattan path from the head to the food
            path = self.__manhattan_path(
                self._logic._snake[0],
                self._logic._food
            )
            for x, y in path:
                self.__scene.addRect(
                    (x * _CELL_SIZE) + 5,
                    (y * _CELL_SIZE) + 5,
                    _CELL_SIZE // 2,
                    _CELL_SIZE // 2,
                    QtGui.QPen(QtGui.QColor("black")),
                    QtGui.QBrush(QtGui.QColor("yellow")),
                )
        x, y = self._logic._snake[0]
        self.__scene.addRect(
            x * _CELL_SIZE,
            y * _CELL_SIZE,
            _CELL_SIZE,
            _CELL_SIZE,
            QtGui.QPen(QtGui.QColor("black")),
            QtGui.QBrush(QtGui.QColor("blue")),
        )
        for x, y in self._logic._snake[1:]:
            self.__scene.addRect(
                x * _CELL_SIZE,
                y * _CELL_SIZE,
                _CELL_SIZE,
                _CELL_SIZE,
                QtGui.QPen(QtGui.QColor("black")),
                QtGui.QBrush(QtGui.QColor("green")),
            )

    def __manhattan_path(
        self,
        start: tuple[int, int],
        end: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """Find the manhattan path from start to end."""
        dif_x, dif_y = (end[0] - start[0], end[1] - start[1])
        add_x, add_y = (int(copysign(1, dif_x)), int(copysign(1, dif_y)))
        path: list[tuple[int, int]] = [start]
        for _ in range(abs(dif_x)):
            path.append((path[-1][0] + add_x, path[-1][1]))
        for _ in range(abs(dif_y)):
            path.append((path[-1][0], path[-1][1] + add_y))
        return path

    def __draw_food(self) -> None:
        """Draw the food on the grid."""
        if self._logic._food is not None:
            x, y = self._logic._food
            self.__scene.addRect(
                x * _CELL_SIZE,
                y * _CELL_SIZE,
                _CELL_SIZE,
                _CELL_SIZE,
                QtGui.QPen(QtGui.QColor("black")),
                QtGui.QBrush(QtGui.QColor("red")),
            )

    def __draw_obstacles(self) -> None:
        """Draw the obstacles on the grid."""
        for x, y in self._logic._obstacles:
            self.__scene.addRect(
                x * _CELL_SIZE,
                y * _CELL_SIZE,
                _CELL_SIZE,
                _CELL_SIZE,
                QtGui.QPen(QtGui.QColor("black")),
                QtGui.QBrush(QtGui.QColor("grey")),
            )

    def __draw_debug(self) -> None:
        """Draw debug information."""
        with self._logic._direction:
            self.__scene.addText(
                f"Direction: {self._logic._direction.get_object()}",
                QtGui.QFont("Arial", 12)
            ).setPos(0, 0)
        self.__scene.addText(
            f"Snake Head: {self._logic._snake[0]}",
            QtGui.QFont("Arial", 12)
        ).setPos(0, 20)
        self.__scene.addText(
            f"Snake Length: {len(self._logic._snake)}",
            QtGui.QFont("Arial", 12)
        ).setPos(0, 40)
        self.__scene.addText(
            f"Food: {self._logic._food}",
            QtGui.QFont("Arial", 12)
        ).setPos(0, 60)

    def __draw_game_over(self) -> None:
        """Draw the game over text."""
        text: QtWidgets.QGraphicsTextItem = self.__scene.addText(
            "Game Over",
            QtGui.QFont("Arial", 72)
        )
        text_size: QtCore.QSizeF = text.boundingRect().size()
        text.setPos(
            ((self.__grid_size[0] * _CELL_SIZE) // 2)
            - (text_size.width() // 2),
            ((self.__grid_size[1] * _CELL_SIZE) // 2)
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
            ((self.__grid_size[0] * _CELL_SIZE) // 2)
            - (text_size.width() // 2),
            ((self.__grid_size[1] * _CELL_SIZE) // 2)
            - (text_size.height() // 2)
        )

    def __update_score(self) -> None:
        self.__score_label.setText(f"Score: {self._logic._score}")


class SnakeGameOptionsJinxWidget(JinxObserverWidget):
    """A widget that allows the user to change the options for the game."""

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        data: JinxGuiData, /, *,
        debug: bool = False
    ) -> None:
        """Create a new snake game options widget."""
        super().__init__(parent, "Snake Game Options", debug=debug)
        self.__data = data
        self.__layout = QtWidgets.QGridLayout()
        self.__layout.setContentsMargins(0, 0, 0, 0)
        self.__layout.setSpacing(10)
        self.__layout.setColumnMinimumWidth(0, 600)
        self.__layout.setColumnMinimumWidth(1, 600)
        self.widget.setLayout(self.__layout)
        self.__create_options()

    def __create_options(self) -> None:
        """Create the options for the game."""
        self.__create_snake_length_option(0, 0)
        self.__create_difficulty_option(0, 1)
        self.__create_food_per_snake_growth_option(1, 0)
        self.__create_speed_option(1, 1)
        self.__create_walls_option(2, 0)
        self.__create_show_path_option(2, 1)

    def __create_snake_length_option(self, row: int, column: int) -> None:
        """Create the option to change the snake length."""
        layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel("Initial Snake Length:")
        layout.addWidget(label)
        self.__snake_length_spinbox = QtWidgets.QSpinBox()
        self.__snake_length_spinbox.setMinimum(1)
        self.__snake_length_spinbox.setMaximum(10)
        self.__snake_length_spinbox.setValue(4)
        self.__snake_length_spinbox.valueChanged.connect(
            self.__set_snake_length)
        layout.addWidget(self.__snake_length_spinbox)
        self.__layout.addLayout(layout, row, column)

    def __create_difficulty_option(self, row: int, column: int) -> None:
        """Create the option to change the difficulty."""
        layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel("Difficulty:")
        layout.addWidget(label)
        self.__difficulty_combobox = QtWidgets.QComboBox()
        self.__difficulty_combobox.addItems(["easy", "medium", "hard"])
        self.__difficulty_combobox.setCurrentText("medium")
        self.__difficulty_combobox.currentTextChanged.connect(
            self.__set_difficulty)
        layout.addWidget(self.__difficulty_combobox)
        self.__layout.addLayout(layout, row, column)

    def __create_speed_option(self, row: int, column: int) -> None:
        """Create the option to change the speed."""
        layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel("Speed:")
        layout.addWidget(label)
        self.__speed_slider = QtWidgets.QSlider()
        self.__speed_slider.setGeometry(50, 50, 100, 50)
        self.__speed_slider.setMinimum(1)
        self.__speed_slider.setMaximum(100)
        self.__speed_slider.setTickInterval(1)
        self.__speed_slider.setTickPosition(
            QtWidgets.QSlider.TickPosition.TicksBelow)
        self.__speed_slider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.__speed_slider.setValue(1)
        self.__speed_slider.valueChanged.connect(
            self.__set_speed)
        layout.addWidget(self.__speed_slider)
        self.__layout.addLayout(layout, row, column)

    def __create_walls_option(self, row: int, column: int) -> None:
        """Create the option to change the walls."""
        layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel("Walls:")
        layout.addWidget(label)
        self.__walls_checkbox = QtWidgets.QCheckBox()
        self.__walls_checkbox.setChecked(False)
        self.__walls_checkbox.stateChanged.connect(
            self.__set_walls)
        layout.addWidget(self.__walls_checkbox)
        self.__layout.addLayout(layout, row, column)

    def __create_show_path_option(self, row: int, column: int) -> None:
        """Create the option to change the show path."""
        layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel("Show Path:")
        layout.addWidget(label)
        self.__show_path_checkbox = QtWidgets.QCheckBox()
        self.__show_path_checkbox.setChecked(False)
        self.__show_path_checkbox.stateChanged.connect(
            self.__set_show_path)
        layout.addWidget(self.__show_path_checkbox)
        self.__layout.addLayout(layout, row, column)

    def __create_food_per_snake_growth_option(
        self,
        row: int,
        column: int
    ) -> None:
        """Create the option to change the food per snake growth."""
        layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel("Food Per Snake Growth:")
        layout.addWidget(label)
        self.__food_per_snake_growth_spinbox = QtWidgets.QSpinBox()
        self.__food_per_snake_growth_spinbox.setMinimum(1)
        self.__food_per_snake_growth_spinbox.setMaximum(10)
        self.__food_per_snake_growth_spinbox.setValue(1)
        self.__food_per_snake_growth_spinbox.valueChanged.connect(
            self.__set_food_per_snake_growth)
        layout.addWidget(self.__food_per_snake_growth_spinbox)
        self.__layout.addLayout(layout, row, column)

    def __set_difficulty(self, value: str) -> None:
        """Update the difficulty."""
        self.__data.set_data("difficulty", value)

    def __set_snake_length(self, value: int) -> None:
        """Update the snake length."""
        self.__data.set_data("initial_snake_length", value)

    def __set_speed(self, value: int) -> None:
        """Update the speed."""
        self.__data.set_data("speed", 1.0 + ((value - 1.0) / 100.0))

    def __set_walls(self, value: int) -> None:
        """Update the walls."""
        self.__data.set_data("walls", value)

    def __set_show_path(self, value: int) -> None:
        """Update the show path."""
        self.__data.set_data("show_path", value)

    def __set_food_per_snake_growth(self, value: int) -> None:
        """Update the food per snake growth."""
        self.__data.set_data("food_per_snake_growth", value)

    def update_observer(self, data: JinxGuiData) -> None:
        """Update the observer."""
        pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-wi", "--width", type=int, default=1200)
    parser.add_argument("-he", "--height", type=int, default=800)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    width: int = args.width
    height: int = args.height
    debug: bool = args.debug

    qapp = QtWidgets.QApplication([])
    qwindow = QtWidgets.QMainWindow()
    qwindow.setWindowTitle("Snake Game")
    qwindow.resize(width, height)

    qtimer = QtCore.QTimer()
    jdata = JinxGuiData("Snake GUI Data", clock=qtimer, debug=debug)
    jgui = JinxGuiWindow(qwindow, jdata, "Snake GUI Window", debug=debug)

    snake_qwidget = QtWidgets.QWidget()
    snake_game_jwidget = SnakeGameJinxWidget(
        snake_qwidget, width, height, debug=debug)
    snake_options_widget = QtWidgets.QWidget()
    snake_game_options_jwidget = SnakeGameOptionsJinxWidget(
        snake_options_widget, jdata, debug=debug)

    jgui.add_view("Snake Game", snake_game_jwidget)
    jgui.add_view("Snake Game Options", snake_game_options_jwidget)
    jdata.desired_view_state = "Snake Game"

    qwindow.show()
    qapp.exec()
