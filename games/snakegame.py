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
import sys
from itertools import count
from math import copysign
import time

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from datastructures.views import ListView
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
_INITIAL_LEVEL: int = 1
_INITIAL_SECONDS_PER_MOVE: float = 0.40
_DEFAULT_DIFFICULTY: str = "medium"
_DEFAULT_SHOW_PATH: bool = False
_DEFAULT_WALLS: bool = False
_DEFAULT_SPEED: int = 1
_DEFAULT_FOOD_TIME_LIMIT: int = 20  # seconds
_DEFAULT_FOOD_PER_LEVEL: int = 5
_DEFAULT_FOOD_PER_SNAKE_GROWTH: int = 1
_DEFAULT_INITIAL_SNAKE_LENGTH: int = 4


class SnakeGameLogic:
    """
    The internal logic of the snake game.
    """

    __slots__ = (
        "__weakref__",

        # Actions
        "__direction",

        # Game state
        "__grid_size",
        "__score",
        "__level",
        "__seconds_per_move",
        "__game_over",
        "__paused",
        "__snake",
        "__food",
        "__obstacles",
        "__manual_update",
        "__last_food_time",
        "__pause_time",

        # Game options
        "difficulty",
        "show_path",
        "walls",
        "speed",
        "food_per_snake_growth",
        "food_time_limit",
        "food_per_level",
        "initial_snake_length"
    )

    def __init__(
        self,
        cells_grid_size: tuple[int, int],
        manual_update: bool = False
    ) -> None:
        """
        Create a new snake game logic object.

        Parameters
        ----------
        `cells_grid_size: tuple[int, int]` - The size of the grid in cells.
        This is the number of cells in the x and y directions.

        `manual_update: bool = False` - TODO
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

        # Actions
        self.__direction: AtomicObject[tuple[int, int]]
        self.__direction = AtomicObject(_INITIAL_DIRECTION)

        # Game state
        self.__grid_size: tuple[int, int] = cells_grid_size
        self.__score: int = _INITIAL_SCORE
        self.__level: int = _INITIAL_LEVEL
        self.__seconds_per_move: float = _INITIAL_SECONDS_PER_MOVE
        self.__game_over: bool = False
        self.__paused: bool = False
        self.__snake: list[tuple[int, int]] = []
        self.__food: tuple[int, int] | None = None
        self.__obstacles: list[tuple[int, int]] = []
        self.__manual_update: bool = manual_update
        self.__last_food_time: float = 0.0
        self.__pause_time: float = 0.0

        # Game options
        self.difficulty: str = _DEFAULT_DIFFICULTY
        self.show_path: bool = _DEFAULT_SHOW_PATH
        self.walls: bool = _DEFAULT_WALLS
        self.speed: int = _DEFAULT_SPEED
        self.food_time_limit: int = _DEFAULT_FOOD_TIME_LIMIT
        self.food_per_level: int = _DEFAULT_FOOD_PER_LEVEL
        self.food_per_snake_growth: int = _DEFAULT_FOOD_PER_SNAKE_GROWTH
        self.initial_snake_length: int = _DEFAULT_INITIAL_SNAKE_LENGTH

    @property
    def direction(self) -> AtomicObject[tuple[int, int]]:
        """Get the direction of the snake."""
        return self.__direction

    @property
    def grid_size(self) -> tuple[int, int]:
        """Get the grid size."""
        return self.__grid_size

    @property
    def score(self) -> int:
        """Get the current score."""
        return self.__score

    @property
    def level(self) -> int:
        """Get the current level."""
        return self.__level

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
        if not self.__manual_update:
            if not self.__paused and paused:
                self.__pause_time = time.perf_counter()
            if self.__paused and not paused:
                self.__last_food_time += (
                    time.perf_counter() - self.__pause_time)
        self.__paused = paused

    @property
    def snake(self) -> ListView[tuple[int, int]]:
        """Get the snake positions, the head is the last element."""
        return ListView(self.__snake)

    @property
    def food(self) -> tuple[int, int] | None:
        """Get the food position."""
        return self.__food

    @property
    def obstacles(self) -> ListView[tuple[int, int]]:
        """Get the obstacle positions."""
        return ListView(self.__obstacles)

    @property
    def time_since_last_food(self) -> float:
        """Get the time since the last food was eaten."""
        return time.perf_counter() - self.__last_food_time

    def move_snake(self) -> None:
        """Move the snake in the current direction."""
        if self.__game_over or self.__paused:
            return

        direction = self.__direction.get_obj()

        # Get the current head of the snake
        x, y = self.snake[0]

        # Get the new head of the snake
        new_x = (x + direction[0]) % self.grid_size[0]
        new_y = (y + direction[1]) % self.grid_size[1]
        new_head = (new_x, new_y)

        # Check if the snake has hit itself or an obstacle
        if new_head in self.__snake or new_head in self.__obstacles:
            self.game_over = True
            return

        # Check if the good time limit has been exceeded
        if (self.time_since_last_food > self.food_time_limit):
            self._random_food()
            self.__last_food_time = time.perf_counter()

        # Check if the snake has eaten the food
        if new_head == self.__food:
            self.__score += 1
            if self.__score % self.food_per_level == 0:
                self.__level += 1
            self._random_food()
            self._random_obstacles()
            self.__last_food_time = time.perf_counter()
            if self.__score % self.food_per_snake_growth == 0:
                self.__seconds_per_move = max(
                    0.1, self.seconds_per_move - 0.005
                )
            else:
                self.__snake.pop()
        else:
            self.__snake.pop()

        # Add the new head to the snake
        self.__snake.insert(0, new_head)

    def _random_start(self) -> None:
        """Start the snake at a random location."""
        valid: bool = False
        while not valid:
            # Place the snake's head at a random location
            x = random.randint(0, self.grid_size[0] - 1)
            y = random.randint(0, self.grid_size[1] - 1)
            if (x, y) in self.__obstacles or (x, y) == self.__food:
                continue
            self.__snake = [(x, y)]
            # Randomly add segments to the snake's tail
            for _ in range(self.initial_snake_length - 1):
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                random.shuffle(directions)
                unsatisfiable: bool = False
                for direction in directions:
                    x += direction[0]
                    y += direction[1]
                    x = x % self.__grid_size[0]
                    y = y % self.__grid_size[1]
                    position = (x, y)
                    # Do not put segments on top of each other
                    if (position not in self.__snake
                            and position not in self.__obstacles
                            and position != self.__food):
                        self.__snake.append(position)
                        break
                    elif direction == directions[-1]:
                        unsatisfiable = True
                    else:
                        x, y = self.__snake[-1]
                if unsatisfiable:
                    break
            if len(self.__snake) != self.initial_snake_length:
                continue
            # Stop the snake from hitting itself immediately
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            random.shuffle(directions)
            for direction in directions:
                next_cell = tuple(vector_add(self.__snake[0], direction))
                if next_cell not in self.__snake:
                    with self.__direction:
                        self.__direction.set_obj(direction)
                    valid = True
                    break

    def _random_food(self) -> None:
        """Place food at a random location."""
        limit: int = ((self.__grid_size[0] * self.__grid_size[1])
                      // _CELL_SIZE) ** 1.5
        for i in count(start=1):
            food = (
                random.randint(0, self.__grid_size[0] - 1),
                random.randint(0, self.__grid_size[1] - 1),
            )
            # Food must not be in the snake,
            if (self.difficulty == "easy"
                    and food not in self.__snake):
                self.__food = food
                break
            # and must not be in the snake or obstacles,
            elif (self.difficulty == "medium"
                    and food not in self.__snake
                    and food not in self.__obstacles):
                self.__food = food
                break
            # and must be at least 10 cells away from the snake's head.
            elif (self.difficulty == "hard"
                    and food not in self.__snake
                    and food not in self.__obstacles
                    and (not self.__snake
                         or vector_distance(
                             self.__snake[0],
                             food,
                             manhattan=True
                         ) > 10)):
                self.__food = food
                break
            if i > limit:
                raise RuntimeError("Unable to find valid food location "
                                   f"after {limit} attempts.")

    def _random_obstacles(self) -> None:
        """Place obstacles at random locations."""
        self.__obstacles = []
        if self.walls:
            self.__obstacles.extend(
                [(x, 0) for x in range(self.grid_size[0])]
            )
            self.__obstacles.extend(
                [(x, self.grid_size[1] - 1)
                 for x in range(self.grid_size[0])]
            )
            self.__obstacles.extend(
                [(0, y) for y in range(self.grid_size[1])]
            )
            self.__obstacles.extend(
                [(self.grid_size[0] - 1, y)
                 for y in range(self.grid_size[1])]
            )
        if self.difficulty == "easy":
            return
        elif self.difficulty == "medium":
            upper = self.score // (2 * self.food_per_level)
            lower = 0
        elif self.difficulty == "hard":
            upper = self.score // self.food_per_level
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
                    random.randint(0, self.grid_size[0] - 1),
                    random.randint(0, self.grid_size[1] - 1),
                )
                # Make sure the obstacle is not in the snake, adjacent to
                # the head of the snake, within three blocks infront of the
                # head of the snake, or on top of the food.
                if (obstacle not in self.snake
                        and obstacle not in self.__adjacent(self.snake[0])
                        and obstacle not in self.__infront(self.snake[0], 4)
                        and obstacle != self.food):
                    self.__obstacles.append(obstacle)
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
        direction = self.direction.get_obj()
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
                    self.grid_size
                )
            )
            for _distance in range(1, distance + 1)
        ]

    def _reset_game_state(self) -> None:
        """Reset the game state."""
        with self.direction as direction:
            direction.set_obj(_INITIAL_DIRECTION)
        self.__score = _INITIAL_SCORE
        self.__seconds_per_move = _INITIAL_SECONDS_PER_MOVE
        self.__game_over = False
        self.__paused = False
        self.__snake = []
        self.__food = None
        self.__obstacles = []
        self.__last_food_time = 0.0
        self.__pause_time = 0.0

    def restart(self) -> None:
        """Reset the game."""
        self._reset_game_state()
        self._random_obstacles()
        self._random_food()
        self._random_start()
        if not self.__manual_update:
            self.__last_food_time = time.perf_counter()

    def get_state(self) -> np.ndarray:
        """
        Get the current state of the game as a numpy array.

        The array is of shape (width, height) and contains the following
        values: 0 for empty cells, 1 for food, 2 for the snake's body, 3 for
        the snake's head, 4 for obstacles, 5 if the snake's head is about to
        move into an empty cell, 6 if the snake's head is about to eat food,
        and 7 if the snake's head is about to hit an obstacle.
        """
        obs = np.zeros(self.grid_size, dtype=np.int8)
        obs[self.food] = 1
        obs[tuple(segment for segment in zip(*self.snake[1:]))] = 2
        obs[self.snake[0]] = 3
        if self.obstacles:
            obs[tuple(obstacle for obstacle in zip(*self.obstacles))] = 4
        next_head = vector_add(self.snake[0], self.direction.get_obj())
        next_head = tuple(vector_modulo(next_head, self.grid_size))
        if obs[next_head] == 1:
            obs[next_head] = 6
        elif obs[next_head] == 4:
            obs[next_head] = 7
        else:
            obs[next_head] = 5
        return obs


_CELL_SIZE: int = 20


class SnakeGameJinxWidget(JinxObserverWidget):
    """
    A class to represent the snake game on a Jinx widget.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget, /,
        size: tuple[int, int], *,
        snake_game_logic: SnakeGameLogic | None = None,
        manual_update: bool = False,
        debug: bool = False
    ) -> None:
        """Create a new snake game widget."""
        super().__init__(parent, "Snake Game", size=size, debug=debug)
        width, height = size
        if width % _CELL_SIZE != 0 or height % _CELL_SIZE != 0:
            raise ValueError(
                f"The width and height must be divisible by {_CELL_SIZE}."
                f"Got; {width=} and {height=}, respectively."
            )
        self.__grid_size: tuple[int, int] = (
            width // _CELL_SIZE,
            height // _CELL_SIZE
        )

        # Set up the timer to update the game
        self.__manual_update: bool = manual_update
        self.__timer = QtCore.QTimer()
        if not manual_update:
            self.__timer.setInterval(100)
            self.__timer.timeout.connect(self.__update_game)

        # Set up the game logic
        self._logic: SnakeGameLogic
        if snake_game_logic is None:
            self._logic = SnakeGameLogic(self.__grid_size)
        else:
            self._logic = snake_game_logic
            self._logic.grid_size = self.__grid_size

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

        # Score label
        self.__score_label = QtWidgets.QLabel("Score: 0")
        self.__score_label.setStyleSheet("color: white;")
        self.__layout.addWidget(self.__score_label, 0, 0)

        # Level label
        self.__level_label = QtWidgets.QLabel("Level: 1")
        self.__level_label.setStyleSheet("color: white;")
        self.__layout.addWidget(self.__level_label, 0, 1)

        # Restart button (reset score, snake, and food)
        self.__restart_button = QtWidgets.QPushButton("Restart")
        self.__restart_button.setStyleSheet("color: white;")
        self.__restart_button.clicked.connect(self._logic.restart)
        self.__layout.addWidget(self.__restart_button, 0, 2)

        # Food time limit label
        self.__food_time_limit_label = QtWidgets.QLabel(
            f"Food time limit: {self._logic.food_time_limit:.2f}"
        )
        self.__food_time_limit_label.setStyleSheet("color: white;")
        self.__layout.addWidget(self.__food_time_limit_label, 1, 0)

        # Food time limit display
        self.__food_time_limit_display = QtWidgets.QProgressBar()
        self.__food_time_limit_display.setStyleSheet(
            "QProgressBar {"
                "border: 1px solid black;"
                "border-radius: 5px;"
                "background-color: white;"
            "}"
            "QProgressBar::chunk {"
                "background-color: red;"
            "}"
        )
        self.__food_time_limit_display.setFixedHeight(10)
        self.__food_time_limit_display.setRange(0, 100)
        self.__food_time_limit_display.setValue(100)
        self.__layout.addWidget(self.__food_time_limit_display, 1, 1, 1, 2)

        # Add a graphics scene to draw the snake and food on
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
        self.__layout.addWidget(self.__display_widget, 2, 0, 1, 3)

        # Set up the key press event
        if not manual_update:
            self.qwidget.keyPressEvent = self.__key_press_event

        # Start the game
        self._logic.restart()
        if not manual_update:
            self.__timer.start()

    def update_observer(self, observable_: JinxGuiData) -> None:
        """Update the observer."""
        self._logic.difficulty = observable_.get_data(
            "difficulty",
            _DEFAULT_DIFFICULTY
        )
        self._logic.show_path = observable_.get_data(
            "show_path",
            _DEFAULT_SHOW_PATH
        )
        self._logic.walls = observable_.get_data(
            "walls",
            _DEFAULT_WALLS
        )
        self._logic.food_time_limit = observable_.get_data(
            "food_time_limit",
            _DEFAULT_FOOD_TIME_LIMIT
        )
        self._logic.food_per_level = observable_.get_data(
            "food_per_level",
            _DEFAULT_FOOD_PER_LEVEL
        )
        self._logic.food_per_snake_growth = observable_.get_data(
            "food_per_snake_growth",
            _DEFAULT_FOOD_PER_SNAKE_GROWTH
        )
        self._logic.initial_snake_length = observable_.get_data(
            "initial_snake_length",
            _DEFAULT_INITIAL_SNAKE_LENGTH
        )
        self._logic.speed = observable_.get_data(
            "speed",
            _DEFAULT_SPEED
        )

    def __key_press_event(self, event: QtCore.QEvent) -> None:
        """
        Handle key press events.

        This handles all the actions that can be taken by the user in this
        game.
        """
        with self._logic.direction:
            if event.key() == QtCore.Qt.Key.Key_W:  # type: ignore
                self._logic.direction.set_obj((0, -1))
            elif event.key() == QtCore.Qt.Key.Key_S:  # type: ignore
                self._logic.direction.set_obj((0, 1))
            elif event.key() == QtCore.Qt.Key.Key_A:  # type: ignore
                self._logic.direction.set_obj((-1, 0))
            elif event.key() == QtCore.Qt.Key.Key_D:  # type: ignore
                self._logic.direction.set_obj((1, 0))
            elif event.key() == QtCore.Qt.Key.Key_R:  # type: ignore
                self._logic.restart()
            elif event.key() == QtCore.Qt.Key.Key_Space:  # type: ignore
                self._logic.paused = not self._logic.paused

    def manual_update_game(self, action: str | int) -> None:
        """
        Update the game manually.

        This can be used to simulate a game with an autonomous agent.

        Valid actions are:
            - "up" | "w" | 0
            - "down" | "s" | 1
            - "left" | "a" | 2
            - "right" | "d" | 3
            - "restart" | "r"
            - "pause" | "p"
        """
        if not self.__manual_update:
            raise RuntimeError("Cannot manually update the game.")
        with self._logic.direction:
            match action:
                case "up" | "w" | 0:
                    self._logic.direction.set_obj((0, -1))
                case "down" | "s" | 1:
                    self._logic.direction.set_obj((0, 1))
                case "left" | "a" | 2:
                    self._logic.direction.set_obj((-1, 0))
                case "right" | "d" | 3:
                    self._logic.direction.set_obj((1, 0))
                case "restart" | "r":
                    self._logic.restart()
                case "pause" | "p":
                    self._logic.paused = not self._logic.paused
                case _:
                    raise ValueError(f"Invalid action: {action}")
        self._logic.move_snake()
        self.__draw_all()

    def __update_game(self) -> None:
        """
        Update the game.

        This method is called by the widget's internal timer to update the
        game when it is being played interactively by a human.
        """
        self._logic.move_snake()
        self.__update_timer()
        self.__draw_all()

    def __update_timer(self) -> None:
        """Update the timer."""
        self.__timer.setInterval(
            int((1000 * self._logic.seconds_per_move) / self._logic.speed))

    def __draw_all(self) -> None:
        """Draw the snake and food on the grid."""
        self.__scene.clear()
        self.__draw_snake()
        self.__draw_food()
        self.__draw_obstacles()
        self.__update_score()
        if not self.__manual_update:
            self.__update_food_timer()
        if self.debug:
            self.__draw_debug()
        if self._logic.game_over:
            self.__draw_game_over()
        elif self._logic.paused:
            self.__draw_paused()
        self.qwidget.update()

    def __draw_snake(self) -> None:
        """Draw the snake on the grid."""
        if self._logic.show_path:
            # Draw the manhattan path from the head to the food
            path = self.__manhattan_path(
                self._logic.snake[0],
                self._logic.food  # type: ignore
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
        x, y = self._logic.snake[0]
        self.__scene.addRect(
            x * _CELL_SIZE,
            y * _CELL_SIZE,
            _CELL_SIZE,
            _CELL_SIZE,
            QtGui.QPen(QtGui.QColor("black")),
            QtGui.QBrush(QtGui.QColor("blue")),
        )
        for x, y in self._logic.snake[1:]:
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
        if self._logic.food is not None:
            x, y = self._logic.food
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
        for x, y in self._logic.obstacles:
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
        self.__scene.addText(
            f"Direction: {self._logic.direction.get_obj()}",
            QtGui.QFont("Arial", 12)
        ).setPos(0, 0)
        self.__scene.addText(
            f"Snake Head: {self._logic.snake[0]}",
            QtGui.QFont("Arial", 12)
        ).setPos(0, 20)
        self.__scene.addText(
            f"Snake Length: {len(self._logic.snake)}",
            QtGui.QFont("Arial", 12)
        ).setPos(0, 40)
        self.__scene.addText(
            f"Food: {self._logic.food}",
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
        self.__score_label.setText(f"Score: {self._logic.score}")
        self.__level_label.setText(f"Level: {self._logic.score // 10 + 1}")

    def __update_food_timer(self) -> None:
        self.__food_time_limit_label.setText(
            f"Food Time Limit: {self._logic.food_time_limit}"
        )
        if not self._logic.paused:
            self.__food_time_limit_display.setValue(
                int(100 * (self._logic.time_since_last_food
                           / self._logic.food_time_limit))
            )


# pylint: disable=W0201
class SnakeGameOptionsJinxWidget(JinxObserverWidget):
    """A widget that allows the user to change the options for the game."""

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        data: JinxGuiData, /,
        size: tuple[int, int], *,
        debug: bool = False
    ) -> None:
        """Create a new snake game options widget."""
        super().__init__(parent, "Snake Game Options", size=size, debug=debug)
        self.__data = data
        self.__layout = QtWidgets.QGridLayout()
        self.__layout.setContentsMargins(0, 0, 0, 0)
        self.__layout.setSpacing(10)
        width = self.size.width
        self.__layout.setColumnMinimumWidth(0, int((width // 2) * 0.95))
        self.__layout.setColumnMinimumWidth(1, int((width // 2) * 0.95))
        self.qwidget.setLayout(self.__layout)
        self.__create_options()

    def __create_options(self) -> None:
        """Create the options for the game."""
        self.__create_snake_length_option(0, 0)
        self.__create_difficulty_option(0, 1)
        self.__create_food_per_snake_growth_option(1, 0)
        self.__create_speed_option(1, 1)
        self.__create_food_per_level_option(2, 0)
        self.__create_food_time_limit_option(2, 1)
        self.__create_walls_option(3, 0)
        self.__create_show_path_option(3, 1)

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
        self.__food_per_snake_growth_spinbox.setValue(
            _DEFAULT_FOOD_PER_SNAKE_GROWTH
        )
        self.__food_per_snake_growth_spinbox.valueChanged.connect(
            self.__set_food_per_snake_growth)
        layout.addWidget(self.__food_per_snake_growth_spinbox)
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
        self.__speed_slider.setValue(_DEFAULT_SPEED)
        self.__speed_slider.setTickPosition(
            QtWidgets.QSlider.TickPosition.TicksBelow)
        self.__speed_slider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.__speed_slider.setValue(1)
        self.__speed_slider.valueChanged.connect(
            self.__set_speed)
        layout.addWidget(self.__speed_slider)
        self.__layout.addLayout(layout, row, column)

    def __create_food_per_level_option(self, row: int, column: int) -> None:
        """Create the option to change the food per level."""
        layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel("Food Per Level:")
        layout.addWidget(label)
        self.__food_per_level_spinbox = QtWidgets.QSpinBox()
        self.__food_per_level_spinbox.setMinimum(1)
        self.__food_per_level_spinbox.setMaximum(20)
        self.__food_per_level_spinbox.setValue(_DEFAULT_FOOD_PER_LEVEL)
        self.__food_per_level_spinbox.valueChanged.connect(
            self.__set_food_per_level)
        layout.addWidget(self.__food_per_level_spinbox)
        self.__layout.addLayout(layout, row, column)

    def __create_food_time_limit_option(self, row: int, column: int) -> None:
        """Create the option to change the food time limit."""
        layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel("Food Time Limit:")
        layout.addWidget(label)
        self.__food_time_limit_slider = QtWidgets.QSlider()
        self.__food_time_limit_slider.setGeometry(50, 50, 100, 50)
        self.__food_time_limit_slider.setMinimum(1)
        self.__food_time_limit_slider.setMaximum(20)
        self.__food_time_limit_slider.setTickInterval(1)
        self.__food_time_limit_slider.setValue(_DEFAULT_FOOD_TIME_LIMIT)
        self.__food_time_limit_slider.setTickPosition(
            QtWidgets.QSlider.TickPosition.TicksBelow)
        self.__food_time_limit_slider.setOrientation(
            QtCore.Qt.Orientation.Horizontal)
        self.__food_time_limit_slider.valueChanged.connect(
            self.__set_food_time_limit)
        layout.addWidget(self.__food_time_limit_slider)
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

    def __set_difficulty(self, value: str) -> None:
        """Update the difficulty."""
        self.__data.set_data("difficulty", value)

    # @QtCore.pyqtSlot(int)
    def __set_snake_length(self, value: int) -> None:
        """Update the snake length."""
        self.__data.set_data("initial_snake_length", value)

    # @QtCore.pyqtSlot(int)
    def __set_food_per_snake_growth(self, value: int) -> None:
        """Update the food per snake growth."""
        self.__data.set_data("food_per_snake_growth", value)

    def __set_speed(self, value: int) -> None:
        """Update the speed."""
        self.__data.set_data("speed", 1.0 + ((value - 1.0) / 100.0))

    # @QtCore.pyqtSlot(int)
    def __set_food_per_level(self, value: int) -> None:
        """Update the food per level."""
        self.__data.set_data("food_per_level", value)

    def __set_food_time_limit(self, value: int) -> None:
        """Update the food time limit."""
        self.__data.set_data("food_time_limit", value)

    def __set_walls(self, value: int) -> None:
        """Update the walls."""
        self.__data.set_data("walls", value)

    def __set_show_path(self, value: int) -> None:
        """Update the show path."""
        self.__data.set_data("show_path", value)

    def update_observer(self, observable_: JinxGuiData) -> None:
        """Update the observer."""
        return None


def play_snake_game(
    width: int,
    height: int,
    debug: bool = False
) -> None:
    """Play the snake game."""
    size = (width, height)

    qapp = QtWidgets.QApplication([])
    qwindow = QtWidgets.QMainWindow()
    qwindow.setWindowTitle("Snake Game")
    qwindow.resize(width, height)

    qtimer = QtCore.QTimer()
    jdata = JinxGuiData("Snake GUI Data", clock=qtimer, debug=debug)
    jgui = JinxGuiWindow(qwindow, jdata, "Snake GUI Window", debug=debug)

    snake_qwidget = QtWidgets.QWidget()
    snake_game_jwidget = SnakeGameJinxWidget(
        snake_qwidget, size, debug=debug
    )
    snake_options_qwidget = QtWidgets.QWidget()
    snake_game_options_jwidget = SnakeGameOptionsJinxWidget(
        snake_options_qwidget, jdata, size, debug=debug
    )

    jgui.add_view("Snake Game", snake_game_jwidget)
    jgui.add_view("Snake Game Options", snake_game_options_jwidget)
    jdata.desired_view_state = "Snake Game"

    qwindow.show()
    sys.exit(qapp.exec())
