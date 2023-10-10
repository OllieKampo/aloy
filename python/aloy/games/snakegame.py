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

"""Module defining a snake game."""

import random
import sys
import time
from itertools import count
from math import copysign

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from aloy.concurrency.atomic import AtomicObject
from aloy.datastructures.views import ListView
from aloy.games.gamerecorder import GameRecorder, GameSpec
from aloy.guis.gui import AloyGuiWindow, AloySystemData, AloyWidget
from aloy.moremath.vectors import (vector_add, vector_between_torus_wrapped,
                                   vector_cast, vector_distance, vector_modulo,
                                   vector_multiply, vector_subtract)

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"

__all__ = (
    "SnakeGameLogic",
    "SnakeGameAloyWidget",
    "SnakeGameOptionsAloyWidget"
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
_DEFAULT_SPEED: float = 1.0
_DEFAULT_FOOD_TIME_LIMIT: int = 20  # seconds
_DEFAULT_FOOD_PER_LEVEL: int = 5
_DEFAULT_FOOD_PER_SNAKE_GROWTH: int = 1
_DEFAULT_INITIAL_SNAKE_LENGTH: int = 4
_DEFAULT_RECORD_PATH: str = "games/recordings/snakegame_record.json"


class SnakeGameLogic:
    """
    The internal logic of the snake game.
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
        "__new_grid_size",
        "difficulty",
        "show_path",
        "walls",
        "speed",
        "food_per_snake_growth",
        "food_time_limit",
        "food_per_level",
        "initial_snake_length",

        # Game recording
        "__recorder",
        "__recorder_enabled",
        "__enable_recording_on_restart",
        "__match_number",
        "__recorder_path"
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

        `manual_update: bool = False` - Whether the game is in manual update
        mode. In this mode
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
        self.__playing: bool = False
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
        self.__new_grid_size: tuple[int, int] | None = None
        self.difficulty: str = _DEFAULT_DIFFICULTY
        self.show_path: bool = _DEFAULT_SHOW_PATH
        self.walls: bool = _DEFAULT_WALLS
        self.speed: float = _DEFAULT_SPEED
        self.food_time_limit: int = _DEFAULT_FOOD_TIME_LIMIT
        self.food_per_level: int = _DEFAULT_FOOD_PER_LEVEL
        self.food_per_snake_growth: int = _DEFAULT_FOOD_PER_SNAKE_GROWTH
        self.initial_snake_length: int = _DEFAULT_INITIAL_SNAKE_LENGTH

        # Game recording
        self.__recorder: GameRecorder | None = None
        self.__recorder_enabled: bool = False
        self.__enable_recording_on_restart: bool = False
        self.__match_number: int = 0
        self.__recorder_path: str = _DEFAULT_RECORD_PATH

    @property
    def direction(self) -> AtomicObject[tuple[int, int]]:
        """Get the direction of the snake."""
        return self.__direction

    @property
    def playing(self) -> bool:
        """
        Get whether the game is being played.

        This is `True` if the snake has moved at least once since the game
        was last reset. Otherwise, this is `False` if the game was just reset,
        and the snake has not yet moved.
        """
        return self.__playing

    @property
    def grid_size(self) -> tuple[int, int]:
        """Get the grid size."""
        return self.__grid_size

    @grid_size.setter
    def grid_size(self, grid_size: tuple[int, int]) -> None:
        """
        Set the grid size.

        If the game is currently being played, then the grid size will be
        changed after the game has been reset. Otherwise, the grid size will
        be changed immediately.
        """
        if not isinstance(grid_size, tuple):
            raise TypeError("The grid size must be a tuple. "
                            f"Got; {type(grid_size)!r}.")
        if len(grid_size) != 2:
            raise ValueError("The grid size must be a tuple of length 2. "
                             f"Got; {len(grid_size)}.")
        if any(size <= 0 for size in grid_size):
            raise ValueError("The grid size must be positive. "
                             f"Got; {grid_size}.")
        if self.__playing:
            self.__new_grid_size = grid_size
        else:
            self.__grid_size = grid_size

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

    @property
    def recording_enabled(self) -> bool:
        """Get whether the game is being recorded."""
        return self.__recorder_enabled

    @recording_enabled.setter
    def recording_enabled(self, recording_enabled: bool) -> None:
        if recording_enabled:
            if self.__playing:
                self.__enable_recording_on_restart = True
            elif self.__recorder is None:
                self.__create_recorder()
                self.__recorder_enabled = True
        elif self.__recorder is not None:
            self.__recorder_enabled = False
            self.__recorder.clear_last_match()

    def __create_recorder(self) -> None:
        game_spec = GameSpec(
            game_name="Snake Game",
            match_name=f"Match {self.__match_number}",
            game_options={
                "difficulty": self.difficulty,
                "show_path": self.show_path,
                "walls": self.walls,
                "speed": self.speed,
                "food_time_limit": self.food_time_limit,
                "food_per_level": self.food_per_level,
                "food_per_snake_growth": self.food_per_snake_growth,
                "initial_snake_length": self.initial_snake_length,
                "grid_size": self.grid_size
            }
        )
        self.__recorder = GameRecorder(game_spec)

    @property
    def recording_path(self) -> str:
        """Get the recording path."""
        return self.__recorder_path

    @recording_path.setter
    def recording_path(self, recording_path: str) -> None:
        self.__recorder_path = recording_path

    def move_snake(self) -> None:
        """Move the snake in the current direction."""
        if not self.__playing:
            self.__playing = True

        if self.__game_over or self.__paused:
            return

        direction = self.__direction.get_obj()

        # Get the current head of the snake
        cur_x, cur_y = self.__snake[0]

        # Get the new head of the snake
        new_x = (cur_x + direction[0]) % self.grid_size[0]
        new_y = (cur_y + direction[1]) % self.grid_size[1]
        new_head = (new_x, new_y)

        # Check if the snake has hit itself or an obstacle
        if new_head in self.__snake or new_head in self.__obstacles:
            self.__game_over = True
            return

        # Check if the food time limit has been exceeded
        if (self.time_since_last_food > (self.food_time_limit / self.speed)):
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

        if self.__recorder_enabled:
            self.__recorder.record(
                action=direction,
                state={
                    "snake": self.__snake,
                    "food": self.__food,
                    "obstacles": self.__obstacles
                }
            )

    def _random_start(self) -> None:
        """Start the snake at a random location."""
        while True:
            # Place the snake's head at a random location
            x_pos = random.randint(0, self.__grid_size[0] - 1)
            y_pos = random.randint(0, self.__grid_size[1] - 1)
            if (x_pos, y_pos) in self.__obstacles or (x_pos, y_pos) == self.__food:
                continue
            self.__snake = [(x_pos, y_pos)]
            # Randomly add segments to the snake's tail
            for _ in range(self.initial_snake_length - 1):
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                random.shuffle(directions)
                unsatisfiable: bool = False
                for direction in directions:
                    x_pos += direction[0]
                    y_pos += direction[1]
                    x_pos = x_pos % self.__grid_size[0]
                    y_pos = y_pos % self.__grid_size[1]
                    position = (x_pos, y_pos)
                    # Do not put segments on top of each other
                    if (position not in self.__snake
                            and position not in self.__obstacles
                            and position != self.__food):
                        self.__snake.append(position)
                        break
                    elif direction == directions[-1]:
                        unsatisfiable = True
                    else:
                        x_pos, y_pos = self.__snake[-1]
                if unsatisfiable:
                    break
            if len(self.__snake) != self.initial_snake_length:
                continue
            # Stop the snake from hitting itself immediately
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            random.shuffle(directions)
            for direction in directions:
                next_cell = tuple(
                    vector_cast(
                        vector_add(self.__snake[0], direction),
                        int
                    )
                )
                if next_cell not in self.__snake:
                    with self.__direction:
                        self.__direction.set_obj(direction)
                    return

    def _random_food(self) -> None:
        """Place food at a random location."""
        limit: int = (self.__grid_size[0] * self.__grid_size[1]) ** 1.5
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
                         ) > 10.0)):
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
                [(x_pos, 0) for x_pos in range(self.grid_size[0])]
            )
            self.__obstacles.extend(
                [(x_pos, self.grid_size[1] - 1)
                 for x_pos in range(self.grid_size[0])]
            )
            self.__obstacles.extend(
                [(0, y_pos) for y_pos in range(self.grid_size[1])]
            )
            self.__obstacles.extend(
                [(self.grid_size[0] - 1, y_pos)
                 for y_pos in range(self.grid_size[1])]
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
        # TODO: We can construct a set of all valid locations and then
        # randomly select from that set instead of randomly generating
        # locations until we find a valid one.
        for _ in range(total_obstacles):
            unsatisfiable: bool = False
            limit: int = (self.__grid_size[0] * self.__grid_size[1])
            limit -= len(self.__snake) + len(self.__obstacles) + 1
            for i in count(start=1):
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
                if i > limit:
                    unsatisfiable = True
                    break
            if unsatisfiable:
                break

    def __adjacent(self, point: tuple[int, int]) -> list[tuple[int, int]]:
        """Get the adjacent points to a given point."""
        x_pos, y_pos = point
        return [
            (x_pos + 1, y_pos),
            (x_pos - 1, y_pos),
            (x_pos, y_pos + 1),
            (x_pos, y_pos - 1),
        ]

    def __infront(
        self,
        head: tuple[int, int],
        distance: int
    ) -> list[tuple[int, int]]:
        """Get the cells infront of the snake's head."""
        direction = self.direction.get_obj()
        return [
            tuple(
                vector_cast(
                    vector_modulo(
                        vector_add(
                            head,
                            vector_multiply(
                                direction,
                                _distance
                            )
                        ),
                        self.grid_size
                    ),
                    int
                )
            )
            for _distance in range(1, distance + 1)
        ]

    def _reset_game_state(self) -> None:
        """Reset the game state."""
        self.__playing = False
        if self.__new_grid_size is not None:
            self.__grid_size = self.__new_grid_size
            self.__new_grid_size = None
        with self.direction as direction:
            direction.set_obj(_INITIAL_DIRECTION)
        self.__score = _INITIAL_SCORE
        self.__level = _INITIAL_LEVEL
        self.__seconds_per_move = _INITIAL_SECONDS_PER_MOVE
        self.__game_over = False
        self.__paused = False
        self.__snake = []
        self.__food = None
        self.__obstacles = []
        self.__last_food_time = 0.0
        self.__pause_time = 0.0

    def restart(self) -> None:
        """Restart the game."""
        self.__match_number += 1
        if self.__recorder_enabled:
            self.__recorder.save(self.__recorder_path)
            self.__recorder.new_match()
        if self.__enable_recording_on_restart:
            self.__enable_recording_on_restart = False
            if self.__recorder is None:
                self.__create_recorder()
            self.__recorder_enabled = True
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
        obs[tuple(segment for segment in zip(*self.__snake[1:]))] = 2
        obs[self.snake[0]] = 3
        if self.obstacles:
            obs[tuple(obstacle for obstacle in zip(*self.obstacles))] = 4
        next_head = vector_add(self.snake[0], self.direction.get_obj())
        next_head = tuple(vector_cast(vector_modulo(next_head, self.grid_size), int))
        if obs[next_head] == 1:
            obs[next_head] = 6
        elif obs[next_head] == 4:
            obs[next_head] = 7
        else:
            obs[next_head] = 5
        return obs


_CELL_SIZE: int = 20


class SnakeGameAloyWidget(AloyWidget):
    """
    A class to represent the snake game on a Aloy widget.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        size: tuple[int, int],
        snake_game_logic: SnakeGameLogic | None = None,
        manual_update: bool = False,
        debug: bool = False
    ) -> None:
        """Create a new snake game widget."""
        super().__init__(parent, name="Snake Game", size=size, debug=debug)
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

        # Set up the game logic
        self._logic: SnakeGameLogic
        if snake_game_logic is None:
            self._logic = SnakeGameLogic(self.__grid_size)
        else:
            self._logic = snake_game_logic
            self._logic.grid_size = self.__grid_size

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
        time_ = self._logic.food_time_limit / self._logic.speed
        self.__food_time_limit_label = QtWidgets.QLabel(
            f"Food time limit: {time_:.2f}s"
        )
        self.__food_time_limit_label.setStyleSheet("color: white;")
        self.__layout.addWidget(self.__food_time_limit_label, 1, 0)

        # Food time limit display
        self.__food_time_limit_display = QtWidgets.QProgressBar()
        self.__food_time_limit_display.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid black;
                border-radius: 5px;
                background-color: white;
            }
            QProgressBar::chunk {
                background-color: red;
            }
            """
        )
        self.__food_time_limit_display.setFixedHeight(20)
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

    def update_observer(self, observable_: AloySystemData) -> None:
        """Update the observer."""
        self._logic.difficulty = observable_.get_var(
            "difficulty",
            _DEFAULT_DIFFICULTY
        )
        self._logic.show_path = observable_.get_var(
            "show_path",
            _DEFAULT_SHOW_PATH
        )
        self._logic.walls = observable_.get_var(
            "walls",
            _DEFAULT_WALLS
        )
        self._logic.food_time_limit = observable_.get_var(
            "food_time_limit",
            _DEFAULT_FOOD_TIME_LIMIT
        )
        self._logic.food_per_level = observable_.get_var(
            "food_per_level",
            _DEFAULT_FOOD_PER_LEVEL
        )
        self._logic.food_per_snake_growth = observable_.get_var(
            "food_per_snake_growth",
            _DEFAULT_FOOD_PER_SNAKE_GROWTH
        )
        self._logic.initial_snake_length = observable_.get_var(
            "initial_snake_length",
            _DEFAULT_INITIAL_SNAKE_LENGTH
        )
        self._logic.speed = observable_.get_var(
            "speed",
            _DEFAULT_SPEED
        )
        self._logic.recording_enabled = observable_.get_var(
            "record",
            False
        )
        self._logic.recording_path = observable_.get_var(
            "record_path",
            _DEFAULT_RECORD_PATH
        )

    def __key_press_event(self, event: QtGui.QKeyEvent) -> None:
        """
        Handle key press events.

        This handles all the actions that can be taken by the user in this
        game.
        """
        key = event.key()
        with self._logic.direction:
            if key == QtCore.Qt.Key.Key_W:
                self._logic.direction.set_obj((0, -1))
            elif key == QtCore.Qt.Key.Key_S:
                self._logic.direction.set_obj((0, 1))
            elif key == QtCore.Qt.Key.Key_A:
                self._logic.direction.set_obj((-1, 0))
            elif key == QtCore.Qt.Key.Key_D:
                self._logic.direction.set_obj((1, 0))
            elif key == QtCore.Qt.Key.Key_R:
                self._logic.restart()
            elif key == QtCore.Qt.Key.Key_Space:
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
        if self._logic.paused:
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
            for x_pos, y_pos in path:
                self.__scene.addRect(
                    (x_pos * _CELL_SIZE) + 5,
                    (y_pos * _CELL_SIZE) + 5,
                    _CELL_SIZE // 2,
                    _CELL_SIZE // 2,
                    QtGui.QPen(QtGui.QColor("black")),
                    QtGui.QBrush(QtGui.QColor("yellow")),
                )
        x_pos, y_pos = self._logic.snake[0]
        self.__scene.addRect(
            x_pos * _CELL_SIZE,
            y_pos * _CELL_SIZE,
            _CELL_SIZE,
            _CELL_SIZE,
            QtGui.QPen(QtGui.QColor("black")),
            QtGui.QBrush(QtGui.QColor("blue")),
        )
        for x_pos, y_pos in self._logic.snake[1:]:
            self.__scene.addRect(
                x_pos * _CELL_SIZE,
                y_pos * _CELL_SIZE,
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
        if self._logic.walls:
            dif_x, dif_y = vector_subtract(end, start)
        else:
            dif_x, dif_y = vector_between_torus_wrapped(
                start, end, self._logic.grid_size)
        add_x, add_y = (int(copysign(1, dif_x)), int(copysign(1, dif_y)))
        path: list[tuple[int, int]] = [start]
        for _ in range(abs(dif_x)):
            path.append((path[-1][0] + add_x, path[-1][1]))
        for _ in range(abs(dif_y)):
            path.append((path[-1][0], path[-1][1] + add_y))
        if not self._logic.walls:
            for index, step in enumerate(path):
                path[index] = vector_cast(
                    vector_modulo(
                        step,
                        self._logic.grid_size
                    ),
                    int
                )
        return path

    def __draw_food(self) -> None:
        """Draw the food on the grid."""
        if self._logic.food is not None:
            x_pos, y_pos = self._logic.food
            self.__scene.addRect(
                x_pos * _CELL_SIZE,
                y_pos * _CELL_SIZE,
                _CELL_SIZE,
                _CELL_SIZE,
                QtGui.QPen(QtGui.QColor("black")),
                QtGui.QBrush(QtGui.QColor("red")),
            )

    def __draw_obstacles(self) -> None:
        """Draw the obstacles on the grid."""
        for x_pos, y_pos in self._logic.obstacles:
            self.__scene.addRect(
                x_pos * _CELL_SIZE,
                y_pos * _CELL_SIZE,
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
        self.__level_label.setText(f"Level: {self._logic.level}")

    def __update_food_timer(self) -> None:
        time_ = self._logic.food_time_limit / self._logic.speed
        self.__food_time_limit_label.setText(
            f"Food Time Limit: {time_:.2f}s"
        )
        if not self._logic.game_over and not self._logic.paused:
            self.__food_time_limit_display.setValue(
                int(100
                    * (self._logic.time_since_last_food
                       / time_))
            )


# pylint: disable=W0201
class SnakeGameOptionsAloyWidget(AloyWidget):
    """A widget that allows the user to change the options for the game."""

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        data: AloySystemData,
        size: tuple[int, int],
        debug: bool = False
    ) -> None:
        """Create a new snake game options widget."""
        super().__init__(
            qwidget=parent,
            data=data,
            name="Snake Game Options",
            size=size,
            debug=debug
        )
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
        self.__create_record_option(4, 0)
        self.__create_record_path_option(4, 1)

    def __create_snake_length_option(self, row: int, column: int) -> None:
        """Create the option to change the snake length."""
        layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel("Initial Snake Length:")
        layout.addWidget(label)
        self.__snake_length_spinbox = QtWidgets.QSpinBox()
        self.__snake_length_spinbox.setMinimum(1)
        self.__snake_length_spinbox.setMaximum(10)
        self.__snake_length_spinbox.setValue(_DEFAULT_INITIAL_SNAKE_LENGTH)
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
        self.__speed_slider.setMinimum(0)
        self.__speed_slider.setMaximum(100)
        self.__speed_slider.setTickInterval(1)
        self.__speed_slider.setValue(0)
        self.__speed_slider.setTickPosition(
            QtWidgets.QSlider.TickPosition.TicksBelow)
        self.__speed_slider.setOrientation(QtCore.Qt.Orientation.Horizontal)
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
        self.__food_time_limit_slider.setMinimum(10)
        self.__food_time_limit_slider.setMaximum(30)
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

    def __create_record_option(self, row: int, column: int) -> None:
        """Create the option to change the record."""
        layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel("Record:")
        layout.addWidget(label)
        self.__record_checkbox = QtWidgets.QCheckBox()
        self.__record_checkbox.setChecked(False)
        self.__record_checkbox.stateChanged.connect(
            self.__set_record)
        layout.addWidget(self.__record_checkbox)
        self.__layout.addLayout(layout, row, column)

    def __create_record_path_option(self, row: int, column: int) -> None:
        """Create the option to change the record path."""
        layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel("Record Path:")
        layout.addWidget(label)
        self.__record_path_lineedit = QtWidgets.QLineEdit()
        self.__record_path_lineedit.setText(_DEFAULT_RECORD_PATH)
        self.__record_path_lineedit.textChanged.connect(
            self.__set_record_path)
        layout.addWidget(self.__record_path_lineedit)
        self.__layout.addLayout(layout, row, column)

    def __set_difficulty(self, value: str) -> None:
        """Update the difficulty."""
        self.data.set_var("difficulty", value)

    def __set_snake_length(self, value: int) -> None:
        """Update the snake length."""
        self.data.set_var("initial_snake_length", value)

    def __set_food_per_snake_growth(self, value: int) -> None:
        """Update the food per snake growth."""
        self.data.set_var("food_per_snake_growth", value)

    def __set_speed(self, value: int) -> None:
        """Update the speed."""
        self.data.set_var("speed", _DEFAULT_SPEED + (value / 100.0))

    def __set_food_per_level(self, value: int) -> None:
        """Update the food per level."""
        self.data.set_var("food_per_level", value)

    def __set_food_time_limit(self, value: int) -> None:
        """Update the food time limit."""
        self.data.set_var("food_time_limit", value)

    def __set_walls(self, value: int) -> None:
        """Update the walls."""
        self.data.set_var("walls", value)

    def __set_show_path(self, value: int) -> None:
        """Update the show path."""
        self.data.set_var("show_path", value)

    def __set_record(self, value: bool) -> None:
        """Update the record."""
        self.data.set_var("record", value)

    def __set_record_path(self, value: str) -> None:
        """Update the record path."""
        self.data.set_var("record_path", value)

    def update_observer(self, observable_: AloySystemData) -> None:
        """Update the observer."""
        return None


def play_snake_game(
    width: int,
    height: int,
    walls: bool = _DEFAULT_WALLS,
    show_path: bool = _DEFAULT_SHOW_PATH,
    debug: bool = False
) -> None:
    """Play the snake game."""
    size = (width, height)

    qapp = QtWidgets.QApplication([])
    qtimer = QtCore.QTimer()

    jdata = AloySystemData(
        name="Snake GUI Data",
        clock=qtimer,
        debug=debug
    )
    jgui = AloyGuiWindow(
        qapp=qapp,
        data=jdata,
        name="Snake GUI Window",
        size=size,
        debug=debug
    )

    snake_qwidget = QtWidgets.QWidget()
    snake_game_jwidget = SnakeGameAloyWidget(
        snake_qwidget, size, debug=debug
    )
    snake_options_qwidget = QtWidgets.QWidget()
    snake_game_options_jwidget = SnakeGameOptionsAloyWidget(
        snake_options_qwidget, jdata, size, debug=debug
    )

    jgui.add_view(snake_game_jwidget)
    jgui.add_view(snake_game_options_jwidget)
    jdata.desired_view_state = snake_game_jwidget.observer_name

    jgui.qwindow.show()
    sys.exit(qapp.exec())
