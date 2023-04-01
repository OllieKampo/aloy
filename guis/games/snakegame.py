
from PyQt6.QtWidgets import QApplication, QWidget, QGridLayout, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QSizePolicy, QSpacerItem, QSpacerItem, QGraphicsScene, QGraphicsView, QMainWindow, QGraphicsItem, QSpinBox
from PyQt6.QtCore import Qt, QTimer, QEvent, QPoint, QRect, QSize
from PyQt6.QtGui import QPainter, QBrush, QPen, QColor, QPalette, QKeySequence
from guis.gui import JinxGuiData, JinxGuiWindow, JinxObserverWidget

from guis.observable import Observable, Observer

import random

from moremath.vectors import vector_add, vector_magnitude, vector_subtract


class SnakeGame(JinxObserverWidget):
    """
    A class to represent the snake game.
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)

        ## Game data
        self.__score: int = 0
        self.__direction: tuple[int, int] = (1, 0)
        self.__seconds_per_move: float = 0.40
        self.__game_over: bool = False
        self.__paused: bool = False
        self.__grid_size: tuple[int, int] = (20, 20)
        self.__snake: list[tuple[int, int]] = []
        self.__food: tuple[int, int] | None = None
        self.__obstacles: list[tuple[int, int]] = []

        ## Game options
        self.__difficulty: str = "hard"
        self.__show_path: bool = False
        self.__walls: bool = False
        self.__food_time_limit: float = 0.0
        self.__food_per_level: int = 1
        self.__food_per_snake_growth: int = 1
        self.__seconds_per_move_reduction_per_snake_growth: float = 0.01
        self.__min_seconds_per_move: float = 0.10
        self.__initial_snake_length: int = 3
        self.__snake_color: QColor = QColor(0, 255, 0)
        self.__snake_head_color: QColor = QColor(0, 255, 255)
        self.__food_color: QColor = QColor(255, 0, 0)
        self.__obstacle_color: QColor = QColor(255, 255, 255)
        self.__grid_color: QColor = QColor(255, 255, 255)

        ## Set up the parent widget and layout
        self.__layout = QGridLayout()
        self.__layout.setContentsMargins(0, 0, 0, 0)
        self.__layout.setSpacing(0)
        self.widget.setLayout(self.__layout)
        self.widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.widget.setStyleSheet("background-color: black;")
        self.widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        ## Score label
        self.__score_label: QLabel = QLabel("Score: 0")
        self.__score_label.setStyleSheet("color: white;")
        self.__layout.addWidget(self.__score_label, 0, 0)

        ## Restart button (reset score, snake, and food)
        self.__restart_button: QPushButton = QPushButton("Restart")
        self.__restart_button.setStyleSheet("color: white;")
        self.__restart_button.clicked.connect(self.__restart)
        self.__layout.addWidget(self.__restart_button, 0, 1)

        ## Add are scene to draw the snake and food on
        self.__scene = QGraphicsScene()
        self.__view = QGraphicsView(self.__scene)
        self.__view.setStyleSheet("background-color: white;")
        self.__view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.__layout.addWidget(self.__view, 1, 0, 1, 2)

        ## Add a spacer to the bottom of the layout
        self.__spacer = QSpacerItem(
            0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.__layout.addItem(self.__spacer, 1, 0, 1, 2)

        ## Set up the timer to move the snake
        self.__timer = QTimer()
        self.__timer.setInterval(int(1000 * self.__seconds_per_move))
        self.__timer.timeout.connect(self.__move_snake)
        self.__timer.start()

        ## Set up the key press event
        self.widget.keyPressEvent = self.__key_press_event

        self.__restart()

    def update_observer(self, data: JinxGuiData) -> None:
        """Update the observer."""
        self.__difficulty = data.get_data("difficulty", "hard")
        self.__show_path = data.get_data("show_path", False)
        self.__walls = data.get_data("walls", False)
        self.__food_time_limit = data.get_data("food_time_limit", 0.0)
        self.__food_per_level = data.get_data("food_per_level", 1)
        self.__food_per_snake_growth = data.get_data("food_per_snake_growth", 1)
        self.__seconds_per_move_reduction_per_snake_growth = data.get_data(
            "seconds_per_move_reduction_per_snake_growth", 0.01
        )
        self.__min_seconds_per_move = data.get_data("min_seconds_per_move", 0.10)
        self.__initial_snake_length = data.get_data("initial_snake_length", 3)
        self.__snake_color = data.get_data("snake_color", QColor(0, 255, 0))
        self.__snake_head_color = data.get_data("snake_head_color", QColor(0, 255, 255))
        self.__food_color = data.get_data("food_color", QColor(255, 0, 0))
        self.__obstacle_color = data.get_data("obstacle_color", QColor(255, 255, 255))
        self.__grid_color = data.get_data("grid_color", QColor(255, 255, 255))
    
    def __key_press_event(self, event: QEvent) -> None:
        """Handle key press events."""
        print("Key Press: ", event.key())
        if event.key() == Qt.Key.Key_W:
            self.__direction = (0, -1)
        elif event.key() == Qt.Key.Key_S:
            self.__direction = (0, 1)
        elif event.key() == Qt.Key.Key_A:
            self.__direction = (-1, 0)
        elif event.key() == Qt.Key.Key_D:
            self.__direction = (1, 0)
        elif event.key() == Qt.Key.Key_Space:
            self.__paused = not self.__paused

    def __move_snake(self) -> None:
        """Move the snake in the current direction."""
        if self.__game_over or self.__paused:
            return

        ## Get the current head of the snake
        x, y = self.__snake[0]

        ## Get the new head of the snake
        new_x = (x + self.__direction[0]) % self.__grid_size[0]
        new_y = (y + self.__direction[1]) % self.__grid_size[1]
        new_head = (new_x, new_y)

        ## Check if the snake has hit itself or an obstacle
        if new_head in self.__snake or new_head in self.__obstacles:
            self.__game_over = True
            return

        ## Check if the snake has eaten the food
        if new_head == self.__food:
            self.__score += 1
            self.__update_score(self.__score)
            self.__random_food()
            self.__random_obstacles()
            self.__seconds_per_move = max(
                0.1, self.__seconds_per_move - 0.005
            )
            self.__timer.setInterval(int(1000 * self.__seconds_per_move))
        else:
            self.__snake.pop()

        ## Add the new head to the snake
        self.__snake.insert(0, new_head)

        ## Draw the snake and food
        ## TODO: This should be done in a separate thread
        self.__draw_all()
        self.widget.update()

    def __draw_all(self) -> None:
        """Draw the snake and food on the grid."""
        self.__scene.clear()
        self.__draw_snake()
        self.__draw_food()
        self.__draw_obstacles()

    def __draw_snake(self) -> None:
        """Draw the snake on the grid."""
        x, y = self.__snake[0]
        self.__scene.addRect(
            x * 20,
            y * 20,
            20,
            20,
            QPen(QColor("black")),
            QBrush(QColor("blue")),
        )
        for x, y in self.__snake[1:]:
            self.__scene.addRect(
                x * 20,
                y * 20,
                20,
                20,
                QPen(QColor("black")),
                QBrush(QColor("green")),
            )

    def __draw_food(self) -> None:
        """Draw the food on the grid."""
        if self.__food is not None:
            x, y = self.__food
            self.__scene.addRect(
                x * 20,
                y * 20,
                20,
                20,
                QPen(QColor("black")),
                QBrush(QColor("red")),
            )

    def __draw_obstacles(self) -> None:
        """Draw the obstacles on the grid."""
        for x, y in self.__obstacles:
            self.__scene.addRect(
                x * 20,
                y * 20,
                20,
                20,
                QPen(QColor("black")),
                QBrush(QColor("grey")),
            )

    def __random_start(self, snake_length: int = 4) -> None:
        """Start the snake at a random location."""
        ## Place the snake's head at a random location
        x = random.randint(0, self.__grid_size[0] - 1)
        y = random.randint(0, self.__grid_size[1] - 1)
        self.__snake = [(x, y)]
        ## Randomly add segments to the snake's tail
        while len(self.__snake) < snake_length:
            direction = random.choice(
                [(-1, 0), (1, 0), (0, -1), (0, 1)]
            )
            x += direction[0]
            y += direction[1]
            x = x % self.__grid_size[0]
            y = y % self.__grid_size[1]
            position = (x, y)
            ## Do not put segments on top of each other
            if position not in self.__snake:
                self.__snake.append(position)
            else:
                x, y = self.__snake[-1]
        ## Stop the snake from hitting itself immediately
        while vector_add(self.__snake[0], self.__direction) in self.__snake:
            self.__direction = random.choice(
                [(-1, 0), (1, 0), (0, -1), (0, 1)]
            )

    def __random_food(self) -> None:
        """Place food at a random location."""
        while True:
            food = (
                random.randint(0, self.__grid_size[0] - 1),
                random.randint(0, self.__grid_size[1] - 1),
            )
            if (self.__difficulty == "easy"
                    and food not in self.__snake):
                self.__food = food
                break
            elif (self.__difficulty == "medium"
                    and food not in self.__snake
                    and food not in self.__obstacles):
                self.__food = food
                break
            elif (self.__difficulty == "hard"
                    and food not in self.__snake
                    and food not in self.__obstacles
                    and abs(sum(vector_subtract(food, self.__snake[0]))) > 10):
                self.__food = food
                break

    def __random_obstacles(self) -> None:
        """Place obstacles at random locations."""
        self.__obstacles = []
        if self.__difficulty == "easy":
            return
        elif self.__difficulty == "medium":
            upper = self.__score // 20
            lower = 0
        elif self.__difficulty == "hard":
            upper = self.__score // 10
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
                    random.randint(0, self.__grid_size[0] - 1),
                    random.randint(0, self.__grid_size[1] - 1),
                )
                ## Make sure the obstacle is not in the snake, adjacent to
                ## the head of the snake, or on top of the food.
                if (obstacle not in self.__snake
                        and obstacle not in self.__adjacent(self.__snake[0])
                        and obstacle != self.__food):
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

    def __restart(self) -> None:
        """Reset the game."""
        self.__score = 0
        self.__update_score(self.__score)
        self.__random_start()
        self.__random_food()
        self.__random_obstacles()
        self.__game_over = False
        self.__paused = False
        self.__seconds_per_move = 0.40
        self.__timer.setInterval(int(1000 * self.__seconds_per_move))
        self.__draw_all()
        self.widget.update()

    def __update_score(self, score: int) -> None:
        self.__score_label.setText(f"Score: {score}")


class SnakeGameOptionsWidget(JinxObserverWidget):
    """A widget that allows the user to change the options for the game."""

    def __init__(self, parent: QWidget, data: JinxGuiData) -> None:
        super().__init__(parent)
        self.__data = data
        self.__layout = QVBoxLayout()
        # self.__layout.setAlignment(Qt.AlignTop)
        self.widget.setLayout(self.__layout)
        self.__create_options()

    def __create_options(self) -> None:
        """Create the options for the game."""
        self.__create_grid_size_option()
        self.__create_snake_length_option()

    def __create_grid_size_option(self) -> None:
        """Create the option to change the grid size."""
        layout = QHBoxLayout()
        # layout.setAlignment(Qt.AlignTop)
        label = QLabel("Grid Size:")
        layout.addWidget(label)
        self.__grid_size_spinbox = QSpinBox()
        self.__grid_size_spinbox.setMinimum(5)
        self.__grid_size_spinbox.setMaximum(100)
        self.__grid_size_spinbox.setValue(20)
        self.__grid_size_spinbox.valueChanged.connect(self.__update_grid_size)
        layout.addWidget(self.__grid_size_spinbox)
        self.__layout.addLayout(layout)

    def __create_snake_length_option(self) -> None:
        """Create the option to change the snake length."""
        layout = QHBoxLayout()
        # layout.setAlignment(Qt.AlignTop)
        label = QLabel("Snake Length:")
        layout.addWidget(label)
        self.__snake_length_spinbox = QSpinBox()
        self.__snake_length_spinbox.setMinimum(3)
        self.__snake_length_spinbox.setMaximum(100)
        self.__snake_length_spinbox.setValue(4)
        self.__snake_length_spinbox.valueChanged.connect(
            self.__update_snake_length)
        layout.addWidget(self.__snake_length_spinbox)
        self.__layout.addLayout(layout)

    def __update_all(self) -> None:
        """Update all of the options."""
        self.__update_grid_size(self.__grid_size_spinbox.value())
        self.__update_snake_length(self.__snake_length_spinbox.value())

    def __update_grid_size(self, value: int) -> None:
        """Update the grid size."""
        self.__data.set_data("grid_size", (value, value))

    def __update_snake_length(self, value: int) -> None:
        """Update the snake length."""
        self.__data.set_data("snake_length", value)

    def update_observer(self, data: JinxGuiData) -> None:
        """Update the observer."""
        print(data.get_data("grid_size"))
        print(data.get_data("snake_length"))


if __name__ == "__main__":
    # You need one (and only one) QApplication instance per application.
    app = QApplication([])
    window = QMainWindow()
    window.setWindowTitle("Snake Game")
    window.resize(800, 600)

    jinx_data = JinxGuiData()
    jinx_gui = JinxGuiWindow(window, jinx_data)

    snake_widget = QWidget()
    snake_game = SnakeGame(snake_widget)
    snake_options_widget = QWidget()
    snake_game_options = SnakeGameOptionsWidget(snake_options_widget, jinx_data)

    jinx_gui.add_view("Snake Game", snake_game)
    jinx_gui.add_view("Snake Game Options", snake_game_options)
    jinx_data.desired_view_state = "Snake Game"

    window.show()

    app.exec()
