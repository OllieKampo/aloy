
from PyQt6.QtWidgets import QApplication, QWidget, QGridLayout, QLabel, QPushButton, QSizePolicy, QSpacerItem, QSpacerItem, QGraphicsScene, QGraphicsView, QMainWindow, QGraphicsItem
from PyQt6.QtCore import Qt, QTimer, QEvent, QPoint, QRect, QSize
from PyQt6.QtGui import QPainter, QBrush, QPen, QColor, QPalette, QKeySequence
from guis.gui import JinxGuiData, JinxGuiWindow, JinxObserverWidget

from guis.observable import Observable, Observer

import random

def vector_add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    """Add two vectors."""
    return (a[0] + b[0], a[1] + b[1])

class SnakeGame(JinxObserverWidget):
    """
    A class to represent the snake game.
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)

        self.__score: int = 0
        self.__direction: tuple[int, int] = (1, 0)
        self.__seconds_per_move: float = 0.40
        self.__game_over: bool = False
        self.__paused: bool = False
        self.__grid_size: tuple[int, int] = (20, 20)
        self.__snake: list[tuple[int, int]] = []
        self.__food: tuple[int, int] | None = None
        self.__obstacles: list[tuple[int, int]] = []

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
            if food not in self.__snake:
                self.__food = food
                break
    
    def __random_obstacles(self) -> None:
        """Place obstacles at random locations."""
        self.__obstacles = []
        upper = self.__score // 10
        lower = max(0, upper - 2)
        total_obstacles = random.randint(lower, upper)
        for _ in range(total_obstacles):
            while True:
                obstacle = (
                    random.randint(0, self.__grid_size[0] - 1),
                    random.randint(0, self.__grid_size[1] - 1),
                )
                ## Make sure the obstacle is not in the snake, adjacent to
                ## the head of the snake, or on top of the food.
                if (obstacle not in self.__snake
                        and obstacle in self.__adjacent(self.__snake[0])
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

if __name__ == "__main__":
    # You need one (and only one) QApplication instance per application.
    app = QApplication([])
    window = QMainWindow()
    window.setWindowTitle("Snake Game")
    window.resize(800, 600)

    data = JinxGuiData()
    jinx_gui = JinxGuiWindow(window, data)

    widget = QWidget()
    window.setCentralWidget(widget)
    snake_game = SnakeGame(widget)
    data.assign_observers(snake_game)

    window.show()

    app.exec()
