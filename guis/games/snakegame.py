
from PyQt6.QtWidgets import QApplication, QWidget, QGridLayout, QLabel, QPushButton, QSizePolicy, QSpacerItem, QSpacerItem, QGraphicsScene, QGraphicsView, QMainWindow, QGraphicsItem
from PyQt6.QtCore import Qt, QTimer, QEvent, QPoint, QRect, QSize
from PyQt6.QtGui import QPainter, QBrush, QPen, QColor, QPalette, QKeySequence
from guis.gui import JinxGuiData

from guis.observable import Observable, Observer

import random

class SnakeGame(Observer):
    """
    A class to represent the snake game.
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__()
        self.__gui_data = JinxGuiData()
        self.__gui_data.assign_observers(self)

        self.__score: int = 0
        self.__direction: tuple[int, int] = (1, 0)
        self.__speed: int = 1
        self.__seconds_per_move: float = 0.5
        self.__game_over: bool = False
        self.__paused: bool = False
        self.__grid_size: tuple[int, int] = (20, 20)
        self.__snake: list[tuple[int, int]] = []
        self.__food: tuple[int, int] | None = None

        ## Set up the parent widget and layout
        self.__parent: QWidget = parent
        self.__layout = QGridLayout()
        self.__layout.setContentsMargins(0, 0, 0, 0)
        self.__layout.setSpacing(0)
        self.__parent.setLayout(self.__layout)
        self.__parent.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.__parent.setStyleSheet("background-color: black;")
        self.__parent.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

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
        self.__restart()

        ## Draw the snake and food
        self.__draw_snake_and_food()
        
        ## Add a spacer to the bottom of the layout
        self.__spacer = QSpacerItem(
            0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.__layout.addItem(self.__spacer, 1, 0, 1, 2)

        ## Set up the timer to move the snake
        self.__timer = QTimer()
        self.__timer.setInterval(1000)
        self.__timer.timeout.connect(self.__move_snake)
        self.__timer.start()

        ## Set up the key press event
        self.__parent.keyPressEvent = self.__key_press_event
    
    def __key_press_event(self, event: QEvent) -> None:
        """Handle key press events."""
        print("Key Press: ", event.key())
        if event.key() == Qt.Key.Key_Up:
            self.__direction = (0, -1)
        elif event.key() == Qt.Key.Key_Down:
            self.__direction = (0, 1)
        elif event.key() == Qt.Key.Key_Left:
            self.__direction = (-1, 0)
        elif event.key() == Qt.Key.Key_Right:
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

        ## Check if the snake has hit itself
        if new_head in self.__snake:
            self.__game_over = True
            return

        ## Check if the snake has eaten the food
        if new_head == self.__food:
            self.__score += 1
            self.__update_score(self.__score)
            self.__random_food()
        else:
            self.__snake.pop()

        ## Add the new head to the snake
        self.__snake.insert(0, new_head)

        ## Draw the snake and food
        self.__draw_snake_and_food()
        self.__parent.update()

    def __draw_snake_and_food(self) -> None:
        """Draw the snake and food on the grid."""
        self.__scene.clear()
        self.__draw_snake()
        self.__draw_food()

    def __draw_snake(self) -> None:
        """Draw the snake on the grid."""
        for x, y in self.__snake:
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

    def __random_start(self) -> None:
        """Start the snake at a random location."""
        self.__snake = [
            (random.randint(0, self.__grid_size[0] - 1),
             random.randint(0, self.__grid_size[1] - 1))
        ]
    
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
    
    def __restart(self) -> None:
        """Reset the game."""
        self.__score = 0
        self.__update_score(self.__score)
        self.__random_start()
        self.__random_food()
        self.__game_over = False
        self.__paused = False
        self.__draw_snake_and_food()
        self.__parent.update()

    def __update_score(self, score: int) -> None:
        self.__score_label.setText(f"Score: {score}")

if __name__ == "__main__":
    # You need one (and only one) QApplication instance per application.
    app = QApplication([])
    window = QMainWindow()
    window.setWindowTitle("Snake Game")
    window.resize(800, 600)
    widget = QWidget()
    window.setCentralWidget(widget)
    snake_game = SnakeGame(widget)
    window.show()

    app.exec()
