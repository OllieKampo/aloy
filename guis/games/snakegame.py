
from PyQt6.QtWidgets import QApplication, QWidget, QGridLayout, QLabel, QPushButton, QSizePolicy, QSpacerItem, QSpacerItem
from guis.gui import JinxGuiData

from guis.observable import Observable, Observer

class SnakeGame(Observer):
    """
    A class to represent the snake game.
    """

    def __init__(self, parent: QWidget) -> None:
        self.__gui_data: JinxGuiData = JinxGuiData()
        self.__gui_data.assign_observers(self)

        self.__snake = None
        self.__food = None

        self.__parent: QWidget = parent
        self.__layout: QGridLayout = QGridLayout()
        self.__layout.setContentsMargins(0, 0, 0, 0)
        self.__layout.setSpacing(0)
        self.__parent.setLayout(self.__layout)
        self.__parent.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.__parent.setStyleSheet("background-color: black;")

        self.__score_label: QLabel = QLabel("Score: 0")
        self.__score_label.setStyleSheet("color: white;")
        self.__layout.addWidget(self.__score_label, 0, 0)

        self.__restart_button: QPushButton = QPushButton("Restart")
        self.__restart_button.setStyleSheet("color: white;")
        self.__restart_button.clicked.connect(self.__restart)
        self.__layout.addWidget(self.__restart_button, 0, 1)

        self.__spacer: QSpacerItem = QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.__layout.addItem(self.__spacer, 1, 0, 1, 2)
    
    def __restart(self) -> None:
        pass

    def __update_score(self, score: int) -> None:
        self.__score_label.setText(f"Score: {score}")

if __name__ == "__main__":
    # You need one (and only one) QApplication instance per application.
    app = QApplication([])
    window = QWidget()
    window.setWindowTitle("Snake Game")
    window.resize(800, 600)
    snake_game = SnakeGame(window)
    window.show()

    app.exec()
