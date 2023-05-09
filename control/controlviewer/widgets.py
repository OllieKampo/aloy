import itertools
import random
import sys
from collections import deque
from typing import Final
from PyQt6.QtWidgets import QApplication, QWidget, QGraphicsScene, QGraphicsLineItem, QGraphicsView
from PyQt6 import QtWidgets, QtCore, QtGui

from guis.gui import JinxObserverWidget


class PositionGraph(JinxObserverWidget):
    DEFAULT_WIDTH: Final[int] = 400
    DEFAULT_HEIGHT: Final[int] = 400
    DEFAULT_MAX_HISTORY: Final[int] = 100

    def __init__(
        self,
        parent: QWidget, /,
        name: str = "Position Graph",
        x_label: str = "Cart position (mm)",
        y_label: str = "Pendulum angle (rads)",
        x_limits: tuple[float, float] = (0.0, 100.0),
        y_limits: tuple[float, float] = (0.0, 100.0),
        max_history: int = DEFAULT_MAX_HISTORY,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT, *,
        debug: bool = False
    ) -> None:
        super().__init__(
            parent,
            name,
            size=(width, height),
            debug=debug
        )

        self.__x_label: str = x_label
        self.__y_label: str = y_label

        self.__x_limits: tuple[float, float] = x_limits
        self.__y_limits: tuple[float, float] = y_limits

        self.__history_x = deque[float](maxlen=max_history)
        self.__history_y = deque[float](maxlen=max_history)

        self.__create_scene()
        self.__paint_display()

    @property
    def history_x(self) -> deque[float]:
        return self.__history_x

    @property
    def history_y(self) -> deque[float]:
        return self.__history_y

    def __create_scene(self) -> None:
        self.__layout = QtWidgets.QGridLayout()
        self.__layout.setContentsMargins(0, 0, 0, 0)
        self.__layout.setSpacing(0)
        self.widget.setLayout(self.__layout)
        self.widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Fixed
        )

        self.__scene = QGraphicsScene(0, 0, *self.size)
        self.__view = QGraphicsView(self.__scene)
        self.__view.setStyleSheet("background-color: white;")
        self.__view.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.__view.setFixedSize(*self.size)
        self.__view.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.__view.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.__layout.addWidget(self.__view, 0, 0, 1, 1)

    def __paint_display(self) -> None:
        self.__x_line = QGraphicsLineItem(
            self.DEFAULT_WIDTH // 2,
            self.DEFAULT_HEIGHT // 8,
            self.DEFAULT_WIDTH // 2,
            self.DEFAULT_HEIGHT - (self.DEFAULT_HEIGHT // 8)
        )
        self.__scene.addItem(self.__x_line)

        self.__y_line = QGraphicsLineItem(
            self.DEFAULT_WIDTH // 8,
            self.DEFAULT_HEIGHT // 2,
            self.DEFAULT_WIDTH - (self.DEFAULT_WIDTH // 8),
            self.DEFAULT_HEIGHT // 2
        )
        self.__scene.addItem(self.__y_line)

        text = self.__scene.addText(
            self.__x_label,
            QtGui.QFont("Arial", 15)
        )
        rect = text.boundingRect()
        text.setPos(
            int(self.DEFAULT_WIDTH * (1.25 / 20)) - rect.center().y(),
            (self.DEFAULT_HEIGHT // 2) + rect.center().x()
        )
        text.setRotation(-90)

        text = self.__scene.addText(
            self.__y_label,
            QtGui.QFont("Arial", 15)
        )
        rect = text.boundingRect()
        text.setPos(
            (self.DEFAULT_WIDTH // 2) - rect.center().x(),
            int(self.DEFAULT_HEIGHT * (18.75 / 20)) - rect.center().y()
        )

        # Add limits to lines.
        text = self.__scene.addText(
            f"{self.__x_limits[0]}",
            QtGui.QFont("Arial", 15)
        )
        text.setPos(
            (self.DEFAULT_WIDTH // 8),
            (self.DEFAULT_HEIGHT // 2)
        )

        text = self.__scene.addText(
            f"{self.__x_limits[1]}",
            QtGui.QFont("Arial", 15)
        )
        rect = text.boundingRect()
        text.setPos(
            (self.DEFAULT_WIDTH - (self.DEFAULT_WIDTH // 8)) - rect.right(),
            (self.DEFAULT_HEIGHT // 2)
        )

        text = self.__scene.addText(
            f"{self.__y_limits[0]}",
            QtGui.QFont("Arial", 15)
        )
        rect = text.boundingRect()
        text.setPos(
            (self.DEFAULT_WIDTH // 2) - rect.right(),
            (self.DEFAULT_HEIGHT
             - (self.DEFAULT_HEIGHT // 8)) - rect.center().y()
        )

        text = self.__scene.addText(
            f"{self.__y_limits[1]}",
            QtGui.QFont("Arial", 15)
        )
        rect = text.boundingRect()
        text.setPos(
            (self.DEFAULT_WIDTH // 2) - rect.right(),
            (self.DEFAULT_HEIGHT // 8) - rect.center().y()
        )

    def __convert_values_to_position(
        self,
        x_value: float,
        y_value: float
    ) -> tuple[int, int]:
        x_max, x_min = self.__x_limits
        y_max, y_min = self.__y_limits
        width, height = self.size

        x_pos = (
            (width / 8)
            + ((x_value - x_min) / (x_max - x_min))
            * (width * 0.75)
        )
        y_pos = (
            (height / 8)
            + ((y_value - y_min) / (y_max - y_min))
            * (height * 0.75)
        )

        return int(x_pos), int(y_pos)

    def update_graph(self, x_value: float, y_value: float) -> None:
        self.__history_x.append(x_value)
        self.__history_y.append(y_value)

        self.__scene.clear()
        self.__paint_display()

        if len(self.__history_x) > 1:
            # Draw lines between points.
            pen = QtGui.QPen(
                QtCore.Qt.GlobalColor.cyan,
                2,
                QtCore.Qt.PenStyle.SolidLine
            )
            for (x_1, y_1), (x_2, y_2) in itertools.pairwise(
                zip(self.__history_x, self.__history_y)
            ):
                x_1, y_1 = self.__convert_values_to_position(x_1, y_1)
                x_2, y_2 = self.__convert_values_to_position(x_2, y_2)
                self.__scene.addLine(x_1, y_1, x_2, y_2, pen)

            # Draw last point.
            pen = QtGui.QPen(
                QtCore.Qt.GlobalColor.red,
                2,
                QtCore.Qt.PenStyle.SolidLine
            )
            brush = QtGui.QBrush(
                QtCore.Qt.GlobalColor.red,
                QtCore.Qt.BrushStyle.SolidPattern
            )
            x, y = self.__convert_values_to_position(
                self.__history_x[-1],
                self.__history_y[-1]
            )
            self.__scene.addEllipse(x - 5, y - 5, 10, 10, pen, brush)

            # Draw marks on axis.
            pen = QtGui.QPen(
                QtCore.Qt.GlobalColor.magenta,
                2,
                QtCore.Qt.PenStyle.SolidLine
            )
            self.__scene.addLine(
                x, int(self.DEFAULT_HEIGHT * 0.475),
                x, int(self.DEFAULT_HEIGHT * 0.525),
                pen
            )
            self.__scene.addLine(
                int(self.DEFAULT_WIDTH * 0.475), y,
                int(self.DEFAULT_WIDTH * 0.525), y,
                pen
            )


if __name__ == "__main__":
    app = QApplication([])
    qwindow = QtWidgets.QMainWindow()

    graph_qwidget = QtWidgets.QWidget()
    graph_jwidget = PositionGraph(graph_qwidget)
    qwindow.setCentralWidget(graph_qwidget)

    def test_update_graph():
        x = (graph_jwidget.history_x[-1]
             if len(graph_jwidget.history_x) > 0
             else 50)
        y = (graph_jwidget.history_y[-1]
             if len(graph_jwidget.history_y) > 0
             else 50)
        print(x, y)
        x = x + random.randint(-5, 5)
        x = max(0, min(x, 100))
        y = y + random.randint(-5, 5)
        y = max(0, min(y, 100))
        print(x, y)
        graph_jwidget.update_graph(x, y)

    qtimer = QtCore.QTimer()
    qtimer.timeout.connect(test_update_graph)
    qtimer.setInterval(100)
    qtimer.start()

    qwindow.show()
    sys.exit(app.exec())
