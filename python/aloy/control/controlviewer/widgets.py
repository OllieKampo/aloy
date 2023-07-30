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

"""Module for Aloy widgets that display control variables."""

import itertools
import random
import sys
from collections import deque
from PySide6 import QtWidgets, QtCore, QtGui
from aloy.datastructures.views import ListView

from aloy.guis.gui import AloySystemData, AloyWidget


# TODO: Change to normal qwidget, we would update this from the same thread
# that updates the simulation view.
# A thread continuously runs the simulation, and a separate thread updates the
# simulation view by directly grabbing the simulation state, so the simulation
# view needs to take the simulation logic as argument.
class PositionGraph(AloyWidget):
    """A widget that displays a radial graph of two control variables."""

    def __init__(
        self,
        parent: QtWidgets.QWidget, /,
        name: str = "Position Graph",
        x_label: str = "x variable",
        y_label: str = "y variable",
        x_limits: tuple[float, float] = (0.0, 100.0),
        y_limits: tuple[float, float] = (0.0, 100.0),
        max_history: int = 100,
        size: tuple[int, int] = (400, 400), *,
        debug: bool = False
    ) -> None:
        """
        Create a new position graph Aloy widget within the given parent widget.

        A position graph is a widget that displays a radial graph of the values
        of two control variables over time.

        Parameters
        ----------
        `parent: QtWidgets.QWidget` - The parent widget to draw this widget on.
        Note that this will modify the parent widget's layout.

        `name: str = "Position Graph"` - The name of the widget.

        `x_label: str = "x variable"` - The label for the x axis.

        `y_label: str = "y variable"` - The label for the y axis.

        `x_limits: tuple[float, float] = (0.0, 100.0)` - The limits for the x
        axis.

        `y_limits: tuple[float, float] = (0.0, 100.0)` - The limits for the y
        axis.

        `max_history: int = 100` - The maximum number of points to display on
        the graph.

        `size: tuple[int, int] = (400, 400)` - The size of the widget in
        pixels. The widget is resized to this size.

        `debug: bool = False` - Whether to log debug messages.
        """
        super().__init__(
            parent,
            name=name,
            size=size,
            resize=True,
            debug=debug
        )

        self.__x_label: str = x_label
        self.__y_label: str = y_label

        self.__x_limits: tuple[float, float] = x_limits
        self.__y_limits: tuple[float, float] = y_limits

        self.__max_history: int = max_history
        self.__history_x = deque[float](maxlen=max_history)
        self.__history_y = deque[float](maxlen=max_history)

        self.__create_scene()
        self.__paint_display()

    @property
    def x_label(self) -> str:
        return self.__x_label

    @x_label.setter
    def x_label(self, value: str) -> None:
        self.__x_label = value

    @property
    def y_label(self) -> str:
        return self.__y_label

    @y_label.setter
    def y_label(self, value: str) -> None:
        self.__y_label = value

    @property
    def x_limits(self) -> tuple[float, float]:
        return self.__x_limits

    @x_limits.setter
    def x_limits(self, value: tuple[float, float]) -> None:
        if len(value) != 2:
            raise ValueError("x_limits must be a tuple of length 2.")
        if value[0] >= value[1]:
            raise ValueError("x_limits[0] must be less than x_limits[1].")
        self.__x_limits = value

    @property
    def y_limits(self) -> tuple[float, float]:
        return self.__y_limits

    @y_limits.setter
    def y_limits(self, value: tuple[float, float]) -> None:
        if len(value) != 2:
            raise ValueError("y_limits must be a tuple of length 2.")
        if value[0] >= value[1]:
            raise ValueError("y_limits[0] must be less than y_limits[1].")
        self.__y_limits = value

    @property
    def max_history(self) -> int:
        return self.__max_history

    @max_history.setter
    def max_history(self, value: int) -> None:
        if value < 1:
            raise ValueError("max_history must be greater than 0.")
        self.__max_history = value
        history_x = deque[float](maxlen=value)
        history_x.extend(self.__history_x)
        self.__history_x = history_x
        history_y = deque[float](maxlen=value)
        history_y.extend(self.__history_y)
        self.__history_y = history_y

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
        self.qwidget.setLayout(self.__layout)
        self.qwidget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Fixed
        )

        self.__scene = QtWidgets.QGraphicsScene(0, 0, *self.size)
        self.__view = QtWidgets.QGraphicsView(self.__scene)
        self.__view.setStyleSheet("background-color: white;")
        self.__view.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.__view.setFixedSize(*self.size)
        self.__view.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.__view.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.__layout.addWidget(self.__view, 0, 0, 1, 1)

    def __paint_display(self) -> None:
        self.__x_line = QtWidgets.QGraphicsLineItem(
            self.size.width // 8,
            self.size.height // 2,
            self.size.width - (self.size.width // 8),
            self.size.height // 2
        )
        self.__scene.addItem(self.__x_line)

        self.__y_line = QtWidgets.QGraphicsLineItem(
            self.size.width // 2,
            self.size.height // 8,
            self.size.width // 2,
            self.size.height - (self.size.height // 8)
        )
        self.__scene.addItem(self.__y_line)

        text = self.__scene.addText(
            self.__x_label,
            QtGui.QFont("Arial", 15)
        )
        rect = text.boundingRect()
        text.setPos(
            int(self.size.width * (1.25 / 20)) - rect.center().y(),
            (self.size.height // 2) + rect.center().x()
        )
        text.setRotation(-90)

        text = self.__scene.addText(
            self.__y_label,
            QtGui.QFont("Arial", 15)
        )
        rect = text.boundingRect()
        text.setPos(
            (self.size.width // 2) - rect.center().x(),
            int(self.size.height * (18.75 / 20)) - rect.center().y()
        )

        # Add limits to lines.
        text = self.__scene.addText(
            f"{self.__x_limits[0]}",
            QtGui.QFont("Arial", 15)
        )
        text.setPos(
            (self.size.width // 8),
            (self.size.height // 2)
        )

        text = self.__scene.addText(
            f"{self.__x_limits[1]}",
            QtGui.QFont("Arial", 15)
        )
        rect = text.boundingRect()
        text.setPos(
            (self.size.width - (self.size.width // 8)) - rect.right(),
            (self.size.height // 2)
        )

        text = self.__scene.addText(
            f"{self.__y_limits[0]}",
            QtGui.QFont("Arial", 15)
        )
        rect = text.boundingRect()
        text.setPos(
            (self.size.width // 2) - rect.right(),
            (self.size.height
             - (self.size.height // 8)) - rect.center().y()
        )

        text = self.__scene.addText(
            f"{self.__y_limits[1]}",
            QtGui.QFont("Arial", 15)
        )
        rect = text.boundingRect()
        text.setPos(
            (self.size.width // 2) - rect.right(),
            (self.size.height // 8) - rect.center().y()
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
                x, int(self.size.height * 0.475),
                x, int(self.size.height * 0.525),
                pen
            )
            self.__scene.addLine(
                int(self.size.width * 0.475), y,
                int(self.size.width * 0.525), y,
                pen
            )

    def clear_graph(self) -> None:
        """Clear the graph."""
        self.__history_x.clear()
        self.__history_y.clear()
        self.__scene.clear()
        self.__paint_display()

    def update_observer(self, observable_: AloySystemData) -> None:
        pass


class ResponseGraph(AloyWidget):
    """A graph that displays the value of a control variable over time."""

    def __init__(
        self,
        parent: QtWidgets.QWidget, /,
        name: str = "Response Graph",
        label: str = "variable",
        limits: tuple[float, float] = (0.0, 100.0),
        time_period: float = 100.0,
        size: tuple[int, int] = (400, 400), *,
        debug: bool = False
    ) -> None:
        """
        Create a new response graph Aloy widget within the given parent widget.

        A response graph is a widget that displays the value of a control
        variable over time.

        Parameters
        ----------
        `parent: QtWidgets.QWidget` - The parent widget to draw this widget on.
        Note that this will modify the parent widget's layout.

        `name: str = "Response Graph"` - The name of the widget.

        `label: str = "variable"` - The label for the y axis.

        `limits: tuple[float, float] = (0.0, 100.0)` - The limits for the y
        axis.

        `max_history: int = 100` - The maximum number of points to display on
        the graph.

        `size: tuple[int, int] = (400, 400)` - The size of the widget in
        pixels. The widget is resized to this size.

        `debug: bool = False` - Whether to log debug messages.
        """
        super().__init__(
            parent,
            name=name,
            size=size,
            resize=True,
            debug=debug
        )

        self.__label: str = label

        self.__limits: tuple[float, float] = limits

        self.__history: list[float] = []
        self.__time_period: float = time_period
        self.__times: list[float] = []
        self.__min_time: float = 0.0
        self.__max_time: float = 0.0
        self.__min_value_index: int = 0
        self.__max_value_index: int = 0

        self.__create_scene()
        self.__paint_display()

    @property
    def label(self) -> str:
        return self.__label

    @label.setter
    def label(self, value: str) -> None:
        self.__label = value

    @property
    def limits(self) -> tuple[float, float]:
        return self.__limits

    @limits.setter
    def limits(self, value: tuple[float, float]) -> None:
        if len(value) != 2:
            raise ValueError("limits must be a tuple of length 2.")
        if value[0] >= value[1]:
            raise ValueError("limits[0] must be less than limits[1].")
        self.__limits = value

    @property
    def history(self) -> ListView[float]:
        return ListView(self.__history)

    @property
    def time_period(self) -> float:
        return self.__time_period

    @time_period.setter
    def time_period(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("time_period must be greater than 0.")
        self.__time_period = value
        if self.__max_time - value < self.__min_time:
            self.__min_time = self.__max_time - value
        history = []
        times = []
        for value, time in zip(self.__history, self.__times):
            if time < self.__min_time:
                history.append(value)
                times.append(time)
        self.__history = history
        self.__times = times

    @property
    def times(self) -> ListView[float]:
        return ListView(self.__times)

    def __create_scene(self) -> None:
        self.__layout = QtWidgets.QGridLayout()
        self.__layout.setContentsMargins(0, 0, 0, 0)
        self.__layout.setSpacing(0)
        self.qwidget.setLayout(self.__layout)
        self.qwidget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Fixed
        )

        self.__scene = QtWidgets.QGraphicsScene(0, 0, *self.size)
        self.__view = QtWidgets.QGraphicsView(self.__scene)
        self.__view.setStyleSheet("background-color: white;")
        self.__view.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.__view.setFixedSize(*self.size)
        self.__view.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.__view.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.__layout.addWidget(self.__view, 0, 0, 1, 1)

    def __paint_display(self) -> None:
        self.__x_line = QtWidgets.QGraphicsLineItem(
            self.size.width // 8,
            self.size.height // 2,
            self.size.width - (self.size.width // 8),
            self.size.height // 2
        )
        self.__scene.addItem(self.__x_line)

        self.__y_line = QtWidgets.QGraphicsLineItem(
            self.size.width // 8,
            self.size.height // 8,
            self.size.width // 8,
            self.size.height - (self.size.height // 8)
        )
        self.__scene.addItem(self.__y_line)

        text = self.__scene.addText(
            self.__label,
            QtGui.QFont("Arial", 15)
        )
        rect = text.boundingRect()
        text.setPos(
            int(self.size.width * (1.25 / 20)) - rect.center().y(),
            (self.size.height // 2) + rect.center().x()
        )
        text.setRotation(-90)

        # Add limits to lines.
        text = self.__scene.addText(
            f"{self.__min_time:.3f}",
            QtGui.QFont("Arial", 15)
        )
        text.setPos(
            (self.size.width // 8),
            (self.size.height // 2)
        )

        text = self.__scene.addText(
            f"{self.__max_time:.3f}",
            QtGui.QFont("Arial", 15)
        )
        rect = text.boundingRect()
        text.setPos(
            (self.size.width - (self.size.width // 8)) - rect.right(),
            (self.size.height // 2)
        )

        text = self.__scene.addText(
            f"{self.__limits[0]:.3f}",
            QtGui.QFont("Arial", 15)
        )
        rect = text.boundingRect()
        text.setPos(
            (self.size.width // 16),
            (self.size.height
             - (self.size.height // 12)) - rect.center().y()
        )

        text = self.__scene.addText(
            f"{self.__limits[1]:.3f}",
            QtGui.QFont("Arial", 15)
        )
        rect = text.boundingRect()
        text.setPos(
            (self.size.width // 16),
            (self.size.height // 12) - rect.center().y()
        )

    def __convert_values_to_position(
        self,
        value: float,
        time: float
    ) -> tuple[int, int]:
        max_, min_ = self.__limits
        width, height = self.size

        x_pos = (
            (width / 8)
            + ((time - self.__min_time) / self.__time_period)
            * (width * 0.75)
        )
        y_pos = (
            (height / 8)
            + ((value - min_) / (max_ - min_))
            * (height * 0.75)
        )

        return int(x_pos), int(y_pos)

    def update_graph(self, value: float, delta_time: float) -> None:
        if delta_time <= 0.0:
            raise ValueError("delta_time must be greater than 0.")

        self.__history.append(value)
        if value < self.__history[self.__min_value_index]:
            self.__min_value_index = len(self.__history) - 1
        elif value > self.__history[self.__max_value_index]:
            self.__max_value_index = len(self.__history) - 1

        self.__max_time += delta_time
        self.__times.append(self.__max_time)
        if len(self.__times) != 1 and self.__max_time - self.__min_time > self.__time_period:
            self.__min_time = self.__max_time - self.__time_period
            time_ = self.__times[0]
            while time_ < self.__min_time:
                self.__history.pop(0)
                self.__times.pop(0)
                self.__min_value_index -= 1
                self.__max_value_index -= 1
                time_ = self.__times[0]
            if self.__min_value_index < 0:
                self.__min_value_index = min(
                    range(len(self.__history)),
                    key=self.__history.__getitem__
                )
            if self.__max_value_index < 0:
                self.__max_value_index = max(
                    range(len(self.__history)),
                    key=self.__history.__getitem__
                )

        self.__scene.clear()
        self.__paint_display()

        if len(self.__history) > 1:
            # Draw lines between points.
            pen = QtGui.QPen(
                QtCore.Qt.GlobalColor.cyan,
                2,
                QtCore.Qt.PenStyle.SolidLine
            )
            for (x_1, y_1), (x_2, y_2) in itertools.pairwise(
                zip(self.__history, self.__times)
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
                self.__history[-1],
                self.__times[-1]
            )
            self.__scene.addEllipse(x - 5, y - 5, 10, 10, pen, brush)

            # Draw line to highlight current value.
            pen = QtGui.QPen(
                QtGui.QColor(255, 0, 255, 150),
                2,
                QtCore.Qt.PenStyle.SolidLine
            )
            self.__scene.addLine(
                (self.size.width // 8), y,
                self.size.width - (self.size.width // 8), y,
                pen
            )

            # Draw line to highlight minimum value.
            pen = QtGui.QPen(
                QtGui.QColor(255, 0, 0, 150),
                2,
                QtCore.Qt.PenStyle.SolidLine
            )
            y = self.__convert_values_to_position(
                self.__history[self.__min_value_index],
                self.__times[self.__min_value_index]
            )[1]
            self.__scene.addLine(
                (self.size.width // 8), y,
                self.size.width - (self.size.width // 8), y,
                pen
            )

            # Draw line to highlight maximum value.
            pen = QtGui.QPen(
                QtGui.QColor(0, 255, 0, 150),
                2,
                QtCore.Qt.PenStyle.SolidLine
            )
            y = self.__convert_values_to_position(
                self.__history[self.__max_value_index],
                self.__times[self.__max_value_index]
            )[1]
            self.__scene.addLine(
                (self.size.width // 8), y,
                self.size.width - (self.size.width // 8), y,
                pen
            )

    def clear_graph(self) -> None:
        """Clear the graph."""
        self.__history.clear()
        self.__times.clear()
        self.__min_time = 0.0
        self.__max_time = 0.0
        self.__scene.clear()
        self.__paint_display()

    def update_observer(self, observable_: AloySystemData) -> None:
        pass


def __test_position_graph(qwindow: QtWidgets.QMainWindow) -> QtCore.QTimer:
    graph_qwidget = QtWidgets.QWidget()
    graph_jwidget = PositionGraph(graph_qwidget)
    qwindow.setCentralWidget(graph_qwidget)

    def test_update_graph() -> None:
        """Update the graph with random jitter."""
        x_point = (
            graph_jwidget.history_x[-1]
            if len(graph_jwidget.history_x) > 0
            else 50
        )
        y_point = (
            graph_jwidget.history_y[-1]
            if len(graph_jwidget.history_y) > 0
            else 50
        )
        x_point = x_point + random.randint(-5, 5)
        x_point = max(
            graph_jwidget.x_limits[0],
            min(
                x_point,
                graph_jwidget.x_limits[1]
            )
        )
        y_point = y_point + random.randint(-5, 5)
        y_point = max(
            graph_jwidget.y_limits[0],
            min(
                y_point,
                graph_jwidget.y_limits[1]
            )
        )
        graph_jwidget.update_graph(x_point, y_point)

    qtimer = QtCore.QTimer()
    qtimer.timeout.connect(test_update_graph)
    qtimer.setInterval(100)
    qtimer.start()
    return qtimer


def __test_response_graph(qwindow: QtWidgets.QMainWindow) -> QtCore.QTimer:
    graph_qwidget = QtWidgets.QWidget()
    graph_jwidget = ResponseGraph(graph_qwidget)
    qwindow.setCentralWidget(graph_qwidget)

    def test_update_graph() -> None:
        """Update the graph with random jitter."""
        point = (
            graph_jwidget.history[-1]
            if len(graph_jwidget.history) > 0
            else 50
        )
        point = point + random.randint(-5, 5)
        point = max(
            graph_jwidget.limits[0],
            min(
                point,
                graph_jwidget.limits[1]
            )
        )
        delta_time = random.randint(1, 10) / 10
        graph_jwidget.update_graph(point, delta_time)

    qtimer = QtCore.QTimer()
    qtimer.timeout.connect(test_update_graph)
    qtimer.setInterval(100)
    qtimer.start()
    return qtimer


def __main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        choices=[
            "positiongraph",
            "responsegraph"
        ],
        help="Run the test suite."
    )
    args: argparse.Namespace = parser.parse_args()

    app = QtWidgets.QApplication([])
    qwindow = QtWidgets.QMainWindow()

    if args.test == "positiongraph":
        qtimer = __test_position_graph(qwindow)
    elif args.test == "responsegraph":
        qtimer = __test_response_graph(qwindow)

    qwindow.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    __main()
