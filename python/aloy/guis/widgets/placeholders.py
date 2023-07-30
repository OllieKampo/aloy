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
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

"""Module defining placeholder PySide6 widgets."""

from PySide6 import QtWidgets, QtCore, QtGui


class PlaceholderWidget(QtWidgets.QWidget):
    """
    Simple placeholder widget.

    Useful for designing GUIs with complex layouts, where you want to see how
    the layout looks like without having to implement the actual widgets.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        text: str = "Placeholder",
        color: QtGui.QColor = QtGui.QColor("black"),
        alignment: QtCore.Qt.AlignmentFlag
        = QtCore.Qt.AlignmentFlag.AlignCenter,
        horizontal_size_policy: QtWidgets.QSizePolicy.Policy
        = QtWidgets.QSizePolicy.Policy.Expanding,
        vertical_size_policy: QtWidgets.QSizePolicy.Policy
        = QtWidgets.QSizePolicy.Policy.Expanding
    ) -> None:
        """Create a new place holder widget."""
        super().__init__(parent)

        self.__layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.__layout)

        self.__label = QtWidgets.QLabel()
        self.__label.setText(text)
        if color is not None:
            self.__label.setStyleSheet(
                f"background-color: {color.name()};")
        self.__label.setAlignment(alignment)
        self.__label.setSizePolicy(
            horizontal_size_policy,
            vertical_size_policy
        )
        self.__label.paintEvent = self.__label_paint_event  # type: ignore
        self.__layout.addWidget(self.__label)

    def __label_paint_event(
        self,
        event: QtGui.QPaintEvent  # pylint: disable=unused-argument
    ) -> None:
        """Paint the placeholder widget."""
        painter = QtGui.QPainter(self.__label)

        # Paint green cross and border.
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(QtGui.QColor("green")))
        painter.drawLine(0, 0, self.__label.width(),
                         self.__label.height())
        painter.drawLine(0, self.__label.height(),
                         self.__label.width(), 0)
        painter.drawRect(self.__label.rect())

        # Red size 16 text.
        painter.setPen(QtGui.QPen(QtGui.QColor("red")))
        font = painter.font()
        font.setPointSize(16)
        painter.setFont(font)

        # Paint the widget size in the top left corner.
        painter.drawText(
            self.__label.rect(),
            QtCore.Qt.AlignmentFlag.AlignLeft
            | QtCore.Qt.AlignmentFlag.AlignTop,
            f"size={self.__label.width()}x"
            f"{self.__label.height()}"
        )

        # Paint the widget text.
        painter.drawText(
            self.__label.rect(),
            self.__label.alignment(),
            self.__label.text()
        )

        painter.end()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    widget = PlaceholderWidget()
    widget.show()
    sys.exit(app.exec())
