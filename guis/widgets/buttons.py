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

"""Module defining some custom PySide6 buttons."""

from PySide6 import QtCore, QtWidgets


class LabelledArrowButton(QtWidgets.QToolButton):
    """Widget defining a labelled arrow button."""

    def __init__(
        self,
        text: str,
        qwidget: QtWidgets.QWidget,
        arrow_type: QtCore.Qt.ArrowType
    ) -> None:
        """Create a new labelled arrow button QtWidget widget."""
        super().__init__(qwidget)

        self.__layout = QtWidgets.QVBoxLayout()

        self.__label = QtWidgets.QLabel()
        self.__label.setText(text)
        self.__layout.addWidget(self.__label)

        self.__button = QtWidgets.QToolButton()
        self.__button.setArrowType(arrow_type)
        self.__button.setAutoRepeat(True)
        self.__layout.addWidget(self.__button)

        self.__layout.setStretch(0, 1)
        self.__layout.setStretch(1, 1)

        self.setLayout(self.__layout)
