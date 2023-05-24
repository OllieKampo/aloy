"""
Module containing Jinx PyQt6 widgets defining interfaces for teleoperate
control of robots.
"""

from abc import abstractmethod
from PyQt6 import QtWidgets
from PyQt6 import QtCore
from PyQt6 import QtGui

from guis.gui import JinxObserverWidget


class DirectionalControlInterface(JinxObserverWidget):
    """
    Widget defining a directional control interface for teleoperation of
    robots.
    """

    def __init__(
        self,
        qwidget: QtWidgets.QWidget, /,
        name: str,
        size: tuple[int, int], *,
        resize: bool = True,
        debug: bool = False
    ) -> None:
        """Create a new directional control interface Jinx widget."""
        super().__init__(
            qwidget,
            name,
            size,
            resize=resize,
            debug=debug
        )

        self._layout = QtWidgets.QGridLayout(self)

        self._forward_button = QtWidgets.QPushButton(self)
        self._forward_button.setText("Forward")
        self._forward_button.pressed.connect(self._forward_pressed)
        self._forward_button.released.connect(self._forward_released)
        self._layout.addWidget(self._forward_button, 0, 1)

        self._left_button = QtWidgets.QPushButton(self)
        self._left_button.setText("Left")
        self._left_button.pressed.connect(self._left_pressed)
        self._left_button.released.connect(self._left_released)
        self._layout.addWidget(self._left_button, 1, 0)

        self._right_button = QtWidgets.QPushButton(self)
        self._right_button.setText("Right")
        self._right_button.pressed.connect(self._right_pressed)
        self._right_button.released.connect(self._right_released)
        self._layout.addWidget(self._right_button, 1, 2)

        self._backward_button = QtWidgets.QPushButton(self)
        self._backward_button.setText("Backward")
        self._backward_button.pressed.connect(self._backward_pressed)
        self._backward_button.released.connect(self._backward_released)
        self._layout.addWidget(self._backward_button, 2, 1)

        self._turn_left_button = QtWidgets.QPushButton(self)
        self._turn_left_button.setText("Turn Left")
        self._turn_left_button.pressed.connect(self._turn_left_pressed)
        self._turn_left_button.released.connect(self._turn_left_released)
        self._layout.addWidget(self._turn_left_button, 2, 0)

        self._turn_right_button = QtWidgets.QPushButton(self)
        self._turn_right_button.setText("Turn Right")
        self._turn_right_button.pressed.connect(self._turn_right_pressed)
        self._turn_right_button.released.connect(self._turn_right_released)
        self._layout.addWidget(self._turn_right_button, 2, 2)

        self._layout.setRowStretch(0, 1)
        self._layout.setRowStretch(1, 1)
        self._layout.setRowStretch(2, 1)
        self._layout.setColumnStretch(0, 1)
        self._layout.setColumnStretch(1, 1)
        self._layout.setColumnStretch(2, 1)

    @abstractmethod
    def _forward_pressed(self) -> None:
        """Called when the forward button is pressed."""
        ...

    @abstractmethod
    def _forward_released(self) -> None:
        """Called when the forward button is released."""
        ...

    @abstractmethod
    def _left_pressed(self) -> None:
        """Called when the left button is pressed."""
        ...

    @abstractmethod
    def _left_released(self) -> None:
        """Called when the left button is released."""
        ...

    @abstractmethod
    def _right_pressed(self) -> None:
        """Called when the right button is pressed."""
        ...

    @abstractmethod
    def _right_released(self) -> None:
        """Called when the right button is released."""
        ...

    @abstractmethod
    def _backward_pressed(self) -> None:
        """Called when the backward button is pressed."""
        ...

    @abstractmethod
    def _backward_released(self) -> None:
        """Called when the backward button is released."""
        ...

    @abstractmethod
    def _turn_left_pressed(self) -> None:
        """Called when the turn left button is pressed."""
        ...

    @abstractmethod
    def _turn_left_released(self) -> None:
        """Called when the turn left button is released."""
        ...

    @abstractmethod
    def _turn_right_pressed(self) -> None:
        """Called when the turn right button is pressed."""
        ...

    @abstractmethod
    def _turn_right_released(self) -> None:
        """Called when the turn right button is released."""
        ...
