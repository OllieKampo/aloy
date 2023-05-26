"""
Module containing Jinx PyQt6 widgets defining interfaces for teleoperate
control of robots.
"""

from abc import abstractmethod
from PyQt6 import QtWidgets
from PyQt6 import QtCore
from PyQt6 import QtGui

from guis.gui import JinxGuiData, JinxObserverWidget


class LabelledArrowButton(QtWidgets.QToolButton):
    """
    Widget defining a labelled arrow button.
    """

    def __init__(
        self,
        qwidget: QtWidgets.QWidget, /,
        name: str,
        arrow_type: QtCore.Qt.ArrowType
    ) -> None:
        """Create a new labelled arrow button Jinx widget."""
        super().__init__(qwidget)

        self.__layout = QtWidgets.QVBoxLayout()

        self.__label = QtWidgets.QLabel()
        self.__label.setText(name)
        self.__layout.addWidget(self.__label)

        self.__button = QtWidgets.QToolButton()
        self.__button.setArrowType(arrow_type)
        self.__button.setAutoRepeat(True)
        self.__layout.addWidget(self.__button)

        self.__layout.setStretch(0, 1)
        self.__layout.setStretch(1, 1)

        self.setLayout(self.__layout)


class EstopInterface(JinxObserverWidget):
    """
    Widget defining an emergency stop interface for teleoperation of robots.
    """

    def __init__(
        self,
        qwidget: QtWidgets.QWidget, /,
        name: str,
        size: tuple[int, int], *,
        resize: bool = True,
        debug: bool = False
    ) -> None:
        """Create a new emergency stop interface Jinx widget."""
        super().__init__(
            qwidget,
            name,
            size,
            resize=resize,
            debug=debug
        )

        self.__layout = QtWidgets.QGridLayout()

        self.__start_time = QtCore.QTime.currentTime()

        self.__estop_button = QtWidgets.QPushButton()
        self.__estop_button.setText("E-Stop")
        self.__estop_button.setObjectName("estop_button")
        self.__estop_button.setFixedSize(
            self.size.width,
            self.size.height // 3
        )
        self.__layout.addWidget(self.__estop_button, 0, 0, 1, 3)

        self.__release_button = QtWidgets.QPushButton()
        self.__release_button.setText("Release E-Stop")
        self.__release_button.setObjectName("release_button")
        self.__release_button.setFixedSize(
            self.size.width,
            self.size.height // 3
        )
        self.__layout.addWidget(self.__release_button, 1, 0, 1, 3)

        self.__clock_label = QtWidgets.QLabel()
        self.__clock_label.setText("Time Elapsed:")
        self.__clock_label.setFixedSize(
            self.size.width // 3,
            self.size.height // 3
        )
        self.__layout.addWidget(self.__clock_label, 2, 0, 1, 1)

        self.__clock_display = QtWidgets.QLCDNumber()
        self.__clock_display.setDigitCount(8)
        self.__clock_display.setSegmentStyle(
            QtWidgets.QLCDNumber.SegmentStyle.Filled
        )
        # self.__clock_display.setFrameStyle(
        #     QtWidgets.QLCDNumber.FrameStyle.NoFrame
        # )
        self.__clock_display.setFixedSize(
            int(self.size.width * (2 / 3)),
            self.size.height // 3
        )
        self.__layout.addWidget(self.__clock_display, 2, 1, 1, 2)

        self.__timer = QtCore.QTimer()
        self.__timer.timeout.connect(self.__update_clock)
        self.__timer.start(1000)

        # self.__layout.setRowStretch(0, 1)
        # self.__layout.setColumnStretch(0, 1)

        self.qwidget.setLayout(self.__layout)

    def __update_clock(self) -> None:
        """Update the clock display."""
        time_ = QtCore.QTime.currentTime()
        time_ = self.__start_time.secsTo(time_)
        time_ = QtCore.QTime(0, 0, 0).addSecs(time_)
        self.__clock_display.display(
            time_.toString("hh:mm:ss")
        )

    def update_observer(self, observable_: JinxGuiData) -> None:
        pass


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

        self.__layout = QtWidgets.QGridLayout()

        self.__forward_button = QtWidgets.QPushButton()
        self.__forward_button.setText("Forward")
        self.__forward_button.setObjectName("forward_button")
        self.__forward_button.pressed.connect(self.button_pressed)
        self.__forward_button.released.connect(self.button_released)
        self.__forward_button.setFixedSize(  # TODO: Account for padding.
            self.size.width // 3,
            self.size.height // 2
        )
        self.__layout.addWidget(self.__forward_button, 0, 1)

        self.__left_button = QtWidgets.QPushButton()
        self.__left_button.setText("Left")
        self.__left_button.setObjectName("left_button")
        self.__left_button.pressed.connect(self.button_pressed)
        self.__left_button.released.connect(self.button_released)
        self.__left_button.setFixedSize(
            self.size.width // 3,
            self.size.height // 2
        )
        self.__layout.addWidget(self.__left_button, 1, 0)

        self.__right_button = QtWidgets.QPushButton()
        self.__right_button.setText("Right")
        self.__right_button.setObjectName("right_button")
        self.__right_button.pressed.connect(self.button_pressed)
        self.__right_button.released.connect(self.button_released)
        self.__right_button.setFixedSize(
            self.size.width // 3,
            self.size.height // 2
        )
        self.__layout.addWidget(self.__right_button, 1, 2)

        self.__backward_button = QtWidgets.QPushButton()
        self.__backward_button.setText("Backward")
        self.__backward_button.setObjectName("backward_button")
        self.__backward_button.pressed.connect(self.button_pressed)
        self.__backward_button.released.connect(self.button_released)
        self.__backward_button.setFixedSize(
            self.size.width // 3,
            self.size.height // 2
        )
        self.__layout.addWidget(self.__backward_button, 1, 1)

        self.__turn_left_button = QtWidgets.QPushButton()
        self.__turn_left_button.setText("Turn Left")
        self.__turn_left_button.setObjectName("turn_left_button")
        self.__turn_left_button.pressed.connect(self.button_pressed)
        self.__turn_left_button.released.connect(self.button_released)
        self.__turn_left_button.setFixedSize(
            self.size.width // 3,
            self.size.height // 2
        )
        self.__layout.addWidget(self.__turn_left_button, 0, 0)

        self.__turn_right_button = QtWidgets.QPushButton()
        self.__turn_right_button.setText("Turn Right")
        self.__turn_right_button.setObjectName("turn_right_button")
        self.__turn_right_button.pressed.connect(self.button_pressed)
        self.__turn_right_button.released.connect(self.button_released)
        self.__turn_right_button.setFixedSize(
            self.size.width // 3,
            self.size.height // 2
        )
        self.__layout.addWidget(self.__turn_right_button, 0, 2)

        self.__speed_slider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal
        )
        self.__speed_slider.setMinimum(0)
        self.__speed_slider.setMaximum(100)
        self.__speed_slider.setValue(50)
        self.__speed_slider.setTickPosition(
            QtWidgets.QSlider.TickPosition.TicksBelow
        )
        self.__speed_slider.setTickInterval(10)
        self.__speed_slider.setSingleStep(1)
        self.__speed_slider.setPageStep(10)
        self.__layout.addWidget(self.__speed_slider, 2, 0, 1, 3)

        self.__layout.setRowStretch(0, 1)
        self.__layout.setRowStretch(1, 1)
        self.__layout.setRowStretch(2, 1)
        self.__layout.setColumnStretch(0, 1)
        self.__layout.setColumnStretch(1, 1)
        self.__layout.setColumnStretch(2, 1)

        self.qwidget.setLayout(self.__layout)

    def update_observer(self, observable_: JinxGuiData) -> None:
        pass

    # @abstractmethod
    def button_pressed(self) -> None:
        """Called when a button is pressed."""
        button = self.qwidget.sender()
        print(f"Pressed: {button.objectName()!s}")

    def button_released(self) -> None:
        """Called when a button is released."""
        button = self.qwidget.sender()
        print(f"Released: {button.objectName()!s}")


qapp = QtWidgets.QApplication([])
qwindow = QtWidgets.QMainWindow()
qwindow.setWindowTitle("Control Interface")
qwindow.resize(300, 200)

estop_qwidget = QtWidgets.QWidget()
estop_jwidget = EstopInterface(estop_qwidget, "E-Stop Interface", (300, 200))

control_qwidget = QtWidgets.QWidget()
control_jwidget = DirectionalControlInterface(control_qwidget, "Control Interface", (300, 200))

conbined_qwidget = QtWidgets.QWidget()
combined_layout = QtWidgets.QVBoxLayout()
combined_layout.addWidget(estop_qwidget)
combined_layout.addWidget(control_qwidget)
conbined_qwidget.setLayout(combined_layout)

qwindow.setCentralWidget(conbined_qwidget)

qwindow.show()
qapp.exec()
