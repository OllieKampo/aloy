"""
Module containing Jinx PyQt6 widgets defining interfaces for teleoperate
control of robots.
"""

from abc import abstractmethod
from PyQt6 import QtWidgets
from PyQt6 import QtCore
from PyQt6 import QtGui

from guis.gui import JinxGuiData, JinxObserverWidget, JinxWidgetSize, scale_size, scale_size_for_grid


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
            set_size="fix",
            debug=debug
        )

        self.__layout = QtWidgets.QGridLayout()

        self.__start_time = QtCore.QTime.currentTime()

        padding = (10, 10)
        margins = (10, 10, 10, 10)
        widget_size = scale_size_for_grid(
            self.size,
            (3, 5),
            (3, 1),
            padding,
            margins
        )
        self.__layout.setContentsMargins(*margins)
        self.__layout.setHorizontalSpacing(padding[0])
        self.__layout.setVerticalSpacing(padding[1])

        # Bold text, red background
        self.__estop_button = QtWidgets.QPushButton()
        self.__estop_button.setText("E-Stop")
        self.__estop_button.setObjectName("estop_button")
        self.__estop_button.setFixedSize(*widget_size)
        self.__estop_button.setStyleSheet(
            "QPushButton#estop_button {"
            "   font-weight: bold;"
            "   background-color: red;"
            "}"
        )
        self.__layout.addWidget(self.__estop_button, 0, 0, 1, 3)

        widget_size = scale_size_for_grid(
            self.size,
            (3, 5),
            (2, 1),
            padding,
            margins
        )
        self.__estop_control_text = QtWidgets.QLabel()
        self.__estop_control_text.setText("You have E-Stop Control")
        self.__estop_control_text.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.__estop_control_text.setFixedSize(*widget_size)
        self.__layout.addWidget(self.__estop_control_text, 1, 0, 1, 2)

        widget_size = scale_size_for_grid(
            self.size,
            (3, 5),
            (1, 1),
            padding,
            margins
        )
        self.__release_button = QtWidgets.QPushButton()
        self.__release_button.setText("Release E-Stop")
        self.__release_button.setObjectName("release_button")
        self.__release_button.setFixedSize(*widget_size)
        self.__layout.addWidget(self.__release_button, 1, 2, 1, 1)

        widget_size = scale_size_for_grid(
            self.size,
            (3, 5),
            (1, 1),
            padding,
            margins
        )
        self.__clock_label = QtWidgets.QLabel()
        self.__clock_label.setText("Time Elapsed:")
        self.__clock_label.setFixedSize(*widget_size)
        self.__layout.addWidget(self.__clock_label, 2, 0, 1, 1)

        widget_size = scale_size_for_grid(
            self.size,
            (3, 5),
            (2, 1),
            padding,
            margins
        )
        self.__clock_display = QtWidgets.QLCDNumber()
        self.__clock_display.setDigitCount(8)
        self.__clock_display.setSegmentStyle(
            QtWidgets.QLCDNumber.SegmentStyle.Filled
        )
        # self.__clock_display.setFrameStyle(
        #     QtWidgets.QLCDNumber.FrameStyle.NoFrame
        # )
        self.__clock_display.setFixedSize(*widget_size)
        self.__layout.addWidget(self.__clock_display, 2, 1, 1, 2)

        self.__timer = QtCore.QTimer()
        self.__timer.timeout.connect(self.__update_clock)
        self.__timer.start(1000)

        widget_size = scale_size_for_grid(
            self.size,
            (3, 5),
            (3, 1),
            padding,
            margins
        )
        self.__power_label = QtWidgets.QLabel()
        self.__power_label.setText("Power:")
        self.__power_label.setFixedSize(*widget_size)
        self.__layout.addWidget(self.__power_label, 3, 0, 1, 1)

        self.__on_button = QtWidgets.QRadioButton()
        self.__on_button.setText("On")
        self.__on_button.setFixedSize(*widget_size)
        self.__layout.addWidget(self.__on_button, 3, 1, 1, 1)

        self.__off_button = QtWidgets.QRadioButton()
        self.__off_button.setText("Off")
        self.__off_button.setFixedSize(*widget_size)
        self.__off_button.setChecked(True)
        self.__layout.addWidget(self.__off_button, 3, 2, 1, 1)

        self.__on_off_button_group = QtWidgets.QButtonGroup()
        self.__on_off_button_group.setExclusive(True)
        self.__on_off_button_group.addButton(self.__on_button)
        self.__on_off_button_group.addButton(self.__off_button)
        self.__on_off_button_group.buttonClicked.connect(
            self.__on_off_button_clicked
        )

        widget_size = scale_size_for_grid(
            self.size,
            (3, 5),
            (1, 1),
            padding,
            margins
        )
        self.__motor_power_action_label = QtWidgets.QLabel()
        self.__motor_power_action_label.setText("Motor Power Action:")
        self.__motor_power_action_label.setFixedSize(*widget_size)
        self.__layout.addWidget(self.__motor_power_action_label, 4, 0, 1, 1)

        class AlignDelegate(QtWidgets.QStyledItemDelegate):
            def initStyleOption(self, option, index):
                super().initStyleOption(option, index)
                option.displayAlignment = QtCore.Qt.AlignmentFlag.AlignCenter

        widget_size = scale_size_for_grid(
            self.size,
            (3, 5),
            (2, 1),
            padding,
            margins
        )
        # Align options in the center of the combo box
        self.__motor_power_action_combo = QtWidgets.QComboBox()
        self.__motor_power_action_combo.setItemDelegate(AlignDelegate())
        self.__motor_power_action_combo.setEditable(True)
        self.__motor_power_action_combo.lineEdit().setReadOnly(True)
        self.__motor_power_action_combo.lineEdit().setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.__motor_power_action_combo.addItems(
            [
                "Cut Immediately",
                "Stop, Sit, and Cut"
            ]
        )
        self.__motor_power_action_combo.setCurrentIndex(1)
        self.__motor_power_action_combo.setFixedSize(*widget_size)
        self.__layout.addWidget(self.__motor_power_action_combo, 4, 1, 1, 2)

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

    def __on_off_button_clicked(self, button: QtWidgets.QRadioButton) -> None:
        """Handle the on/off button being clicked."""
        if button.text() == "On":
            self.__start_time = QtCore.QTime.currentTime()
            self.__update_clock()
            self.__timer.start()
        else:
            self.__timer.stop()

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
            set_size="fix",
            debug=debug
        )

        self.__group_box = QtWidgets.QGroupBox()
        self.__group_box.setTitle(name)
        self.__layout = QtWidgets.QGridLayout()
        self.__group_box.setLayout(self.__layout)
        _group_box_layout = QtWidgets.QVBoxLayout()
        _group_box_layout.addWidget(self.__group_box)
        self.qwidget.setLayout(_group_box_layout)

        padding = (10, 10)
        margins = (10, 10, 10, 10)
        widget_size = scale_size_for_grid(
            # scale_size(self.size, (0.9, 0.8)),
            (3, 2),
            (1, 1),
            padding
        )
        self.__layout.setContentsMargins(*margins)
        self.__layout.setHorizontalSpacing(padding[0])
        self.__layout.setVerticalSpacing(padding[1])

        self.__forward_button = QtWidgets.QPushButton()
        self.__forward_button.setText("Forward")
        self.__forward_button.setObjectName("forward_button")
        self.__connect_slots_and_set_size(
            self.__forward_button, widget_size
        )
        self.__layout.addWidget(self.__forward_button, 0, 1)

        self.__left_button = QtWidgets.QPushButton()
        self.__left_button.setText("Left")
        self.__left_button.setObjectName("left_button")
        self.__connect_slots_and_set_size(
            self.__left_button, widget_size
        )
        self.__layout.addWidget(self.__left_button, 1, 0)

        self.__right_button = QtWidgets.QPushButton()
        self.__right_button.setText("Right")
        self.__right_button.setObjectName("right_button")
        self.__connect_slots_and_set_size(
            self.__right_button, widget_size
        )
        self.__layout.addWidget(self.__right_button, 1, 2)

        self.__backward_button = QtWidgets.QPushButton()
        self.__backward_button.setText("Backward")
        self.__backward_button.setObjectName("backward_button")
        self.__connect_slots_and_set_size(
            self.__backward_button, widget_size
        )
        self.__layout.addWidget(self.__backward_button, 1, 1)

        self.__turn_left_button = QtWidgets.QPushButton()
        self.__turn_left_button.setText("Turn Left")
        self.__turn_left_button.setObjectName("turn_left_button")
        self.__connect_slots_and_set_size(
            self.__turn_left_button, widget_size
        )
        self.__layout.addWidget(self.__turn_left_button, 0, 0)

        self.__turn_right_button = QtWidgets.QPushButton()
        self.__turn_right_button.setText("Turn Right")
        self.__turn_right_button.setObjectName("turn_right_button")
        self.__connect_slots_and_set_size(
            self.__turn_right_button, widget_size
        )
        self.__layout.addWidget(self.__turn_right_button, 0, 2)

        # self.__speed_label = QtWidgets.QLabel()
        # self.__speed_label.setText("Speed:")
        # self.__layout.addWidget(self.__speed_label, 2, 0)

        # self.__speed_slider = QtWidgets.QSlider(
        #     QtCore.Qt.Orientation.Horizontal
        # )
        # self.__speed_slider.setMinimum(0)
        # self.__speed_slider.setMaximum(100)
        # self.__speed_slider.setValue(50)
        # self.__speed_slider.setTickPosition(
        #     QtWidgets.QSlider.TickPosition.TicksBelow
        # )
        # self.__speed_slider.setTickInterval(10)
        # self.__speed_slider.setSingleStep(1)
        # self.__speed_slider.setPageStep(10)
        # self.__layout.addWidget(self.__speed_slider, 2, 1, 1, 2)

        # self.__step_height_label = QtWidgets.QLabel()
        # self.__step_height_label.setText("Step Height:")
        # self.__layout.addWidget(self.__step_height_label, 3, 0)

        # self.__step_height_slider = QtWidgets.QSlider(
        #     QtCore.Qt.Orientation.Horizontal
        # )
        # self.__step_height_slider.setMinimum(0)
        # self.__step_height_slider.setMaximum(100)
        # self.__step_height_slider.setValue(50)
        # self.__step_height_slider.setTickPosition(
        #     QtWidgets.QSlider.TickPosition.TicksBelow
        # )
        # self.__step_height_slider.setTickInterval(10)
        # self.__step_height_slider.setSingleStep(1)
        # self.__step_height_slider.setPageStep(10)
        # self.__layout.addWidget(self.__step_height_slider, 3, 1, 1, 2)

        # self.__walk_height_label = QtWidgets.QLabel()
        # self.__walk_height_label.setText("Walk Height:")
        # self.__layout.addWidget(self.__walk_height_label, 4, 0)

        # self.__walk_height_slider = QtWidgets.QSlider(
        #     QtCore.Qt.Orientation.Horizontal
        # )
        # self.__walk_height_slider.setMinimum(0)
        # self.__walk_height_slider.setMaximum(100)
        # self.__walk_height_slider.setValue(50)
        # self.__walk_height_slider.setTickPosition(
        #     QtWidgets.QSlider.TickPosition.TicksBelow
        # )
        # self.__walk_height_slider.setTickInterval(10)
        # self.__walk_height_slider.setSingleStep(1)
        # self.__walk_height_slider.setPageStep(10)
        # self.__layout.addWidget(self.__walk_height_slider, 4, 1, 1, 2)

        # self.__layout.setRowStretch(0, 1)
        # self.__layout.setRowStretch(1, 1)
        # self.__layout.setRowStretch(2, 1)
        # self.__layout.setColumnStretch(0, 1)
        # self.__layout.setColumnStretch(1, 1)
        # self.__layout.setColumnStretch(2, 1)

    def __connect_slots_and_set_size(self, button: QtWidgets.QPushButton, size: JinxWidgetSize) -> None:
        """Connect slots and set the size of the button."""
        button.pressed.connect(self.button_pressed)
        button.released.connect(self.button_released)
        button.sizeHint = lambda: QtCore.QSize(size.width, size.height)
        button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )

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


if __name__ == "__main__":
    qapp = QtWidgets.QApplication([])
    qwindow = QtWidgets.QMainWindow()
    qwindow.setWindowTitle("Control Interface")
    qwindow.resize(400, 500)

    estop_qwidget = QtWidgets.QWidget()
    estop_jwidget = EstopInterface(estop_qwidget, "E-Stop Interface", (400, 233))

    control_qwidget = QtWidgets.QWidget()
    control_jwidget = DirectionalControlInterface(control_qwidget, "Control Interface", (400, 266))

    conbined_qwidget = QtWidgets.QWidget()
    combined_layout = QtWidgets.QVBoxLayout()
    combined_layout.addWidget(estop_qwidget)
    combined_layout.addWidget(control_qwidget)
    combined_layout.setSpacing(0)
    conbined_qwidget.setLayout(combined_layout)

    qwindow.setCentralWidget(conbined_qwidget)

    qwindow.show()
    qapp.exec()
