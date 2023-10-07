###############################################################################
# Copyright (C) 2023 Oliver Michael Kamperis
# Email: olliekampo@gmail.com
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

"""
Module containing Aloy PySide6 widgets defining interfaces for teleoperate
control of robots.
"""

from PySide6 import QtWidgets, QtCore

from aloy.guis.gui import (
    AloySystemData,
    AloyWidget,
    AloyWidgetSize,
    combine_aloy_widgets,
    scale_size,
    scale_size_for_grid
)
from aloy.robots.robotcontrol import AloyRobotControlData, AloyRobotControl


class EstopInterface(AloyWidget):
    """
    Widget defining an emergency stop interface for teleoperation of robots.
    """

    def __init__(
        self,
        qwidget: QtWidgets.QWidget | None = None,
        data: AloyRobotControlData | None = None,
        name: str = "E-Stop Interface",
        size: tuple[int, int] | None = None,
        resize: bool = True,
        debug: bool = False
    ) -> None:
        """Create a new emergency stop interface Aloy widget."""
        super().__init__(
            qwidget=qwidget,
            data=data,
            name=name,
            size=size,
            resize=resize,
            set_size="fix",
            debug=debug
        )

        self.__layout = QtWidgets.QGridLayout()

        self.__start_time = QtCore.QTime.currentTime()

        spacing = (10, 10)
        margins = (10, 10, 10, 10)
        widget_size = scale_size_for_grid(
            self.size,
            (3, 5),
            (3, 1),
            spacing,
            margins
        )
        self.__layout.setContentsMargins(*margins)
        self.__layout.setHorizontalSpacing(spacing[0])
        self.__layout.setVerticalSpacing(spacing[1])

        # Bold text, red background
        self.__estop_button = QtWidgets.QPushButton()
        self.__estop_button.setText("E-Stop")
        self.__estop_button.setObjectName("estop_button")
        self.__estop_button.setFixedSize(*widget_size)
        self.__estop_button.setStyleSheet(
            """
            QPushButton#estop_button {
               font-weight: bold;
               background-color: red;
            }
            """
        )
        self.__layout.addWidget(self.__estop_button, 0, 0, 1, 3)

        self.__has_estop_control: bool = False

        estop_control_layout = QtWidgets.QHBoxLayout()

        widget_size = scale_size_for_grid(
            self.size,
            (2, 5),
            (1, 1),
            spacing,
            margins
        )
        self.__estop_control_text = QtWidgets.QLabel()
        if self.__has_estop_control:
            self.__estop_control_text.setText("You have E-Stop Control")
        else:
            self.__estop_control_text.setText("You don't have E-Stop Control")
        self.__estop_control_text.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.__estop_control_text.setFixedSize(*widget_size)
        estop_control_layout.addWidget(self.__estop_control_text)

        widget_size = scale_size_for_grid(
            self.size,
            (2, 5),
            (1, 1),
            spacing,
            margins
        )
        self.__release_button = QtWidgets.QPushButton()
        if self.__has_estop_control:
            self.__release_button.setText("Release E-Stop")
        else:
            self.__release_button.setText("Acquire E-Stop Control")
        self.__release_button.setObjectName("release_button")
        self.__release_button.setFixedSize(*widget_size)
        self.__release_button.setStyleSheet(
            """
            QPushButton#release_button {
               font-weight: bold;
               background-color: green;
            }
            """
        )
        self.__release_button.clicked.connect(self.__release_button_clicked)
        estop_control_layout.addWidget(self.__release_button)

        self.__layout.addLayout(estop_control_layout, 1, 0, 1, 3)

        widget_size = scale_size_for_grid(
            self.size,
            (3, 5),
            (1, 1),
            spacing,
            margins
        )
        self.__clock_label = QtWidgets.QLabel()
        self.__clock_label.setText("Time Elapsed:")
        self.__clock_label.setFixedSize(*widget_size)
        self.__layout.addWidget(self.__clock_label, 2, 0, 1, 1)

        self.__power_on: bool = False

        widget_size = scale_size_for_grid(
            self.size,
            (3, 5),
            (2, 1),
            spacing,
            margins
        )
        self.__clock_display = QtWidgets.QLCDNumber()
        self.__clock_display.setDigitCount(8)
        self.__clock_display.setSegmentStyle(
            QtWidgets.QLCDNumber.SegmentStyle.Filled
        )
        time_ = QtCore.QTime(0, 0, 0)
        self.__clock_display.display(
            time_.toString("hh:mm:ss")
        )
        self.__clock_display.setFixedSize(*widget_size)
        self.__layout.addWidget(self.__clock_display, 2, 1, 1, 2)

        self.__timer = QtCore.QTimer()
        self.__timer.timeout.connect(self.__update_clock)
        if self.__power_on:
            self.__timer.start(1000)

        widget_size = scale_size_for_grid(
            self.size,
            (3, 5),
            (3, 1),
            spacing,
            margins
        )
        self.__power_label = QtWidgets.QLabel()
        self.__power_label.setText("Power:")
        self.__power_label.setFixedSize(*widget_size)
        self.__layout.addWidget(self.__power_label, 3, 0, 1, 1)

        self.__on_button = QtWidgets.QRadioButton()
        self.__on_button.setText("On")
        self.__on_button.setFixedSize(*widget_size)
        self.__on_button.setChecked(self.__power_on)
        self.__layout.addWidget(self.__on_button, 3, 1, 1, 1)

        self.__off_button = QtWidgets.QRadioButton()
        self.__off_button.setText("Off")
        self.__off_button.setFixedSize(*widget_size)
        self.__off_button.setChecked(not self.__power_on)
        self.__layout.addWidget(self.__off_button, 3, 2, 1, 1)

        self.__on_off_button_group = QtWidgets.QButtonGroup()
        self.__on_off_button_group.setExclusive(True)
        self.__on_off_button_group.addButton(self.__on_button)
        self.__on_off_button_group.addButton(self.__off_button)
        self.__on_off_button_group.buttonClicked.connect(
            self.__on_off_button_clicked
        )
        self.__waiting_for_power_on: bool = False
        self.__waiting_for_power_off: bool = False

        widget_size = scale_size_for_grid(
            self.size,
            (3, 5),
            (1, 1),
            spacing,
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
            spacing,
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
                "Immediate Stop",
                "Safe Stop"
            ]
        )
        self.__motor_power_action_combo.setCurrentIndex(1)
        self.__motor_power_action_combo.setFixedSize(*widget_size)
        self.__layout.addWidget(self.__motor_power_action_combo, 4, 1, 1, 2)

        self.qwidget.setLayout(self.__layout)

    @property
    def data(self) -> AloyRobotControlData:
        return super().data

    def __update_clock(self) -> None:
        """Update the clock display."""
        qtime = QtCore.QTime.currentTime()
        stime = self.__start_time.secsTo(qtime)
        qtime = QtCore.QTime(0, 0, 0).addSecs(stime)
        self.__clock_display.display(
            qtime.toString("hh:mm:ss")
        )

    def __release_button_clicked(self) -> None:
        """Handle the released button being clicked."""
        self.__has_estop_control = not self.__has_estop_control
        if self.__has_estop_control:
            self.data.robot_control.acquire_estop_control()
            self.__estop_control_text.setText("You have E-Stop Control")
            self.__release_button.setText("Release E-Stop")
        else:
            self.data.robot_control.release_estop_control()
            self.__estop_control_text.setText("You don't have E-Stop Control")
            self.__release_button.setText("Acquire E-Stop Control")

    def __on_off_button_clicked(self, button: QtWidgets.QRadioButton) -> None:
        """Handle the on/off button being clicked."""
        if button.text() == "On":
            self.__waiting_for_power_on = True
            self.data.robot_control.request_motor_power_on()
        else:
            self.__waiting_for_power_off = True
            self.data.robot_control.request_motor_power_off()
        for button_ in self.__on_off_button_group.buttons():
            button_.setEnabled(False)

    def __confirm_power_on(self) -> bool:
        """Confirm that the power is on."""
        if self.data.robot_control.power_on:
            self.__waiting_for_power_on = False
            self.__power_on = True
            self.__start_time = QtCore.QTime.currentTime()
            self.__update_clock()
            self.__timer.start()
            return True
        return False

    def __confirm_power_off(self) -> bool:
        """Confirm that the power is off."""
        if not self.data.robot_control.power_on:
            self.__waiting_for_power_off = False
            self.__power_on = False
            self.__timer.stop()
            return True
        return False

    def update_observer(self, observable_: AloySystemData) -> None:
        """Update the observer."""
        if self.data is observable_:
            if self.__waiting_for_power_on:
                confirmed = self.__confirm_power_on()
            if self.__waiting_for_power_off:
                confirmed = self.__confirm_power_off()
            if confirmed:
                for button in self.__on_off_button_group.buttons():
                    button.setEnabled(True)


class DirectionalControlInterface(AloyWidget):
    """
    Widget defining a directional control interface for teleoperation of
    robots.
    """

    def __init__(
        self,
        qwidget: QtWidgets.QWidget | None = None,
        data: AloySystemData | None = None,
        name: str = "Directional Control Interface",
        size: tuple[int, int] | None = None,
        resize: bool = True,
        debug: bool = False
    ) -> None:
        """Create a new directional control interface Aloy widget."""
        super().__init__(
            qwidget=qwidget,
            data=data,
            name=name,
            size=size,
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

        spacing = (10, 10)
        margins = (10, 10, 10, 10)
        widget_size = scale_size_for_grid(
            self.size,
            (3, 2),
            (1, 1),
            spacing,
            margins
        )
        self.__layout.setContentsMargins(*margins)
        self.__layout.setHorizontalSpacing(spacing[0])
        self.__layout.setVerticalSpacing(spacing[1])

        self.__forward_button = QtWidgets.QPushButton()
        self.__forward_button.setText("Forward")
        self.__forward_button.setObjectName("forward_button")
        self.__set_size(self.__forward_button, widget_size)
        self.__layout.addWidget(self.__forward_button, 0, 1)

        self.__left_button = QtWidgets.QPushButton()
        self.__left_button.setText("Left")
        self.__left_button.setObjectName("left_button")
        self.__set_size(self.__left_button, widget_size)
        self.__layout.addWidget(self.__left_button, 1, 0)

        self.__right_button = QtWidgets.QPushButton()
        self.__right_button.setText("Right")
        self.__right_button.setObjectName("right_button")
        self.__set_size(self.__right_button, widget_size)
        self.__layout.addWidget(self.__right_button, 1, 2)

        self.__backward_button = QtWidgets.QPushButton()
        self.__backward_button.setText("Backward")
        self.__backward_button.setObjectName("backward_button")
        self.__set_size(self.__backward_button, widget_size)
        self.__layout.addWidget(self.__backward_button, 1, 1)

        self.__turn_left_button = QtWidgets.QPushButton()
        self.__turn_left_button.setText("Turn Left")
        self.__turn_left_button.setObjectName("turn_left_button")
        self.__set_size(self.__turn_left_button, widget_size)
        self.__layout.addWidget(self.__turn_left_button, 0, 0)

        self.__turn_right_button = QtWidgets.QPushButton()
        self.__turn_right_button.setText("Turn Right")
        self.__turn_right_button.setObjectName("turn_right_button")
        self.__set_size(self.__turn_right_button, widget_size)
        self.__layout.addWidget(self.__turn_right_button, 0, 2)

        self.__directional_buttons = QtWidgets.QButtonGroup()
        self.__directional_buttons.addButton(self.__forward_button)
        self.__directional_buttons.addButton(self.__left_button)
        self.__directional_buttons.addButton(self.__right_button)
        self.__directional_buttons.addButton(self.__backward_button)
        self.__directional_buttons.addButton(self.__turn_left_button)
        self.__directional_buttons.addButton(self.__turn_right_button)
        self.__directional_buttons.buttonPressed.connect(
            self.directional_button_pressed
        )
        self.__directional_buttons.buttonReleased.connect(
            self.directional_button_released
        )

        widget_size = scale_size(widget_size, (2.0, 0.20))
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
        self.__set_size(self.__speed_slider, widget_size)
        self.__speed_slider.valueChanged.connect(self.speed_slider_changed)
        self.__layout.addWidget(self.__speed_slider, 2, 1, 1, 2)

        self.__speed_label = QtWidgets.QLabel()
        self.__speed_label.setText(
            f"Desired speed: {self.__speed_slider.value()!s}%")
        self.__set_size(self.__speed_label, widget_size)
        self.__layout.addWidget(self.__speed_label, 2, 0, 1, 1)

    def __set_size(
        self,
        button: QtWidgets.QWidget,
        size: AloyWidgetSize
    ) -> None:
        """Connect slots and set the size of the button."""
        def get_size() -> QtCore.QSize:
            return QtCore.QSize(size.width, size.height)
        button.sizeHint = get_size  # type: ignore
        button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )

    def update_observer(self, observable_: AloySystemData) -> None:
        """Update the observer with the latest data."""
        pass

    def directional_button_pressed(
        self,
        button: QtWidgets.QPushButton
    ) -> None:
        """Called when a button is pressed."""
        print(f"Pressed: {button.objectName()!s}")

    def directional_button_released(
        self,
        button: QtWidgets.QPushButton
    ) -> None:
        """Called when a button is released."""
        print(f"Released: {button.objectName()!s}")

    def speed_slider_changed(self, value: int) -> None:
        """Called when the speed slider is changed."""
        self.__speed_label.setText(f"Desired speed: {value!s}%")
        print(f"Speed set to: {value!s}")


def __main() -> None:
    size = (425, 500)
    qapp = QtWidgets.QApplication([])
    qwindow = QtWidgets.QMainWindow()
    qwindow.setWindowTitle("Control Interface")
    qwindow.resize(*size)

    qtimer = QtCore.QTimer()
    arobotcontrol = AloyRobotControl(qtimer=qtimer)
    arobotcontroldata = AloyRobotControlData(arobotcontrol, clock=qtimer)
    estop_awidget = EstopInterface(
        data=arobotcontroldata,
        size=scale_size(size, (1.0, 0.45))
    )
    control_awidget = DirectionalControlInterface(
        data=arobotcontroldata,
        size=scale_size(size, (1.0, 0.55))
    )

    combine_aloy_widgets(
        awidgets=[
            estop_awidget,
            control_awidget
        ],
        kind="vertical",
        stretches=[1, 1],
        spacing=0,
        alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
        parent=qwindow
    )

    qwindow.show()
    qapp.exec()


if __name__ == "__main__":
    __main()
