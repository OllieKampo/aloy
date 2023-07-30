"""Module defining the Aloy robot control system."""

from typing import Any
from PySide6.QtCore import QTimer  # pylint: disable=E0611

from aloy.concurrency.clocks import ClockThread
from aloy.guis.gui import AloyGuiWindow, AloySystemData
from aloy.guis.observable import Observable, notifies_observers


class AloyRobotControl(Observable):
    """A class defining a Aloy robot control object."""

    def __init__(self, qtimer: QTimer) -> None:
        """Create a new Aloy robot control object."""
        super().__init__(clock=qtimer)
        self.power: bool = False

    def acquire_estop_control(self) -> None:
        pass

    def release_estop_control(self) -> None:
        pass

    def request_robot_control(self) -> None:
        pass

    def release_robot_control(self) -> None:
        pass

    @notifies_observers()
    def request_motor_power_on(self) -> None:
        self.power = True

    @notifies_observers()
    def request_motor_power_off(self) -> None:
        self.power = False

    @property
    def power_on(self) -> bool:
        return self.power


class AloyRobotControlData(AloySystemData):
    """A class defining a Aloy robot control data object."""

    __slots__ = {
        "__robot_control": "The robot control object connected to this data "
                           "object."
    }

    def __init__(
        self,
        robot_control: AloyRobotControl,
        name: str | None = None,
        gui: AloyGuiWindow | None = None,
        data_dict: dict[str, Any] | None = None,
        clock: ClockThread | QTimer | None = None, *,
        debug: bool = False
    ) -> None:
        """
        Create a new Aloy robot control data object.

        Parameters
        ----------
        `robot_control: AloyRobotControl` - The robot control object
        connected to this data object.

        `name: str | None = None` - The name of the object.
        If not given or None, a unique name will be generated.
        See `aloy.guis.observable.Observable` for details.

        `gui: AloyGuiWindow | None = None` - The gui object to be
        connected to this system data object.

        `data_dict: dict[str, Any] | None = None` - A data dictionary
        to be copied into the system data object.

        `clock: ClockThread | QTimer | None = None` - The clock
        thread or timer to be used for the observable object.
        If not given or None, a new clock thread will be created.
        See `aloy.guis.observable.Observable` for details.

        `debug: bool = False` - Whether to log debug messages.
        """
        super().__init__(name, gui, data_dict, clock, debug=debug)
        self.__robot_control: AloyRobotControl = robot_control
        self.__robot_control.chain_notifies_to(self)

    @property
    def robot_control(self) -> AloyRobotControl:
        """Get the robot control object connected to this data object."""
        return self.__robot_control
