###########################################################################
###########################################################################
## Module defining GUI classes.                                          ##
##                                                                       ##
## Copyright (C)  2023  Oliver Michael Kamperis                          ##
## Email: o.m.kamperis@gmail.com                                         ##
##                                                                       ##
## This program is free software: you can redistribute it and/or modify  ##
## it under the terms of the GNU General Public License as published by  ##
## the Free Software Foundation, either version 3 of the License, or     ##
## any later version.                                                    ##
##                                                                       ##
## This program is distributed in the hope that it will be useful,       ##
## but WITHOUT ANY WARRANTY; without even the implied warranty of        ##
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          ##
## GNU General Public License for more details.                          ##
##                                                                       ##
## You should have received a copy of the GNU General Public License     ##
## along with this program. If not, see <https://www.gnu.org/licenses/>. ##
###########################################################################
###########################################################################

"""Module defining GUI classes."""

from typing import Any
from PyQt6 import QtWidgets
from PyQt6.QtCore import QTimer  # pylint: disable=E0611
from concurrency.clocks import ClockThread
from concurrency.synchronization import atomic_update

import guis.observable as observable

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"

__all__ = (
    "JinxGuiData",
    "JinxObserverWidget",
    "JinxGuiWindow"
)


def __dir__() -> tuple[str, ...]:
    """Get the names of module attributes."""
    return __all__


class JinxGuiData(observable.Observable):
    """A class defining a gui data object."""

    __slots__ = {
        "__weakref__": "Weak reference to the object.",
        "__desired_view_state": "The currently desired view state.",
        "__view_states": "The list of view states.",
        "__data": "Arbitrary data associated with the gui."
    }

    def __init__(
        self,
        name: str | None = None,
        data_dict: dict[str, Any] | None = None, /,
        clock: ClockThread | QTimer | None = None, *,
        debug: bool = False
    ) -> None:
        """
        Create a new Jinx gui data.

        Parameters
        ----------
        `name: str | None = None` - The name of the object.
        If not given or None, a unique name will be generated.
        See `jinx.guis.observable.Observable` for details.

        `data_dict: dict[str, Any] | None = None` - A data dictionary
        to be copied into the gui data object.

        `clock: ClockThread | QTimer | None = None` - The clock
        thread or timer to be used for the observable object.
        If not given or None, a new clock thread will be created.
        See `jinx.guis.observable.Observable` for details.

        `debug: bool = False`  - Whether to log debug messages.
        """
        super().__init__(name, clock, debug=debug)
        self.__desired_view_state: str | None = None
        self.__view_states: list[str] = []
        self.__data: dict[str, Any]
        if data_dict is None:
            self.__data = {}
        else:
            self.__data = data_dict.copy()

    def add_view_state(self, view_state: str) -> None:
        """Add a new view state name."""
        self.__view_states.append(view_state)

    def remove_view_state(self, view_state: str) -> None:
        """Remove a view state name."""
        self.__view_states.remove(view_state)

    @property
    def desired_view_state(self) -> str | None:
        """Get the current desired view state name."""
        return self.__desired_view_state

    @desired_view_state.setter
    @observable.notifies_observers()
    def desired_view_state(self, desired_view_state: str | None) -> None:
        """Set the current desired view state name."""
        self.__desired_view_state = desired_view_state

    @atomic_update("data", method=True)
    def get_data(self, key: str, default: Any = None) -> Any:
        """Get the data associated with the given key."""
        return self.__data.get(key, default)

    @observable.notifies_observers()
    def set_data(self, key: str, value: Any) -> None:
        """Set the data associated with the given key."""
        self.__data[key] = value


class JinxObserverWidget(observable.Observer):
    """A class defining an observer widget."""

    __slots__ = {
        "__weakref__": "Weak reference to the object.",
        "__widget": "The encapsulated widget."
    }

    def __init__(
        self,
        widget: QtWidgets.QWidget, /,
        name: str | None = None, *,
        debug: bool = False
    ) -> None:
        """Create a new Jinx widget within the given parent widget."""
        super().__init__(name, debug=debug)
        self.__widget: QtWidgets.QWidget = widget

    @property
    def widget(self) -> QtWidgets.QWidget:
        """Get the widget."""
        return self.__widget

    def update_observer(self, observable: JinxGuiData) -> None:
        """
        Update the observer.

        This method should be overridden by subclasses.
        """
        pass


class JinxGuiWindow(observable.Observer):
    """A class defining a PyQt6 window used by Jinx."""

    __slots__ = {
        "__weakref__": "Weak reference to the object.",
        "__window": "The main window.",
        "__main_widget": "The main widget.",
        "__vbox": "The main vertical box layout.",
        "__stack": "The stacked widget for the views.",
        "__combo_box": "The combo box for the views selection.",
        "__data": "The Jinx gui data for the window.",
        "__views": "The views of the window.",
        "__current_view_state": "The current view state of the window.",
        "__default_widget": "The default widget."
    }

    def __init__(
        self,
        window: QtWidgets.QMainWindow,
        data: JinxGuiData,
        name: str | None = None, *,
        debug: bool = False
    ) -> None:
        """Create a new Jinx window within the given parent widget."""
        super().__init__(name, debug=debug)

        # self.__window: QtWidgets.QWidget = window
        self.__main_widget = QtWidgets.QWidget()
        # self.__window.setCentralWidget(self.__main_widget)
        window.setCentralWidget(self.__main_widget)

        self.__vbox = QtWidgets.QVBoxLayout()
        self.__stack = QtWidgets.QStackedWidget()
        self.__combo_box = QtWidgets.QComboBox()
        self.__combo_box.currentTextChanged.connect(
            self.__combo_box_changed
        )
        self.__vbox.addWidget(self.__combo_box)
        self.__vbox.addWidget(self.__stack)
        self.__main_widget.setLayout(self.__vbox)

        self.__data: JinxGuiData = data
        self.__data.assign_observers(self)

        self.__views: dict[str, JinxObserverWidget] = {}
        self.__current_view_state: str | None = None
        self.__default_widget = QtWidgets.QWidget()

        self.__data.notify_all()

    @property
    def current_view_state(self) -> str | None:
        """Get the current view state of the window."""
        return self.__current_view_state

    def __combo_box_changed(self, view_name: str) -> None:
        """Handle a change in the combo box."""
        self.__data.desired_view_state = view_name

    def add_view(self, name: str, widget: JinxObserverWidget) -> None:
        """Add a new view state to the window."""
        self.__views[name] = widget
        self.__stack.addWidget(widget.widget)
        self.__combo_box.addItem(name)
        self.__data.add_view_state(name)
        if self.__current_view_state is None:
            self.__data.assign_observers(widget)
            self.__current_view_state = name
            self.__data.desired_view_state = name

    def remove_view(self, name: str) -> None:
        """Remove a view state from the window."""
        self.__views.pop(name)
        self.__stack.removeWidget(self.__views[name].widget)
        self.__combo_box.removeItem(self.__combo_box.findText(name))
        self.__data.remove_view_state(name)
        if self.__current_view_state == name:
            self.__data.remove_observers(self.__views[name])
            self.__data.desired_view_state = None

    def update_observer(self, observable: JinxGuiData) -> None:
        """Update the observer."""
        desired_view_state: str | None = self.__data.desired_view_state
        if self.__current_view_state != desired_view_state:
            if self.__current_view_state is not None:
                jinx_widget = self.__views[self.__current_view_state]
                self.__data.remove_observers(jinx_widget)
            if desired_view_state is not None:
                jinx_widget = self.__views[desired_view_state]
                qt_widget = jinx_widget.widget
                self.__stack.setCurrentWidget(qt_widget)
                self.__data.assign_observers(jinx_widget)
                self.__data.notify(jinx_widget)
            else:
                self.__stack.setCurrentWidget(self.__default_widget)
            self.__current_view_state = desired_view_state
