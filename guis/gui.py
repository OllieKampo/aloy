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

from abc import abstractmethod
from typing import Any, Literal, NamedTuple
from PyQt6 import QtWidgets
from PyQt6 import QtCore
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


def scale_size(
    size: tuple[int, int],
    scale: tuple[float, float]
) -> tuple[int, int]:
    """Scale a size by the given factor."""
    return (
        int(round(size[0] * scale[0])),
        int(round(size[1] * scale[1]))
    )


def scale_size_for_grid(
    size: tuple[int, int],
    grid_size: tuple[int, int],
    widget_size: tuple[int, int]
) -> tuple[int, int]:
    """Scale a size to fit a grid."""
    scale = (
        grid_size[0] / widget_size[0],
        grid_size[1] / widget_size[1]
    )
    return scale_size(size, scale)


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

    @atomic_update("view_states", method=True)
    def add_view_state(self, view_state: str) -> None:
        """Add a new view state name."""
        self.__view_states.append(view_state)

    @atomic_update("view_states", method=True)
    def remove_view_state(self, view_state: str) -> None:
        """Remove a view state name."""
        self.__view_states.remove(view_state)

    @property
    @atomic_update("desired_view_state", method=True)
    def desired_view_state(self) -> str | None:
        """Get the current desired view state name."""
        return self.__desired_view_state

    @desired_view_state.setter
    @atomic_update("desired_view_state", method=True)
    @observable.notifies_observers()
    def desired_view_state(self, desired_view_state: str | None) -> None:
        """Set the current desired view state name."""
        self.__desired_view_state = desired_view_state

    @atomic_update("data", method=True)
    def get_data(self, key: str, default: Any = None) -> Any:
        """Get the data associated with the given key."""
        return self.__data.get(key, default)

    @atomic_update("data", method=True)
    @observable.notifies_observers()
    def set_data(self, key: str, value: Any) -> None:
        """Set the data associated with the given key."""
        self.__data[key] = value

    @atomic_update("data", method=True)
    @observable.notifies_observers()
    def del_data(self, key: str) -> None:
        """Delete the data associated with the given key."""
        self.__data.pop(key)


class JinxWidgetSize(NamedTuple):
    """Tuple representing the size of a Jinx widget."""

    width: int
    height: int


class JinxObserverWidget(observable.Observer):
    """A class defining an observer widget."""

    __slots__ = {
        "__weakref__": "Weak reference to the object.",
        "__qwidget": "The encapsulated widget.",
        "__size": "The size of the widget."
    }

    def __init__(
        self,
        qwidget: QtWidgets.QWidget, /,
        name: str | None = None,
        size: tuple[int, int] | None = None, *,
        resize: bool = True,
        set_size: Literal["fix", "base", "min", "max"] | None = None,
        debug: bool = False
    ) -> None:
        """
        Create a new Jinx widget wrapping the given parent widget.

        Parameters
        ----------
        `qwidget: QtWidgets.QWidget` - The parent widget.

        `name: str | None = None` - The name of the object. If None, the
        class name and id of the object are used.

        `size: tuple[int, int] | None = None` - The size of the widget in
        pixels (width, height). If None, the size of the widget is not set.

        `resize: bool = True` - Whether to resize the parent widget to the
        given size, or just simply store the size.

        `set_size: Literal["fix", "base", "min", "max"] | None = None` -
        Whether to set the size of the widget to the given size. If 'fix'
        then the size is fixed, if 'base' then the base size is set, if
        'min' then the minimum size is set, if 'max' then the maximum size
        is set. If None, the size of the widget is not set.

        `debug: bool = False` - Whether to log debug messages.
        """
        super().__init__(name, debug=debug)
        self.__qwidget: QtWidgets.QWidget = qwidget
        self.__size: JinxWidgetSize | None = None
        if size is not None:
            self.__size = JinxWidgetSize(*size)
            if resize:
                self.__qwidget.resize(*size)
            if set_size is not None:
                if set_size == "fix":
                    self.__qwidget.setFixedSize(*size)
                elif set_size == "base":
                    self.__qwidget.setBaseSize(*size)
                elif set_size == "min":
                    self.__qwidget.setMinimumSize(*size)
                elif set_size == "max":
                    self.__qwidget.setMaximumSize(*size)
                else:
                    raise ValueError(
                        f"Invalid set_size value: {set_size}."
                        f"Choose from: fix, base, min, max."
                    )
        if name is not None:
            self.__qwidget.setObjectName(name)

    @property
    def qwidget(self) -> QtWidgets.QWidget:
        """Get the qt widget."""
        return self.__qwidget

    @property
    def size(self) -> JinxWidgetSize | None:
        """Get the size of the widget."""
        return self.__size

    @abstractmethod
    def update_observer(self, observable_: JinxGuiData) -> None:
        """Update the observer."""
        return super().update_observer(observable_)


class JinxGuiWindow(observable.Observer):
    """A class defining a PyQt6 window used by Jinx."""

    view_changed = QtCore.pyqtSignal(str)
    view_added = QtCore.pyqtSignal(str)
    view_removed = QtCore.pyqtSignal(str)

    __slots__ = {
        "__weakref__": "Weak reference to the object.",
        "__qwindow": "The main window.",
        "__main_qwidget": "The main widget.",
        "__vbox": "The main vertical box layout.",
        "__stack": "The stacked widget for the views.",
        "__kind": "Whether the views are tabbed or not.",
        "__combo_box": "The combo box for the views selection.",
        "__tab_bar": "The tab bar for the views selection.",
        "__data": "The Jinx gui data for the window.",
        "__views": "The views of the window.",
        "__current_view_state": "The current view state of the window.",
        "__default_qwidget": "The default widget."
    }

    def __init__(
        self,
        qwindow: QtWidgets.QMainWindow,
        data: JinxGuiData,
        name: str | None = None,
        size: tuple[int, int] | None = None, /,
        kind: Literal["tabbed", "combo"] | None = "tabbed",
        set_title: bool = True,
        resize: bool = True,
        debug: bool = False
    ) -> None:
        """
        Create a new Jinx window within the given main window.

        This creates a new widget and sets it as the central widget of
        the main window.

        Parameters
        ----------
        `qwindow: QtWidgets.QMainWindow` - The main window.

        `data: JinxGuiData` - The Jinx gui data for the window.

        `name: str | None = None` - The name of the object. If None, the
        class name and id of the object are used.

        `size: tuple[int, int] | None = None` - The size of the window in
        pixels (width, height). If None, the size of the window is not set.

        `tabbed: bool = True` - Whether the views are tabbed or to use a
        combo box drop-down menu to select the views.

        `set_title: bool = True` - Whether to set the title of the window
        to the given name.

        `resize: bool = True` - Whether to resize the window to the given
        size, or just simply store the size.

        `debug: bool = False` - Whether to log debug messages.
        """
        super().__init__(name, debug=debug)

        self.__qwindow: QtWidgets.QWidget = qwindow
        if name is not None and set_title:
            self.__qwindow.setWindowTitle(name)
        if size is not None and resize:
            self.__qwindow.resize(*size)
        self.__main_qwidget = QtWidgets.QWidget()
        self.__qwindow.setCentralWidget(self.__main_qwidget)

        self.__vbox = QtWidgets.QVBoxLayout()
        self.__stack = QtWidgets.QStackedWidget()
        self.__kind: str | None = kind
        if self.__kind == "tabbed":
            self.__tab_bar = QtWidgets.QTabBar()
            self.__tab_bar.currentChanged.connect(
                self.__tab_bar_changed
            )
            self.__vbox.addWidget(self.__tab_bar)
        elif self.__kind == "combo":
            self.__combo_box = QtWidgets.QComboBox()
            self.__combo_box.currentTextChanged.connect(
                self.__combo_box_changed
            )
            self.__vbox.addWidget(self.__combo_box)
        self.__vbox.addWidget(self.__stack)
        self.__main_qwidget.setLayout(self.__vbox)

        self.__data: JinxGuiData = data
        self.__data.assign_observers(self)

        self.__views: dict[str, JinxObserverWidget] = {}
        self.__current_view_state: str | None = None
        self.__default_qwidget = QtWidgets.QWidget()

        self.__data.notify_all()

    @property
    def current_view_state(self) -> str | None:
        """Get the current view state of the window."""
        return self.__current_view_state

    def __combo_box_changed(self, view_name: str) -> None:
        """Handle a change in the combo box."""
        self.__data.desired_view_state = view_name

    def __tab_bar_changed(self, index: int) -> None:
        """Handle a change in the tab bar."""
        self.__data.desired_view_state = self.__tab_bar.tabText(index)

    def add_view(self, name: str, widget: JinxObserverWidget) -> None:
        """Add a new view state to the window."""
        self.__views[name] = widget
        self.__stack.addWidget(widget.qwidget)
        if self.__kind == "tabbed":
            self.__tab_bar.addTab(name)
        elif self.__kind == "combo":
            self.__combo_box.addItem(name)
        else:
            self.view_added.emit(name)
        self.__data.add_view_state(name)
        if self.__current_view_state is None:
            self.__data.assign_observers(widget)
            self.__current_view_state = name
            self.__data.desired_view_state = name

    def remove_view(self, name: str) -> None:
        """Remove a view state from the window."""
        self.__stack.removeWidget(self.__views[name].widget)
        self.__views.pop(name)
        if self.__kind == "tabbed":
            for index in range(self.__tab_bar.count()):
                if self.__tab_bar.tabText(index) == name:
                    self.__tab_bar.removeTab(index)
                    break
        elif self.__kind == "combo":
            self.__combo_box.removeItem(self.__combo_box.findText(name))
        else:
            self.view_removed.emit(name)
        self.__data.remove_view_state(name)
        if self.__current_view_state == name:
            self.__data.remove_observers(self.__views[name])
            self.__data.desired_view_state = None

    def update_observer(self, observable_: JinxGuiData) -> None:
        """Update the observer."""
        desired_view_state: str | None = self.__data.desired_view_state
        if self.__current_view_state != desired_view_state:
            self.view_changed.emit(desired_view_state)
            if self.__current_view_state is not None:
                jwidget = self.__views[self.__current_view_state]
                self.__data.remove_observers(jwidget)
            if desired_view_state is not None:
                jwidget = self.__views[desired_view_state]
                qwidget = jwidget.qwidget
                self.__stack.setCurrentWidget(qwidget)
                self.__data.assign_observers(jwidget)
                self.__data.notify(jwidget)
            else:
                self.__stack.setCurrentWidget(self.__default_qwidget)
            self.__current_view_state = desired_view_state
