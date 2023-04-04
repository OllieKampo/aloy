
import functools
from typing import Any, Callable
from PyQt6 import QtCore, QtGui, QtWidgets

import guis.observable as observable


class JinxGuiData(observable.Observable):

    __slots__ = {
        "__desired_view_state": "The currently desired view state.",
        "__view_states": "The list of view states.",
        "__data": "Arbitrary data associated with the gui."
    }

    def __init__(self, data_dict: dict[str, Any] = {}) -> None:
        """Create a new Jinx gui data."""
        super().__init__()
        self.__desired_view_state: str | None = None
        self.__view_states: list[str] = []
        self.__data: dict[str, Any] = data_dict.copy()

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
    @observable.notifies_observers
    def desired_view_state(self, desired_view_state: str | None) -> None:
        """Set the current desired view state name."""
        self.__desired_view_state = desired_view_state

    def get_data(self, key: str, default: Any = None) -> Any:
        """Get the data associated with the given key."""
        return self.__data.get(key, default)

    @observable.notifies_observers
    def set_data(self, key: str, value: Any) -> None:
        """Set the data associated with the given key."""
        self.__data[key] = value


class JinxObserverWidget(observable.Observer):
    """A class defining an observer widget."""

    __slots__ = {
        "__widget": "The encapsulated widget."
    }

    def __init__(self, widget: QtWidgets.QWidget) -> None:
        """Create a new Jinx widget within the given parent widget."""
        super().__init__()
        self.__widget: QtWidgets.QWidget = widget

    @property
    def widget(self) -> QtWidgets.QWidget:
        """Get the widget."""
        return self.__widget

    def update_observer(self, data: JinxGuiData) -> None:
        """
        Update the observer.

        This method should be overridden by subclasses.
        """
        pass


def make_observer_widget(
    widget: QtWidgets.QWidget,
    update_observer: Callable[
        [JinxObserverWidget, JinxGuiData], None] = lambda data: None
) -> JinxObserverWidget:
    """Create a new observer widget."""
    return type("AnonymousObserverWidget", (JinxObserverWidget,),
                {"update_observer": update_observer})(widget)


def combo_box_changed(view_name: str, data: JinxGuiData) -> None:
    """Handle a change in the combo box."""
    print("combo box changed", view_name)
    data.desired_view_state = view_name


class JinxGuiWindow(observable.Observer):
    """A class defining a PyQt6 window used by Jinx."""

    __slots__ = {
        "__window": "The main window.",
        "__main_widget": "The main widget.",
        "__vbox": "The main vertical box layout.",
        "__stack": "The stacked widget for the views.",
        "__combo_box": "The combo box for the views selection.",
        "__data": "The Jinx gui data for the window.",
        "__views": "The views of the window.",
        "__current_view_state": "The current view state of the window."
    }

    def __init__(
        self,
        window: QtWidgets.QMainWindow,
        data: JinxGuiData,
        upper_tabs: bool = False,
        lower_tabs: bool = False,
        left_tabs: bool = False,
        right_tabs: bool = False
    ) -> None:
        """Create a new Jinx window within the given parent widget."""
        super().__init__()

        # self.__window: QtWidgets.QWidget = window
        self.__main_widget = QtWidgets.QWidget()
        # self.__window.setCentralWidget(self.__main_widget)
        window.setCentralWidget(self.__main_widget)

        self.__vbox = QtWidgets.QVBoxLayout()
        self.__stack = QtWidgets.QStackedWidget()
        self.__combo_box = QtWidgets.QComboBox()
        self.__combo_box.currentTextChanged.connect(
            functools.partial(combo_box_changed, data=data)
        )
        self.__vbox.addWidget(self.__combo_box)
        self.__vbox.addWidget(self.__stack)
        self.__main_widget.setLayout(self.__vbox)

        self.__data: JinxGuiData = data
        self.__data.assign_observers(self)

        self.__views: dict[str, JinxObserverWidget] = {}
        self.__current_view_state: str | None = None

        self.__data.notify_all()

    @property
    def current_view_state(self) -> str:
        """Get the current view state of the window."""
        return self.__current_view_state

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

    def update_observer(self, data: JinxGuiData) -> None:
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
            # else:
            #     self.__stack.setCurrentWidget(default_widget)
            #     self.__data.assign_observers(default_widget)
            self.__current_view_state = desired_view_state
