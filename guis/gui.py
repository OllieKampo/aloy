
from typing import Callable
from PyQt6 import QtCore, QtGui, QtWidgets

import guis.observable as observable

class JinxGuiData(observable.Observable):

    __slots__ = (
        "__desired_view_state",
        "__view_states"
    )
    
    def __init__(self) -> None:
        super().__init__()
        self.__desired_view_state: str | None = None
        self.__view_states: list[str] = []
    
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

class JinxObserverWidget(observable.Observer):
    """A class defining an observer widget."""

    def __init__(self,
                 parent: QtWidgets.QWidget | None = None,
                 flags: QtCore.Qt.WindowType = QtCore.Qt.WindowType.Widget
                 ) -> None:
        """Create a new Jinx widget within the given parent widget."""
        super().__init__(self)
    
    def update_observer(self, data: JinxGuiData) -> None:
        pass

# def make_observer_widget(parent: QtWidgets.QWidget | None = None,
#                          flags: QtCore.Qt.WindowType = QtCore.Qt.WindowType.Widget,
#                          update_observer: Callable[[JinxGuiData], None] = lambda data: None
#                          ) -> ObserverWidget:
#     """Create a new observer widget."""
#     return type("AnonymousObserverWidget", (ObserverWidget,),
#                 {"update_observer": update_observer})(parent, flags)

# class JinxGuiWindow(observable.Observer):
#     """A class defining a PyQt6 window used by Jinx."""

#     __slots__ = ("__data",
#                  "__view_states",
#                  "__current_view_state",
#                  "__window")
    
#     def __init__(self, parent: QtWidgets.QMainWindow, data: JinxGuiData) -> None:
#         """Create a new Jinx window within the given parent widget."""
#         observable.Observer.__init__(self)

#         self.__parent: QtWidgets.QWidget = parent
#         vbox = QtWidgets.QVBoxLayout()
#         self.__stack = QtWidgets.QStackedWidget()
#         self.__combo_box = QtWidgets.QComboBox()
#         self.__combo_box.currentTextChanged.connect(self.__combo_box_changed)
#         vbox.addWidget(self.__combo_box)
#         vbox.addWidget(self.__stack)
#         self.__parent.setLayout(vbox)

#         self.__data: JinxGuiData = data
#         self.__data.assign_observers(self)
        
#         self.__view_states: dict[str, ObserverWidget] = {}
#         self.__current_view_state: str | None = None
#         self.__data.desired_view_state = None

#         self.__data.notify_all()
    
#     @property
#     def current_view_state(self) -> str:
#         """Get the current view state of the window."""
#         return self.__current_view_state
    
#     def add_view_state(self, view_state: str, widget: ObserverWidget) -> None:
#         """Add a new view state to the window."""
#         self.__view_states[view_state] = widget
    
#     def remove_view_state(self, view_state: str) -> None:
#         """Remove a view state from the window."""
#         self.__view_states.pop(view_state)
    
#     def update_observer(self, data: JinxGuiData) -> None:
#         desired_view_state: str | None = self.__data.desired_view_state
#         if self.__current_view_state != desired_view_state:
#             self.__data.clear_observers()
#             self.__data.assign_observers(self)
#             if desired_view_state is not None:
#                 widget: ObserverWidget = self.__view_states[desired_view_state]
#                 self.__stack.setCurrentWidget(widget)
#                 self.__data.assign_observers(widget)
#             self.__current_view_state = desired_view_state
#             self.__data.notify_all()
    
#     def __combo_box_changed(self, index: int) -> None:
#         """Handle a change in the combo box."""
#         self.__data.desired_view_state = self.__combo_box.currentText()
