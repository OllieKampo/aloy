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
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Module defining Jinx GUI classes for working with PySide6."""

from abc import abstractmethod
from typing import Any, ClassVar, Literal, NamedTuple, Sequence, Union, final
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import QTimer  # pylint: disable=E0611
from concurrency.clocks import ClockThread
from concurrency.synchronization import atomic_update

import guis.observable as observable

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"

__all__ = (
    "JinxWidgetSize",
    "JinxGridShape",
    "JinxWidgetSpacing",
    "JinxWidgetMargins",
    "scale_size",
    "scale_size_for_grid",
    "GridScaler",
    "JinxGuiData",
    "JinxObserverWidget",
    "JinxGuiWindow"
)


def __dir__() -> tuple[str, ...]:
    """Get the names of module attributes."""
    return __all__


class JinxWidgetSize(NamedTuple):
    """Tuple representing the size of a Jinx widget."""

    width: int
    height: int


class JinxGridShape(NamedTuple):
    """Tuple representing the shape of a Jinx grid."""

    columns: int
    rows: int


class JinxWidgetSpacing(NamedTuple):
    """Tuple representing the spacing of a Jinx widget."""

    horizontal: int
    vertical: int


class JinxWidgetMargins(NamedTuple):
    """Tuple representing the margins of a Jinx widget."""

    left: int
    top: int
    right: int
    bottom: int


def scale_size(
    size: tuple[int, int],
    scale: tuple[float, float]
) -> JinxWidgetSize:
    """Scale a size by the given factor."""
    return JinxWidgetSize(
        int(round(size[0] * scale[0])),
        int(round(size[1] * scale[1]))
    )


def scale_size_for_grid(
    size: tuple[int, int],
    grid_shape: tuple[int, int],
    widget_shape: tuple[int, int] = (1, 1),
    spacing: tuple[int, int] = (10, 10),
    margins: tuple[int, int, int, int] = (10, 10, 10, 10)
) -> JinxWidgetSize:
    """
    Scale a size to find the size of a grid cell.

    Parameters
    ----------
    `size: tuple[int, int]` - The size in pixels to be scaled.

    `grid_shape: tuple[int, int]` - The shape of the grid; columns, rows.

    `widget_shape: tuple[int, int]` - The shape of the widget; columns, rows.

    `spacing: tuple[int, int] = (10, 10)` - The spacing in pixels between
    grid cells; horizontal, vertical.

    `margins: tuple[int, int, int, int] = (10, 10, 10, 10)` - The margins
    in pixels around the grid; left, top, right, bottom.

    Returns
    -------
    `tuple[int, int]` - The scaled size in pixels of a grid cell.

    Notes
    -----
    For help of layouts see: https://doc.qt.io/qt-6/layout.html
    For help on size policies see: https://doc.qt.io/qt-6/qsizepolicy.html
    """
    scale = (widget_shape[0] / grid_shape[0], widget_shape[1] / grid_shape[1])
    size = (
        size[0]
        - (spacing[0] * (grid_shape[0] - widget_shape[0]))
        - (margins[0] + margins[2]),
        size[1]
        - (spacing[1] * (grid_shape[1] - widget_shape[1]))
        - (margins[1] + margins[3])
    )
    return scale_size(size, scale)


@final
class GridScaler:
    """A class for scaling widgets in a grid."""

    __slots__ = {
        "__size": "The size of the grid in pixels.",
        "__grid_shape": "The shape of the grid (rows, columns).",
        "__spacing": "The spacing in pixels between grid cells "
                     "(horizontal, vertical).",
        "__margins": "The margins in pixels around the grid "
                     "(left, top, right, bottom)."
    }

    def __init__(
        self,
        size: tuple[int, int],
        grid_shape: tuple[int, int],
        spacing: tuple[int, int] = (10, 10),
        margins: tuple[int, int, int, int] = (10, 10, 10, 10)
    ) -> None:
        """
        Create a new grid scaler object.

        Parameters
        ----------
        `size: tuple[int, int]` - The size in pixels of the grid.

        `grid_shape: tuple[int, int]` - The shape of the grid, i.e. the number
        of rows and columns.

        `spacing: tuple[int, int] = (10, 10)` - The spacing in pixels between
        grid cells. The order is horizontal, vertical.

        `margins: tuple[int, int, int, int] = (10, 10, 10, 10)` - The margins
        in pixels around the grid. The order is left, top, right, bottom.
        """
        self.__size = JinxWidgetSize(*size)
        self.__grid_shape = JinxGridShape(*grid_shape)
        self.__spacing = JinxWidgetSpacing(*spacing)
        self.__margins = JinxWidgetMargins(*margins)

    @property
    def size(self) -> JinxWidgetSize:
        """Get the size of the grid in pixels."""
        return self.__size

    @property
    def grid_shape(self) -> JinxGridShape:
        """Get the shape of the grid, i.e. the number of rows and columns."""
        return self.__grid_shape

    @property
    def spacing(self) -> JinxWidgetSpacing:
        """Get the spacing in pixels between grid cells."""
        return self.__spacing

    @property
    def margins(self) -> JinxWidgetMargins:
        """Get the margins in pixels around the grid."""
        return self.__margins

    def get_size(self, widget_shape: tuple[int, int]) -> JinxWidgetSize:
        """
        Get the size of a widget of the given shape in the grid.

        Parameters
        ----------
        `widget_shape: tuple[int, int]` - The shape of the widget, i.e. the
        number of rows and columns it occupies.

        Returns
        -------
        `tuple[int, int]` - The scaled size in pixels of a grid cell.
        """
        return scale_size_for_grid(
            self.__size,
            self.__grid_shape,
            widget_shape,
            self.__spacing,
            self.__margins
        )


def combine_jinx_widgets(
    jwidgets: Sequence["JinxObserverWidget"],
    stretches: Sequence[int],
    alignment: QtCore.Qt.AlignmentFlag,
    kind: Literal["horizontal", "vertical"],
    spacing: int = 0,
    parent: QtWidgets.QWidget | QtWidgets.QMainWindow | None = None
) -> QtWidgets.QWidget | QtWidgets.QMainWindow:
    """
    Combine a sequence of Jinx widgets onto a single widget.

    Parameters
    ----------
    `jwidgets: Sequence[JinxObserverWidget]` - The Jinx widgets to combine.

    `stretches: Sequence[int]` - The stretch factors for each widget.

    `alignment: QtCore.Qt.AlignmentFlag` - The alignment of the widgets.

    `kind: Literal["horizontal", "vertical"]` - The kind of layout to use.

    `spacing: int = 0` - The spacing between widgets in pixels.

    `parent: QtWidgets.QWidget | QtWidgets.QMainWindow | None = None` - The
    parent widget to use. If None, a new widget will be created.

    Returns
    -------
    `QtWidgets.QWidget | QtWidgets.QMainWindow` - The parent widget.
    """
    layout: QtWidgets.QLayout
    if kind == "horizontal":
        layout = QtWidgets.QHBoxLayout()
    elif kind == "vertical":
        layout = QtWidgets.QVBoxLayout()
    else:
        raise ValueError(f"Unknown kind: {kind!s}")
    layout.setSpacing(spacing)

    for jwidget, stretch in zip(jwidgets, stretches):
        layout.addWidget(jwidget.qwidget, stretch=stretch, alignment=alignment)

    if isinstance(parent, QtWidgets.QMainWindow):
        combined_qwidget = QtWidgets.QWidget()
        combined_qwidget.setLayout(layout)
        parent.setCentralWidget(combined_qwidget)
    else:
        if parent is None:
            parent = QtWidgets.QWidget()
        parent.setLayout(layout)

    return parent


class JinxGuiData(observable.Observable):
    """A class defining a Jinx GUI data object."""

    __slots__ = {
        "__weakref__": "Weak reference to the object.",
        "__connected_gui": "The gui object connected to this data object.",
        "__view_states": "The list of view states.",
        "__desired_view_state": "The currently desired view state.",
        "__data": "Arbitrary data associated with the gui."
    }

    def __init__(
        self,
        name: str | None = None,
        gui: Union["JinxGuiWindow", None] = None,
        data_dict: dict[str, Any] | None = None,
        clock: ClockThread | QTimer | None = None, *,
        debug: bool = False
    ) -> None:
        """
        Create a new Jinx gui data object.

        Parameters
        ----------
        `name: str | None = None` - The name of the object.
        If not given or None, a unique name will be generated.
        See `jinx.guis.observable.Observable` for details.

        `gui: JinxGuiWindow | None = None` - The gui object to be
        connected to this gui data object.

        `data_dict: dict[str, Any] | None = None` - A data dictionary
        to be copied into the gui data object.

        `clock: ClockThread | QTimer | None = None` - The clock
        thread or timer to be used for the observable object.
        If not given or None, a new clock thread will be created.
        See `jinx.guis.observable.Observable` for details.

        `debug: bool = False`  - Whether to log debug messages.

        Notes
        -----
        For help on windows see: https://doc.qt.io/qt-6/mainwindow.html
        """
        super().__init__(name, clock, debug=debug)
        self.__connected_gui: JinxGuiWindow | None = None
        if gui is not None:
            self.connect_gui(gui)
        self.__desired_view_state: str | None = None
        self.__view_states: list[str] = []
        self.__data: dict[str, Any]
        if data_dict is None:
            self.__data = {}
        else:
            self.__data = data_dict.copy()

    @atomic_update("gui", method=True)
    @observable.notifies_observers()
    def connect_gui(self, gui: "JinxGuiWindow") -> None:
        """Connect the gui to this gui data object."""
        if self.__connected_gui is not None:
            raise RuntimeError("Gui data object already connected to a gui.")
        self.__connected_gui = gui

    @property
    @atomic_update("gui", method=True)
    def connected_gui(self) -> Union["JinxGuiWindow", None]:
        """Get the connected gui."""
        return self.__connected_gui

    @atomic_update("view_states", method=True)
    @observable.notifies_observers()
    def update_view_states(self) -> None:
        """Update the view states of the connected gui."""
        if self.__connected_gui is None:
            raise RuntimeError("Gui data object not connected to a gui.")
        self.__view_states = self.__connected_gui.view_states

    @property
    @atomic_update("view_states", method=True)
    def view_states(self) -> list[str]:
        """Get the list of view states."""
        return self.__view_states

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


class JinxObserverWidget(observable.Observer):
    """A class defining Jinx observer widgets."""

    __slots__ = {
        "__weakref__": "Weak reference to the object.",
        "__qwidget": "The encapsulated widget.",
        "__data": "The gui data object.",
        "__size": "The size of the widget."
    }

    def __init__(
        self,
        qwidget: QtWidgets.QWidget | None = None,
        data: JinxGuiData | None = None,
        name: str | None = None,
        size: tuple[int, int] | None = None,
        resize: bool = True,
        set_size: Literal[
            "fix",
            "min",
            "max",
            "hint",
            "hint-min"
        ] | None = None,
        size_policy: tuple[
            QtWidgets.QSizePolicy.Policy,
            QtWidgets.QSizePolicy.Policy
        ] | None = None,
        debug: bool = False
    ) -> None:
        """
        Create a new Jinx widget wrapping the given parent widget.

        Parameters
        ----------
        `qwidget: QtWidgets.QWidget | None = None` - The parent widget to
        be wrapped. If None, a new widget will be created.

        `data: JinxGuiData | None = None` - The gui data object to be
        attached to the widget. If None, no gui data object will be
        attached. Gui data objects can be attached later using the
        `attach_data()` method. Gui data is automatically attached to
        the widget when the widget is added to a Jinx gui window.

        `name: str | None = None` - The name of the object. If None, the
        class name and id of the object are used.

        `size: tuple[int, int] | None = None` - The size of the widget in
        pixels (width, height). If None, the size of the widget is not set.

        `resize: bool = True` - Whether to resize the parent widget to the
        given size, or just simply store the size.

        `set_size: "fix" | "min" | "max" | "hint" | "hint-min" | None = None`
        - Whether to set the size of the widget to the given size.

        `size_policy: tuple[QtWidgets.QSizePolicy.Policy,
        QtWidgets.QSizePolicy.Policy] | None = None` - The size horizontal
        and vertical size policies of the widget. If None, the size policies
        are not set.

        `debug: bool = False` - Whether to log debug messages.
        """
        super().__init__(name, debug=debug)

        if qwidget is None:
            qwidget = QtWidgets.QWidget()
        self.__qwidget: QtWidgets.QWidget = qwidget

        self.__data: JinxGuiData | None = data

        self.__size: JinxWidgetSize | None = None
        if size is not None:
            self.__size = JinxWidgetSize(*size)
            if resize:
                qwidget.resize(*size)
            if set_size is not None:
                if set_size == "fix":
                    qwidget.setFixedSize(*size)
                elif set_size == "min":
                    qwidget.setMinimumSize(*size)
                elif set_size == "max":
                    qwidget.setMaximumSize(*size)
                elif "hint" in set_size:
                    def get_size() -> QtCore.QSize:
                        return QtCore.QSize(*size)
                    if set_size == "hint":
                        qwidget.sizeHint = get_size  # type: ignore
                    elif set_size == "hint-min":
                        qwidget.minimumSizeHint = get_size  # type: ignore
                else:
                    raise ValueError(
                        f"Invalid set_size value: {set_size}."
                        f"Choose from: fix, min, max, hint, hint-min"
                    )

        if size_policy is not None:
            qwidget.setSizePolicy(
                size_policy[0],
                size_policy[1]
            )

        if name is not None:
            qwidget.setObjectName(name)

    @property
    def qwidget(self) -> QtWidgets.QWidget:
        """Get the qt widget."""
        return self.__qwidget

    @property
    def data(self) -> JinxGuiData | None:
        """Get the data object."""
        return self.__data

    @property
    def size(self) -> JinxWidgetSize | None:
        """Get the size of the widget."""
        return self.__size

    def attach_data(self, data: JinxGuiData) -> None:
        """Attach the given data object to the widget."""
        if self.__data is not None:
            self.__data.remove_observers(self)
        self.__data = data
        self.__data.assign_observers(self)

    def detach_data(self) -> None:
        """Detach the data object from the widget."""
        if self.__data is not None:
            self.__data.remove_observers(self)
            self.__data = None

    @abstractmethod
    def update_observer(self, observable_: JinxGuiData) -> None:
        """Update the observer."""
        return super().update_observer(observable_)


class JinxGuiWindow(observable.Observer):
    """
    A class defining a PySide6 window used by Jinx.

    A Jinx window is a window that can contain multiple views. The views
    can optionally be selected using a combo box or a tab bar, or a custom
    interface can be used to select the views. In the case where a custom
    interface is desired, the window object emites the following signals;
    `view_changed(str)`, `view_added(str)`, and `view_removed(str)`, to
    allow the custom interface to be updated.
    """

    view_changed: ClassVar[QtCore.Signal] = QtCore.Signal(str)
    view_added: ClassVar[QtCore.Signal] = QtCore.Signal(str)
    view_removed: ClassVar[QtCore.Signal] = QtCore.Signal(str)

    __slots__ = {
        "__weakref__": "Weak reference to the object.",
        "__qapp": "The main application.",
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
        qapp: QtWidgets.QApplication | None = None,
        qwindow: QtWidgets.QMainWindow | None = None,
        data: JinxGuiData | None = None,
        name: str | None = None,
        size: tuple[int, int] | None = None, *,
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
        `qapp: QtWidgets.QApplication | None = None` - The main application.

        `qwindow: QtWidgets.QMainWindow | None = None` - The main window.
        If not given or None, a new main window is created.

        `data: JinxGuiData | None = None` - The Jinx gui data for the
        window. If not given or None, a new Jinx gui data object is
        created.

        `name: str | None = None` - The name of the window. If not given or
        None, then the class name and id of the object are used. The
        attributes `observer_name` and `window_name` are set to this value.

        `size: tuple[int, int] | None = None` - The size of the window in
        pixels (width, height). If not given or None, the size of the window
        is not set.

        `kind: Literal["tabbed", "combo"] | None = "tabbed"` - Whether the
        view selecter is a tab bar or a combo box. If None, then no view
        selecter is used.

        `set_title: bool = True` - Whether to set the title of the window
        to the given name.

        `resize: bool = True` - Whether to resize the window to the given
        size, or just simply store the size.

        `debug: bool = False` - Whether to log debug messages.
        """
        super().__init__(name, debug=debug)

        self.__qapp: QtWidgets.QApplication
        if qapp is None:
            self.__qapp = QtWidgets.QApplication.instance()  # type: ignore
            if self.__qapp is None:
                self.__qapp = QtWidgets.QApplication([])
        else:
            self.__qapp = qapp

        self.__qwindow: QtWidgets.QMainWindow
        if qwindow is None:
            self.__qwindow = QtWidgets.QMainWindow()
        else:
            self.__qwindow = qwindow
        qwindow = self.__qwindow

        if set_title:
            self.__qwindow.setWindowTitle(self.window_name)
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

        self.__data: JinxGuiData
        if data is None:
            self.__data = JinxGuiData(
                name=f"Gui Data for {self.window_name}",
                gui=self,
                debug=debug
            )
        else:
            self.__data = data
            self.__data.connect_gui(self)
        self.__data.assign_observers(self)

        self.__views: dict[str, JinxObserverWidget] = {}
        self.__current_view_state: str | None = None
        self.__default_qwidget = QtWidgets.QLabel()
        self.__setup_default_qwidget()
        self.__stack.addWidget(self.__default_qwidget)
        self.__stack.setCurrentWidget(self.__default_qwidget)

        self.__data.notify_all()

    def __default_size_hint(
        self
    ) -> QtCore.QSize:
        """
        Return the default size hint.

        The size hint is a reasonable size for the widget,
        no minimum size hint is provided, as the minimum size
        is handled by the  text size in the default widget.
        """
        return QtCore.QSize(
            int(self.__qwindow.width() * 0.9),
            int(self.__qwindow.height() * 0.9)
        )

    def __setup_default_qwidget(self) -> None:
        """Create the default widget."""
        self.__default_qwidget = QtWidgets.QLabel()
        self.__default_qwidget.setStyleSheet(
            "QLabel { background-color: black; }"
        )
        self.__default_qwidget.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.__default_qwidget.setText(
            "No views have been added to this window."
        )
        self.__default_qwidget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.__default_qwidget.sizeHint = (     # type: ignore
            self.__default_size_hint
        )
        self.__default_qwidget.paintEvent = (   # type: ignore
            self.__default_paint_event
        )

    def __default_paint_event(
        self,
        arg__1: QtGui.QPaintEvent
    ) -> None:
        """Paint the default widget."""
        painter = QtGui.QPainter(self.__default_qwidget)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(QtGui.QColor("green")))
        painter.drawLine(0, 0, self.__default_qwidget.width(),
                         self.__default_qwidget.height())
        painter.drawLine(0, self.__default_qwidget.height(),
                         self.__default_qwidget.width(), 0)
        painter.setPen(QtGui.QPen(QtGui.QColor("red")))
        # Paint the widget size in the top left corner.
        painter.drawText(
            self.__default_qwidget.rect(),
            QtCore.Qt.AlignmentFlag.AlignLeft
            | QtCore.Qt.AlignmentFlag.AlignTop,
            f"size={self.__default_qwidget.width()}x"
            f"{self.__default_qwidget.height()}"
        )
        painter.drawText(
            self.__default_qwidget.rect(),
            self.__default_qwidget.alignment(),
            self.__default_qwidget.text()
        )
        painter.end()

    @property
    def qapp(self) -> QtWidgets.QApplication:
        """Get the main application."""
        return self.__qapp

    @property
    def qwindow(self) -> QtWidgets.QMainWindow:
        """Get the main window."""
        return self.__qwindow

    @property
    def data(self) -> JinxGuiData:
        """Get the Jinx gui data for the window."""
        return self.__data

    @property
    def window_name(self) -> str:
        """Get the name of the object."""
        return self.observer_name

    @property
    def kind(self) -> str | None:
        """Get the kind of the window."""
        return self.__kind

    @property
    def view_states(self) -> list[str]:
        """Get the view states of the window."""
        return list(self.__views.keys())

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

    def add_view(
        self,
        jwidget: JinxObserverWidget,
        name: str | None = None
    ) -> None:
        """
        Add a Jinx widget as a new view state to the window.

        If the name is None, the name of the Jinx widget is used.
        """
        if name is None:
            name = jwidget.observer_name

        self.__views[name] = jwidget
        jwidget.attach_data(self.__data)
        self.__data.update_view_states()

        self.__stack.addWidget(jwidget.qwidget)
        if self.__kind == "tabbed":
            self.__tab_bar.addTab(name)
        elif self.__kind == "combo":
            self.__combo_box.addItem(name)

        if self.__current_view_state is None:
            self.__data.desired_view_state = name

    def remove_view(self, name: str) -> None:
        """Remove a view state from the window."""
        if name not in self.__views or name is None:
            return
        jwidget = self.__views.pop(name)
        self.__data.update_view_states()

        self.__stack.removeWidget(jwidget.qwidget)
        if self.__kind == "tabbed":
            for index in range(self.__tab_bar.count()):
                if self.__tab_bar.tabText(index) == name:
                    self.__tab_bar.removeTab(index)
                    break
        elif self.__kind == "combo":
            self.__combo_box.removeItem(self.__combo_box.findText(name))

        if self.__current_view_state == name:
            if self.__views:
                self.__data.desired_view_state = next(iter(self.__views))
            else:
                self.__data.desired_view_state = None

    def update_observer(self, observable_: JinxGuiData) -> None:
        """Update the observer."""
        if observable_ is not self.__data:
            return
        desired_view_state: str | None = self.__data.desired_view_state
        if self.__current_view_state != desired_view_state:
            if self.__current_view_state is not None:
                jwidget = self.__views[self.__current_view_state]
                self.__data.remove_observers(jwidget)
            if desired_view_state is not None:
                jwidget = self.__views[desired_view_state]
                self.__stack.setCurrentWidget(jwidget.qwidget)
                self.__data.assign_observers(jwidget)
                self.__data.notify(jwidget)
            else:
                self.__current_view_state = None
                self.__stack.setCurrentWidget(self.__default_qwidget)
            self.__current_view_state = desired_view_state
