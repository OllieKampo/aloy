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
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

"""
Module defining Jinx GUI components which wrap PySide6.

The main Jinx GUI components are; JinxGuiWindow which wrap PySide6 QMainWindow
and JinxWidget which wraps PySide6 QWidget. Both of these are observers.
"""

from abc import abstractmethod
from collections import defaultdict
import itertools
from typing import Any, Literal, NamedTuple, Sequence, Union, final
from PySide6 import QtWidgets, QtCore, QtGui  # pylint: disable=unused-import

from concurrency.clocks import ClockThread
from concurrency.synchronization import atomic_update
import guis.observable as observable
from guis.widgets.placeholders import PlaceholderWidget
from moremath.mathutils import closest_integer_factors

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
    "JinxSystemData",
    "JinxWidget",
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
    jwidgets: Sequence["JinxWidget"],
    kind: Literal["horizontal", "vertical", "grid"] = "vertical",
    stretches: (Sequence[int] | tuple[Sequence[int], Sequence[int]]
                | None) = None,
    spacing: int | tuple[int, int] = 0,
    contents_margins: tuple[int, int, int, int] | None = None,
    alignment: QtCore.Qt.AlignmentFlag = QtCore.Qt.AlignmentFlag.AlignCenter,
    parent: QtWidgets.QWidget | QtWidgets.QMainWindow | None = None,
    grid_size_max: tuple[int, int] | None = None
) -> QtWidgets.QWidget | QtWidgets.QMainWindow:
    """
    Combine a sequence of Jinx widgets onto a single widget.

    Parameters
    ----------
    `jwidgets: Sequence[JinxWidget]` - The Jinx widgets to combine.

    `kind: Literal["horizontal", "vertical", "grid"]` - The kind of layout to
    use. If "grid", the widgets will be initially placed along the columns of
    the first row, additional rows are added whenever a row is filled.

    `stretches: Sequence[int] | tuple[Sequence[int], Sequence[int]] | None =
    None` - The stretch factors for the widgets. If None, all widgets will have
    a stretch factor of 0. If a sequence of integers, the stretch factors will
    be applied to the widgets in the order they are given. For grid layouts,
    a tuple of two sequences of integers must be given, where the first
    sequence is the stretch factors for the rows and the second sequence is the
    stretch factors for the columns. If there are more widgets than stretch
    factors, the sequence of stretch factors will be cycled.

    `spacing: int | tuple[int, int] = 0` - The spacing between widgets in
    pixels. If a single integer is given, the same spacing will be used
    horizontally and vertically. For grid layouts, a tuple of two integers must
    be given for the horizontal and vertical spacing.

    `contents_margins: tuple[int, int, int, int] | None = None` - The margins
    around the widgets in pixels. If None, no margins will be used. The order
    is left, top, right, bottom.

    `alignment: QtCore.Qt.AlignmentFlag` - The alignment of the widgets.

    `parent: QtWidgets.QWidget | QtWidgets.QMainWindow | None = None` - The
    parent widget to use. If None, a new widget will be created.

    `grid_size_max: tuple[int, int] | None = None` - The maximum size of the
    grid in pixels. If None, the grid size will be made the tightest possible
    square that fits all widgets, i.e. the number of rows and columns will be
    closest possible two factors of the closest square number that is greater
    than or equal to the number of widgets. If a tuple of two integers is
    given, try to fit the widgets into a grid of the given size, filling the
    grid from left to right, then top to bottom (expand columns first, then
    rows).

    Returns
    -------
    `QtWidgets.QWidget | QtWidgets.QMainWindow` - The parent widget.
    """
    layout: QtWidgets.QLayout

    if kind == "grid":
        layout = QtWidgets.QGridLayout()

        if not isinstance(spacing, tuple):
            raise TypeError(
                f"Spacing {spacing!r} must be a tuple of two integers for "
                f"grid layout."
            )
        layout.setHorizontalSpacing(spacing[0])
        layout.setVerticalSpacing(spacing[1])

        if stretches is None:
            row_stretches = itertools.repeat(0)
            col_stretches = itertools.repeat(0)
        else:
            if len(next(iter(stretches))) != 2:  # type: ignore
                raise ValueError(
                    "Grid layout requires a tuple of two sequences of "
                    "stretch factors, one for rows and one for columns."
                )
            row_stretches = itertools.cycle(stretches[0])  # type: ignore
            col_stretches = itertools.cycle(stretches[1])  # type: ignore

        if grid_size_max is not None:
            if (grid_size_max[0] * grid_size_max[1]) < len(jwidgets):
                raise ValueError(
                    f"Grid size {grid_size_max!r} is too small for "
                    f"{len(jwidgets)!r} widgets."
                )
        else:
            grid_size_max = closest_integer_factors(len(jwidgets))

        for index, jwidget in enumerate(jwidgets):
            index_row, index_col = divmod(index, grid_size_max[1])
            layout.addWidget(
                jwidget.qwidget,
                index_row,
                index_col,
                alignment=alignment
            )
            layout.setRowStretch(index_row, next(row_stretches))
            layout.setColumnStretch(index_col, next(col_stretches))

    elif kind in ("horizontal", "vertical"):
        if kind == "horizontal":
            layout = QtWidgets.QHBoxLayout()
        elif kind == "vertical":
            layout = QtWidgets.QVBoxLayout()

        if not isinstance(spacing, int):
            raise TypeError(
                f"Spacing must be an integer for {kind!r} layout."
            )
        layout.setSpacing(spacing)

        if stretches is None:
            stretches = itertools.repeat(0)  # type: ignore
        elif not isinstance(next(iter(stretches)), int):
            raise TypeError(
                f"Stretches must be a sequence of integers for {kind!r} "
                f"layout."
            )

        for jwidget, stretch in zip(jwidgets, stretches):  # type: ignore
            layout.addWidget(
                jwidget.qwidget,
                stretch=stretch,  # type: ignore
                alignment=alignment
            )

    else:
        raise ValueError(f"Unknown kind: {kind!s}")

    if contents_margins is not None:
        layout.setContentsMargins(*contents_margins)

    if isinstance(parent, QtWidgets.QMainWindow):
        combined_qwidget = QtWidgets.QWidget()
        combined_qwidget.setLayout(layout)
        parent.setCentralWidget(combined_qwidget)
    else:
        if parent is None:
            parent = QtWidgets.QWidget()
        parent.setLayout(layout)

    return parent


class JinxSystemData(observable.Observable):
    """A class defining a Jinx system data object."""

    __slots__ = {
        "__weakref__": "Weak reference to the object.",
        "__linked_gui": "The gui object linked to this data object.",
        "__view_states": "The list of view states.",
        "__desired_view_state": "The currently desired view state.",
        "__data": "Arbitrary data associated with the gui.",
        "__messages": "Log messages associated with the gui.",
    }

    def __init__(
        self,
        name: str | None = None,
        gui: Union["JinxGuiWindow", None] = None,
        data_dict: dict[str, Any] | None = None,
        clock: ClockThread | QtCore.QTimer | None = None, *,
        debug: bool = False
    ) -> None:
        """
        Create a new Jinx system data object.

        Parameters
        ----------
        `name: str | None = None` - The name of the object.
        If not given or None, a unique name will be generated.
        See `jinx.guis.observable.Observable` for details.

        `gui: JinxGuiWindow | None = None` - The gui object to be
        linked to this system data object.

        `data_dict: dict[str, Any] | None = None` - A data dictionary
        to be copied into the system data object.

        `clock: ClockThread | QTimer | None = None` - The clock
        thread or timer to be used for the observable object.
        If not given or None, a new clock thread will be created.
        See `jinx.guis.observable.Observable` for details.

        `debug: bool = False`  - Whether to log debug messages.
        """
        super().__init__(name, clock, debug=debug)
        self.__linked_gui: JinxGuiWindow | None = None
        if gui is not None:
            self.link_gui(gui)
        self.__desired_view_state: str | None = None
        self.__view_states: list[str] = []
        self.__data: dict[str, Any]
        if data_dict is None:
            self.__data = {}
        else:
            self.__data = data_dict.copy()
        self.__messages: dict[str, list[str]] = defaultdict(list)

    @atomic_update("gui", method=True)
    @observable.notifies_observers()
    def link_gui(self, gui: "JinxGuiWindow") -> None:
        """Connect the gui to this system data object."""
        if self.__linked_gui is not None:
            raise RuntimeError("System data object already linked to a gui.")
        self.__linked_gui = gui

    @property
    @atomic_update("gui", method=True)
    def linked_gui(self) -> Union["JinxGuiWindow", None]:
        """Get the linked gui."""
        return self.__linked_gui

    @atomic_update("view_states", method=True)
    @observable.notifies_observers()
    def update_view_states(self) -> None:
        """Update the view states of the linked gui."""
        if self.__linked_gui is None:
            raise RuntimeError("System data object not linked to a gui.")
        self.__view_states = self.__linked_gui.view_states

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

    @atomic_update("messages", method=True)
    @observable.notifies_observers()
    def add_log_message(self, kind: str, message: str) -> None:
        """Add a log message."""
        self.__messages[kind].append(message)

    @atomic_update("messages", method=True)
    @observable.notifies_observers()
    def clear_log_messages(self, kind: str | None = None) -> None:
        """Clear log messages."""
        if kind is None:
            self.__messages.clear()
        else:
            self.__messages[kind].clear()

    @atomic_update("messages", method=True)
    def get_log_messages(
        self,
        kind: str | None = None
    ) -> dict[str, list[str]] | list[str]:
        """Get log messages."""
        if kind is None:
            return self.__messages
        else:
            return self.__messages[kind]


class JinxWidget(observable.Observer):
    """
    Class defining Jinx GUI widgets.

    Jinx widgets are wrappers around PySide6 Qt widgets. Jinx widgets are
    observer objects and can be assigned to system data objects, and are
    notified when the data objects are changed. Similarly, data objects can be
    attached to Jinx widgets, so that they can directly access or modify the
    data objects. A Jinx widget can also be directly added to a Jinx gui
    window as a view.
    """

    __slots__ = {
        "__weakref__": "Weak reference to the object.",
        "__qwidget": "The encapsulated widget.",
        "__data": "The system data object.",
        "__size": "The size of the widget."
    }

    def __init__(
        self,
        qwidget: QtWidgets.QWidget | None = None,
        data: JinxSystemData | None = None,
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
        Create a new Jinx GUI widget wrapping the given PySide6 Qt widget.

        Parameters
        ----------
        `qwidget: QtWidgets.QWidget | None = None` - The parent widget to
        be wrapped. If None, a new widget will be created.

        `data: JinxSystemData | None = None` - The system data object to be
        attached to the widget. If None, no system data object will be
        attached. system data objects can be attached later using the
        `attach_data()` method. system data is automatically attached to
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

        self.__data: JinxSystemData | None = None
        if data is not None:
            self.attach_data(data)

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
                        "Choose from: fix, min, max, hint, hint-min."
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
    def data(self) -> JinxSystemData | None:
        """Get the data object."""
        return self.__data

    @property
    def size(self) -> JinxWidgetSize | None:
        """Get the size of the widget."""
        return self.__size

    def attach_data(self, data: JinxSystemData) -> None:
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
    def update_observer(self, observable_: JinxSystemData) -> None:
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
        "__data": "The Jinx system data for the window.",
        "__views": "The views of the window.",
        "__current_view_state": "The current view state of the window.",
        "__default_qwidget": "The default widget."
    }

    def __init__(
        self,
        qapp: QtWidgets.QApplication | None = None,
        qwindow: QtWidgets.QMainWindow | None = None,
        data: JinxSystemData | None = None,
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

        `data: JinxSystemData | None = None` - The Jinx system data for the
        window. If not given or None, a new Jinx system data object is
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

        self.__data: JinxSystemData
        if data is None:
            self.__data = JinxSystemData(
                name=f"System data for {self.window_name}",
                gui=self,
                debug=debug
            )
        else:
            self.__data = data
            self.__data.link_gui(self)
        self.__data.assign_observers(self)

        self.__views: dict[str, JinxWidget] = {}
        self.__current_view_state: str | None = None
        self.__default_qwidget = PlaceholderWidget(
            "No views have been added to this window."
        )
        self.__default_qwidget.sizeHint = self.__default_size_hint
        self.__stack.addWidget(self.__default_qwidget)
        self.__stack.setCurrentWidget(self.__default_qwidget)

        self.__data.notify_all()

    def __default_size_hint(
        self
    ) -> QtCore.QSize:
        """
        Return the default size hint.

        The size hint is a reasonable size for the widget, no minimum size
        hint is provided, as the minimum size is handled by the text size
        in the default widget.
        """
        return QtCore.QSize(
            int(self.__qwindow.width() * 0.9),
            int(self.__qwindow.height() * 0.9)
        )

    @property
    def qapp(self) -> QtWidgets.QApplication:
        """Get the main application."""
        return self.__qapp

    @property
    def qwindow(self) -> QtWidgets.QMainWindow:
        """Get the main window."""
        return self.__qwindow

    @property
    def data(self) -> JinxSystemData:
        """Get the Jinx system data for the window."""
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
        jwidget: JinxWidget,
        name: str | None = None
    ) -> None:
        """
        Add a Jinx widget as a new view state to the window.

        If `name` is None, the name of the Jinx widget is used. This window's
        linked data object will be attached to the Jinx widget. The Jinx
        widget is then added to the window's view stack. If this is the first
        view state added to the window, it will be set as the current view
        state.
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
        """Remove the view state with the given name from the window."""
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

    def update_observer(self, observable_: JinxSystemData) -> None:
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
