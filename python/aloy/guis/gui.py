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
Module defining Aloy GUI components which wrap PySide6.

The main Aloy GUI components are; AloyGuiWindow which wrap PySide6 QMainWindow
and AloyWidget which wraps PySide6 QWidget. Both of these are observers.
"""

import itertools
from abc import abstractmethod
from typing import Any, Literal, NamedTuple, Sequence, Union, final

from PySide6 import QtCore, QtWidgets

import aloy.guis.observable as observable
from aloy.concurrency.clocks import ClockThread
from aloy.concurrency.synchronization import sync
from aloy.guis.widgets.placeholders import PlaceholderWidget
from aloy.moremath.mathutils import closest_integer_factors

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"
__version__ = "1.0.0"

__all__ = (
    "AloyWidgetSize",
    "AloyGridShape",
    "AloyWidgetSpacing",
    "AloyWidgetMargins",
    "scale_size",
    "scale_size_for_grid",
    "GridScaler",
    "AloySystemData",
    "AloyWidget",
    "AloyGuiWindow"
)


def __dir__() -> tuple[str, ...]:
    """Get the names of module attributes."""
    return __all__


class AloyWidgetSize(NamedTuple):
    """Tuple representing the size of a Aloy widget."""

    width: int
    height: int


class AloyGridShape(NamedTuple):
    """Tuple representing the shape of a Aloy grid."""

    columns: int
    rows: int


class AloyWidgetSpacing(NamedTuple):
    """Tuple representing the spacing of a Aloy widget."""

    horizontal: int
    vertical: int


class AloyWidgetMargins(NamedTuple):
    """Tuple representing the margins of a Aloy widget."""

    left: int
    top: int
    right: int
    bottom: int


def scale_size(
    size: tuple[int, int],
    scale: tuple[float, float]
) -> AloyWidgetSize:
    """Scale a size by the given factor."""
    return AloyWidgetSize(
        int(round(size[0] * scale[0])),
        int(round(size[1] * scale[1]))
    )


def scale_size_for_grid(
    size: tuple[int, int],
    grid_shape: tuple[int, int],
    widget_shape: tuple[int, int] = (1, 1),
    spacing: tuple[int, int] = (10, 10),
    margins: tuple[int, int, int, int] = (10, 10, 10, 10)
) -> AloyWidgetSize:
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
        self.__size = AloyWidgetSize(*size)
        self.__grid_shape = AloyGridShape(*grid_shape)
        self.__spacing = AloyWidgetSpacing(*spacing)
        self.__margins = AloyWidgetMargins(*margins)

    @property
    def size(self) -> AloyWidgetSize:
        """Get the size of the grid in pixels."""
        return self.__size

    @property
    def grid_shape(self) -> AloyGridShape:
        """Get the shape of the grid, i.e. the number of rows and columns."""
        return self.__grid_shape

    @property
    def spacing(self) -> AloyWidgetSpacing:
        """Get the spacing in pixels between grid cells."""
        return self.__spacing

    @property
    def margins(self) -> AloyWidgetMargins:
        """Get the margins in pixels around the grid."""
        return self.__margins

    def get_size(self, widget_shape: tuple[int, int]) -> AloyWidgetSize:
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


def combine_aloy_widgets(
    awidgets: Sequence["AloyWidget"],
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
    Combine a sequence of Aloy widgets onto a single widget.

    Parameters
    ----------
    `awidgets: Sequence[AloyWidget]` - The Aloy widgets to combine.

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
            if (grid_size_max[0] * grid_size_max[1]) < len(awidgets):
                raise ValueError(
                    f"Grid size {grid_size_max!r} is too small for "
                    f"{len(awidgets)!r} widgets."
                )
        else:
            grid_size_max = closest_integer_factors(len(awidgets))

        for index, awidget in enumerate(awidgets):
            index_row, index_col = divmod(index, grid_size_max[1])
            layout.addWidget(
                awidget.qwidget,
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

        for awidget, stretch in zip(awidgets, stretches):  # type: ignore
            layout.addWidget(
                awidget.qwidget,
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


class AloySystemData(observable.Observable):
    """A class defining a Aloy system data object."""

    __slots__ = {
        "__linked_gui": "The gui object linked to this data object.",
        "__view_states": "The list of view states.",
        "__desired_view_state": "The currently desired view state."
    }

    def __init__(
        self,
        name: str | None = None,
        gui: Union["AloyGuiWindow", None] = None,
        var_dict: dict[str, Any] | None = None,
        clock: ClockThread | QtCore.QTimer | None = None,
        tick_rate: int = 10,
        start_clock: bool = True,
        debug: bool = False
    ) -> None:
        """
        Create a new Aloy system data object.

        Parameters
        ----------
        `name: str | None = None` - The name of the object.
        If not given or None, a unique name will be generated.
        See `aloy.guis.observable.Observable` for details.

        `gui: AloyGuiWindow | None = None` - The gui object to be
        linked to this system data object.

        `var_dict: dict[str, Any] | None = None` - A data dictionary of
        variables to be stored in the observable.

        `clock: ClockThread | QTimer | None = None` - The clock thread or
        timer to be used for the observable object. If not given or None, a
        new clock thread will be created. For Qt applications, a QTimer is
        highly recommended. See `aloy.guis.observable.Observable` for details.

        `tick_rate: int = 10` - The tick rate of the clock if a new clock is
        created. Ignored if an existing clock is given.

        `start_clock: bool = True` - Whether to start the clock if an existing
        clock is given. Ignored if a new clock is created (the clock is always
        started in this case).

        `debug: bool = False`  - Whether to log debug messages.
        """
        super().__init__(
            name=name,
            var_dict=var_dict,
            clock=clock,
            tick_rate=tick_rate,
            start_clock=start_clock,
            debug=debug
        )
        self.__linked_gui: AloyGuiWindow | None = None
        if gui is not None:
            self.link_gui(gui)
        self.__desired_view_state: str | None = None
        self.__view_states: list[str] = []

    @observable.notifies_observers()
    @sync(group_name="__linked_gui__")
    def link_gui(self, gui: "AloyGuiWindow") -> None:
        """Connect the gui to this system data object."""
        if self.__linked_gui is not None:
            raise RuntimeError("System data object already linked to a gui.")
        self.__linked_gui = gui

    @property
    @sync(group_name="__linked_gui__")
    def linked_gui(self) -> Union["AloyGuiWindow", None]:
        """Get the linked gui."""
        return self.__linked_gui

    @observable.notifies_observers()
    @sync(group_name="__view_states__")
    def update_view_states(self) -> None:
        """Update the view states from the linked gui."""
        if self.__linked_gui is None:
            raise RuntimeError("System data object not linked to a gui.")
        self.__view_states = self.__linked_gui.view_states

    @property
    @sync(group_name="__view_states__")
    def view_states(self) -> list[str]:
        """Get the list of view states."""
        return self.__view_states

    @property
    @sync(group_name="__view_states__")
    def desired_view_state(self) -> str | None:
        """Get the current desired view state name."""
        return self.__desired_view_state

    @desired_view_state.setter
    @observable.notifies_observers()
    @sync(group_name="__view_states__")
    def desired_view_state(self, desired_view_state: str | None) -> None:
        """Set the current desired view state name."""
        self.__desired_view_state = desired_view_state


class AloyWidget(observable.Observer):
    """
    Class defining Aloy GUI widgets.

    Aloy widgets are wrappers around PySide6 Qt widgets. Aloy widgets are
    observer objects and can be assigned to system data objects, and are
    notified when the data objects are changed. Similarly, data objects can be
    attached to Aloy widgets, so that they can directly access or modify the
    data objects. An Aloy widget can also be directly added to a Aloy gui
    window as a view.
    """

    __slots__ = {
        "__qwidget": "The encapsulated widget.",
        "__data": "The system data object.",
        "__size": "The size of the widget."
    }

    def __init__(
        self,
        qwidget: QtWidgets.QWidget | None = None,
        data: AloySystemData | None = None,
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
        Create a new Aloy GUI widget wrapping the given PySide6 Qt widget.

        Parameters
        ----------
        `qwidget: QtWidgets.QWidget | None = None` - The parent widget to
        be wrapped. If None, a new widget will be created.

        `data: AloySystemData | None = None` - The system data object to be
        attached to the widget. If None, no system data object will be
        attached. System data objects can be attached later using the
        `attach_data()` method. System data is automatically attached to
        the widget when the widget is added to a Aloy GUI window.

        `name: str | None = None` - The name of the object. If None, the
        class name and id of the object are used.

        `size: tuple[int, int] | None = None` - The size of the widget in
        pixels (width, height). If None, the size of the widget is not set.

        `resize: bool = True` - Whether to resize the parent widget to the
        given size, or just simply store the size.

        `set_size: "fix" | "min" | "max" | "hint" | "hint-min" | None =
        None` - Whether to set the size of the widget to the given size.

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

        self.__data: AloySystemData | None = None
        if data is not None:
            self.attach_data(data)

        self.__size: AloyWidgetSize | None = None
        if size is not None:
            self.__size = AloyWidgetSize(*size)
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
    def data(self) -> AloySystemData | None:
        """Get the data object."""
        return self.__data

    @property
    def size(self) -> AloyWidgetSize | None:
        """Get the size of the widget."""
        return self.__size

    def attach_data(self, data: AloySystemData) -> None:
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
    def update_observer(self, observable_: AloySystemData) -> None:
        """Update the observer."""
        return super().update_observer(observable_)


class AloyGuiWindow(observable.Observer):
    """
    A class defining a PySide6 window used by Aloy.

    A Aloy window is a window that can contain multiple views. The views
    can optionally be selected using a combo box or a tab bar, or a custom
    interface can be used to select the views. In the case where a custom
    interface is desired, the window object emites the following signals;
    `view_changed(str)`, `view_added(str)`, and `view_removed(str)`, to
    allow the custom interface to be updated.
    """

    __slots__ = {
        "__qapp": "The main application.",
        "__qwindow": "The main window.",
        "__main_qwidget": "The main widget.",
        "__vbox": "The main vertical box layout.",
        "__stack": "The stacked widget for the views.",
        "__kind": "Whether the views are tabbed or not.",
        "__combo_box": "The combo box for the views selection.",
        "__tab_bar": "The tab bar for the views selection.",
        "__data": "The Aloy system data for the window.",
        "__views": "The views of the window.",
        "__current_view_state": "The current view state of the window.",
        "__default_qwidget": "The default widget."
    }

    def __init__(
        self,
        qapp: QtWidgets.QApplication | None = None,
        qwindow: QtWidgets.QMainWindow | None = None,
        data: AloySystemData | None = None,
        name: str | None = None,
        size: tuple[int, int] | None = None, *,
        kind: Literal["tabbed", "combo"] | None = "tabbed",
        set_title: bool = True,
        resize: bool = True,
        debug: bool = False
    ) -> None:
        """
        Create a new Aloy window within the given main window.

        This creates a new widget and sets it as the central widget of
        the main window.

        Parameters
        ----------
        `qapp: QtWidgets.QApplication | None = None` - The main application.

        `qwindow: QtWidgets.QMainWindow | None = None` - The main window.
        If not given or None, a new main window is created.

        `data: AloySystemData | None = None` - The Aloy system data for the
        window. If not given or None, a new Aloy system data object is
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

        self.__data: AloySystemData
        if data is None:
            self.__data = AloySystemData(
                name=f"System data for {self.window_name}",
                gui=self,
                clock=QtCore.QTimer(),
                debug=debug
            )
        else:
            self.__data = data
            self.__data.link_gui(self)
        self.__data.assign_observers(self)

        self.__views: dict[str, AloyWidget] = {}
        self.__current_view_state: str | None = None
        self.__default_qwidget = PlaceholderWidget(
            text="No views have been added to this window."
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
    def data(self) -> AloySystemData:
        """Get the Aloy system data for the window."""
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
        awidget: AloyWidget,
        name: str | None = None
    ) -> None:
        """
        Add a Aloy widget as a new view state to the window.

        If `name` is None, the name of the Aloy widget is used. This window's
        linked data object will be attached to the Aloy widget. The Aloy
        widget is then added to the window's view stack. If this is the first
        view state added to the window, it will be set as the current view
        state.
        """
        if name is None:
            name = awidget.observer_name

        self.__views[name] = awidget
        awidget.attach_data(self.__data)
        self.__data.update_view_states()

        self.__stack.addWidget(awidget.qwidget)
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
        awidget = self.__views.pop(name)
        self.__data.update_view_states()

        self.__stack.removeWidget(awidget.qwidget)
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

    def update_observer(self, observable_: AloySystemData) -> None:
        """Update the observer."""
        if observable_ is not self.__data:
            return
        desired_view_state: str | None = self.__data.desired_view_state
        if self.__current_view_state != desired_view_state:
            if self.__current_view_state is not None:
                awidget = self.__views[self.__current_view_state]
                self.__data.remove_observers(awidget)
            if desired_view_state is not None:
                awidget = self.__views[desired_view_state]
                self.__stack.setCurrentWidget(awidget.qwidget)
                self.__data.assign_observers(awidget)
                self.__data.notify(awidget)
            else:
                self.__current_view_state = None
                self.__stack.setCurrentWidget(self.__default_qwidget)
            self.__current_view_state = desired_view_state
