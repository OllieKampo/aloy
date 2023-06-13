"""Module defining widgets for displaying text."""

from abc import abstractmethod
import PySide6.QtWidgets as QtWidgets
import PySide6.QtCore as QtCore


class ScrollTextPanel(QtWidgets.QWidget):
    """Base class mixin for scrollable text panels."""

    @abstractmethod
    def set_text(self, text: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_line(self, line: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError


class ScrollTextEditPanel(ScrollTextPanel):
    """Text panel that displays text in a text edit with a scrollbar."""

    def __init__(
        self,
        *text: str,
        parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.__text: str = "\n".join(text)

        self.__layout = QtWidgets.QVBoxLayout(self)
        self.__layout.setContentsMargins(0, 0, 0, 0)
        self.__layout.setSpacing(0)

        self.__text_edit = QtWidgets.QTextEdit(self.__text, self)
        self.__text_edit.setReadOnly(True)
        self.__text_edit.setLineWrapMode(
            QtWidgets.QTextEdit.LineWrapMode.WidgetWidth
        )
        self.__text_edit.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )
        self.__text_edit.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.__text_edit.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.__layout.addWidget(self.__text_edit)

        self.setLayout(self.__layout)

    def set_text(self, text: str) -> None:
        self.__text = text
        self.__text_edit.setText(self.__text)

    def add_line(self, line: str) -> None:
        if self.__text:
            self.__text += "\n" + line
        else:
            self.__text = line
        self.__text_edit.setText(self.__text)

    def clear(self) -> None:
        self.__text = ""
        self.__text_edit.setText(self.__text)


class ScrollTextListPanel(ScrollTextPanel):
    """Text panel that displays text in a list widget with a scrollbar."""

    def __init__(
        self,
        *text: str,
        parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.__text: list[str] = list(text)

        self.__layout = QtWidgets.QVBoxLayout(self)
        self.__layout.setContentsMargins(0, 0, 0, 0)
        self.__layout.setSpacing(0)

        self.__list_widget = QtWidgets.QListWidget(self)
        self.__list_widget.setWordWrap(True)
        self.__list_widget.setAutoScroll(True)
        self.__list_widget.setAlternatingRowColors(True)
        self.__list_widget.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )
        self.__list_widget.setViewMode(
            QtWidgets.QListView.ViewMode.ListMode
        )
        self.__list_widget.setResizeMode(
            QtWidgets.QListView.ResizeMode.Adjust
        )
        self.__list_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.__layout.addWidget(self.__list_widget)

        self.setLayout(self.__layout)

    def set_text(self, *text: str) -> None:
        self.__text = list(text)
        self.__list_widget.clear()
        self.__list_widget.addItems(self.__text)

    def add_line(self, line: str) -> None:
        self.__text.append(line)
        self.__list_widget.addItem(line)

    def clear(self) -> None:
        self.__text.clear()
        self.__list_widget.clear()


class TabbedScrollTextListPanel(QtWidgets.QWidget):
    """
    A panel with a tabbed interface for displaying text.
    """

    def __init__(self, *tab_names: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.__tabs: dict[str, ScrollTextListPanel] = {}

        self.__layout = QtWidgets.QVBoxLayout(self)
        self.__layout.setContentsMargins(0, 0, 0, 0)
        self.__layout.setSpacing(0)

        self.__tab_widget = QtWidgets.QTabWidget(self)
        for tab_name in tab_names:
            self.add_tab(tab_name)
        self.__layout.addWidget(self.__tab_widget)

        self.setLayout(self.__layout)

    def tabs(self) -> list[str]:
        """
        Get the names of the tabs in the panel.

        :return: A list of the names of the tabs in the panel.
        """
        return list(self.__tabs.keys())

    def get_tab(self, tab_name: str) -> ScrollTextListPanel:
        """
        Get the tab with the given name.

        :param tab_name: The name of the tab to get.
        :return: The tab with the given name, or None if no such tab exists.
        """
        return self.__tabs[tab_name]

    def add_tab(self, tab_name: str, *text: str) -> None:
        """
        Add a new tab to the panel.

        :param tab_name: The name of the tab to add.
        """
        self.__tabs[tab_name] = ScrollTextListPanel(
            text,
            parent=self.__tab_widget
        )
        self.__tab_widget.addTab(self.__tabs[tab_name], tab_name)

    def remove_tab(self, tab_name: str) -> None:
        """
        Remove a tab from the panel.

        :param tab_name: The name of the tab to remove.
        """
        self.__tab_widget.removeTab(self.__tab_widget.indexOf(self.__tabs[tab_name]))
        del self.__tabs[tab_name]


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window_1 = QtWidgets.QMainWindow()
    window_2 = QtWidgets.QMainWindow()
    window_3 = QtWidgets.QMainWindow()
    list_panel = ScrollTextListPanel(parent=window_1)
    edit_panel = ScrollTextEditPanel(parent=window_2)
    tabbed_panel = TabbedScrollTextListPanel("Tab 1", "Tab 2", parent=window_1)
    window_1.setCentralWidget(list_panel)
    window_1.show()
    window_2.setCentralWidget(edit_panel)
    window_2.show()
    window_3.setCentralWidget(tabbed_panel)
    window_3.show()
    def add_line():
        list_panel.add_line("Hello" + str(list(range(100))))
        edit_panel.add_line("Hello" + str(list(range(100))))
        tabbed_panel.get_tab("Tab 1").add_line("Hello" + str(list(range(100, 200))))
        tabbed_panel.get_tab("Tab 2").add_line("Hello" + str(list(range(200, 300))))
    timer = QtCore.QTimer()
    timer.timeout.connect(add_line)
    timer.start(1000)
    sys.exit(app.exec_())
