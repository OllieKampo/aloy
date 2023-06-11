from PyQt6.QtWidgets import QMainWindow, QToolBar, QPushButton, QStatusBar, QApplication
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QAction, QActionGroup

from guis.gui import JinxGuiWindow


class TestWindow(QMainWindow):
    def __init__(self, app: QApplication) -> None:
        super().__init__()
        self.__app = app

        # Add a menu-bar
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("File")
        file_menu.addSeparator()
        file_menu.addAction("New")
        file_menu.addAction("Open")
        file_menu.addAction("Save")
        file_menu.addAction("Save as...")
        file_menu.addSeparator()
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.exit_app)

        # Edit menu with a sub-menu
        edit_menu = menu_bar.addMenu("Edit")
        edit_menu.addAction("Copy")
        edit_menu.addAction("Cut")
        edit_menu.addAction("Paste")
        edit_menu.addSeparator()
        edit_menu.addAction("Undo")
        edit_menu.addAction("Redo")
        edit_menu.addSeparator()
        find_edit_menu = edit_menu.addMenu("Find")
        find_edit_menu.addAction("Find...")
        find_edit_menu.addAction("Find next")
        find_edit_menu.addAction("Find previous")
        find_edit_menu.addSeparator()
        find_edit_menu.addAction("Replace...")
        find_edit_menu.addAction("Replace next")
        find_edit_menu.addAction("Replace previous")

        # Settings menu with sub-sections
        settings_menu = menu_bar.addMenu("Settings")
        settings_menu.addAction("Preferences")
        settings_menu.addSection("Appearance")
        settings_menu.addAction("Theme")
        settings_menu.addAction("Font")
        settings_menu.addAction("Color")
        settings_menu.addSection("Layout")
        settings_menu.addAction("Window size")
        settings_menu.addAction("Window position")
        settings_menu.addSection("Profile")
        settings_menu.addAction("Export profile")
        settings_menu.addAction("Import profile")

        # Add a tool-bar
        toolbar = QToolBar("My main toolbar")
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(toolbar)

        # Exit tool-bar action
        toolbar.addAction(exit_action)
        toolbar.addSeparator()

        # Two separate actions
        action1 = QAction("Action 1", self)
        action1.setStatusTip("Status message from action 1")
        action1.triggered.connect(self.toolbar_action1_triggered)
        toolbar.addAction(action1)
        action2 = QAction("Action 2", self)
        action2.setStatusTip("Status message from action 2")
        action2.triggered.connect(self.toolbar_action2_triggered)
        toolbar.addAction(action2)
        toolbar.addSeparator()

        # Three grouped actions
        grouped_action1 = QAction("Grouped action 1", self)
        grouped_action2 = QAction("Grouped action 2", self)
        grouped_action3 = QAction("Grouped action 3", self)
        action_group = QActionGroup(self)
        action_group.addAction(grouped_action1)
        action_group.addAction(grouped_action2)
        action_group.addAction(grouped_action3)
        action_group.triggered.connect(self.toolbar_button_group_clicked)
        toolbar.addActions(action_group.actions())
        for action in action_group.actions():
            action.setObjectName(action.text())
            action.setStatusTip(f"Status message from {action.objectName()}")
        toolbar.addSeparator()

        # Add a standard button to the tool-bar
        toolbar.addWidget(QPushButton("Standard Button"))

        # Add a status-bar at the bottom of the window
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)

    def exit_app(self) -> None:
        self.__app.quit()

    def toolbar_action1_triggered(self) -> None:
        self.statusBar().showMessage("Action 1 triggered", 3000)

    def toolbar_action2_triggered(self) -> None:
        self.statusBar().showMessage("Action 2 triggered", 3000)

    def toolbar_button_group_clicked(self, action: QAction) -> None:
        self.statusBar().showMessage(
            f"Message from {action.objectName()}",
            3000
        )


if __name__ == "__main__":
    qapp = QApplication([])
    qwindow = TestWindow(qapp)
    jgui = JinxGuiWindow(qwindow=qwindow, name="Window Test", size=(800, 600))
    qwindow.show()
    qapp.exec()
