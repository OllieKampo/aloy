"""
Module defining some example and tests of windows with menus, tool-bars and
status-bars.
"""

from PySide6.QtWidgets import (  # pylint: disable=E0611
    QMainWindow,
    QToolBar,
    QPushButton,
    QStatusBar,
    QApplication,
    QMessageBox,
    QErrorMessage,
    QLineEdit
)
from PySide6.QtCore import QSize  # pylint: disable=E0611
from PySide6.QtGui import QAction, QActionGroup  # pylint: disable=E0611

from aloy.guis.gui import AloyGuiWindow


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
        save_action = file_menu.addAction("Save")
        save_action.triggered.connect(self.create_save_message_box)
        save_as_action = file_menu.addAction("Save as...")
        save_as_action.triggered.connect(self.create_save_as_message_box)
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
        action_group = QActionGroup(self)
        action_group.addAction(grouped_action1)
        action_group.addAction(grouped_action2)
        action_group.triggered.connect(self.toolbar_button_group_clicked)
        toolbar.addActions(action_group.actions())
        for action in action_group.actions():
            action.setObjectName(action.text())
            action.setStatusTip(f"Status message from {action.objectName()}")
        toolbar.addSeparator()

        # Add a standard button to the tool-bar
        toolbar.addWidget(QPushButton("Message", self, clicked=self.create_message_box))
        toolbar.addWidget(QPushButton("Critical message", self, clicked=self.create_critical_message_box))
        toolbar.addSeparator()

        # Create an error message box
        # https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QErrorMessage.html
        self.__error_message_box = QErrorMessage()
        self.__error_message_box.setWindowTitle("Error message")
        toolbar.addWidget(QPushButton("Error message", self, clicked=self.show_error_message_box))

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

    def create_message_box(self) -> None:
        # Create a message box.
        # https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QMessageBox.html
        message_box = QMessageBox()
        message_box.setWindowTitle("Message box")
        message_box.setText("This is a message box")
        message_box.setInformativeText("This is some informative text")
        message_box.setDetailedText("This is some detailed text")
        message_box.setIcon(QMessageBox.Icon.Information)
        message_box.setStandardButtons(
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        )
        # Add a custom button
        # https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QMessageBox.html#advanced-usage
        message_box.addButton("Custom button", QMessageBox.ButtonRole.ActionRole)
        message_box.setDefaultButton(QMessageBox.StandardButton.Ok)
        message_box.setEscapeButton(QMessageBox.StandardButton.Cancel)
        return_ = message_box.exec()
        if return_ == QMessageBox.StandardButton.Ok:
            self.statusBar().showMessage("Ok clicked", 3000)
            print("Ok clicked")
        elif return_ == QMessageBox.StandardButton.Cancel:
            self.statusBar().showMessage("Cancel clicked", 3000)
            print("Cancel clicked")

    def create_save_message_box(self) -> None:
        # Create a save message box
        message_box = QMessageBox()
        message_box.setText("The document has been modified.")
        message_box.setInformativeText("Do you want to save your changes?")
        message_box.setIcon(QMessageBox.Icon.Question)
        message_box.setStandardButtons(
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel
        )
        message_box.setDefaultButton(QMessageBox.StandardButton.Save)
        return_ = message_box.exec()
        if return_ == QMessageBox.StandardButton.Save:
            self.statusBar().showMessage("Save clicked", 3000)
            print("Save clicked")
        elif return_ == QMessageBox.StandardButton.Discard:
            self.statusBar().showMessage("Discard clicked", 3000)
            print("Discard clicked")
        elif return_ == QMessageBox.StandardButton.Cancel:
            self.statusBar().showMessage("Cancel clicked", 3000)
            print("Cancel clicked")

    def create_save_as_message_box(self) -> None:
        # Create a save as message box
        message_box = QMessageBox()
        message_box.setText("The document has been modified.")
        message_box.setInformativeText("Choose a name and directory for the document.")
        # Add a line edit to the message box
        line_edit = QLineEdit(message_box)
        line_edit.setPlaceholderText("File name")
        message_box.layout().addWidget(line_edit, 1, 1, 1, 1)
        message_box.setIcon(QMessageBox.Icon.Question)
        message_box.setStandardButtons(
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel
        )
        message_box.setDefaultButton(QMessageBox.StandardButton.Save)
        return_ = message_box.exec()
        if return_ == QMessageBox.StandardButton.Save:
            self.statusBar().showMessage("Save clicked", 3000)
            print("Save clicked")
        elif return_ == QMessageBox.StandardButton.Discard:
            self.statusBar().showMessage("Discard clicked", 3000)
            print("Discard clicked")
        elif return_ == QMessageBox.StandardButton.Cancel:
            self.statusBar().showMessage("Cancel clicked", 3000)
            print("Cancel clicked")

    def create_critical_message_box(self) -> None:
        # Create a critical message box using the static functions API.
        # https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QMessageBox.html#the-static-functions-api
        return_ = QMessageBox.critical(
            self,
            "Critical message box",
            "This is a critical message box",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Ok
        )
        if return_ == QMessageBox.StandardButton.Ok:
            self.statusBar().showMessage("Ok clicked", 3000)
            print("Ok clicked")
        elif return_ == QMessageBox.StandardButton.Cancel:
            self.statusBar().showMessage("Cancel clicked", 3000)
            print("Cancel clicked")

    def show_error_message_box(self) -> None:
        self.__error_message_box.showMessage("This is an error message box")


if __name__ == "__main__":
    qapp = QApplication([])
    qwindow = TestWindow(qapp)
    jgui = AloyGuiWindow(qwindow=qwindow, name="Window Test", size=(800, 600))
    qwindow.show()
    qapp.exec()
