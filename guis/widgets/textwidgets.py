
import PySide6.QtWidgets as QtWidgets
import PySide6.QtCore as QtCore
import PySide6.QtGui as QtGui


class ScrollTextPanel(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget, text: str) -> None:
        super().__init__(parent)
        self.__text = text

        self.__layout = QtWidgets.QVBoxLayout(self)
        self.__layout.setContentsMargins(0, 0, 0, 0)
        self.__layout.setSpacing(0)

        self.__scroll_area = QtWidgets.QScrollArea(self)
        self.__scroll_area.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )
        self.__scroll_area.setWidgetResizable(True)
        self.__scroll_area.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.__layout.addWidget(self.__scroll_area)

        self.__text_edit = QtWidgets.QTextEdit(self.__text, self)
        self.__text_edit.setReadOnly(True)
        self.__text_edit.setLineWrapMode(
            QtWidgets.QTextEdit.LineWrapMode.WidgetWidth
        )
        self.__text_edit.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )
        self.__text_edit.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.__scroll_area.setWidget(self.__text_edit)

        self.setLayout(self.__layout)

    def set_text(self, text: str) -> None:
        pass


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    window.setCentralWidget(ScrollTextPanel(window, str(list(range(1000)))))
    window.show()
    sys.exit(app.exec_())