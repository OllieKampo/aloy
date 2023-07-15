import time
from PySide6.QtGui import QPixmap, QColor, QPainter, QPen, Qt, QFont
from PySide6.QtWidgets import QApplication, QSplashScreen, QMainWindow, QWidget, QVBoxLayout, QPushButton, QTabWidget
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("My App")
        self.setGeometry(100, 100, 800, 600)

        # Create the tab widget and add it to the main window
        self.tab_widget = QTabWidget()
        self.tab_widget.tabBar().setVisible(False)
        self.setCentralWidget(self.tab_widget)

        # Create the initial view widget with a launch button
        initial_widget = QWidget()
        initial_layout = QVBoxLayout()
        initial_widget.setLayout(initial_layout)
        launch_button = QPushButton("Launch")
        launch_button.clicked.connect(self.launch_clicked)
        initial_layout.addWidget(launch_button)

        # Add the initial view widget to the tab widget
        self.tab_widget.addTab(initial_widget, "Initial View")

    def launch_clicked(self):
        # Create the second view widget with four numbered buttons
        second_widget = QWidget()
        second_layout = QVBoxLayout()
        second_widget.setLayout(second_layout)
        for i in range(1, 5):
            button = QPushButton(str(i))
            second_layout.addWidget(button)

        # Add the second view widget to the tab widget and switch to it
        self.tab_widget.addTab(second_widget, "Second View")
        self.tab_widget.setCurrentWidget(second_widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    screen = app.screens()[1]

    # Create a blank pixmap.
    pixmap = QPixmap(500, 500)
    pixmap.fill(QColor('transparent'))

    # Draw a red cross.
    painter = QPainter(pixmap)
    painter.setPen(QPen(QColor('red'), 2))
    painter.drawLine(0, 0, 500, 500)
    painter.drawLine(0, 500, 500, 0)
    painter.end()

    # Create and show the splash screen.
    splash = QSplashScreen(screen, pixmap)
    splash.show()

    # Set the font of the text.
    font = QFont("Arial", 20)
    splash.setFont(font)

    splash.showMessage(
        "Loading...",
        alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
        color=QColor('white')
    )
    app.processEvents()
    time.sleep(2)

    # Simulate something that takes time.
    splash.showMessage(
        "Loading more...",
        alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
        color=QColor('white')
    )
    time.sleep(2)

    # Load the main window.
    main_window = MainWindow()
    main_window.show()

    # Close the splash screen.
    splash.finish(main_window)

    sys.exit(app.exec())
