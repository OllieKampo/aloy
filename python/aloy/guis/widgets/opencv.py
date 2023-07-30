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

"""Widgets for displaying OpenCV video streams."""

import threading
from PySide6 import QtCore, QtWidgets, QtGui
import cv2
import numpy as np


class CvVideoViewer(QtWidgets.QWidget):
    """Widget defining a labelled arrow button."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        video_source: int | str = 0,
        fps: int = 30,
        display_size: tuple[int, int] | None = None
    ) -> None:
        """Create a new labelled arrow button QtWidget widget."""
        super().__init__(parent)

        self.__layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.__layout)

        self.__video_label = QtWidgets.QLabel()
        self.__video_label.setScaledContents(True)
        self.__video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.__layout.addWidget(self.__video_label)

        self.__enable_button = QtWidgets.QPushButton("Enable Video")
        self.__enable_button.setCheckable(True)
        self.__enable_button.clicked.connect(self.__enable_video)
        self.__layout.addWidget(self.__enable_button)

        self.__image_updater_thread = ImageUpdaterThread(
            video_source=video_source,
            fps=fps,
            display_size=display_size
        )
        self.__image_updater_thread.image_updated.connect(
            self.__update_image
        )
        self.__image_updater_thread.start()

    @QtCore.Slot(QtGui.QImage)
    def __update_image(self, image: QtGui.QImage) -> None:
        """Update the image on the label."""
        self.__video_label.setPixmap(QtGui.QPixmap.fromImage(image))

    def __enable_video(self) -> None:
        """Enable or disable the video."""
        if self.__enable_button.isChecked():
            self.__image_updater_thread.enable()
        else:
            self.__image_updater_thread.disable()

    # pylint: disable=invalid-name
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Stop the thread when the widget is closed."""
        self.__image_updater_thread.kill()
        event.accept()


class ImageUpdaterThread(QtCore.QThread):
    """Thread for updating an image."""

    __slots__ = {
        "__video_source": "The video source.",
        "__fps": "The frames per second.",
        "__display_size": "The size to display the image at.",
        "__capture": "The video capture object.",
        "__streaming": "Event indicating whether the thread is streaming.",
        "__killed": "Event indicating whether to stop the thread."
    }

    image_updated = QtCore.Signal(QtGui.QImage)

    def __init__(
        self,
        parent: QtCore.QObject | None = None,
        video_source: int | str = 0,
        fps: int = 30,
        display_size: tuple[int, int] | None = None
    ) -> None:
        """Create a new image updater thread."""
        super().__init__(parent)

        self.__video_source: int | str = video_source
        self.__fps: int = fps
        self.__display_size: tuple[int, int] | None = display_size

        self.__capture = cv2.VideoCapture(self.__video_source)
        self.__streaming = threading.Event()
        self.__killed = threading.Event()

    def run(self) -> None:
        """Run the video updater thread."""
        kill: bool = False
        while True:
            self.__streaming.wait()
            while not (kill := self.__killed.wait(1 / self.__fps)):
                ret, frame = self.__capture.read()
                if ret:
                    image = self.__convert_frame(frame)
                    self.image_updated.emit(image)
            if kill:
                self.__capture.release()
                break

    def __convert_frame(self, frame: np.ndarray) -> QtGui.QImage:
        """Convert a frame to a QImage."""
        cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv_image_flipped = cv2.flip(cv_image, 1)
        height, width, channel = cv_image_flipped.shape
        bytes_per_line = channel * width
        qt_image = QtGui.QImage(
            cv_image_flipped.data,
            width,
            height,
            bytes_per_line,
            QtGui.QImage.Format.Format_RGB888
        )
        if self.__display_size is not None:
            qt_image = qt_image.scaled(
                *self.__display_size,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio
            )
        return qt_image

    def enable(self) -> None:
        """Enable the thread."""
        self.__streaming.set()

    def disable(self) -> None:
        """Disable the thread."""
        self.__streaming.clear()

    def kill(self) -> None:
        """Kill the thread."""
        self.__killed.set()
        self.wait()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = CvVideoViewer()
    window.show()
    sys.exit(app.exec())
