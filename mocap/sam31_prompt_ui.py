"""Text prompt and optional person markers on the paused source frame."""

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QLabel, QLineEdit,
                               QPushButton, QVBoxLayout, QWidget)


class MarkingFrame(QWidget):
    def __init__(self, image):
        super().__init__()
        self.image = image
        self.points = []
        self.setMinimumSize(280, 200)
        self.setAccessibleName('Frame for marking people')

    def image_rect(self):
        size = self.image.size().scaled(self.size(), Qt.KeepAspectRatio)
        return QRectF((self.width() - size.width()) / 2, (self.height() - size.height()) / 2,
                      size.width(), size.height())

    def mousePressEvent(self, event):
        pos = event.position()
        rect = self.image_rect()
        if event.button() == Qt.LeftButton and rect.contains(pos) and len(self.points) < 16:
            self.points.append((min(.999999, (pos.x()-rect.x())/rect.width()),
                                min(.999999, (pos.y()-rect.y())/rect.height())))
        elif event.button() == Qt.RightButton and self.points:
            self.points.pop()
        self.update()

    def clear(self):
        self.points.clear()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.image_rect()
        painter.fillRect(self.rect(), QColor('#202020'))
        painter.drawImage(rect, self.image)
        painter.setPen(QPen(QColor('#1b1b1b'), 2))
        painter.setBrush(QColor('#80dbff'))
        for x, y in self.points:
            painter.drawEllipse(QPointF(rect.x()+x*rect.width(), rect.y()+y*rect.height()), 5, 5)


class PromptDialog(QDialog):
    def __init__(self, parent, image, prompt):
        super().__init__(parent)
        self.setWindowTitle('Find person with SAM 3.1')
        self.resize(620, 640)
        layout = QVBoxLayout(self)
        text = QLabel('Enter a text prompt. To keep only specific people, click once on each person in this frame. '
                      'With no markers, all matches are tracked. Right-click to undo the last marker.')
        text.setWordWrap(True)
        layout.addWidget(text)
        self.prompt = QLineEdit(prompt)
        self.prompt.setMaxLength(256)
        self.prompt.setPlaceholderText('person')
        self.prompt.setAccessibleName('Tracking text prompt')
        layout.addWidget(self.prompt)
        self.frame = MarkingFrame(image)
        layout.addWidget(self.frame, 1)
        clear = QPushButton('Clear Markers')
        clear.clicked.connect(self.frame.clear)
        layout.addWidget(clear)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText('Track')
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
