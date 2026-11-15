"""Shared options for exporting facial expression tracks beside motion files."""
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QWidget


class FaceExportControls(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.checkbox = QCheckBox('Export Face')
        self.checkbox.setToolTip('Save ARKit-style expressions from the source person beside the motion file.')
        layout.addWidget(self.checkbox)
        self.format = QComboBox()
        self.format.addItem('RGB MoCap Face (.txt)', 'txt')
        self.format.addItem('Unreal Live Link Face (.csv)', 'livelink_csv')
        self.format.setAccessibleName('Face export format')
        self.format.setToolTip('TXT includes tracking status and source timing. Unreal CSV is imported as a Level Sequence.')
        self.format.hide()
        self.checkbox.toggled.connect(self.format.setVisible)
        layout.addWidget(self.format, 1)
        layout.addStretch(1)

    def options(self):
        return dict(export_face=self.checkbox.isChecked(), face_format=self.format.currentData())
