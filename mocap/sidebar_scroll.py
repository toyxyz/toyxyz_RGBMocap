"""Keep sidebar wheel input scrolling instead of editing inference settings."""
from PySide6.QtCore import QEvent,QPointF,Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication,QScrollArea,QComboBox,QAbstractSpinBox


class SettingsScrollArea(QScrollArea):
    def setWidget(self,widget):
        super().setWidget(widget)
        self._wheel_controls={}
        for control in [*widget.findChildren(QComboBox),*widget.findChildren(QAbstractSpinBox)]:
            self._wheel_controls[control]=control
            control.installEventFilter(self)
            if isinstance(control,QAbstractSpinBox):
                edit=control.lineEdit()
                self._wheel_controls[edit]=control;edit.installEventFilter(self)

    def eventFilter(self,watched,event):
        if event.type()!=QEvent.Wheel:return super().eventFilter(watched,event)
        control=getattr(self,'_wheel_controls',{}).get(watched)
        if control is None:return super().eventFilter(watched,event)
        # A deliberately opened list retains its normal popup scrolling.
        if isinstance(control,QComboBox) and control.view().isVisible():return False
        # Deliberate focused editing remains available without capturing ordinary scrolls.
        focused=control.hasFocus() or (isinstance(control,QAbstractSpinBox) and control.lineEdit().hasFocus())
        if focused and event.modifiers()&Qt.ControlModifier:return False
        point=QPointF(self.viewport().mapFromGlobal(event.globalPosition().toPoint()))
        forwarded=QWheelEvent(point,event.globalPosition(),event.pixelDelta(),event.angleDelta(),
            event.buttons(),event.modifiers(),event.phase(),event.inverted(),event.source(),event.pointingDevice())
        QApplication.sendEvent(self.viewport(),forwarded)
        event.accept()
        return True
