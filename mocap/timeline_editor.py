"""Frame ruler with bounded cached tracks and direct seeking."""
import math
from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QSlider, QSizePolicy
from .theme import current_appearance, theme_color, ui_px
from .footlock_editor import FootLockEditor


class TimelineSlider(FootLockEditor,QSlider):
    clip_changed = Signal(int, int)  # Inclusive in, exclusive out, in media frames.
    trim_started = Signal()
    contacts_changed=Signal(object)
    contacts_edit_started=Signal()
    contacts_reset=Signal()
    contacts_preview_changed=Signal()

    def __init__(self, orientation=Qt.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.init_contacts()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.StrongFocus)
        self.fps = 30.0
        self.has_motion = False
        self.person_color = self.person_name = None
        self.video_track_label = 'Source clip'
        self.view_start = 0.0
        self.view_span = None
        self._cache = None
        self._cache_key = None
        self._pan_origin = None
        self._playhead_drag = None
        self.clip_in, self.clip_out = 0, 1
        self.trim_enabled = True
        self._trim_drag = None
        self.setMouseTracking(True)
        self.rangeChanged.connect(self.frame_all)
        self.rangeChanged.connect(self.reset_trim)
        self.valueChanged.connect(self._follow)
        self.refresh_appearance()

    def refresh_appearance(self):
        self.LEFT, self.RIGHT = ui_px(64), ui_px(14)
        self.TRACK_TOP, self.TRACK_HEIGHT = ui_px(28), ui_px(26)
        self.RULER_HEIGHT = ui_px(25)
        self.setMinimumHeight(ui_px(190))
        self._cache_key = None
        self.update()

    def frame_all(self, *_):
        self.view_start = float(self.minimum())
        self.view_span = None
        self.update()

    def set_media(self, fps, has_motion=None):
        self.fps = fps
        if has_motion is not None:
            self.has_motion = has_motion
        self.update()

    def minimum_clip_frames(self):
        # Reconstruction prepares at 30 fps and requires at least four frames.
        return min(self.maximum() + 1, max(1, math.ceil(self.fps * 4 / 30)))

    def reset_trim(self, *_):
        self.set_clip(0, self.maximum() + 1)

    def set_clip(self, first, end):
        total, minimum = self.maximum() + 1, self.minimum_clip_frames()
        first = max(0, min(total - minimum, int(first)))
        end = max(first + minimum, min(total, int(end)))
        if (first, end) != (self.clip_in, self.clip_out):
            self.clip_in, self.clip_out = first, end
            self.clip_changed.emit(first, end)
            self.update()

    def clip_edges(self):
        return self.frame_x(self.clip_in), self.frame_x(self.clip_out - 1)

    def _trim_handle(self, event):
        if not self.trim_enabled or not ui_px(29) <= event.position().y() <= ui_px(52):
            return None
        x = event.position().x()
        edges = [(abs(x - edge), name) for edge, name in zip(self.clip_edges(), ("in", "out"))
                 if self.LEFT - 1 <= edge <= self.width() - self.RIGHT + 1]
        distance, name = min(edges, default=(float("inf"), None))
        return name if distance <= ui_px(8) else None

    def _drag_trim(self, event):
        frame = self._position(event)
        if self._trim_drag == "in":
            self.set_clip(min(frame, self.clip_out - self.minimum_clip_frames()), self.clip_out)
        else:
            self.set_clip(self.clip_in, max(frame + 1, self.clip_in + self.minimum_clip_frames()))
        self.setValue(self.clip_in if self._trim_drag == "in" else self.clip_out - 1)

    def span(self):
        return self.view_span if self.view_span is not None else max(1, self.maximum() - self.minimum())

    def frame_x(self, frame):
        return self.LEFT + (frame - self.view_start) / self.span() * max(1, self.width() - self.LEFT - self.RIGHT)

    def _position(self, event):
        width = max(1, self.width() - self.LEFT - self.RIGHT)
        value = self.view_start + (event.position().x() - self.LEFT) / width * self.span()
        return max(self.minimum(), min(self.maximum(), round(value)))

    def playhead_rect(self):
        """Share painted and clickable geometry; the line and pointer are inert."""
        width = max(ui_px(25), self.fontMetrics().horizontalAdvance(str(self.value() + 1)) + ui_px(12))
        return QRectF(self.frame_x(self.value()) - width / 2, ui_px(1), width, ui_px(21))

    def _in_ruler(self,event):
        point=event.position()
        return 0 <= point.y() < self.RULER_HEIGHT and self.LEFT <= point.x() <= self.width()-self.RIGHT

    def _drag_playhead(self, event):
        pixel, frame, scale = self._playhead_drag
        value = round(frame + (event.position().x() - pixel) * scale)
        self.setValue(max(self.minimum(), min(self.maximum(), value)))

    def _clamp_view(self):
        self.view_start = max(self.minimum(), min(max(self.minimum(), self.maximum() - self.span()), self.view_start))

    def _follow(self, value):
        if self.view_span is not None and not self.view_start <= value <= self.view_start + self.span():
            self.view_start = value - self.span() * .2
            self._clamp_view()
        self.update()

    def mousePressEvent(self, event):
        if self._foot_press(event):return
        if event.button() == Qt.MiddleButton:
            self._pan_origin = (event.position().x(), self.view_start)
            event.accept()
        elif event.button() == Qt.LeftButton:
            self._trim_drag = self._trim_handle(event)
            if self._trim_drag:
                self.trim_started.emit()
                self.setCursor(Qt.SizeHorCursor)
            elif self.playhead_rect().contains(event.position()):
                self.setFocus(Qt.MouseFocusReason)
                self._playhead_drag = (event.position().x(), self.value(),
                    self.span() / max(1, self.width() - self.LEFT - self.RIGHT))
                self.setSliderDown(True)
                self.setCursor(Qt.ClosedHandCursor)
            elif self._in_ruler(event):
                self.setFocus(Qt.MouseFocusReason)
                self.setValue(self._position(event))
                self._playhead_drag = (event.position().x(), self.value(),
                    self.span() / max(1, self.width() - self.LEFT - self.RIGHT))
                self.setSliderDown(True)
                self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._foot_move(event):return
        if self._pan_origin is not None:
            pixel, start = self._pan_origin
            self.view_start = start - (event.position().x() - pixel) * self.span() / max(1, self.width() - self.LEFT - self.RIGHT)
            self._clamp_view()
            self.update()
            event.accept()
        elif self._trim_drag:
            self._drag_trim(event)
            event.accept()
        elif self._playhead_drag is not None:
            self._drag_playhead(event)
            event.accept()
        else:
            foot=self._contact_row(event)
            if foot is not None and self.contact_edit_enabled and event.position().x()>=self.LEFT:
                _,mode=self._contact_hit(foot,event)
                self.setCursor(Qt.SizeHorCursor if mode in ('left','right') else Qt.OpenHandCursor if mode=='move' else Qt.CrossCursor)
            else:self.setCursor(Qt.OpenHandCursor if self.playhead_rect().contains(event.position())
                else Qt.PointingHandCursor if self._in_ruler(event)
                else Qt.SizeHorCursor if self._trim_handle(event) else Qt.ArrowCursor)
            event.accept()

    def mouseReleaseEvent(self, event):
        if self._foot_release(event):return
        if event.button() == Qt.MiddleButton:
            self._pan_origin = None
            event.accept()
        elif event.button() == Qt.LeftButton:
            if self._trim_drag:
                self._drag_trim(event)
                self._trim_drag = None
            elif self._playhead_drag is not None:
                self._drag_playhead(event)
                self._playhead_drag = None
                self.setSliderDown(False)
            self.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        old_span = self.span()
        if event.modifiers() & Qt.ShiftModifier:
            self.view_start -= event.angleDelta().y() / 120 * old_span * .12
        else:
            fraction = max(0, min(1, (event.position().x() - self.LEFT) / max(1, self.width() - self.LEFT - self.RIGHT)))
            anchor = self.view_start + fraction * old_span
            total = max(1, self.maximum() - self.minimum())
            self.view_span = max(min(12, total), min(total, old_span * math.exp(-event.angleDelta().y() / 120 * .22)))
            self.view_start = anchor - fraction * self.view_span
        self._clamp_view()
        self.update()
        event.accept()

    def paintEvent(self, _):
        key = (current_appearance(), self.size(), self.devicePixelRatioF(), self.minimum(), self.maximum(), self.view_start,
               self.span(), self.fps, self.has_motion, self.clip_in, self.clip_out,self.contact_revision,self.contact_edit_enabled,self.contact_track_enabled,
               self.person_color,self.person_name,self.video_track_label,
               (self._foot_drag['foot'],self._foot_drag['index']) if self._foot_drag else None,self._selected_contact)
        if key != self._cache_key:
            ratio = self.devicePixelRatioF()
            cache = QPixmap(round(self.width() * ratio), round(self.height() * ratio))
            cache.setDevicePixelRatio(ratio)
            cache.fill(QColor(theme_color("#242424")))
            p = QPainter(cache)
            p.setFont(self.font())
            p.fillRect(0, 0, self.width(), self.RULER_HEIGHT, QColor(theme_color("#303030")))
            target = self.span() / max(1, (self.width() - self.LEFT) / ui_px(85))
            power = 10 ** math.floor(math.log10(max(1, target)))
            step = next((v * power for v in (1, 2, 5, 10) if v * power >= target), power * 10)
            first = math.ceil((self.view_start + 1) / step) * step - 1
            row_height = self.TRACK_HEIGHT
            rows = ((0, "Video", "#6289ad"), (1, "Motion", (self.person_color or "#6aa186") if self.has_motion else "#303030"))
            # Rows keep an editor-like height; extra panel space is available
            # below them for future channels rather than stretching controls.
            for row, label, color in rows:
                y = self.TRACK_TOP + row * row_height
                tint = QColor(color)
                tint.setAlpha(80)
                left, right = self.clip_edges() if row == 0 else (self.LEFT, self.width() - self.RIGHT)
                left, right = max(self.LEFT, left), min(self.width() - self.RIGHT, right)
                p.fillRect(QRectF(left, y + 2, max(0, right - left), row_height - 5), tint)
            self.paint_contact_tracks(p)
            # Draw the time grid above the translucent tracks. The visible tick
            # count remains bounded by viewport width, independent of duration.
            p.save()
            p.setClipRect(QRectF(self.LEFT, self.RULER_HEIGHT, max(1,self.width()-self.LEFT-self.RIGHT),self.height()-self.RULER_HEIGHT))
            minor_step = max(1,step//5)
            minor_first = math.ceil((self.view_start+1)/minor_step)*minor_step-1
            p.setPen(QPen(QColor(133,133,133,40),1))
            for frame in range(minor_first,math.ceil(self.view_start+self.span())+1,minor_step):
                x = self.frame_x(frame)
                p.drawLine(QPointF(x, self.RULER_HEIGHT),QPointF(x,self.height()))
            p.setPen(QPen(QColor(115,115,115,145),1))
            for frame in range(first,math.ceil(self.view_start+self.span())+1,step):
                x = self.frame_x(frame)
                p.drawLine(QPointF(x, self.RULER_HEIGHT),QPointF(x,self.height()))
            p.restore()
            p.setPen(QColor(theme_color("#999999")))
            p.drawText(ui_px(9), ui_px(17), "Frame")
            for frame in range(first,math.ceil(self.view_start+self.span())+1,step):
                x = self.frame_x(frame)
                p.setPen(QColor(theme_color("#b3b3b3")))
                p.drawText(QRectF(x-ui_px(35), ui_px(2), ui_px(70), ui_px(20)),Qt.AlignCenter,str(frame+1))
            for row,label,color in rows:
                y = self.TRACK_TOP + row * row_height
                p.setPen(QColor(theme_color("#aaaaaa")))
                p.drawText(QRectF(ui_px(8), y, ui_px(53), row_height), Qt.AlignVCenter, label)
                p.setPen(QColor(theme_color("#b1b9c1" if row == 0 else "#aebeb4")))
                text = self.video_track_label if row == 0 else ((self.person_name or "Body animation") if self.has_motion else "No motion data")
                left, right = self.clip_edges() if row == 0 else (self.LEFT, self.width() - self.RIGHT)
                left, right = max(self.LEFT, left), min(self.width() - self.RIGHT, right)
                p.drawText(QRectF(left + 12, y, max(0, right - left - 24), row_height), Qt.AlignVCenter, text)
            # Edge grips belong to the video clip, never to the playhead/ruler.
            p.setPen(QPen(QColor(theme_color("#a6bed5")), 1))
            left, right = self.clip_edges()
            for edge, direction in ((left, 1), (right, -1)):
                if self.LEFT <= edge <= self.width() - self.RIGHT:
                    p.fillRect(QRectF(edge if direction == 1 else edge - ui_px(6), ui_px(30), ui_px(6), ui_px(21)), QColor(139,177,211,150))
                    p.drawLine(QPointF(edge, ui_px(30)), QPointF(edge, ui_px(51)))
            p.end()
            self._cache, self._cache_key = cache, key
        p = QPainter(self)
        p.setFont(self.font())
        p.drawPixmap(0, 0, self._cache)
        self.paint_contact_selection(p)
        x = self.frame_x(self.value())
        p.setPen(QPen(QColor(theme_color("#74a6e0")), 1.4))
        p.drawLine(QPointF(x, ui_px(23)), QPointF(x, self.height()))
        p.setRenderHint(QPainter.Antialiasing)
        text = str(self.value() + 1)
        head = self.playhead_rect()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(theme_color("#477bb7")))
        p.drawRoundedRect(head, 3, 3)
        p.drawPolygon(QPolygonF([QPointF(x - ui_px(4), ui_px(21)), QPointF(x + ui_px(4), ui_px(21)), QPointF(x, ui_px(26))]))
        p.setPen(QColor("#ffffff"))
        p.drawText(head, Qt.AlignCenter, text)
