from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget
from .video_preview import PreviewDecoder
from .viewport import MotionView

COCO_BONES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]


class VideoView(QWidget):
    frame_presented = Signal(int)
    frame_painted = Signal(int)
    decode_error = Signal(str)
    metadata_ready = Signal(object)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(440, 280)
        cv2.setNumThreads(1)  # Preview uses one decoder, not a pool for each resize.
        self.external_mode=False
        self.preview_rotation=0.
        self._external_saved=None
        self.frame = None
        self.boxes = self.observed = self.keypoints = self.confidence = None
        self.index = 0
        self.requested_index = 0
        self.decoder = None
        self.source_size = (1, 1)
        self.generation = 0
        self.logical_size = None
        self.timeline_fps = None
        self.native_fps = None
        self._source_key = None
        self.timeline_requested_index = 0
        self.show_observations = True
        self.projected_overlay = None
        self.actor_color = None
        self.zoom = 1.0
        self.pan = QPointF()
        self._pan_position = None

    def reset_view(self):
        self.zoom = 1.0
        self.pan = QPointF()
        self._end_pan()
        self.update()

    def image_rect(self):
        """One source-coordinate mapping for the image and every observation."""
        w, h = self.source_size
        angle=0. if self.external_mode else self.preview_rotation%360
        if angle:
            a=math.radians(angle);w,h=abs(w*math.cos(a))+abs(h*math.sin(a)),abs(w*math.sin(a))+abs(h*math.cos(a))
        scale = min(self.width() / w, self.height() / h) * self.zoom
        return QRectF((self.width() - w * scale) / 2 + self.pan.x(),
                      (self.height() - h * scale) / 2 + self.pan.y(),
                      w * scale, h * scale)

    def set_preview_rotation(self,angle):
        angle=float(angle)
        if angle!=self.preview_rotation:self.preview_rotation=angle;self.update()

    def wheelEvent(self, event):
        if self.frame is None:
            event.ignore()
            return
        delta = event.angleDelta().y() / 120
        if not delta:
            delta = event.pixelDelta().y() / 120
        if not delta:
            event.ignore()
            return
        old_zoom = self.zoom
        new_zoom = max(0.25, min(16.0, old_zoom * math.pow(1.2, max(-8, min(8, delta)))))
        # Keep the source point beneath the mouse fixed as the image grows.
        anchor = event.position() - QPointF(self.width() / 2, self.height() / 2)
        self.pan = anchor - (anchor - self.pan) * (new_zoom / old_zoom)
        self.zoom = new_zoom
        self.update()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton and self.frame is not None:
            self._pan_position = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan_position is not None:
            if event.buttons() & Qt.MiddleButton:
                self.pan += event.position() - self._pan_position
                self._pan_position = event.position()
                self.update()
                event.accept()
                return
            self._end_pan()
        super().mouseMoveEvent(event)

    def _end_pan(self):
        self._pan_position = None
        self.unsetCursor()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton and self._pan_position is not None:
            self._end_pan()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton and self.frame is not None:
            self.reset_view()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def hideEvent(self, event):
        self._end_pan()
        super().hideEvent(event)

    def open(self, path: Path, source_size=None, timeline_fps=None, timeline_start=0, *, reuse=False):
        try:
            stamp=Path(path).stat()
            key=(str(Path(path).resolve()),stamp.st_size,stamp.st_mtime_ns,
                 tuple(source_size) if source_size is not None else None,timeline_fps,timeline_start)
        except OSError:key=None
        if reuse and key is not None and key==self._source_key and self.decoder is not None and self.native_fps and self.frame is not None:
            return False
        self._source_key=key
        self.reset_view()
        self.logical_size = source_size
        self.timeline_fps = timeline_fps
        self.timeline_start = timeline_start
        self.native_fps = None
        self.frame = None
        self.boxes = self.observed = self.keypoints = self.confidence = None
        if self.decoder is None:
            self.decoder = PreviewDecoder(parent=self)
            self.decoder.available.connect(self._available)
            self.decoder.opened.connect(self._opened)
            self.decoder.failed.connect(self._failed)
            self.decoder.start()
        self.generation = self.decoder.change_source(path)
        self.seek(0)
        return True

    def _opened(self, generation, metadata):
        if self.decoder is not None and self.sender() is self.decoder and generation == self.generation:
            self.source_size = self.logical_size or (metadata["width"], metadata["height"])
            self.native_fps = metadata['fps']
            self.metadata_ready.emit(metadata)

    def _failed(self, generation, message):
        if self.decoder is not None and self.sender() is self.decoder and generation == self.generation:
            self._source_key = None
            self.decode_error.emit(message)

    def shutdown(self):
        self._end_pan()
        self._source_key = None
        if self.decoder:
            decoder = self.decoder
            self.decoder = None
            self.generation += 1
            decoder.stop()
            decoder.wait()
            decoder.deleteLater()
        self.frame = None

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)

    def _available(self, generation, index):
        if self.decoder is not None and self.sender() is self.decoder and generation == self.generation and index == self.requested_index:
            image = self.decoder.image(index)
            if image is not None:
                self._present(index, image)

    def set_external_frame(self,image):
        if image is None:
            if not self.external_mode:return
            self.external_mode=False
            self.frame,self.source_size=self._external_saved
            self._external_saved=None
            self.seek(self.timeline_requested_index);self.update();return
        if not self.external_mode:self._external_saved=(self.frame,self.source_size)
        self.external_mode=True
        self.frame=None if image.isNull() else image
        if self.frame is not None:self.source_size=(image.width(),image.height())
        self.update()

    def _present(self, index, image):
        if self.external_mode:return
        self.index, self.frame = (self.timeline_requested_index if self.timeline_fps else index), image
        self.update()
        self.frame_presented.emit(self.index)

    def set_observations(self, arrays):
        self.boxes, self.observed = arrays.get("boxes"), arrays.get("observed")
        self.keypoints = arrays.get("keypoints")
        confidence = arrays.get("confidence")
        self.confidence = confidence
        self.update()

    def seek(self, index: int):
        if self.decoder is None:
            return
        self.timeline_requested_index = index
        if self.timeline_fps and self.native_fps:
            index = max(0,round((index + self.timeline_start) * self.native_fps / self.timeline_fps))
            if self.decoder.length:index=min(index,self.decoder.length-1)
        self.requested_index = index
        if self.external_mode:return
        image = self.decoder.request(index)
        if image is not None:
            self._present(index, image)

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#202020"))
        if self.frame is None:
            painter.setPen(QColor("#afbdd2"))
            painter.drawText(self.rect(), Qt.AlignCenter, "Open a video to begin\nMP4 · MOV · AVI")
            return
        rect = self.image_rect()
        angle=0. if self.external_mode else self.preview_rotation%360
        if angle:
            w,h=self.source_size;a=math.radians(angle);ratio=rect.width()/(abs(w*math.cos(a))+abs(h*math.sin(a)))
            painter.translate(rect.center());painter.rotate(angle);rect=QRectF(-w*ratio/2,-h*ratio/2,w*ratio,h*ratio)
        else:ratio = rect.width() / self.source_size[0]
        offset_x, offset_y = rect.x(), rect.y()
        painter.drawImage(rect, self.frame)
        if self.show_observations and self.boxes is not None and self.index < len(self.boxes):
            box = self.boxes[self.index]
            seen = bool(self.observed[self.index])
            pen = QPen(QColor((self.actor_color or "#40e7bf") if seen else "#ffbd62"), 2)
            if not seen:
                pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            if np.isfinite(box).all():
                painter.drawRect(QRectF(offset_x + box[0] * ratio, offset_y + box[1] * ratio,
                                        (box[2] - box[0]) * ratio, (box[3] - box[1]) * ratio))
        if self.show_observations and self.keypoints is not None and self.index < len(self.keypoints):
            points = self.keypoints[self.index]
            confidence = self.confidence[self.index].reshape(-1)
            painter.setPen(QPen(QColor("#7cecff"), 3))
            for a, b in COCO_BONES:
                if min(confidence[a], confidence[b]) >= 0.25:
                    painter.drawLine(QPointF(offset_x + points[a, 0] * ratio, offset_y + points[a, 1] * ratio),
                                     QPointF(offset_x + points[b, 0] * ratio, offset_y + points[b, 1] * ratio))
            for point, score in zip(points, confidence):
                if score >= 0.25:
                    painter.setBrush(QColor("#fff6cf"))
                    painter.drawEllipse(QPointF(offset_x + point[0] * ratio, offset_y + point[1] * ratio), 3, 3)
        if self.show_observations and self.projected_overlay is not None:
            self.projected_overlay(painter,rect)
        painter.end()
        self.frame_painted.emit(self.index)


