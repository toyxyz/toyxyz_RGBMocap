"""One persistent decoder, bounded cache and latest-request-wins seeking."""
from collections import OrderedDict
from threading import Condition
import math

import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage


class PreviewDecoder(QThread):
    available = Signal(int, int)
    opened = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, path=None, parent=None, budget_bytes=64 * 1024**2):
        super().__init__(parent)
        self.condition = Condition()
        self.cache = OrderedDict()
        self.target = 0
        self.stopping = False
        self.capacity = 24
        self.budget = budget_bytes
        self.seek_count = self.decoded_count = 0
        self.generation = 0
        self.path = str(path) if path else None
        self.length = 0

    def change_source(self, path):
        with self.condition:
            self.generation += 1
            self.path = str(path)
            self.target = self.length = 0
            self.cache.clear()
            self.seek_count = self.decoded_count = 0
            self.condition.notify()
            return self.generation

    def request(self, index):
        with self.condition:
            self.target = max(0, int(index))
            if self.length:
                self.target = min(self.target, self.length - 1)
            end = self.target + max(1, self.capacity - 5)
            for key in list(self.cache):
                if key >= 3 and (key < self.target - 2 or key >= end):
                    del self.cache[key]
            image = self.cache.get(self.target)
            self.condition.notify()
            return image

    def image(self, index):
        with self.condition:
            return self.cache.get(index)

    def stop(self):
        with self.condition:
            self.stopping = True
            self.condition.notify()

    def _obsolete(self, generation, wanted=None):
        with self.condition:
            return self.stopping or generation != self.generation or (wanted is not None and not
                (wanted < 3 or self.target - 2 <= wanted < self.target + max(1, self.capacity - 5)))

    def run(self):
        capture = None
        generation, cursor = -1, 0
        try:
            while True:
                with self.condition:
                    if self.stopping:
                        return
                    if self.path is None:
                        self.condition.wait()
                        continue
                    opening = generation != self.generation
                    if opening:
                        path, generation = self.path, self.generation
                    else:
                        end = min(self.length, self.target + max(1, self.capacity - 5))
                        wanted = next((i for i in range(self.target, end) if i not in self.cache), None)
                        if wanted is None:
                            self.condition.wait()
                            continue
                try:
                    if opening:
                        if capture is not None:
                            capture.release()
                        capture = cv2.VideoCapture(path, cv2.CAP_FFMPEG,
                            [cv2.CAP_PROP_N_THREADS, 2, cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 1500,
                             cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1500])
                        if not capture.isOpened():
                            raise ValueError("Cannot open the video.")
                        width, height = int(capture.get(3)), int(capture.get(4))
                        length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = float(capture.get(cv2.CAP_PROP_FPS))
                        if width <= 0 or height <= 0 or length <= 0 or not math.isfinite(fps) or fps <= 0:
                            raise ValueError("Cannot determine the video length and frame rate.")
                        if self._obsolete(generation):
                            continue
                        with self.condition:
                            self.length = length
                            self.target = min(self.target, length - 1)
                        cursor = 0
                        self.opened.emit(generation, {"width": width, "height": height, "frames": length,
                                                      "fps": fps, "duration": length / fps, "path": path})
                        continue
                    if self._obsolete(generation, wanted):
                        continue
                    if cursor < wanted <= cursor + 120:
                        for _ in range(wanted - cursor):
                            if self._obsolete(generation, wanted):
                                break
                            if not capture.grab():
                                raise ValueError("Cannot read the frame.")
                            cursor += 1
                        if self._obsolete(generation, wanted):
                            continue
                    elif cursor != wanted:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, wanted)
                        self.seek_count += 1
                        cursor = wanted
                    ok, frame = capture.read()
                    cursor += 1
                    if not ok:
                        raise ValueError(f"Cannot read video frame {wanted}.")
                    if self._obsolete(generation, wanted):
                        continue
                    height, width = frame.shape[:2]
                    scale = min(1.0, 1280 / width, 720 / height,
                                math.sqrt(self.budget / (4 * 3 * width * height)))
                    if scale < 1:
                        frame = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))),
                                           interpolation=cv2.INTER_AREA)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()
                    with self.condition:
                        if generation != self.generation:
                            continue
                        self.capacity = max(4, min(90, self.budget // image.sizeInBytes()))
                        self.decoded_count += 1
                        self.cache[wanted] = image
                        while len(self.cache) > self.capacity:
                            removable = next(key for key in self.cache if key >= 3)
                            del self.cache[removable]
                        notify = wanted == self.target
                    if notify:
                        self.available.emit(generation, wanted)
                except Exception as error:
                    if capture is not None:
                        capture.release()
                        capture = None
                    if not self._obsolete(generation):
                        from .storage import path_error_message
                        self.failed.emit(generation, path_error_message(error))
                        with self.condition:
                            if generation == self.generation:
                                self.path = None
                                self.cache.clear()
        finally:
            if capture is not None:
                capture.release()
            with self.condition:
                self.cache.clear()
