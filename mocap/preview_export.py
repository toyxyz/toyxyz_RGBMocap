"""Bounded preview-video encoder. GUI widgets are never used by this worker."""
from __future__ import annotations

from collections import deque
import math
import os
from pathlib import Path
import subprocess
from threading import Condition, Event, Lock, Thread
import uuid

from PySide6.QtCore import QRectF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QImage, QPainter


def file_stamp(path):
    try:
        stat = Path(path).stat()
        return stat.st_size, stat.st_mtime_ns
    except FileNotFoundError:
        return None


def output_layout(video, viewport):
    """Keep both complete images; even MP4 dimensions, at most a 4K frame."""
    if min(video.width(), video.height(), viewport.width(), viewport.height()) <= 0:
        raise ValueError('The video and 3D viewport must both be ready.')
    height = viewport.height()
    left = height * video.width() / video.height()
    right = viewport.width()
    scale = min(1., 3840 / (left + right), 2160 / height)
    even = lambda x: max(2, int(x * scale) // 2 * 2)
    return even(left), even(right), even(height)


def compose_frame(video, viewport, layout):
    left, right, height = layout
    frame = QImage(left + right, height, QImage.Format_RGB888)
    frame.fill(QColor('#202020'))
    painter = QPainter(frame)
    try:
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        for image, x, width in ((video, 0, left), (viewport, left, right)):
            scale = min(width / image.width(), height / image.height())
            w, h = image.width() * scale, image.height() * scale
            painter.drawImage(QRectF(x + (width - w) / 2, (height - h) / 2, w, h), image)
    finally:
        painter.end()
    # RGB scanlines may contain padding. Never send padding to rawvideo.
    width_bytes = frame.width() * 3
    bits = frame.constBits()
    if frame.bytesPerLine() == width_bytes:
        return bytes(bits[:width_bytes * frame.height()])
    stride = frame.bytesPerLine()
    return b''.join(bytes(bits[y * stride:y * stride + width_bytes]) for y in range(frame.height()))


class PreviewEncoder(QThread):
    frame_written = Signal(int)

    def __init__(self, destination, layout, fps, frames, audio=None, audio_start=0., parent=None):
        super().__init__(parent)
        self.destination = Path(destination)
        self.temporary = self.destination.with_name('.' + self.destination.stem + '-' + uuid.uuid4().hex + '.partial.mp4')
        self.expected_destination = file_stamp(self.destination)
        self.layout, self.fps, self.frames = layout, float(fps), int(frames)
        self.audio = Path(audio) if audio else None
        self.audio_start = float(audio_start)
        self.condition, self.process_lock = Condition(), Lock()
        self.stop_requested = Event()
        self.pending = None
        self.ending = False
        self.process = None
        self.error = None
        self.succeeded = False
        self.written = 0

    def submit(self, index, video, viewport):
        with self.condition:
            if self.pending is not None or self.ending or self.stop_requested.is_set():
                raise RuntimeError('Preview export is no longer accepting frames.')
            self.pending = index, QImage(video), QImage(viewport)
            self.condition.notify()

    def finish_frames(self):
        with self.condition:
            self.ending = True
            self.condition.notify()

    def cancel(self):
        self.stop_requested.set()
        with self.condition:
            self.pending = None
            self.condition.notify()
        with self.process_lock:
            if self.process is not None and self.process.poll() is None:
                try:
                    self.process.terminate()
                except OSError:
                    pass

    def run(self):
        errors = deque(maxlen=8)  # At most 32 KiB of FFmpeg output.
        reader = None
        drained = Event()
        process = None
        try:
            from .video import executable
            if not math.isfinite(self.fps) or self.fps <= 0 or self.frames <= 0:
                raise ValueError('Invalid preview video timing.')
            left, right, height = self.layout
            command = [executable('ffmpeg'), '-hide_banner', '-loglevel', 'error', '-y',
                       '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{left + right}x{height}',
                       '-framerate', format(self.fps, '.12g'), '-i', 'pipe:0']
            if self.audio is not None:
                command += ['-ss', format(self.audio_start, '.12g'), '-i', str(self.audio)]
            command += ['-map', '0:v:0']
            if self.audio is not None:
                command += ['-map', '1:a:0?', '-c:a', 'aac', '-b:a', '192k']
            command += ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-threads', '4',
                        '-pix_fmt', 'yuv420p', '-t', format(self.frames / self.fps, '.12g'),
                        '-movflags', '+faststart', str(self.temporary)]
            with self.process_lock:
                if self.stop_requested.is_set():
                    return
                process = self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            def drain():
                try:
                    while True:
                        chunk = process.stderr.read(4096)
                        if not chunk:
                            return
                        errors.append(chunk)
                finally:
                    drained.set()
            reader = Thread(target=drain, daemon=True)
            reader.start()
            while not self.stop_requested.is_set():
                with self.condition:
                    while self.pending is None and not self.ending and not self.stop_requested.is_set():
                        self.condition.wait()
                    if self.stop_requested.is_set():
                        break
                    if self.pending is None:
                        break
                    index, video, viewport = self.pending
                    self.pending = None
                payload = compose_frame(video, viewport, self.layout)
                process.stdin.write(payload)
                process.stdin.flush()
                self.written += 1
                self.frame_written.emit(index)
                del payload, video, viewport
            if self.stop_requested.is_set():
                return
            process.stdin.close()
            if process.wait(timeout=120):
                raise RuntimeError('FFmpeg could not finish the preview video.')
            if self.written != self.frames:
                raise RuntimeError('Preview export ended before every frame was written.')
            if file_stamp(self.destination) != self.expected_destination:
                raise RuntimeError('The destination changed during export. Choose another filename.')
            if not self.temporary.is_file() or self.temporary.stat().st_size == 0:
                raise RuntimeError('FFmpeg did not produce a preview video.')
            # Only this completed file replaces the chosen destination.
            with self.process_lock:
                if self.stop_requested.is_set():
                    return
                os.replace(self.temporary, self.destination)
                self.succeeded = True
        except Exception as exc:
            if not self.stop_requested.is_set():
                self.error = str(exc)
        finally:
            if process is not None:
                if process.poll() is None:
                    try:
                        process.kill()
                    except OSError:
                        pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                if reader is not None:
                    # Event waiting avoids Python 3.10 registering each Qt
                    # worker as a permanent DummyThread through Thread.join.
                    drained.wait(timeout=5)
                for stream in (process.stdin, process.stderr):
                    if stream is not None:
                        try:
                            stream.close()
                        except OSError:
                            pass
            if self.error and errors:
                self.error += '\n' + b''.join(errors).decode('utf-8', errors='replace')[-4000:]
            with self.process_lock:
                self.process = None
            with self.condition:
                self.pending = None
            if not self.succeeded:
                try:
                    self.temporary.unlink(missing_ok=True)
                except OSError as exc:
                    self.error = (self.error or 'Preview export cancelled.') + '\nCannot remove temporary file: ' + str(exc)
