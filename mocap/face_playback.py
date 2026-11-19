"""One cancellable background face cache shared by overlay and streaming."""
from pathlib import Path
from threading import Condition

from PySide6.QtCore import QObject, QThread, Signal
from .face_data import FILE


def archive_key(root):
    if root is None:
        return None
    root = Path(root)
    if not (root / FILE).is_file():
        return None
    rows = []
    for name in (FILE, 'project.json', 'people.json', 'input.mp4', 'tracking.npz', 'face_neutral.json'):
        path = root / name
        stamp = path.stat() if path.is_file() else None
        rows.append((stamp.st_size, stamp.st_mtime_ns) if stamp else None)
    return root, tuple(rows)


class FacePlaybackLoader(QThread):
    ready = Signal(int, object, object, str)

    def __init__(self, parent):
        super().__init__(parent)
        self.condition = Condition()
        self.generation = 0
        self.pending = None
        self.stopping = False

    def request(self, key):
        with self.condition:
            self.generation += 1
            self.pending = (self.generation, key) if key else None
            self.condition.notify()
            return self.generation

    def stop(self):
        with self.condition:
            self.stopping = True
            self.generation += 1
            self.pending = None
            self.condition.notify()

    def run(self):
        from .face_smoothing import prepare_archive

        while True:
            with self.condition:
                while self.pending is None and not self.stopping:
                    self.condition.wait()
                if self.stopping:
                    return
                generation, key = self.pending
                self.pending = None
            cancelled = lambda: self.stopping or generation != self.generation
            try:
                archive = prepare_archive(key[0], cancelled)
                if cancelled():
                    continue
                if archive_key(key[0]) != key:
                    self.ready.emit(generation, key, None, '')
                    continue
                self.ready.emit(generation, key, archive, '')
            except InterruptedError:
                pass
            except Exception as error:
                if not cancelled():
                    self.ready.emit(generation, key, None, str(error))


class FacePlayback(QObject):
    changed = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self.key = None
        self.archive = None
        self.error = ''
        self.generation = 0
        self.closed = False
        self.loader = FacePlaybackLoader(self)
        self.loader.ready.connect(self.completed)
        self.loader.start()

    def request(self, root):
        if self.closed:
            return None
        key = archive_key(root)
        if key != self.key:
            self.key, self.archive, self.error = key, None, ''
            self.generation = self.loader.request(key)
        return self.archive

    def completed(self, generation, key, archive, error):
        if self.closed or generation != self.generation or key != self.key:
            return
        try:
            current = archive_key(key[0])
        except OSError:
            current = None
        if current != key:
            self.key = None
            self.archive = None
        else:
            self.archive, self.error = archive, error
        self.changed.emit()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.loader.stop()
        self.loader.wait()
        self.archive = None
