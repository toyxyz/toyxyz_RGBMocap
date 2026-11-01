"""Unicode filesystem paths with OpenCV's unchanged image codecs.

Use Python for file I/O on Windows, where OpenCV's narrow filename API can
depend on the system code page. No resize, colour conversion or lossy recode
is introduced. The compatibility scope is only for isolated upstream workers.
"""
from contextlib import contextmanager
from pathlib import Path


def read_image(filename, flags=1):
    import cv2
    import numpy as np
    try:
        data = Path(filename).read_bytes()
    except OSError:
        return None
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags)


def write_image(filename, image, params=None):
    import cv2
    path = Path(filename)
    ok, data = cv2.imencode(path.suffix, image, [] if params is None else params)
    if not ok:
        return False
    try:
        with path.open('wb') as stream:
            stream.write(memoryview(data))
    except OSError:
        return False
    return True


@contextmanager
def unicode_image_paths():
    """Bridge upstream cv2 reads/writes in one dedicated Windows worker.

    Restore functions even on cancellation/errors. Never use this scope in
    the editor or a process running concurrent inference jobs.
    """
    import os
    if os.name != 'nt':
        yield
        return
    import cv2
    original_read, original_write = cv2.imread, cv2.imwrite
    cv2.imread, cv2.imwrite = read_image, write_image
    try:
        yield
    finally:
        cv2.imread, cv2.imwrite = original_read, original_write
