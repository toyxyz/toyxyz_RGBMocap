"""Same-machine, on-demand prepared takes for the Unreal editor baker."""
from dataclasses import dataclass
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, HTTPServer
import secrets
import struct
from threading import Thread
from urllib.parse import parse_qs, urlsplit

import numpy as np


HEADER = struct.Struct('<4sIIII')
MAX_SAMPLES = 54_000
MAX_BYTES = 128 * 1024 * 1024
FACE_CHANNELS = 52


@dataclass(frozen=True)
class BakeFace:
    archive: object
    person: int
    start: float
    fps: float

    @property
    def revision(self):
        return id(self.archive), self.person, self.start, self.fps

    def samples(self, count):
        # Resolve the whole take through the same source clock and held samples
        # used by the live preview, including trims at a different FPS.
        state = {'start': self.start, 'fps': self.fps}
        held = np.fromiter(
            (self.archive.resolve(None, frame, person=self.person, state=state)[1]
             for frame in range(count)),
            dtype=np.intp, count=count)
        available = held >= 0
        if not available.any():
            return None
        values = np.zeros((count, FACE_CHANNELS), dtype='<f4')
        values[available] = self.archive.values[self.archive.index[self.person], held[available]]
        if np.isfinite(values).all():
            return np.clip(values, 0., 1., out=values)
        return None


@dataclass(frozen=True)
class BakeTake:
    frames: object
    count: int
    fps: float
    rig: object
    transform: object
    face: BakeFace | None = None

    def encode(self, convert, stopping, include_face=False):
        ratio = Fraction(self.fps).limit_denominator(100_000)
        count, bones = self.count, len(self.rig.bones)
        size = HEADER.size + count * bones * 7 * 4
        if not 2 <= count <= MAX_SAMPLES or not 1 <= bones <= 256 or size > MAX_BYTES:
            raise ValueError('The take exceeds the Unreal bake size limit.')
        if not 1 <= ratio <= 240:
            raise ValueError('Unsupported bake frame rate.')
        frames = self.frames
        if not isinstance(frames, np.ndarray):
            frames = np.load(frames, mmap_mode='r', allow_pickle=False)
        if len(frames) != count:
            raise ValueError('The prepared take changed. Reconnect it and try again.')
        parents = [bone[1] for bone in self.rig.bones]
        face = self.face.samples(count) if include_face and self.face else None
        if face is not None:
            size += face.nbytes
            if size > MAX_BYTES:
                raise ValueError('The take exceeds the Unreal bake size limit.')
        magic = b'RMB2' if face is not None else b'RMB1'
        yield size, HEADER.pack(magic, ratio.numerator, ratio.denominator, count, bones)
        block = bytearray()
        for index in range(count):
            if stopping.is_set():
                return
            world = np.frombuffer(self.rig.pose(frames[index], self.transform), dtype='<f4').reshape(bones, 3, 4)
            block.extend(np.asarray(convert(world, parents), dtype='<f4').tobytes())
            if face is not None:
                block.extend(face[index].tobytes())
            if (index + 1) % 32 == 0:
                yield bytes(block)
                block.clear()
        if block:
            yield bytes(block)


class BakeServer(HTTPServer):
    """A single in-flight transfer, bound only to IPv4 loopback."""

    request_queue_size = 1

    def __init__(self, convert, stopping):
        super().__init__(('127.0.0.1', 0), BakeRequest)
        self.token = secrets.token_hex(32)
        self.convert = convert
        self.stopping = stopping
        self.takes = {}
        self.thread = Thread(target=self.serve_forever, kwargs={'poll_interval': .1},
                             name='Unreal take transfer', daemon=True)
        self.thread.start()

    def stop(self):
        self.shutdown()
        self.server_close()

    def get_request(self):
        request, address = super().get_request()
        request.settimeout(3)
        return request, address


class BakeRequest(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        expected = 'Bearer ' + self.server.token
        auth = self.headers.get('Authorization', '')
        if not auth.isascii() or not secrets.compare_digest(auth, expected):
            self.send_error(403, 'This bake request is not from the current Unreal stream.')
            return
        try:
            url = urlsplit(self.path)
            args = parse_qs(url.query, strict_parsing=True, max_num_fields=3)
            key = tuple(args[field][0] for field in ('subject', 'rig', 'take'))
            take = self.server.takes.get(key) if url.path == '/take' else None
        except (ValueError, KeyError):
            take = None
        if take is None:
            self.send_error(409, 'The connected take changed. Select Bake again.')
            return
        blocks = take.encode(self.server.convert, self.server.stopping,
                             include_face=self.headers.get('X-RGBMoCap-Face') == '1')
        try:
            size, header = next(blocks)
        except (OSError, ValueError, OverflowError):
            self.send_error(422, 'The prepared take is unavailable or too large to bake.')
            return
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', str(size))
            self.end_headers()
            self.wfile.write(header)
            for block in blocks:
                self.wfile.write(block)
        except (OSError, ValueError, OverflowError):
            # A short body is rejected by the editor. Never publish partial keys.
            self.close_connection = True
