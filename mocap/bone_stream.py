"""Local, latest-frame transport for offline Blender bone streaming."""
from hashlib import sha256
from bisect import bisect_right
import json
from queue import Empty, Full, Queue
from select import select
import socket
import struct
from threading import Event, Thread

import numpy as np


PORT = 49168
LIMIT = 1024 * 1024
MAX_ACTORS = 128
MAX_BAKE_FRAMES = 36000
Y_TO_Z = np.array(((1, 0, 0), (0, 0, -1), (0, 1, 0)), dtype=np.float32)


class StreamRig:
    def __init__(self, names, parents, heads, bind, display, lengths):
        rest = Y_TO_Z @ np.concatenate((bind, heads[..., None]), axis=-1)
        self.display = np.asarray(display, dtype=np.float32)
        self.bones = [
            [name, int(parent), matrix.ravel().tolist(), float(length)]
            for name, parent, matrix, length in zip(names, parents, rest, lengths)
        ]
        raw = json.dumps(self.bones, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        self.key = sha256(raw.encode('utf-8')).hexdigest()

    @classmethod
    def mixamo(cls, names, asset):
        parents, heads, bind, display, tips = (asset[key] for key in (
            'parents', 'fbx_rest', 'fbx_rest_frames', 'fbx_display_frames', 'fbx_tip_local'))
        lengths = np.linalg.norm(tips, axis=1)
        for index in range(len(names)):
            children = heads[parents == index]
            if len(children):
                lengths[index] = np.linalg.norm(children.mean(axis=0) - heads[index])
        return cls(names, parents, heads, bind, display, np.maximum(lengths, .01))

    @classmethod
    def soma(cls, asset):
        from .bone_overlay import reference_frame

        names, parents = asset['names'].tolist(), asset['parents']
        heads, native = asset['rest'][:, :3, 3], asset['rest'][:, :3, :3]
        directions = asset['ends'].copy()
        for index, name in enumerate(names):
            children = np.flatnonzero(parents == index)
            if len(children):
                preferred = {'Hips': 'Spine1', 'Head': 'HeadEnd'}.get(name)
                child = names.index(preferred) if preferred in names else -1
                tail = heads[child] if child in children else heads[children].mean(axis=0)
                directions[index] = tail - heads[index]
        directions[np.linalg.norm(directions, axis=1) < .001] = (0., .02, 0.)
        bind = np.stack([reference_frame(direction) for direction in directions])
        return cls(names, parents, heads, bind, native.transpose(0, 2, 1) @ bind,
                   np.linalg.norm(directions, axis=1))

    def pose(self, world, transform):
        pose = world[:, :3, :4].copy()
        pose[:, :, :3] = pose[:, :, :3] @ self.display
        if transform is not None:
            pose = transform[:3, :3] @ pose
            pose[:, :, 3] += transform[:3, 3]
        return (Y_TO_Z @ pose).astype('<f4', copy=False).tobytes()


def packet(tag, payload):
    size = len(payload) + 1
    if size > LIMIT:
        raise ValueError('Bone stream packet exceeds the size limit.')
    return struct.pack('<I', size) + tag + payload


def bake_packets(plan, stopping):
    """Read recorded frames in bounded chunks on the sender thread."""
    if isinstance(plan, str):
        yield packet(b'F', plan.encode('utf-8'))
        return
    actor, key, fps, origin, count, spans, starts, preferred = plan
    current_source = frames = None
    block = []
    yield packet(b'H', json.dumps(dict(actor=actor, rig=key, fps=fps, frames=count)).encode('utf-8'))
    for index in range(count):
        if stopping.is_set():
            return
        seconds = origin + index / fps
        slot = bisect_right(starts, seconds + 1e-7) - 1
        span = spans[slot] if slot >= 0 and seconds < spans[slot][1] - 1e-7 else None
        if preferred and preferred[0] - 1e-7 <= seconds < preferred[1] - 1e-7:
            span = preferred
        if span is None:
            continue
        _, _, start, rate, source, rig, transform = span
        if source is not current_source:
            frames = source if isinstance(source, np.ndarray) else np.load(
                source, mmap_mode='r', allow_pickle=False)
            current_source = source
        frame = int(np.floor((seconds - start) * rate + 1e-6))
        if 0 <= frame < len(frames):
            block.extend((struct.pack('<I', index), rig.pose(frames[frame], transform)))
        if len(block) == 16:
            yield packet(b'D', b''.join(block))
            block.clear()
    if block:
        yield packet(b'D', b''.join(block))
    yield packet(b'Z', b'\x01')


class BoneStream:
    def __init__(self, port=PORT):
        self.listener = socket.socket()
        try:
            self.listener.bind(('127.0.0.1', port))
            self.listener.listen(1)
            self.listener.settimeout(.1)
        except OSError:
            self.listener.close()
            raise
        self.latest = (), None
        self.changed = Event()
        self.stopping = Event()
        self.connected = False
        self.errors = Queue(maxsize=1)
        self.commands = Queue(maxsize=8)
        self.bake_reply = None
        self.connection = 0
        self.thread = Thread(target=self._serve, name='Blender bone stream', daemon=True)
        self.thread.start()

    def publish(self, actors, selected):
        self.latest = actors, selected
        self.changed.set()

    def error(self):
        try:
            return self.errors.get_nowait()
        except Empty:
            return None

    def close(self):
        self.stopping.set()
        self.changed.set()
        self.listener.close()
        self.thread.join(timeout=.4)

    def _serve(self):
        while not self.stopping.is_set():
            try:
                client, _ = self.listener.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            with client:
                client.settimeout(.2)
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.connection += 1
                self.bake_reply = None
                self.connected = True
                try:
                    self._send(client)
                except OSError:
                    pass
                except (ValueError, struct.error) as error:
                    try:
                        self.errors.put_nowait(str(error))
                    except Full:
                        pass
                finally:
                    self.connected = False

    def _send(self, client):
        known = {}
        sequence = 0
        last = None
        incoming = bytearray()
        baking = None
        include_selection = False
        while not self.stopping.is_set():
            self.changed.clear()
            snapshot = self.latest
            if snapshot is not last:
                actors, selected = snapshot
                # Drop declarations for inactive actors. A returning actor can
                # declare its rig again and Blender resolves its persistent ID.
                known = {actor: known[actor] for actor, *_ in actors if actor in known}
                parts = [struct.pack('<QH', sequence, len(actors))]
                for actor, name, rig, frame, world, transform in actors:
                    entry = (name, rig.key)
                    if actor not in known or known[actor][0] != entry:
                        occupied = {slot for _, slot in known.values()}
                        slot = known[actor][1] if actor in known else next(
                            index for index in range(1, MAX_ACTORS + 1) if index not in occupied)
                        schema = dict(version=2, actor=actor, name=name, slot=slot, bones=rig.bones,
                                      bake=1, selection=1)
                        client.sendall(packet(b'R', json.dumps(schema, ensure_ascii=False,
                                                               separators=(',', ':')).encode('utf-8')))
                        known[actor] = entry, slot
                    parts.extend((struct.pack('<II', known[actor][1], frame), rig.pose(world, transform)))
                if include_selection and selected is not None:
                    # Selection is independent of the poses present at this time.
                    parts.append(struct.pack('<H', len(selected)))
                    parts.extend(bytes.fromhex(actor) for actor in selected)
                client.sendall(packet(b'S', b''.join(parts)))
                last = snapshot
                sequence += 1
            reply = self.bake_reply
            if reply is not None:
                self.bake_reply = None
                connection, plan = reply
                if connection == self.connection:
                    baking = bake_packets(plan, self.stopping)
            if baking is not None:
                try:
                    chunk = next(baking, None)
                except (ValueError, OSError) as error:
                    chunk = packet(b'F', str(error).encode('utf-8'))
                    baking = None
                if chunk is None:
                    baking = None
                else:
                    client.settimeout(1.5)
                    client.sendall(chunk)
                    client.settimeout(.2)
            if select([client], [], [], 0)[0]:
                data = client.recv(4096)
                if not data:
                    return
                incoming.extend(data)
                while incoming:
                    kind = incoming[0]
                    if kind == ord('V'):
                        if not include_selection:
                            include_selection = True
                            last = None
                        del incoming[0]
                        continue
                    if kind not in (ord('B'), ord('A')):
                        return
                    size = 65 if kind == ord('B') else 33
                    if len(incoming) < size:
                        break
                    if kind == ord('B') and not include_selection:
                        include_selection = True
                        last = None
                    try:
                        self.commands.put_nowait((self.connection, kind, incoming[1:33].hex(),
                                                   incoming[33:size].hex()))
                    except Full:
                        return
                    del incoming[:size]
            if baking is None:
                self.changed.wait(.06)
