"""Optional Unreal-hosted Live Link provider with body and facial data; no mesh transfer."""
from hashlib import sha256
import json
import os
import re
from pathlib import Path
from queue import Empty, Full, Queue
import secrets
import socket
import struct
import subprocess
from threading import Event, Thread
import time

import numpy as np

from .bone_stream import MAX_ACTORS, Y_TO_Z


LIMIT = 4 * 1024 * 1024
SOURCE = 'developer-build/unreal-live-link'
# Same convention as the verified MetaHuman pilot: source (-Z, X, Y),
# followed by -90 degrees world yaw on BOTH reference and animation.
SOURCE_TO_UE = np.array(((0., 0., -1.), (1., 0., 0.), (0., 1., 0.)))
YAW = np.array(((0., 1., 0.), (-1., 0., 0.), (0., 0., 1.)))
BLENDER_TO_UE = YAW @ SOURCE_TO_UE @ Y_TO_Z.T


def source_identity(root):
    return source_tree_identity(Path(root) / SOURCE)


def source_tree_identity(folder):
    folder = Path(folder)
    digest = sha256()
    for path in sorted(folder.rglob('*')):
        if path.relative_to(folder).parts[0] in ('Binaries', 'Intermediate', 'Saved', 'DerivedDataCache'):
            continue
        if path.is_file() and path.suffix in ('.h', '.cpp', '.cs', '.uproject', '.ini'):
            digest.update(path.relative_to(folder).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def configuration(root):
    root = Path(root)
    try:
        data = json.loads((root / 'runtime_unreal_stream/bridge.json').read_text(encoding='utf-8'))
        runtime = (root / 'runtime_unreal_stream').resolve()
        relative = Path(data.get('project', 'Bridge/RGBMoCapLiveLink.uproject'))
        project = (runtime / relative).resolve()
        if relative.is_absolute() or not project.is_relative_to(runtime):
            raise ValueError('Invalid optional Unreal project path.')
        module = project.parent / 'Binaries/Win64/UnrealEditor-RGBMoCapLiveLink.dll'
        engine = Path(data['engine'])
        if (data['version'] != 1 or data['source'] != source_identity(root)
                or not project.is_file() or not module.is_file()
                or sha256(module.read_bytes()).hexdigest() != data['module']
                or sha256((engine / 'Engine/Build/Build.version').read_bytes()).hexdigest() != data['engine_build']
                or not (engine / 'Engine/Binaries/Win64/UnrealEditor-Cmd.exe').is_file()):
            raise ValueError('Unreal Live Link support changed. Run Setup_UnrealStream.cmd again.')
        return engine, project
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError('Run Setup_UnrealStream.cmd first (Unreal Engine 5.6 or later and Visual Studio C++ tools required).') from error


def local_transforms(world, parents):
    """Blender Z-up world matrices -> parent-local Unreal cm / xyzw, without SciPy."""
    world = np.asarray(world, dtype=np.float64)
    count = len(parents)
    if world.shape != (count, 3, 4) or not np.isfinite(world).all():
        raise ValueError('Invalid streaming bone matrices.')
    rotation = BLENDER_TO_UE @ world[:, :, :3] @ SOURCE_TO_UE.T
    position = (BLENDER_TO_UE @ world[:, :, 3, None])[..., 0] * 100.
    if (not np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=.003)
            or np.any(np.linalg.det(rotation) <= 0)):
        raise ValueError('Live Link requires rigid bone transforms.')
    local_r, local_p = rotation.copy(), position.copy()
    for i, parent in enumerate(parents):
        if parent >= 0:
            local_r[i] = rotation[parent].T @ rotation[i]
            local_p[i] = rotation[parent].T @ (position[i] - position[parent])
    # Unit quaternion is the largest-eigenvalue vector of this symmetric matrix.
    r = local_r
    k = np.empty((count, 4, 4))
    k[:, 0, 0] = r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2]
    k[:, 1, 1] = r[:, 1, 1] - r[:, 0, 0] - r[:, 2, 2]
    k[:, 2, 2] = r[:, 2, 2] - r[:, 0, 0] - r[:, 1, 1]
    k[:, 3, 3] = np.trace(r, axis1=1, axis2=2)
    for a, b, value in ((0, 1, r[:, 0, 1]+r[:, 1, 0]),
                        (0, 2, r[:, 0, 2]+r[:, 2, 0]),
                        (1, 2, r[:, 1, 2]+r[:, 2, 1]),
                        (0, 3, r[:, 2, 1]-r[:, 1, 2]),
                        (1, 3, r[:, 0, 2]-r[:, 2, 0]),
                        (2, 3, r[:, 1, 0]-r[:, 0, 1])):
        k[:, a, b] = k[:, b, a] = value
    quat = np.linalg.eigh(k)[1][:, :, -1]
    quat *= np.where(quat[:, 3:] < 0, -1., 1.)
    return np.concatenate((local_p, quat), axis=1).tolist()


def rig_schema(rig):
    bones = rig.bones
    if not 0 < len(bones) <= 256:
        raise ValueError('Live Link supports 1–256 bones per person.')
    names = [b[0] for b in bones]
    parents = [int(b[1]) for b in bones]
    if (len({n.casefold() for n in names}) != len(names) or any(not n or len(n.encode('utf-16-le')) > 240 for n in names)
            or parents[0] != -1 or any(not 0 <= p < i for i, p in enumerate(parents[1:], 1))):
        raise ValueError('Invalid streaming bone hierarchy.')
    lengths = np.asarray([b[3] for b in bones], dtype=float) * 100.
    if not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError('Invalid streaming bone lengths.')
    rest = np.asarray([b[2] for b in bones]).reshape(-1, 3, 4)
    return dict(rig=rig.key, names=names, parents=parents, rest=local_transforms(rest, parents),
                lengths=lengths.tolist())


def subject_name(actor, name):
    """Readable identity, stable across seeks/reconnects with the same person name."""
    if len(actor) != 64 or any(c not in '0123456789abcdef' for c in actor):
        raise ValueError('Invalid streaming person ID.')
    label = re.sub(r'[\W_]+', '_', str(name), flags=re.UNICODE).strip('_')[:24] or 'Person'
    return f'RGBMoCap_{label}_{actor[:8]}'


def snapshot(actors, schemas, faces=None):
    if len(actors) > MAX_ACTORS:
        raise ValueError('Too many streaming people.')
    result, used, subjects = [], set(), set()
    for actor, name, rig, frame, world, transform in actors:
        subject = subject_name(actor, name)
        # Unreal FName is case-insensitive. Never merge people if short IDs collide.
        if subject.casefold() in subjects:
            raise ValueError('Live Link subject names collide. Rename one of the people.')
        subjects.add(subject.casefold())
        used.add(rig.key)
        if rig.key not in schemas:
            schemas[rig.key] = rig_schema(rig)
        schema = schemas[rig.key]
        matrices = np.frombuffer(rig.pose(world, transform), dtype='<f4').reshape(-1, 3, 4)
        # Human-readable names retain a short, stable person identity.
        result.append(dict(schema, subject=subject, label=str(name).encode('utf-16-le')[:240].decode('utf-16-le','ignore'),
                           frame=int(frame), pose=local_transforms(matrices, schema['parents'])))
        face = (faces or {}).get(actor)
        if face is not None:
            values = np.asarray(face['values'], dtype=float)
            if values.shape != (52,) or not np.isfinite(values).all():
                raise ValueError('Invalid streaming face coefficients.')
            sample, source_frame = int(face['sample']), int(face['source_frame'])
            if sample != face['sample'] or source_frame != face['source_frame']:
                raise ValueError('Face sample timing must use integer frame indices.')
            if not 0 <= sample <= source_frame <= 2147483647:
                raise ValueError('Invalid streaming face sample timing.')
            result[-1]['face'] = dict(values=np.clip(values, 0., 1.).tolist(),
                                     sample=sample, source_frame=source_frame,
                                     tracked=bool(face['tracked']))
    for key in list(schemas):
        if key not in used:
            del schemas[key]
    return dict(version=1, actors=result)


def encode(data):
    raw = json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
    if len(raw) > LIMIT:
        raise ValueError('Live Link snapshot exceeds 4 MiB.')
    return struct.pack('<I', len(raw)) + raw


class UnrealStream:
    """Latest-only app IPC. The official provider repeats paused poses in Unreal."""
    target = 'Unreal Live Link'

    def __init__(self, root):
        self.root = Path(root)
        engine, project = configuration(root)
        self.connected = False
        self.ready = False
        self.errors = Queue(maxsize=1)
        self.latest = ((), {})
        self.changed = Event()
        self.stopping = Event()
        self.listener = socket.socket()
        self.listener.bind(('127.0.0.1', 0))
        self.listener.listen(1)
        self.listener.settimeout(.2)
        self.port = self.listener.getsockname()[1]
        token = secrets.token_hex(32)
        folder = self.root / 'cache/unreal_stream'
        folder.mkdir(parents=True, exist_ok=True)
        self.log_path = folder / ('bridge-' + token[:12] + '.log')
        self.log = self.log_path.open('wb')
        command = [str(engine / 'Engine/Binaries/Win64/UnrealEditor-Cmd.exe'), str(project),
                   '-run=RGBMoCapLiveLink',
                   '-unattended', '-nullrhi', '-nosound', '-nosplash', '-NoLiveCoding',
                   '-Messaging', '-NoAssetRegistryCache', '-NoShaderCompile', '-NoAutoLoadPlugins',
                   '-RGBPort='+str(self.port),
                   '-abslog='+str(self.log_path.with_suffix('.engine.log'))]
        try:
            self.process = subprocess.Popen(command, cwd=project.parent, stdin=subprocess.DEVNULL,
                                            stdout=self.log, stderr=subprocess.STDOUT,
                                            env=dict(os.environ, RGB_MOCAP_STREAM_TOKEN=token),
                                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except OSError:
            self.listener.close()
            self.log.close()
            raise
        # Normal interpreter shutdown must wait for owned-process cleanup.
        self.thread = Thread(target=self._serve, args=(token,), name='Unreal Live Link', daemon=False)
        self.thread.start()

    def publish(self, actors, selected=None, faces=None):
        self.latest = (actors, faces or {})
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
        # Reaping the owned bridge happens off the GUI thread.

    def _serve(self, token):
        try:
            deadline = time.monotonic() + 120
            while not self.stopping.is_set():
                if self.process.poll() is not None:
                    raise OSError('Live Link bridge exited. See ' + str(self.log_path))
                if time.monotonic() > deadline:
                    raise TimeoutError('Live Link bridge startup timed out. See ' + str(self.log_path))
                try:
                    client, _ = self.listener.accept()
                except socket.timeout:
                    continue
                with client:
                    client.settimeout(.2)
                    client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    handshake = bytearray()
                    until = time.monotonic() + 3
                    while len(handshake) < 65 and time.monotonic() < until and not self.stopping.is_set():
                        try:
                            data = client.recv(65-len(handshake))
                        except socket.timeout:
                            continue
                        if not data:
                            break
                        handshake.extend(data)
                    if not secrets.compare_digest(bytes(handshake), (token+'\n').encode()):
                        continue
                    self._send(client)
                    break
        except (OSError, ValueError, TypeError, KeyError) as error:
            if not self.stopping.is_set():
                try:
                    self.errors.put_nowait(str(error))
                except Full:
                    pass
        finally:
            self.ready = self.connected = False
            self.listener.close()
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.log.close()

    def _send(self, client):
        last, encoded, sent, status_at = None, None, 0., time.monotonic()
        incoming, schemas = bytearray(), {}
        while not self.stopping.is_set():
            self.changed.clear()
            latest = self.latest
            if latest is not last:
                encoded = encode(snapshot(latest[0], schemas, latest[1]))
                last = latest
            now = time.monotonic()
            if encoded is not None and (now-sent > 1. or latest is not getattr(self, '_sent', None)):
                client.settimeout(.5)
                client.sendall(encoded)
                self._sent, sent = latest, now
            client.settimeout(.01)
            try:
                data = client.recv(1024)
            except socket.timeout:
                data = None
            if data == b'':
                raise OSError('Live Link bridge disconnected. Select Retry to restart it.')
            if data:
                incoming.extend(data)
                if len(incoming) > 4096:
                    raise ValueError('Invalid bridge status.')
                while b'\n' in incoming:
                    line, _, incoming = incoming.partition(b'\n')
                    report = json.loads(line)
                    self.ready = True
                    self.connected = report.get('connected') is True
                    status_at = now
            if now-status_at > 10:
                raise TimeoutError('Live Link bridge stopped responding. Select Retry.')
            self.changed.wait(.02)
