"""Experimental PEAR -> VMC OSC sender. No Torch, Qt or extra packages.

Send source-local SMPL transforms, not display bone-facing matrices. These
transforms reconstruct the PEAR source rig; arbitrary avatar bind poses still
require retargeting at the receiver. Positions are metres, quaternions XYZW.
"""
import ipaddress
import math
import socket
import struct
import threading
import time

import numpy as np

BODY = ('Hips', 'LeftUpperLeg', 'RightUpperLeg', 'Spine', 'LeftLowerLeg',
        'RightLowerLeg', 'Chest', 'LeftFoot', 'RightFoot', 'UpperChest',
        'LeftToes', 'RightToes', 'Neck', 'LeftShoulder', 'RightShoulder',
        'Head', 'LeftUpperArm', 'RightUpperArm', 'LeftLowerArm',
        'RightLowerArm', 'LeftHand', 'RightHand')
NAMES = dict(enumerate(BODY))
for side, start in (('Left', 25), ('Right', 40)):
    for finger_index, finger in enumerate(('Index', 'Middle', 'Little', 'Ring', 'Thumb')):
        for segment_index, segment in enumerate(('Proximal', 'Intermediate', 'Distal')):
            NAMES[start + finger_index * 3 + segment_index] = side + finger + segment

# Native SMPL: +X anatomical left, +Y up. Unity: +X right, +Y up.
UNITY = np.diag([-1., 1., 1.])
# Aligned PEAR packets left-multiply rotations/positions by this display basis.
DISPLAY = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])


def quaternion(matrix):
    """Proper rotation matrix to a normalized XYZW quaternion, including pi turns."""
    r = np.asarray(matrix, dtype=float)
    trace = float(np.trace(r))
    if trace > 0:
        s = math.sqrt(trace + 1.) * 2
        q = np.array([(r[2, 1]-r[1, 2])/s, (r[0, 2]-r[2, 0])/s,
                      (r[1, 0]-r[0, 1])/s, s/4])
    else:
        i = int(np.argmax(np.diag(r))); j = (i+1) % 3; k = (i+2) % 3
        s = math.sqrt(max(0., 1.+r[i, i]-r[j, j]-r[k, k])) * 2
        if s < 1e-10:
            raise ValueError('Invalid PEAR rotation')
        q = np.zeros(4); q[i] = s/4
        q[j] = (r[i, j]+r[j, i])/s; q[k] = (r[i, k]+r[k, i])/s
        q[3] = (r[k, j]-r[j, k])/s
    q /= np.linalg.norm(q)
    return q if q[3] >= 0 else -q


def transforms(pose, parents, aligned=False):
    """World FK to parent-local transforms; never modify the live preview packet."""
    joints = np.asarray(pose['joints'], dtype=float)[:55].copy()
    rotations = np.asarray(pose['rotations'], dtype=float).copy()
    parents = np.asarray(parents)
    if (joints.shape != (55, 3) or rotations.shape != (55, 3, 3)
            or parents.shape != (55,) or parents.dtype.kind not in 'iu'
            or parents[0] != -1 or np.any(parents[1:] < 0)
            or np.any(parents[1:] >= np.arange(1, 55))
            or not np.isfinite(joints).all() or not np.isfinite(rotations).all()):
        raise ValueError('Invalid PEAR VMC pose/hierarchy')
    if (np.max(abs(rotations.swapaxes(-1, -2) @ rotations - np.eye(3))) > .005
            or np.min(np.linalg.det(rotations)) < .99):
        raise ValueError('Invalid PEAR VMC rotation matrices')
    if aligned:
        joints = joints @ DISPLAY
        rotations = DISPLAY.T @ rotations
    result = []
    for index, name in NAMES.items():
        parent = int(parents[index])
        if parent >= 0 and parent not in NAMES:
            raise ValueError('Unsupported PEAR bone parent')
        if parent < 0:
            position = joints[index]
            local = rotations[index]
        else:
            position = rotations[parent].T @ (joints[index]-joints[parent])
            local = rotations[parent].T @ rotations[index]
        result.append((name, UNITY @ position, quaternion(UNITY @ local @ UNITY)))
    return result


def osc_string(value):
    data = value.encode('utf8') + b'\0'
    return data + b'\0' * (-len(data) % 4)


def message(address, *values):
    tags = ','; payload = []
    for value in values:
        if isinstance(value, str):
            tags += 's'; payload.append(osc_string(value))
        elif isinstance(value, int):
            tags += 'i'; payload.append(struct.pack('>i', value))
        else:
            number = float(value)
            if not math.isfinite(number):
                raise ValueError('Nonfinite OSC value')
            tags += 'f'; payload.append(struct.pack('>f', number))
    return osc_string(address) + osc_string(tags) + b''.join(payload)


def bundles(messages, limit=1400):
    """Stay below a normal LAN MTU; final /T closes a frame for common receivers."""
    header = b'#bundle\0' + struct.pack('>Q', 1)
    current = header
    for item in messages:
        entry = struct.pack('>I', len(item)) + item
        if len(header) + len(entry) > limit:
            raise ValueError('VMC message exceeds packet size')
        if len(current) + len(entry) > limit:
            yield current
            current = header
        current += entry
    if current != header:
        yield current


def frame_packets(pose, parents, aligned, elapsed):
    bones = transforms(pose, parents, aligned)
    messages = [message('/VMC/Ext/Root/Pos', 'root', 0., 0., 0., 0., 0., 0., 1.)]
    messages += [message('/VMC/Ext/Bone/Pos', name, *position, *rotation)
                 for name, position, rotation in bones]
    messages += [message('/VMC/Ext/OK', 1), message('/VMC/Ext/T', float(elapsed))]
    return list(bundles(messages))


class Sender:
    """One latest-pose slot. Conversion and nonblocking UDP run off the GUI thread."""
    def __init__(self, host='127.0.0.1', port=39539):
        address = ipaddress.ip_address(host.strip())  # No DNS on the GUI thread.
        if address.is_unspecified or address.is_multicast:
            raise ValueError('Use a unicast destination IP address')
        if not 1 <= int(port) <= 65535:
            raise ValueError('Port must be 1-65535')
        self.target = (str(address), int(port))
        self.socket = socket.socket(socket.AF_INET6 if address.version == 6 else socket.AF_INET,
                                    socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        self.condition = threading.Condition(); self.pending = None
        self.closed = False; self.error = ''; self.frames = 0
        self.started = time.monotonic(); self.last_pose = 0.; self.tracking = False
        self.thread = threading.Thread(target=self._run, name='pear-vmc', daemon=True)
        self.thread.start()

    def submit(self, pose, parents=None, aligned=False):
        # Keep only small numeric arrays, never images/meshes or a pose history.
        value = None if pose is None else (
            dict(joints=np.array(pose['joints'][:55], copy=True),
                 rotations=np.array(pose['rotations'], copy=True)),
            np.array(parents, copy=True), bool(aligned), time.monotonic())
        with self.condition:
            if self.closed:
                return
            self.pending = (value,)
            self.condition.notify()

    def close(self):
        with self.condition:
            self.closed = True; self.pending = None; self.condition.notify()
        # Worker owns and closes the socket. No blocking join in Qt callbacks.

    def _send(self, packets):
        for packet in packets:
            self.socket.sendto(packet, self.target)

    def _unavailable(self):
        self.tracking = False
        self._send(bundles([message('/VMC/Ext/OK', 0),
                            message('/VMC/Ext/T', time.monotonic()-self.started)]))

    def _run(self):
        try:
            self._unavailable()
            while True:
                with self.condition:
                    if not self.closed and self.pending is None:
                        self.condition.wait(.1)
                    if self.closed:
                        break
                    pending, self.pending = self.pending, None
                try:
                    if pending is not None:
                        value = pending[0]
                        if value is None:
                            self._unavailable()
                        else:
                            pose, parents, aligned, submitted = value
                            if time.monotonic()-submitted > .75:
                                self._unavailable(); continue
                            self._send(frame_packets(pose, parents, aligned,
                                                     time.monotonic()-self.started))
                            self.frames += 1; self.last_pose = time.monotonic()
                            self.tracking = True; self.error = ''
                    elif self.tracking and time.monotonic()-self.last_pose > .75:
                        self._unavailable()
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    self.error = str(exc)
                    try:
                        self._unavailable()
                    except OSError:
                        pass
        except OSError as exc:
            self.error = str(exc)
        finally:
            try:
                self._unavailable()
            except OSError:
                pass
            self.socket.close()
