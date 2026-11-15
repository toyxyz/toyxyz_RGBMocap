"""Project-local Face Landmarker results and deterministic gap handling."""
from bisect import bisect_left
from pathlib import Path
import json
import os
from uuid import uuid4
from zipfile import BadZipFile

import numpy as np

from .people_store import load_people, root_for, safe_child
from .result_artifacts import file_stat

VERSION = 2
FILE = 'face_capture.npz'
CHANNELS = '''browDownLeft browDownRight browInnerUp browOuterUpLeft browOuterUpRight
cheekPuff cheekSquintLeft cheekSquintRight eyeBlinkLeft eyeBlinkRight
eyeLookDownLeft eyeLookDownRight eyeLookInLeft eyeLookInRight eyeLookOutLeft
eyeLookOutRight eyeLookUpLeft eyeLookUpRight eyeSquintLeft eyeSquintRight
eyeWideLeft eyeWideRight jawForward jawLeft jawOpen jawRight mouthClose
mouthDimpleLeft mouthDimpleRight mouthFrownLeft mouthFrownRight mouthFunnel
mouthLeft mouthLowerDownLeft mouthLowerDownRight mouthPressLeft mouthPressRight
mouthPucker mouthRight mouthRollLower mouthRollUpper mouthShrugLower
mouthShrugUpper mouthSmileLeft mouthSmileRight mouthStretchLeft mouthStretchRight
mouthUpperUpLeft mouthUpperUpRight noseSneerLeft noseSneerRight tongueOut'''.split()
CONTOURS = (
    (33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 33),
    (263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249, 263),
    (70, 63, 105, 66, 107, 55, 65, 52, 53, 46),
    (300, 293, 334, 296, 336, 285, 295, 282, 283, 276),
    (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61),
    (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78),
    (168, 6, 197, 195, 5, 4, 2), (129, 98, 97, 2, 326, 327, 358),
)
STABLE = (33, 133, 362, 263, 168, 6, 197, 195, 5, 4, 2)
POINTS = sorted({n for contour in CONTOURS for n in contour} | set(STABLE))
POINT_INDEX = {landmark: i for i, landmark in enumerate(POINTS)}
EDGES = np.array([(POINT_INDEX[a], POINT_INDEX[b])
                  for path in CONTOURS for a,b in zip(path, path[1:])], dtype=np.int32)


def manifest(project):
    return json.loads((project/'project.json').read_text(encoding='utf8'))


def capture_inputs(root):
    media = root/'input.mp4'
    info = manifest(root)
    people = load_people(root)
    if people:
        actors = people['people']
        tracks = {}
        project_model = info.get('pipeline', {}).get('model')
        for actor in actors:
            path = safe_child(root, actor['preview']) / 'tracking.npz'
            if path.is_file():
                tracks[actor['id']] = path
            elif project_model in ('mamma', 'prompthmr'):
                from .face_tracks import projected_takes
                tracks[actor['id']] = projected_takes(root, actor)
            else:
                tracks[actor['id']] = None if len(actors) == 1 else ()
    else:
        path = root/'tracking.npz'
        if not path.is_file() and info.get('pipeline',{}).get('model')=='gemx':
            path = root/'gemx_face_boxes.npz'
        tracks = {1: path if path.is_file() else None}
    return media, info, tracks


def fingerprint(root, media, info, tracks):
    files = sorted((key, str(path.relative_to(root)), file_stat(path))
                   for key,path in tracks.items() if isinstance(path, Path))
    data = [VERSION, file_stat(media), info['frames'], info['fps'], info['start'], files]
    projected = [(key, [take.identity(root) for take in takes])
                 for key, takes in sorted(tracks.items()) if isinstance(takes, tuple)]
    if projected:
        data.append(projected)
    return json.dumps(data)


def write(root, **arrays):
    target = root/FILE
    temp = target.with_name('.'+uuid4().hex+'.npz')
    try:
        np.savez_compressed(temp, **arrays)
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def last_success(valid):
    frames = np.arange(len(valid), dtype=np.int32)
    return np.maximum.accumulate(np.where(valid, frames, -1))


class FaceArchive:
    def __init__(self, root, verify=True):
        self.root = Path(root).resolve()
        self.media, self.info, tracks = capture_inputs(self.root)
        self.identity = fingerprint(self.root, self.media, self.info, tracks)
        try:
            with np.load(self.root/FILE, allow_pickle=False) as z:
                if str(z['identity']) != self.identity and verify:
                    raise ValueError('Face tracking is out of date. Run Track Face again.')
                self.ids = z['ids'].tolist()
                self.channels = z['channels'].tolist()
                self.valid, self.values, self.offsets = z['valid'], z['values'], z['offsets']
                self.indices, self.weights = z['indices'], z['weights']
                self.binding_type = str(z['binding_type'])
        except (BadZipFile, EOFError) as error:
            raise ValueError('Face tracking data is damaged. Run Track Face again.') from error
        n, frames = len(self.ids), int(self.info['frames'])
        if (len(set(self.ids)) != n or self.valid.shape != (n, frames)
                or self.values.shape != (n, frames, len(self.channels))
                or self.offsets.shape != (n, frames, len(POINTS), 2)
                or self.indices.shape != (len(POINTS), 3) or self.weights.shape != self.indices.shape
                or self.channels != CHANNELS):
            raise ValueError('Invalid face tracking data.')
        self.index = {person: slot for slot, person in enumerate(self.ids)}
        self.held = np.array([last_success(v) for v in self.valid])

    def resolve(self, project, frame, person=None, state=None):
        state = manifest(Path(project)) if state is None else state
        t0 = float(state['start'])
        fps = float(state['fps'])
        now = t0 + frame/fps
        index = round((now - self.info['start'])*self.info['fps'])
        slot = self.index.get(state.get('person_id', 1) if person is None else person)
        if slot is None or not 0 <= index < self.valid.shape[1]:
            return -1, -1, index
        return slot, int(self.held[slot,index]), index

    def pose(self, person, source_frame):
        slot = self.index.get(person)
        if slot is not None and 0 <= source_frame < self.valid.shape[1]:
            previous = self.held[slot, source_frame]
            if previous >= 0:
                return self.offsets[slot, previous]
        return None


def archive_for(project):
    project = Path(project)
    return FaceArchive(root_for(project))
