"""Per-person neutral-frame calibration; captured face arrays stay untouched."""
import json
import numpy as np
from .storage import atomic_json

FILE = 'face_neutral.json'


def settings(root):
    path = root / FILE
    if not path.is_file():
        return dict(version=1, identity=None, people={})
    value = json.loads(path.read_text(encoding='utf-8'))
    if (not isinstance(value, dict) or value.get('version') != 1 or
            not isinstance(value.get('people'), dict)):
        raise ValueError('Invalid neutral face settings.')
    return value


def neutral_frames(archive):
    value = settings(archive.root)
    if value.get('identity') != archive.identity:
        return {}
    result = {}
    for ident in archive.ids:
        row = value['people'].get(str(ident))
        frame = row.get('frame') if isinstance(row, dict) else None
        slot = archive.index[ident]
        if (type(frame) is int and 0 <= frame < archive.valid.shape[1]
                and bool(archive.valid[slot, frame])):
            result[ident] = frame
    return result


def set_neutral(archive, ident, frame):
    if (ident not in archive.index or type(frame) is not int or
            not 0 <= frame < archive.valid.shape[1] or
            not bool(archive.valid[archive.index[ident], frame])):
        raise ValueError('Choose a frame where this person’s face was tracked successfully.')
    value = settings(archive.root)
    if value.get('identity') != archive.identity:
        value = dict(version=1, identity=archive.identity, people={})
    value['people'][str(ident)] = dict(frame=frame)
    atomic_json(archive.root / FILE, value)


def reset_neutral(root, ident):
    value = settings(root)
    if value['people'].pop(str(ident), None) is not None:
        atomic_json(root / FILE, value)


def relative_values(values, frame):
    baseline = values[frame]
    # Keep the usable [neutral, full expression] range. Near-saturated neutral
    # channels have little usable information; cap gain to avoid huge noise.
    return np.clip((values - baseline) / np.maximum(1. - baseline, .1), 0., 1.)


def expression_values(archive, slot, *, smooth, frames=None):
    from .face_smoothing import smooth_face_values

    values = archive.values[slot]
    if smooth:
        values = smooth_face_values(values, archive.valid[slot], float(archive.info['fps']), archive.channels)
    frames = neutral_frames(archive) if frames is None else frames
    frame = frames.get(archive.ids[slot])
    return relative_values(values, frame) if frame is not None else values
