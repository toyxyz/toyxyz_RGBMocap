"""Deterministic One Euro filtering shared by face playback and export."""
import numpy as np


def expression_profile(name):
    if name.startswith(('eyeBlink', 'eyeLook')):
        return 9., 3.2, 32.
    if name.startswith('jaw') or (name.startswith('mouth') and
            not name.startswith(('mouthSmile', 'mouthFrown', 'mouthDimple'))):
        return 4.5, 1.8, 24.
    return 2.4, .9, 16.


def smooth_valid(values, valid, fps, profiles, *, vector=False, cancelled=None):
    from .pear.temporal_filter import AdaptiveFilter

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid face capture frame rate.')
    minimum, beta, maximum = np.asarray(profiles).T
    if vector:
        minimum, beta, maximum = minimum[:, None], beta[:, None], maximum[:, None]
    smoothing = AdaptiveFilter(min_cutoff=minimum, beta=beta, max_cutoff=maximum,
                               vector=vector)
    filtered = values.copy()
    previous = None
    for count, index in enumerate(np.flatnonzero(valid)):
        if count % 64 == 0 and cancelled is not None and cancelled():
            raise InterruptedError('Face smoothing cancelled.')
        if not np.isfinite(values[index]).all():
            raise ValueError('Face tracking contains non-finite samples.')
        elapsed = 0. if previous is None else (index - previous) / fps
        filtered[index] = smoothing.update(values[index], elapsed)
        previous = index
    return filtered


def smooth_face_values(values, valid, fps, channels, *, cancelled=None):
    return smooth_valid(values, valid, fps, [expression_profile(n) for n in channels],
                        cancelled=cancelled)


def smooth_face_offsets(offsets, valid, fps, *, cancelled=None):
    from .face_data import CONTOURS, POINTS

    eyes = set(CONTOURS[0]) | set(CONTOURS[1])
    lips = set(CONTOURS[4]) | set(CONTOURS[5])
    profiles = [expression_profile('eyeBlink' if n in eyes else 'mouth' if n in lips
                                   else 'landmark') for n in POINTS]
    return smooth_valid(offsets, valid, fps, profiles, vector=True, cancelled=cancelled)


def prepare_archive(root, cancelled=None):
    from .face_data import FaceArchive

    archive = FaceArchive(root)
    from .face_neutral import neutral_frames, relative_values
    archive.neutral_frames = neutral_frames(archive)
    fps = float(archive.info['fps'])
    for slot, valid in enumerate(archive.valid):
        if cancelled is not None and cancelled():
            raise InterruptedError('Face smoothing cancelled.')
        archive.values[slot] = smooth_face_values(archive.values[slot], valid, fps,
                                                  archive.channels, cancelled=cancelled)
        archive.offsets[slot] = smooth_face_offsets(archive.offsets[slot], valid, fps,
                                                    cancelled=cancelled)
        frame = archive.neutral_frames.get(archive.ids[slot])
        if frame is not None:
            archive.values[slot] = relative_values(archive.values[slot], frame)
            archive.offsets[slot] -= archive.offsets[slot, frame].copy()
    archive.values.setflags(write=False)
    archive.offsets.setflags(write=False)
    return archive
