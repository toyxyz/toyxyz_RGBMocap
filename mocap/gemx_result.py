"""Committed numeric Native77 results, readable without Torch or SOMA."""

import json
from pathlib import Path
import zipfile

from .result_artifacts import file_stat

RECEIPT = 'soma-result.json'
ASSET = 'soma-rig.npz'
POSE = 'soma-world.npy'
MESH = 'soma-vertices.npy'
PARAMETERS = 'soma-parameters.npz'
FILES = (ASSET, POSE, MESH, PARAMETERS, 'input.mp4')
JOINT_COUNT = 77
VERTEX_COUNT = 18056
FPS = 30


def manifest(project):
    path = Path(project) / 'project.json'
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def is_project(project):
    try:
        return manifest(project).get('pipeline', {}).get('model') == 'gemx'
    except (OSError, ValueError, AttributeError):
        return False


def current(project, state=None):
    project = Path(project)
    try:
        state = manifest(project) if state is None else state
        receipt = project / RECEIPT
        if receipt.stat().st_size > 65536:
            return False
        record = json.loads(receipt.read_text(encoding='utf-8'))
        return (record['version'] == 1 and record['skeleton'] == 'soma-native77'
                and record['frames'] == state['frames']
                and record['fps'] == state['fps'] == FPS and 4 <= state['frames'] <= 1800
                and state['pipeline']['model'] == 'gemx' and state['pipeline']['result'] in ('initial', 'corrected_initial')
                and set(record['outputs']) == set(FILES)
                and all(file_stat(project / name) == record['outputs'][name] for name in FILES))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def load_rig(project):
    import numpy as np
    path = Path(project) / ASSET
    limit = 4 * 1024**2
    if path.stat().st_size > limit:
        raise ValueError('SOMA rig exceeds the metadata size limit.')
    keys = ('names', 'parents', 'rest', 'faces', 'ends')
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if sorted(e.filename for e in entries) != sorted(key + '.npy' for key in keys) or sum(e.file_size for e in entries) > limit:
            raise ValueError('Invalid SOMA rig archive.')
    with np.load(path, allow_pickle=False) as data:
        rig = {key: data[key] for key in keys}
    names, parents, rest, faces, ends = (rig[k] for k in ('names', 'parents', 'rest', 'faces', 'ends'))
    if (names.shape != (JOINT_COUNT,) or names.dtype.kind != 'U' or names[0] != 'Hips'
            or len(set(names)) != JOINT_COUNT or parents.shape != names.shape or parents.dtype.kind not in 'iu'
            or parents[0] != -1 or (parents[1:] < 0).any() or (parents[1:] >= np.arange(1, JOINT_COUNT)).any()
            or rest.shape != (JOINT_COUNT, 4, 4) or not np.isfinite(rest).all()
            or ends.shape != (JOINT_COUNT, 3) or not np.isfinite(ends).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or not 1 <= len(faces) <= 3 * VERTEX_COUNT
            or faces.dtype.kind not in 'iu' or faces.min() < 0 or faces.max() >= VERTEX_COUNT):
        raise ValueError('Invalid SOMA Native77 rig.')
    return rig


def open_preview(result, cancelled):
    from .bone_overlay import geometry
    from .mesh_data import adjacency
    from .preview_data import FrameArray, LoadCancelled, motion_summary

    project = result.project
    if not current(project):
        return
    n = manifest(project)['frames']
    from .gemx_correct import KIND, contact_track, selected
    variants = [('initial', project)]
    correction = selected(project, ignore_contacts=True, ignore_options=True)
    if correction:
        folder, record = correction
        variants.append((KIND, folder))
        result.contact_tracks['initial'] = contact_track(project, record, manifest(project))
    for kind, folder in variants:
        load_variant(result, n, kind, folder, geometry, adjacency, motion_summary, FrameArray)
    if correction:
        report = record['report']
        summary = result.summaries[KIND + '_joints']
        summary.update(floor=report['floor'], correction_warning=report['warning'],
                       pelvis_auto_drop=report.get('max_reach_pelvis_drop_m', 0.),
                       pelvis_auto_raise=report.get('max_reach_pelvis_raise_m', 0.),
                       pelvis_applied_lift=report.get('max_applied_pelvis_lift_m', 0.))
    if cancelled() or not current(project):
        raise LoadCancelled()


def load_variant(result, n, kind, folder, geometry, adjacency, motion_summary, FrameArray):
    project = result.project
    pose = result.arrays[kind + '_rig'] = FrameArray(folder / POSE, max_blocks=2)
    mesh = result.arrays[kind + '_native'] = FrameArray(folder / MESH, max_blocks=2)
    if (pose.shape != (n, JOINT_COUNT, 4, 4) or mesh.shape != (n, VERTEX_COUNT, 3)
            or pose.dtype.name != 'float32' or mesh.dtype.name != 'float32'
            or any(array.path.stat().st_size != array.offset + n * array.stride for array in (pose, mesh))):
        raise ValueError('Invalid SOMA animation dimensions.')
    rig = load_rig(project)
    from .gemx_retarget import BODY
    indices = [list(rig['names']).index(name) for name in BODY]
    # The viewport's follow/path and floor code expects these 22 semantic slots.
    # These are selections of native joints. The mesh and 77-bone rig stay native.
    # Extract translations first so joint indexing keeps the frame axis first.
    joints = pose[:, :, :3, 3][:, indices, :]
    result.arrays[kind + '_joints'] = joints
    result.summaries[kind + '_joints'] = motion_summary(joints)
    key = (str(folder / POSE), *file_stat(folder / POSE))
    faces = rig['faces'].astype('uint32', copy=False)
    offsets, neighbors = adjacency(faces, VERTEX_COUNT)
    asset = dict(faces=faces, offsets=offsets, neighbors=neighbors, vertex_count=VERTEX_COUNT,
                 mode='raw', key=('soma-surface', *key),
                 topology_key=(str(project / ASSET), *file_stat(project / ASSET)))
    bone_asset = geometry(rig['rest'][:, :3, 3], rig['parents'], rig['ends'], key,
                          count=JOINT_COUNT, rest_axes=rig['rest'][:, :3, :3])
    from .bone_stream import StreamRig
    result.target_assets[kind] = {'bones': bone_asset, 'raw_asset': asset,
                                  'stream_rig': StreamRig.soma(rig)}
