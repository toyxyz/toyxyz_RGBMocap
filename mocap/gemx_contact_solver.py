"""CPU bridge from SOMA joint frames to the shared anatomical contact solver."""

import json
from pathlib import Path

import numpy as np

# Semantic role order in gemx_retarget.BODY, shared with the contact solver.
PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
SOLVED_ROLES = (1, 2, 4, 5, 7, 8, 13, 14, 16, 17, 18, 19, 20, 21)


def solve(project, plan, output, report_path, emit):
    from scipy.spatial.transform import Rotation
    from . import correction
    from .compute_policy import configure_worker_cpu
    from .correction_settings import correction_options
    from .foot_contacts import read_edit
    from .gemx_correct import context
    from .gemx_result import FPS, POSE, load_rig, manifest
    from .gemx_retarget import BODY
    from .hand_lock import apply as apply_hands
    from .storage import atomic_json

    configure_worker_cpu('correct')
    if tuple(correction.PARENTS) != PARENTS:
        raise RuntimeError('The shared solver joint layout changed.')
    state = manifest(project)
    options = correction_options(state.get('correction'))
    rig = load_rig(project)
    ids = [list(rig['names']).index(name) for name in BODY]
    with np.load(plan, allow_pickle=False) as inputs:
        probes, palms = inputs['probes'], inputs['palms']
    raw = np.load(project / POSE, mmap_mode='r', allow_pickle=False)
    try:
        world = raw[:, ids]
    finally:
        raw._mmap.close()
    neutral = rig['rest'][ids]
    rest, root = neutral[:, :3, 3].copy(), world[:, 0, :3, 3].copy()
    rotation = world[:, :, :3, :3] @ neutral[:, :3, :3].swapaxes(-1, -2)
    local = rotation.copy()
    local[:, 1:] = rotation[:, PARENTS[1:]].swapaxes(-1, -2) @ rotation[:, 1:]
    n = len(root)
    quat = Rotation.from_matrix(local.reshape(-1, 3, 3)).as_quat().reshape(n, 22, 4)[..., [3, 0, 1, 2]].astype(np.float32)
    points, _ = correction.fk(rest, root, quat)
    used = [i for i, name in enumerate(BODY) if name != 'Head']
    if np.max(np.abs(points[:, used] - world[:, :, :3, 3][:, used])) > .002:
        raise ValueError('The SOMA body proportions vary across this result. The original motion was kept.')
    edited, _, warning = read_edit(project, 'initial', context(project, state))
    floor = None if options['auto_floor'] else options['floor']
    # GEM-X provides no contact logits. Let the shared solver classify contacts
    # from low/slow sole motion, keeping flight and manual timing precedence.
    scores = np.ones((n, 4), dtype=np.float32)
    corrected_root, corrected_quat, report = correction.correct_arrays(rest, root, quat, probes, scores,
        fps=FPS, floor=floor, preset=options['preset'], mode=options['mode'], pelvis_offset=options['pelvis_offset'],
        foot_enabled=options['foot_lock'], contact_ranges=edited[:2] if edited is not None else None, emit=emit)
    report['hand_lock'] = apply_hands(rest, root, quat, corrected_root, corrected_quat, palms, FPS,
        options['mode'], report['floor'], options['preset'], manual=edited[2:] if edited is not None and len(edited) == 4 else None,
        enabled=options['hand_lock'], emit=emit)
    report['warning'] = ' '.join(filter(None, (report['warning'], report['hand_lock']['warning'], warning)))
    updated = Rotation.from_quat(corrected_quat[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(n, 22, 3, 3)
    np.savez(output, local=updated.astype(np.float32), shift=corrected_root-root)
    atomic_json(report_path, report)


def surface_report(before, after, report, fps):
    # Evaluate the baked SOMA surface over the solver's applied intervals.
    active = np.zeros((len(before), 2), dtype=bool)
    for segment in report['segments']:
        active[segment['start']:segment['end'], segment['foot']] = True
    adjacent = active[1:] & active[:-1]
    if not adjacent.any():
        return {}
    from .contact_trajectory import support_samples, support_weights
    weights = support_weights(before, active, fps)
    result = {}
    for label, points in (('before', before), ('after', after)):
        xy = support_samples(points, weights)[..., [0, 2]]
        speed = np.linalg.norm(np.diff(xy, axis=0), axis=2)[adjacent] * fps
        result[f'sole_speed_{label}_p95_mps'] = float(np.percentile(speed, 95))
    return result


def main(args, emit):
    try:
        solve(*map(Path, args), emit)
    except (OSError, ValueError, RuntimeError) as error:
        print('SOMA contact solve: ' + str(error), flush=True)
        return 1
    return 0
