"""Apply the shared contact solver to a shaped SOMA rig, then bake with SOMA-X."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import uuid
from contextlib import contextmanager

from .gemx_result import ASSET, FPS, JOINT_COUNT, MESH, POSE, PARAMETERS, RECEIPT, VERTEX_COUNT, current, load_rig, manifest
from .result_artifacts import file_stat
from .storage import atomic_json, sha256

KIND = 'corrected_initial'
RECORD = 'soma-correction.json'
STAGE = 'gemx_correct'


@contextmanager
def geometry_assets(models, cache):
    from .models import BY_KEY

    with tempfile.TemporaryDirectory(prefix='soma-', dir=cache) as directory:
        for key in ('gemx_soma_neutral', 'gemx_soma_rig', 'gemx_soma_transforms', 'gemx_soma_correctives'):
            model = BY_KEY[key]
            source = model.path.resolve()
            if sha256(source) != models[key]:
                raise ValueError('SOMA asset changed. Reimport the verified file: ' + model.name)
            target = Path(directory) / Path(model.relative).name
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
        yield Path(directory)


def selected(project, ignore_contacts=False, ignore_options=False):
    from .correction_settings import correction_options
    from .foot_contacts import contact_revision

    project = Path(project).resolve()
    try:
        receipt = project / RECORD
        if receipt.stat().st_size > 4 * 1024**2:
            return None
        record = json.loads(receipt.read_text(encoding='utf-8'))
        generation = record['generation']
        parent = project / 'soma-corrections'
        folder = parent / generation
        if (record['version'] != 1 or not re.fullmatch('[0-9a-f]{32}', generation)
                or folder.resolve() != folder
                or not current(project) or record['source'] != file_stat(project / RECEIPT)
                or (not ignore_options and record['options'] != correction_options(manifest(project).get('correction')))
                or (not ignore_contacts and record['revision'] != contact_revision(project, 'initial'))
                or set(record['outputs']) != {POSE, MESH}
                or any(file_stat(folder / name) != stamp for name, stamp in record['outputs'].items())):
            return None
        return folder, record
    except (OSError, KeyError, TypeError, ValueError):
        return None


def context(project, state):
    return dict(frames=state['frames'], fps=FPS, sources={POSE: file_stat(project / POSE), MESH: file_stat(project / MESH)})


def contact_track(project, record, state):
    from .foot_contacts import read_edit, segment_ranges

    source = context(project, state)
    report = record['report']
    feet = report['segments']
    hand = report['hand_lock']
    generated = segment_ranges(feet) + segment_ranges(hand['segments'])
    automatic = report['automatic_ranges'] + hand['automatic']
    edited, revision, warning = read_edit(project, 'initial', source)
    if edited is not None and len(edited) == 2:
        edited += generated[2:]
    levels = [[[s['start'], s['end'], s['anchor'][1]] for s in feet if s['foot'] == side] for side in range(2)]
    return dict(ranges=edited if edited is not None else (automatic if revision else generated), automatic=automatic,
                context=source, revision=revision, applied_revision=record['revision'], manual=edited is not None,
                warning=warning, mode=report['mode'], levels=levels,
                level_starts=[[start for start, _, _ in side] for side in levels])


def sole_groups(vertices, rest):
    import numpy as np

    groups = []
    for ankle, toe, other in ((7, 10, 8), (8, 11, 7)):
        foot = vertices - rest[ankle]
        heading = rest[toe] - rest[ankle]
        heading[1] = 0
        length = np.linalg.norm(heading)
        if length < .02:
            raise ValueError('SOMA forefoot has no usable neutral direction.')
        heading /= length
        along = foot @ heading
        lateral = foot @ np.cross([0., 1., 0.], heading)
        other_foot = vertices - rest[other]
        mask = ((np.abs(lateral) < .65 * length) & (along > -.65 * length) & (along < 1.75 * length)
                & (foot[:, 1] < .1 * length)
                & (np.linalg.norm(foot[:, [0, 2]], axis=1) < np.linalg.norm(other_foot[:, [0, 2]], axis=1)))
        for region in (along <= .52 * length, along > .52 * length):
            ids = np.flatnonzero(mask & region)
            if len(ids) < 8:
                raise ValueError('SOMA sole surface could not be identified.')
            bottom = ids[np.argsort(vertices[ids, 1])[:max(8, len(ids)//3)]]
            groups.append((ids, bottom[:24]))
    return groups


def sample_surface(surface, groups):
    import numpy as np

    samples = np.empty((len(surface), 4, 3), np.float32)
    for start in range(0, len(surface), 32):
        stop = min(len(surface), start + 32)
        for k, (region, bottom) in enumerate(groups):
            samples[start:stop, k] = surface[start:stop, bottom].mean(1)
            samples[start:stop, k, 1] = surface[start:stop, region, 1].min(1)
    return samples


def run(project, cfg, emit):
    import numpy as np
    import torch
    from gem.motion.geometry import MotionGeometry
    from .compute_policy import configure_worker_cpu, gpu_memory_fraction
    from .correction_settings import correction_options
    from .foot_contacts import contact_revision
    from .paths import HOME, ROOT
    from .runtime_registry import command

    project = Path(project).resolve()
    if not current(project):
        raise ValueError('The original SOMA result is unavailable.')
    state = manifest(project)
    options = correction_options(state.get('correction'))
    stamp = file_stat(project / RECEIPT)
    revision = contact_revision(project, 'initial')
    n = state['frames']
    parent = project / 'soma-corrections'
    if parent.resolve() != parent:
        raise ValueError('SOMA correction directory leaves the result.')
    if shutil.disk_usage(project).free < n * VERTEX_COUNT * 12 + 128 * 1024**2:
        raise OSError('Free more disk space for the corrected SOMA surface.')
    parent.mkdir(exist_ok=True)
    folder = parent / uuid.uuid4().hex
    folder.mkdir()
    committed = False
    try:
        (folder / '.rgbmocap-soma-correction').touch(exist_ok=False)
        if not torch.cuda.is_available() or torch.__version__ != cfg['torch']:
            raise RuntimeError('The verified GEM-X CUDA runtime is unavailable. Run Setup_GEMX.cmd.')
        torch.set_num_threads(configure_worker_cpu(STAGE))
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction(*torch.cuda.mem_get_info()))
        cache = HOME / 'cache/gemx/warp'
        cache.mkdir(parents=True, exist_ok=True)
        os.environ['WARP_CACHE_PATH'] = str(cache)
        raw = np.load(project / POSE, mmap_mode='r', allow_pickle=False)
        surface = np.load(project / MESH, mmap_mode='r', allow_pickle=False)
        rig = load_rig(project)
        names = list(rig['names'])
        from .gemx_retarget import BODY
        indices = [names.index(name) for name in BODY]
        rest = rig['rest'][indices, :3, 3].copy()
        with (project / RECEIPT).open(encoding='utf-8') as stream:
            source = json.load(stream)
        emit('progress', stage=STAGE, message='Motion Correction: reading native SOMA support geometry')
        with geometry_assets(source['models'], cache) as assets, np.load(project / PARAMETERS, allow_pickle=False) as archive:
            params = {name: torch.from_numpy(archive[name]) for name in ('identity_coeffs', 'scale_params', 'transl')}
            engine = MotionGeometry(str(assets), device='cuda', chunk_size=16)
            adapter = engine.body_model
            layer = adapter.runtime
            identity = params['identity_coeffs'][:1].to('cuda')
            gs, local = adapter.shape_spec.split_external_scale(params['scale_params'][:1].to('cuda'))
            with torch.inference_mode():
                layer.prepare_identity(identity, torch.cat((local, local.new_ones(1, 4)), dim=-1), global_scale=gs.reshape(-1))
                zero = layer.pose(torch.zeros((1, JOINT_COUNT, 3), device='cuda'), apply_correctives=True)
                neutral = zero['vertices'][0].cpu().numpy()
                neutral += rest[0] - zero['joints'][0, 0].cpu().numpy()
            groups = sole_groups(neutral, rest)
            probes = sample_surface(surface, groups)
            palms = []
            for side, hand in (('Left', 20), ('Right', 21)):
                center = rig['rest'][[names.index(side + 'Hand' + digit + '2') for digit in ('Index', 'Middle', 'Ring', 'Pinky')], :3, 3].mean(0)
                palms.append((center - rest[hand]) * .65 + np.array([0., -.012, 0.]))
            with tempfile.TemporaryDirectory(prefix='solve-', dir=cache) as temporary:
                plan = Path(temporary) / 'plan.npz'
                solved = Path(temporary) / 'solved.npz'
                report_path = Path(temporary) / 'report.json'
                np.savez(plan, probes=probes, palms=palms)
                arguments = [project, plan, solved, report_path]
                with subprocess.Popen(command(ROOT, 'gemx_contact_solve', arguments, frozen=True),
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8',
                                      errors='replace', creationflags=subprocess.CREATE_NO_WINDOW) as child:
                    for line in child.stdout:
                        print(line, end='', flush=True)
                    if child.wait():
                        raise RuntimeError(f'SOMA contact solving failed (exit {child.returncode}). See Console.')
                report = json.loads(report_path.read_text(encoding='utf-8'))
                with np.load(solved, allow_pickle=False) as value:
                    bake(layer, adapter, params, rig, raw, folder, value['local'], value['shift'], indices,
                         float(source['vertical_offset_m']), emit)
            corrected = np.load(folder / MESH, mmap_mode='r', allow_pickle=False)
            try:
                from .gemx_contact_solver import surface_report
                report.update(surface_report(probes, sample_surface(corrected, groups), report, FPS))
            finally:
                corrected._mmap.close()
        if (not current(project) or file_stat(project / RECEIPT) != stamp or contact_revision(project, 'initial') != revision
                or correction_options(manifest(project).get('correction')) != options):
            raise ValueError('Correction inputs changed while processing. Apply Correction again.')
        previous = selected(project, ignore_contacts=True, ignore_options=True)
        predecessor = previous[0].name if previous else None
        atomic_json(project / RECORD, dict(version=1, generation=folder.name, predecessor=predecessor,
            source=stamp, revision=revision, options=options, report=report,
            outputs={name: file_stat(folder / name) for name in (POSE, MESH)}))
        committed = True
        # Keep the last preview readable. Only retire a generation that a
        # previous successful correction explicitly recorded as its predecessor.
        retired = previous[1].get('predecessor') if previous else None
        if isinstance(retired, str) and retired not in (folder.name, predecessor) and re.fullmatch('[0-9a-f]{32}', retired):
            old = parent / retired
            if old.resolve() == old and (old / '.rgbmocap-soma-correction').is_file():
                shutil.rmtree(old, ignore_errors=True)
        return project / RECORD
    finally:
        if not committed and folder.resolve().parent == parent:
            shutil.rmtree(folder, ignore_errors=True)


def bake(layer, adapter, params, rig, raw, folder, solved_local, shift, indices, vertical_offset, emit):
    import numpy as np
    import torch
    from .gemx_contact_solver import PARENTS, SOLVED_ROLES

    parents = rig['parents'][1:]
    neutral = rig['rest'][:, :3, :3]
    n = len(raw)
    pose = np.lib.format.open_memmap(folder / POSE, 'w+', dtype='float32', shape=raw.shape)
    mesh = None
    try:
        mesh = np.lib.format.open_memmap(folder / MESH, 'w+', dtype='float32', shape=(n, VERTEX_COUNT, 3))
        for start in range(0, n, 16):
            stop = min(start+16, n)
            world = raw[start:stop, :, :3, :3]
            local = np.empty_like(world)
            local[:, 0] = world[:, 0]
            local[:, 1:] = world[:, parents].swapaxes(-1, -2) @ world[:, 1:]
            for role in SOLVED_ROLES:
                j, p = indices[role], indices[PARENTS[role]]
                local[:, j] = neutral[p].T @ solved_local[start:stop, role] @ neutral[j]
            transl = params['transl'][start:stop].to('cuda') + torch.from_numpy(shift[start:stop]).to('cuda')
            transl[:, 1] += vertical_offset
            gs, scales = adapter.shape_spec.split_external_scale(params['scale_params'][start:stop].to('cuda'))
            scales = torch.cat((scales, scales.new_ones(len(local), 4)), dim=-1)
            with torch.inference_mode():
                output = layer(torch.from_numpy(local).to('cuda'), params['identity_coeffs'][start:stop].to('cuda'),
                               scales, transl, pose2rot=False, absolute_pose=True, global_scale=gs.reshape(-1))
                transforms = output['transforms'][:, 1:].cpu().numpy()
                vertices = output['vertices'].cpu().numpy()
            if not np.isfinite(transforms).all() or not np.isfinite(vertices).all():
                raise ValueError('SOMA correction produced non-finite geometry.')
            pose[start:stop] = transforms
            mesh[start:stop] = vertices
            emit('progress', stage=STAGE, current=stop, total=n, message='Motion Correction: baking SOMA bones and surface')
    finally:
        pose.flush()
        pose._mmap.close()
        if mesh is not None:
            mesh.flush()
            mesh._mmap.close()
