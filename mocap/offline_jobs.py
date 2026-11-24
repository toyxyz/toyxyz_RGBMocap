"""Build and validate orchestration for the dedicated offline pipelines."""
import hashlib
import math
from pathlib import Path

from . import project_workspace as workspace
from .paths import HOME, PROJECTS
from .storage import project_key

DEDICATED = ('gemx', 'prompthmr', 'mamma')
READY_EVENT = {model: f'{model}_ready' for model in DEDICATED}


def write_jobs(model, request):
    from .motion_tools import MOTION_TOOLS
    from .storage import atomic_json
    from uuid import uuid4

    folders = {'gemx': 'worker_jobs', 'prompthmr': 'prompthmr-jobs', 'mamma': 'mamma-jobs'}
    path = HOME / 'cache' / folders[model] / (uuid4().hex + '.json')
    atomic_json(path, {'version': 1, **request})
    return path, [(stage, ['--request', str(path)]) for stage in MOTION_TOOLS[model].stages()]


def single_root(video, info):
    return PROJECTS / (video.stem[:80] + '_' + project_key(video, 0, info['duration']) + '_project')


def camera_root(entries):
    token = hashlib.sha256('|'.join(e['path'] for e in entries).encode()).hexdigest()[:14]
    return PROJECTS / (Path(entries[0]['path']).stem[:80] + '_' + token + '_project')


def request_for(args, video):
    from .cli_workflows import chosen_settings, manifest, missing_models, project_directory, requested_workspace
    from .rife_video import recipe
    from .video import probe

    model = args.model
    if args.find_person or args.person or args.tracker != 'default' or any((args.sam31_prompt, args.sam31_time is not None, args.sam31_point)):
        raise ValueError(f'{model} handles its own actor selection. Use --people where supported.')
    if args.smoothing is not None or any(getattr(args, name) is not None for name in
            ('camera_interval', 'fast_sam', 'hamer', 'htd', 'footmr', 'handflow')):
        raise ValueError(f'The selected reconstruction options do not apply to {model}.')
    if model != 'mamma' and (args.camera_video or args.calibration or args.height is not None
                            or args.active_camera is not None):
        raise ValueError('Camera videos, calibration, reference height and active camera are MAMMA options.')
    if model != 'mamma' and args.vggt_opt is not None:
        raise ValueError('--vggt-opt applies to FlowHMR and MAMMA.')
    if model != 'prompthmr' and args.interaction is not None:
        raise ValueError('--interaction applies to PromptHMR.')
    if model == 'gemx' and (args.people is not None or args.camera not in (None, 'static')):
        raise ValueError('GEM-X uses a single actor and fixed camera.')

    info = probe(video)
    state = {}
    previous = {}
    if model == 'mamma':
        from .mamma_inputs import common_media, match_calibration, read_calibration, video_paths
        from .vggt_camera_opt import enabled as vggt_opt_enabled
        from .local_choices import saved
        if args.camera is not None:
            raise ValueError('MAMMA uses fixed cameras. Supply --calibration for manual calibration.')
        paths = video_paths([video, *args.camera_video])
        if len(paths) < 2:
            raise ValueError('MAMMA requires at least two synchronized camera videos.')
        entries = [dict(id=p.stem, path=str(p), metadata=info if p == video else probe(p)) for p in paths]
        root = project_directory(args.project) if args.project is not None else camera_root(entries)
        if (root / 'project.json').is_file():
            from .mamma_inputs import resolved_entries
            previous = workspace.read(root)
            previous_entries = resolved_entries(root, previous.get('mamma_inputs', []))
            if [(e['id'], Path(e['path']).resolve()) for e in previous_entries] != [(e['id'], Path(e['path']).resolve()) for e in entries]:
                raise ValueError('Camera inputs differ from the saved workspace. Choose a new project folder.')
            active = workspace.model_project(root, model, previous)
            if (active / 'project.json').is_file():
                state = manifest(active)
        elif root.exists() and (not root.is_dir() or any(root.iterdir())):
            raise ValueError('Choose an empty MAMMA workspace.')
        prior = state.get('mamma_options', {})
        defaults = saved('default_settings') or {}
        opted = args.vggt_opt if args.vggt_opt is not None else vggt_opt_enabled(
            prior if prior else defaults.get('mamma_options', {}))
        calibration = read_calibration(args.calibration) if args.calibration else prior.get('calibration')
        if calibration:
            match_calibration(calibration, entries)
        count = args.people if args.people is not None else prior.get('people', 0)
        height = args.height if args.height is not None else prior.get('height_m', 1.7)
        camera = args.active_camera if args.active_camera is not None else previous.get('mamma_active_camera', 0)+1
        if not 0 <= count <= 8 or not math.isfinite(height) or not .5 <= height <= 2.5 or not 1 <= camera <= len(entries):
            raise ValueError('MAMMA expects 0–8 people, 0.5–2.5 m height and a valid 1-based camera number.')
        options = dict(camera_mode='manual' if calibration else 'vggt', calibration=calibration,
                       people=count, height_m=height, vggt_opt=opted)
        source = dict(inputs=entries, active_camera=camera-1, options=options)
        duration = common_media(entries)['duration']
        fps_values = [e['metadata']['fps'] for e in entries]
    else:
        root, state, offset = requested_workspace(args.project if args.project is not None else single_root(video, info),
                                                  video, model)
        if offset:
            raise ValueError('Use the original full video for this reconstruction model.')
        state = state or {}
        source = {'video': str(video)}
        duration = float(info['duration'])
        fps_values = [info['fps']]
        if model == 'prompthmr':
            from .prompthmr_pipeline import options
            from .local_choices import saved
            defaults = saved('default_settings') or {}
            values = {**defaults.get('prompthmr_options', {}), **state.get('prompthmr_options', {})}
            for key in ('people', 'interaction', 'camera'):
                v = getattr(args, key)
                if v is not None:
                    values[key] = v
            source['options'] = options(values)
    first = 0.0 if args.start is None else args.start
    last = duration if args.end is None else args.end
    if (not math.isfinite(first) or not math.isfinite(last) or not 0 <= first < last <= duration + 1e-6
            or not 4 <= round((last - first) * 30) <= 1800):
        raise ValueError('Choose 4–1800 frames at 30 fps within the common video range.')
    last = min(duration, last)
    exmr, correction, compute = chosen_settings(args, state)
    selected = state.get('pipeline', {}).get('result', 'corrected_initial')
    enabled = args.correction if args.correction is not None else selected.startswith('corrected_')
    rife = args.rife if args.rife is not None else bool(state.get('interpolation'))
    interpolation = next((value for fps in fps_values if (value := recipe(rife, fps))), '')
    required = [f'{model}_assets']
    if model == 'prompthmr' and source['options'].get('camera') == 'moving':
        required.append('prompthmr_camera_assets')
    if interpolation:
        from .models import BY_KEY, ready
        interpolation_missing = [] if ready(BY_KEY['rife49']) else [BY_KEY['rife49'].name]
    else:
        interpolation_missing = []
    absent = missing_models(required) + interpolation_missing
    if model == 'mamma' and source['options']['calibration']:
        from .models import BY_KEY
        absent = [name for name in absent if name not in {BY_KEY[key].name for key in ('flow_vggt', 'pear_geocalib')}]
    request = dict(root=str(root), start=first, end=last, interpolation=interpolation, correction=correction, **source)
    request['correct' if model == 'gemx' else 'correction_enabled'] = enabled
    return request, enabled, absent, exmr, compute


def verify_runtime(model, request):
    if model == 'gemx':
        from .gemx_runtime import configuration
    elif model == 'mamma':
        from .mamma_runtime import configuration
        if request['options']['camera_mode'] != 'manual':
            from .mamma_gravity import support_identity
            support_identity()
    else:
        from .prompthmr_runtime import configuration
        if request['options'].get('camera') == 'moving':
            from .prompthmr_camera_runtime import configuration as camera_configuration
            camera_configuration()
    configuration()


def reconstruct(args, video, report):
    from .cli_runner import run_worker
    from .cli_workflows import environment, one_event
    from .motion_tools import MOTION_TOOLS

    model = args.model
    request, corrected, missing, exmr, compute = request_for(args, video)
    plan = dict(model=model, project=request['root'], stages=MOTION_TOOLS[model].stages(),
                result='corrected_initial' if corrected else 'initial')
    if args.dry_run:
        return {**plan, 'dry_run': True, 'missing_models': missing,
                'range_seconds': [request['start'], request['end']]}
    if missing:
        raise ValueError('Prepare these models in Model Manager: ' + ', '.join(missing))
    verify_runtime(model, request)
    path, jobs = write_jobs(model, request)
    try:
        env = environment(compute, exmr)
        for stage, flags in jobs:
            events = run_worker(stage, flags, env, report)
        event = one_event(events, READY_EVENT[model])
    finally:
        path.unlink(missing_ok=True)
    return {'model': model, 'workspace': event['root'], 'project': event.get('group', event['project']),
            'result': event['result']}
