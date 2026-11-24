"""ExMR workflow commands backed by the desktop's job builders and workers."""
from __future__ import annotations

import json
import math
from pathlib import Path
import os
import uuid

from . import project_workspace as workspace
from .cli_runner import WorkerError, run_process, run_worker
from .compute_policy import MODES as COMPUTE_MODES
from .correction_settings import MODES, PRESETS, correction_options, effective_options
from .exmr_options import options as exmr_options
from .local_choices import saved
from .motion_tools import MOTION_TOOLS, RESULTS, base_result, result_is_current, result_kind, selected_export
from .paths import PROJECTS, ROOT
from .people_store import load_people, root_for, safe_child, save_people
from .person_names import export_stem
from .resource_safety import load as resource_safety
from .single_actor import preferences
from .storage import atomic_json, project_key, sha256
from .take_selection import diagnose_take
from .workflow_jobs import export_row, grouped_take_jobs, next_job, person_jobs, take_jobs, video_jobs


def manifest(project):
    return json.loads((project / 'project.json').read_text(encoding='utf-8'))


def project_directory(path):
    path = path.resolve()
    return path.parent if path.name == 'project.json' and path.is_file() else path


def model_project(path, model):
    path = project_directory(path)
    state = manifest(path)
    if state.get('workspace_version') == workspace.VERSION:
        path = workspace.model_project(path, model, state)
        state = manifest(path)
    if state.get('pipeline', {}).get('model') != model:
        raise ValueError(f'{path} is not a saved {model} project.')
    return path


def requested_workspace(path, video, model):
    """Validate the source before prepare can reuse a cached working clip."""
    if path is None:
        from .video import probe
        duration = probe(video)['duration']
        candidate = PROJECTS / f'{video.stem}_{project_key(video, 0, duration)}_project'
        if not (candidate / 'project.json').exists():
            return None, None, 0.
        path = candidate
    path = project_directory(path)
    if not (path / 'project.json').exists():
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise ValueError('Choose an empty folder or a saved workspace for this video.')
        return path, None, 0.
    root = workspace.root_for(path)
    if root is None:
        raise ValueError('Choose a model-folder workspace for reconstruction.')
    record = workspace.read(root)
    original = Path(record['source']['path']).resolve()
    managed = workspace.media_path(root, record).resolve()
    path = workspace.model_project(root, model, record)
    settings = manifest(path) if (path / 'project.json').is_file() else {}
    scene = root_for(path, settings) if settings else path
    scene_state = manifest(scene) if scene != path else settings
    if not (record.get('media') and video == managed):
        expected = scene_state.get('source_sha256')
        matches = sha256(video) == expected if expected else video == original
        if not matches:
            raise ValueError('The video differs from this workspace source. Choose a new project folder.')
    if (scene / 'people.json').is_file():
        try:
            group = load_people(scene)
        except ValueError:
            # Find person can replace stale detections. The source was validated above.
            group = None
        people = group['people'] if group is not None else []
        active = next((person for person in people if person['id'] == group.get('active')), None)
        if path == scene and active and active.get('takes'):
            path = safe_child(scene, active['takes'][-1]['project'])
        if path != scene:
            actor_state = manifest(path)
            settings = {**scene_state, **actor_state, 'correction': effective_options(path, actor_state)}
    return root, settings, float(record.get('time_offset', 0.))


def chosen_settings(args, state=None, base_correction=None):
    defaults = saved('default_settings')
    defaults = defaults if isinstance(defaults, dict) else {}
    panels = {key: defaults[key] if isinstance(defaults.get(key), dict) else {}
              for key in ('exmr_options', 'correction', 'processing_options')}
    state = state or {}
    exmr = dict(panels['exmr_options'])
    if type(exmr.get('static_camera')) is not bool:
        exmr.pop('static_camera', None)
    if exmr.get('smoothing') not in ('global', 'none'):
        exmr.pop('smoothing', None)
    exmr.update(state.get('exmr_options', {}))
    if getattr(args, 'camera', None) is not None:
        exmr['static_camera'] = args.camera == 'static'
    if getattr(args, 'smoothing', None) is not None:
        exmr['smoothing'] = 'global' if args.smoothing else 'none'
    fallback = panels['correction']
    correction = {}
    for key, choices in [('preset', PRESETS), ('mode', MODES)]:
        if fallback.get(key) in choices:
            correction[key] = fallback[key]
    correction.update({key: fallback[key] for key in ('foot_lock', 'hand_lock') if type(fallback.get(key)) is bool})
    correction.update(state.get('correction', {}) if base_correction is None else base_correction)
    for key in ('preset', 'mode', 'foot_lock', 'hand_lock', 'pelvis_offset'):
        if getattr(args, key, None) is not None:
            correction[key] = getattr(args, key)
    if args.floor is not None:
        correction.update(auto_floor=False, floor=args.floor)
    elif args.auto_floor:
        correction['auto_floor'] = True
    compute = args.compute or state.get('processing_options', {}).get('resource_mode') or panels['processing_options'].get('resource_mode')
    return exmr_options(exmr), correction_options(correction), compute if compute in COMPUTE_MODES else 'auto'


def environment(compute, settings=None):
    return {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1',
            'RGB_MOCAP_EXMR_OPTIONS': json.dumps(exmr_options(settings)),
            'RGB_MOCAP_COMPUTE_MODE': compute,
            'RGB_MOCAP_RESOURCE_SAFETY': '1' if resource_safety() else '0'}


def missing_models(stages):
    from .models import missing
    return list(dict.fromkeys(model.name for stage in stages for model in missing(stage)))


def video_range(video, start, end, offset):
    import cv2
    capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
    try:
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not math.isfinite(frames) or not math.isfinite(fps) or frames < 1 or fps <= 0:
            raise ValueError('Cannot read the video frame count and frame rate.')
    finally:
        capture.release()
    duration = frames / fps
    first = 0. if start is None else start
    last = duration if end is None else end
    if not math.isfinite(first) or not math.isfinite(last) or not 0. <= first < last <= duration + 1e-6:
        raise ValueError(f'Choose a finite trim within the {duration:.3f} second video.')
    return first + offset, min(last, duration) + offset


def run_jobs(jobs, project, env, report):
    while jobs:
        stage, arguments, jobs = next_job(jobs, project)
        events = run_worker(stage, arguments, env, report)
        if stage == 'prepare':
            project = Path(one_event(events, 'prepared')['project']).resolve()
    return project


def one_event(events, name):
    for event in events:
        if event['type'] == name:
            return event
    raise WorkerError(f'Worker completed without the required {name} event.')


def reconstruct(args, report):
    from .scene_floor import ensure_config, save_config

    video = args.video.resolve(strict=True)
    if args.person and not args.find_person:
        raise ValueError('--person requires --find-person.')
    if args.person and any(person < 1 for person in args.person):
        raise ValueError('Person IDs start at 1.')
    workspace_path, settings, offset = requested_workspace(args.project, video, args.model)
    if workspace_path is not None and settings and not args.find_person:
        active = workspace.model_project(workspace_path, args.model)
        if (root_for(active) / 'people.json').is_file():
            raise ValueError('This workspace uses found people. Include --find-person to reconstruct them.')
        if not preferences(active)['checked']:
            raise ValueError('Person 1 is unchecked in this saved single-person project. Check it in the GUI first.')
    first, last = video_range(video, args.start, args.end, offset)
    exmr, correction, compute = chosen_settings(args, settings)
    corrected = args.correction if args.correction is not None else (settings or {}).get('pipeline', {}).get('result', 'corrected_initial').startswith('corrected_')
    kind = result_kind(correction=corrected)
    interpolation = args.rife if args.rife is not None else bool((settings or {}).get('interpolation'))
    stages = MOTION_TOOLS[args.model].stages(correction=corrected)
    discovery = ['people'] if args.find_person else []
    absent = missing_models([*discovery, *[stage for stage in stages if stage != 'track' or not discovery]])
    if interpolation:
        from .rife_video import recipe
        from .models import BY_KEY, ready
        from .video import probe
        model = BY_KEY['rife49']
        if recipe(True, probe(video)['fps']) and not ready(model):
            absent.append(model.name)
    if args.dry_run:
        return {'dry_run': True, 'model': args.model, 'result': kind, 'range_seconds': [first, last],
                'project': str(workspace_path) if workspace_path else None,
                'stages': ['prepare', *discovery, *(['person'] if discovery else []),
                           *[stage for stage in stages if stage != 'track' or not discovery]],
                'takes_deferred': bool(discovery), 'missing_models': absent}
    if absent:
        raise ValueError('Prepare these models in Model Manager: ' + ', '.join(absent))
    env = environment(compute, exmr)
    jobs = video_jobs(video, first, last, args.model, discovery or stages,
                     result=None if discovery else kind, correction=correction,
                     workspace=workspace_path, interpolation=interpolation)
    project = run_jobs(jobs, None, env, report)
    if not discovery:
        return {'project': str(project), 'result': kind, 'model': args.model}

    group = load_people(project)
    if not group:
        raise WorkerError('No saved people group was published.')
    people = group['people']
    if args.person:
        unknown = set(args.person) - {person['id'] for person in people}
        if unknown:
            raise ValueError('Unknown person IDs: ' + ', '.join(map(str, sorted(unknown))))
        for person in people:
            person['checked'] = person['id'] in args.person
        save_people(project, group)
    selected = [person for person in people if person.get('checked')]
    if not selected:
        raise ValueError('No person is checked. Specify --person with a found person ID.')

    ensure_config(project)
    state = manifest(project)
    state['exmr_options'] = exmr
    atomic_json(project / 'project.json', state)
    if corrected:
        save_config(project, correction)
        active = next((person for person in people if person['id'] == group.get('active')), selected[0])
        preview = safe_child(project, active['preview'])
        preview_state = manifest(preview)
        preview_state['correction'] = correction
        atomic_json(preview / 'project.json', preview_state)
    names = [person['id'] for person in selected]
    preparation = person_jobs(project, names, first, last, state, kind, correction)
    inference, post, published = [], [], []
    process_stages = [stage for stage in stages if stage != 'track']
    for stage, arguments in preparation:
        event = one_event(run_worker(stage, arguments, env, report), 'person_prepared')
        paths = [Path(path).resolve() for path in event['projects']]
        jobs, deferred = take_jobs(paths, process_stages, kind)
        inference.extend(jobs)
        post.extend(deferred)
        published.extend(paths)
    if not published:
        raise WorkerError('No take was prepared for the selected people.')
    run_jobs(grouped_take_jobs(project, inference, [], post), project, env, report)
    prepare_people_scene(project, env, report)
    return {'project': str(project), 'result': kind, 'model': args.model, 'takes': [str(path) for path in published]}


def select_takes(args):
    project = model_project(args.project, args.model)
    root = root_for(project)
    group = load_people(root)
    if group is None:
        if args.take and args.take != ['single'] or args.person and set(args.person) != {1}:
            raise ValueError('This is a single-person project. Use person 1 or take single.')
        person = preferences(project)
        return root, None, [('single', project, person)]
    rows = [(take['project'], safe_child(root, take['project']), person)
            for person in group['people'] for take in person.get('takes', [])]
    if args.take:
        rows = [row for row in rows if row[0] in args.take]
        if {row[0] for row in rows} != set(args.take):
            raise ValueError('Some selected takes are unavailable. Run project inspect.')
    elif args.person:
        rows = [row for row in rows if row[2]['id'] in args.person]
        if {row[2]['id'] for row in rows} != set(args.person):
            raise ValueError('Some selected people have no reconstructed takes. Run project inspect.')
    elif not args.all_takes and root != project:
        rows = [row for row in rows if row[1] == project]
    elif not args.all_takes and args.command != 'project' and len(rows) != 1:
        raise ValueError('Choose --take, --person or --all-takes. Run project inspect to see the IDs.')
    if not rows and args.command != 'project':
        raise ValueError('No saved takes. Reconstruct at least one person first.')
    for _, path, _ in rows:
        model_project(path, args.model)
    return root, group, rows


def selected_kind(path, override):
    kind = override or manifest(path).get('pipeline', {}).get('result', 'initial')
    if kind not in RESULTS or base_result(kind) != 'initial':
        raise ValueError(f'Unsupported ExMR result {kind} in {path}.')
    return kind


def inspect(args):
    root, group, rows = select_takes(args)
    result = []
    for ident, path, person in rows:
        state = manifest(path)
        kind = selected_kind(path, args.result)
        issue = diagnose_take(path, kind) if selected_export(path, kind) is None else None
        if issue is None and group is not None:
            from .scene_export import context
            try:
                context(root, path)
            except (OSError, ValueError, KeyError) as error:
                from .take_selection import TakeIssue
                issue = TakeIssue('Prepare Preview', str(error), 'Open the project and use Prepare Preview.')
        result.append({'id': ident, 'project': str(path), 'person_id': person['id'], 'person': person['name'],
                       'frames': state.get('frames'), 'source_frame_range': state.get('person_frames'),
                       'result': kind, 'export_ready': issue is None,
                       'issue': vars(issue) if issue is not None else None})
    return {'model': args.model, 'project': str(root), 'takes': result}


def prepare_people_scene(root, env, report):
    from .scene_export import context
    from .scene_camera import take_projects
    group = load_people(root)
    projects = take_projects(root, group)
    if not projects:
        return
    try:
        for _, project in projects:
            context(root, project)
    except (OSError, ValueError, KeyError):
        command = [str(ROOT / 'runtime/python.exe'), '-I', '-X', 'utf8', str(ROOT / 'scene_boot.py'), str(root)]
        run_process(command, 'Prepare shared scene', env, report, env['RGB_MOCAP_RESOURCE_SAFETY'] == '1')


def correct(args, report):
    from .scene_floor import read_config, save_config
    root, group, rows = select_takes(args)
    if args.result not in (None, 'corrected_initial'):
        raise ValueError('ExMR correction writes corrected_initial.')
    updates, jobs, compute = [], [], None
    floor_requested = args.floor is not None or args.auto_floor or args.mode is not None
    for _, path, _ in rows:
        if not (path / RESULTS['initial'][0]).is_file() or not result_is_current(path, 'initial'):
            raise ValueError(f'{path} has no current base motion. Reconstruct it first.')
        state = manifest(path)
        base = state.get('correction')
        if group is not None and read_config(root) is not None:
            base = effective_options(path, state)
        _, options, mode = chosen_settings(args, state, base)
        if compute is None:
            compute = mode
        state['correction'] = options
        updates.append((path, state))
    for path, _ in updates:
        jobs.extend((stage, ['--project', str(path), '--result', 'corrected_initial'])
                    for stage in ('correct', 'export'))
    if group is not None:
        from .workflow_jobs import scene_correction_jobs
        jobs, skipped = scene_correction_jobs(root, group['people'], 'corrected_initial')
        for stage, values in jobs:
            if stage == 'correct':
                model_project(Path(values[1]), args.model)
        for name in skipped:
            report(name + ': current base motion unavailable. Reconstruct this take first.')
    if args.dry_run:
        return {'dry_run': True, 'project': str(root), 'result': 'corrected_initial',
                'correction': {str(path): state['correction'] for path, state in updates},
                'stages': [{'stage': stage, 'project': values[1]} for stage, values in jobs]}
    if group is not None and floor_requested:
        save_config(root, updates[0][1]['correction'])
    for path, state in updates:
        atomic_json(path / 'project.json', state)
    run_jobs(jobs, root, environment(compute), report)
    return {'project': str(root), 'result': 'corrected_initial'}


def output_paths(args, rows):
    if args.output is not None and len(rows) != 1:
        raise ValueError('Use --output-dir when exporting more than one take.')
    outputs = []
    for ident, _, person in rows:
        if args.output is not None:
            path = args.output
            if not path.suffix:
                path = path.with_suffix('.fbx')
        else:
            person_id = person['id']
            # Existing take-directory keys include the source-frame interval. Person ID disambiguates names.
            key = Path(ident).name
            name = f'{export_stem(person["name"])}_p{person_id:04d}_{key}'
            path = args.output_dir / (name + '.fbx')
        path = path.resolve()
        if path.suffix.lower() != '.fbx' or path.is_dir():
            raise ValueError(f'Choose an FBX file destination: {path}')
        if path.exists() and not args.overwrite:
            raise ValueError(f'Destination exists: {path}. Pass --overwrite to replace it.')
        outputs.append(path)
    if len({str(path).casefold() for path in outputs}) != len(outputs):
        raise ValueError('The selected takes have duplicate FBX destinations. Export them individually.')
    return outputs


def export(args, report):
    root, group, takes = select_takes(args)
    targets = output_paths(args, takes)
    ue5 = args.fbx_profile == 'ue5_mannequin'
    if ue5 and (args.rest_pose or args.prepend_tpose or args.place_at_origin or args.symmetric_rest or args.fbx_compatibility):
        raise ValueError('The UE5 Mannequin profile uses the stock rest pose and axis settings.')
    if not ue5 and (args.ue5_target or args.in_place or args.include_mesh):
        raise ValueError('Choose --fbx-profile ue5_mannequin to set mannequin options.')
    options = dict(fbx_profile=args.fbx_profile, export_face=False, fbx_compatibility=args.fbx_compatibility,
                   prepend_tpose=args.prepend_tpose, place_at_origin=args.place_at_origin, symmetric_rest=args.symmetric_rest)
    if ue5:
        options.update(ue5_target=args.ue5_target or 'manny', ue5_in_place=args.in_place, ue5_include_mesh=args.include_mesh)
    else:
        options['fbx_rest_pose'] = args.rest_pose or 't'
    env = environment('auto')
    selected, errors = [], []
    for ident, path, person in takes:
        kind = selected_kind(path, args.result)
        source = selected_export(path, kind)
        if source is None:
            issue = diagnose_take(path, kind)
            errors.append(issue.report(f'{person["name"]}, {ident}'))
        selected.append((path, person, kind, source))
    if errors:
        raise ValueError('\n\n'.join(errors))
    if args.dry_run:
        if group is not None:
            from .scene_export import context
            for path, _, _, _ in selected:
                context(root, path)
        return {'dry_run': True, 'project': str(root), 'outputs': [str(target) for target in targets]}
    if group is not None:
        prepare_people_scene(root, env, report)
    plan = []
    for (path, person, kind, source), target in zip(selected, targets):
        row = export_row(path, kind, source, target, person['name'], options, people_root=root if group else None,
                         person_id=person['id'] if group else None)
        row['replace_existing'] = args.overwrite
        plan.append(row)
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
    request = root / ('.fbx-export-' + uuid.uuid4().hex + '.json')
    atomic_json(request, plan)
    try:
        run_worker('save_fbx', ['--project', str(root), '--output', str(request)], env, report)
    finally:
        from .blend_support import cleanup, partial
        for path in [request, *(partial(target, request) for target in targets)]:
            try:
                if path.is_dir():
                    if (path / '.rgbmocap-owner').is_file():
                        cleanup(path, request)
                else:
                    path.unlink(missing_ok=True)
            except (OSError, ValueError) as error:
                report(f'Could not clean an export staging path: {error}')
    return {'project': str(root), 'outputs': [str(target) for target in targets]}


def execute(args, report):
    if args.command == 'check':
        stages = MOTION_TOOLS['exmr'].stages(correction=True) + ['people']
        return {'runtime': 'verified', 'model': 'exmr', 'missing_models': missing_models(stages)}
    if args.command == 'project':
        return inspect(args)
    return {'reconstruct': reconstruct, 'correct': correct, 'export': export}[args.command](args, report)
