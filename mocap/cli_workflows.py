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
from .motion_tools import MOTION_TOOLS, RESULTS, base_result, has_handflow, result_is_current, result_kind, result_options, selected_export
from .paths import PROJECTS, ROOT
from .people_store import load_people, root_for, safe_child, save_people
from .person_names import export_stem
from .resource_safety import load as resource_safety
from .single_actor import preferences
from .storage import atomic_json, project_key, sha256
from .take_selection import diagnose_take
from .workflow_jobs import export_row, grouped_take_jobs, next_job, person_jobs, take_jobs, video_jobs

CAMERA_REFINEMENT_MODELS = ('flowhmr', 'exmr')


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


def check_camera_refinement(args):
    if args.refine_camera_rotation is not None and args.model not in CAMERA_REFINEMENT_MODELS:
        raise ValueError('Refine Camera Rotation is available for FlowHMR and ExMR.')


def workspace_view(state, model):
    return state.get('models', {}).get(model, {}).get('view', state.get('view', {}))


def camera_refinement_enabled(project, model, override):
    if override is not None:
        return override
    if model in CAMERA_REFINEMENT_MODELS:
        root = workspace.root_for(project)
        if root is not None:
            view = workspace_view(workspace.read(root), model)
            return view.get('camera_background', {}).get(model) is True
    return False


def reconstruction_result(args, project, **result):
    if args.refine_camera_rotation is not None:
        root = workspace.root_for(project)
        state = workspace.read(root)
        view = workspace_view(state, args.model)
        background = {**view.get('camera_background', {}), args.model: args.refine_camera_rotation}
        workspace.set_active(root, args.model, view={**view, 'camera_background': background})
        result['refine_camera_rotation'] = args.refine_camera_rotation
    return dict(project=str(project), model=args.model, **result)


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


def environment(compute, settings=None, flow=None):
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1',
           'RGB_MOCAP_EXMR_OPTIONS': json.dumps(exmr_options(settings)),
           'RGB_MOCAP_COMPUTE_MODE': compute,
           'RGB_MOCAP_RESOURCE_SAFETY': '1' if resource_safety() else '0'}
    if flow is not None:
        names = dict(camera_interval='INTERVAL', fast_sam='FAST_SAM', sam_batch4='BATCH4',
                     sam_gpu_crop='GPU_CROP', sam_hand_stride4='HAND_STRIDE4', hamer='HAMER', vggt_opt='VGGT_OPT')
        env.update({'RGB_MOCAP_FLOW_' + name: str(int(flow[key])) for key, name in names.items()})
    return env


def stage_settings(args, state, corrected):
    defaults = saved('default_settings')
    defaults = defaults if isinstance(defaults, dict) else {}
    model = args.model
    if model != 'flowhmr' and any(getattr(args, name) is not None for name in ('fast_sam','hamer','camera_interval')):
        raise ValueError('Those camera sampling or hand settings apply to FlowHMR.')
    if model != 'flowhmr' and args.vggt_opt is not None:
        raise ValueError('--vggt-opt applies to FlowHMR and MAMMA.')
    if model != 'gvhmr' and any(getattr(args, name) is not None for name in ('htd','footmr','handflow')):
        raise ValueError('HTD, FootMR and HandFlow apply to GVHMR.')
    if model != 'exmr' and args.smoothing is not None:
        raise ValueError('--smoothing applies to ExMR.')
    if model == 'gvhmr':
        if args.camera == 'moving':
            raise ValueError('The GVHMR workflow uses a fixed camera. Use --camera static.')
        pipeline = state.get('pipeline', {})
        existing = pipeline.get('result')
        previous = (dict(zip(('htd','footmr'), result_options(existing)), handflow=has_handflow(existing))
                    if existing in RESULTS else {})
        choices = {key: True for key in ('htd','footmr','handflow')}
        choices.update({k:v for k,v in defaults.get('gvhmr_options', {}).items() if k in choices and type(v) is bool})
        choices.update(previous)
        choices.update({key: getattr(args,key) for key in choices if getattr(args,key) is not None})
        return MOTION_TOOLS[model].stages(refine=choices['htd'], footmr=choices['footmr'],
                                         handflow=choices['handflow'], correction=corrected), result_kind(correction=corrected, **choices), None
    flow = None
    if model == 'flowhmr':
        from .vggt_camera_opt import DEFAULT_ENABLED
        flow = dict(camera_interval=0, fast_sam=True, sam_batch4=True, sam_gpu_crop=True,
                    sam_hand_stride4=True, hamer=False, vggt_opt=DEFAULT_ENABLED)
        for values in (defaults.get('flow_options', {}), state.get('flow_options', {})):
            flow.update({key: value for key, value in values.items() if key in flow and type(value) is type(flow[key])})
        if args.camera is not None:
            flow['camera_interval'] = -1 if args.camera == 'static' else 0
        if args.camera_interval is not None:
            flow['camera_interval'] = args.camera_interval
        for key in ('fast_sam', 'hamer', 'vggt_opt'):
            if getattr(args,key) is not None:
                flow[key] = getattr(args,key)
    return MOTION_TOOLS[model].stages(correction=corrected), result_kind(correction=corrected), flow

def missing_models(stages, tool=None):
    from .models import missing
    return list(dict.fromkeys(model.name for stage in stages for model in missing(stage, tool=tool)))


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
    from .offline_jobs import DEDICATED, reconstruct as reconstruct_dedicated

    check_camera_refinement(args)
    video = args.video.resolve(strict=True)
    if args.model in DEDICATED:
        if args.find_only:
            raise ValueError('This model tracks people as part of reconstruction.')
        return reconstruct_dedicated(args, video, report)
    if args.find_only:
        args.find_person = True
    if args.camera_video or args.calibration or args.people is not None or args.height is not None or args.active_camera is not None or args.interaction is not None:
        raise ValueError('Those input options apply to MAMMA or PromptHMR.')
    if args.tracker != 'sam31' and any((args.sam31_prompt, args.sam31_time is not None, args.sam31_point)):
        raise ValueError('SAM 3.1 prompt and point options require --tracker sam31.')
    if args.tracker != 'default' and not args.find_person:
        raise ValueError('--tracker requires --find-person.')
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
    people_options = None
    if args.tracker == 'sam31':
        from .workflow_jobs import sam31_arguments
        prompt = (args.sam31_prompt or 'person').strip()
        anchor = first if args.sam31_time is None else args.sam31_time + offset
        points = args.sam31_point or []
        if not first <= anchor < last or len(prompt)>256 or len(points)>16 or any(
                not 0<=x<1 or not 0<=y<1 for x,y in points):
            raise ValueError('Choose a SAM 3.1 marker frame inside the trim and up to 16 normalized points.')
        people_options = sam31_arguments(prompt, anchor, points)
    exmr, correction, compute = chosen_settings(args, settings)
    corrected = args.correction if args.correction is not None else (settings or {}).get('pipeline', {}).get('result', 'corrected_initial').startswith('corrected_')
    kind = result_kind(correction=corrected)
    interpolation = args.rife if args.rife is not None else bool((settings or {}).get('interpolation'))
    stages, kind, flow = stage_settings(args, settings or {}, corrected)
    discovery = ['people' if args.tracker == 'default' else 'sam31_people'] if args.find_person else []
    if args.find_only:
        stages = []
    required = [*discovery, *[stage for stage in stages if stage != 'track' or not discovery]]
    if flow is not None and flow['camera_interval'] == -1:
        if 'flow_camera' in required:
            required.remove('flow_camera')
    if flow is not None and flow['hamer'] and not args.find_only:
        from .flow_hamer import MODEL_KEYS
        from .models import BY_KEY, ready
        optional = [BY_KEY[k].name for k in MODEL_KEYS if not ready(BY_KEY[k])]
    else:
        optional = []
    absent = missing_models(required, tool=args.model) + optional
    if interpolation:
        from .rife_video import recipe
        from .models import BY_KEY, ready
        from .video import probe
        model = BY_KEY['rife49']
        if recipe(True, probe(video)['fps']) and not ready(model):
            absent.append(model.name)
    if args.dry_run:
        plan = {'dry_run': True, 'model': args.model, 'result': kind, 'range_seconds': [first, last],
                'project': str(workspace_path) if workspace_path else None,
                'stages': ['prepare', *discovery, *(['person'] if discovery and not args.find_only else []),
                           *[stage for stage in stages if stage != 'track' or not discovery]],
                'takes_deferred': bool(discovery), 'missing_models': absent}
        if args.refine_camera_rotation is not None:
            plan['refine_camera_rotation'] = args.refine_camera_rotation
        return plan
    if absent:
        raise ValueError('Prepare these models in Model Manager: ' + ', '.join(absent))
    env = environment(compute, exmr, flow)
    jobs = video_jobs(video, first, last, args.model, discovery or stages,
                     result=None if discovery else kind, correction=correction,
                     workspace=workspace_path, interpolation=interpolation, people_options=people_options)
    project = run_jobs(jobs, None, env, report)
    if not discovery:
        return reconstruction_result(args, project, result=kind)

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
    if args.find_only:
        return reconstruction_result(args, project, people=people)
    selected = [person for person in people if person.get('checked')]
    if not selected:
        raise ValueError('No person is checked. Specify --person with a found person ID.')
    ensure_config(project)
    state = manifest(project)
    if args.model == 'exmr':
        state['exmr_options'] = exmr
    if flow is not None:
        state['flow_options'] = {**state.get('flow_options', {}), **flow}
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
    if args.model != 'flowhmr':
        prepare_people_scene(project, env, report)
    return reconstruction_result(args, project, result=kind, takes=[str(path) for path in published])


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
    pipeline = manifest(path).get('pipeline', {})
    kind = override or pipeline.get('result', 'initial')
    if kind not in RESULTS or (pipeline.get('model') != 'gvhmr' and base_result(kind) != 'initial'):
        raise ValueError(f'Unsupported motion result {kind} in {path}.')
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
    people = [dict(id=person['id'], name=person['name'], checked=bool(person.get('checked')))
              for person in group['people']] if group else [dict(id=1, name=rows[0][2]['name'], checked=bool(rows[0][2]['checked']))]
    return {'model': args.model, 'project': str(root), 'takes': result, 'people': people}


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
    from .workflow_jobs import scene_correction_jobs
    updates, jobs, compute = [], [], None
    floor_requested = args.floor is not None or args.auto_floor or args.mode is not None
    stages = ('gemx_correct',) if args.model == 'gemx' else ('correct', 'flow_export' if args.model == 'flowhmr' else 'export')
    for _, path, _ in rows:
        result = 'corrected_' + base_result(selected_kind(path, args.result))
        base_kind = base_result(result)
        if not result_is_current(path, base_kind) or (args.model != 'gemx' and not (path / RESULTS[base_kind][0]).is_file()):
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
        jobs.extend((stage, ['--project', str(path), '--result', result]) for stage in stages)
    kinds = {values[-1] for _, values in jobs}
    if group is not None:
        if len(kinds) != 1:
            raise ValueError('Choose one result kind with --result to correct the shared scene.')
        result = next(iter(kinds))
        jobs, skipped = scene_correction_jobs(root, group['people'], result, stages=stages)
        for stage, values in jobs:
            if stage == 'correct':
                model_project(Path(values[1]), args.model)
        for name in skipped:
            report(name + ': current base motion unavailable. Reconstruct this take first.')
    if not jobs:
        raise ValueError('No saved take has all the required base motion.')
    result = jobs[0][1][-1] if len(kinds)==1 else None
    if args.dry_run:
        return {'dry_run': True, 'project': str(root), 'result': result,
                'correction': {str(path): state['correction'] for path, state in updates},
                'stages': [{'stage': stage, 'project': values[1]} for stage, values in jobs]}
    if group is not None and floor_requested:
        save_config(root, updates[0][1]['correction'])
    for path, state in updates:
        atomic_json(path / 'project.json', state)
    run_jobs(jobs, root, environment(compute), report)
    return {'project': str(root), 'result': result}


def output_paths(args, rows):
    if args.output is not None and len(rows) != 1 and not (
            getattr(args, 'combined', False) or getattr(args, 'whole_scene', False)):
        raise ValueError('Use --output-dir when exporting more than one take.')
    fmt = getattr(args,'format','fbx')
    bundled = fmt == 'blend'
    combined = bundled and (args.combined or args.whole_scene)
    outputs = []
    for ident, _, person in rows:
        if args.output is not None:
            path = args.output
            if not path.suffix:
                path = path.with_suffix('.' + fmt)
        else:
            person_id = person['id']
            # Existing take-directory keys include the source-frame interval. Person ID disambiguates names.
            key = Path(ident).name
            name = f'{export_stem(person["name"])}_p{person_id:04d}_{key}'
            path = args.output_dir / (name + '.' + fmt)
        path = path.resolve()
        if path.suffix.lower() != '.' + fmt or path.is_dir():
            raise ValueError(f'Choose a .{fmt} destination: {path}')
        if bundled:
            bundle = path.with_suffix('')
            if bundle.exists():
                raise ValueError(f'Choose a new Blender bundle directory: {bundle}')
            path = bundle / path.name
        elif path.exists() and not args.overwrite:
            raise ValueError(f'Destination exists: {path}. Pass --overwrite to replace it.')
        outputs.append(path)
    if not combined and len({str(path).casefold() for path in outputs}) != len(outputs):
        raise ValueError('The selected takes have duplicate destinations. Export them individually.')
    return outputs


def export(args, report):
    check_camera_refinement(args)
    fmt = args.format
    if args.refine_camera_rotation is not None and (fmt != 'blend' or args.no_camera_animation):
        raise ValueError('Refine Camera Rotation requires --format blend with camera animation enabled.')
    root, group, takes = select_takes(args)
    targets = output_paths(args, takes)
    ue5 = args.fbx_profile == 'ue5_mannequin'
    combined = args.combined or args.whole_scene
    if combined and (fmt != 'blend' or args.output is None):
        raise ValueError('Combined and whole-scene export require --format blend and --output.')
    if args.whole_scene and (args.model != 'mamma' or group is None):
        raise ValueError('--whole-scene uses a reconstructed MAMMA people scene.')
    if fmt != 'fbx' and (ue5 or args.rest_pose or args.fbx_compatibility):
        raise ValueError('FBX profile, rest pose and compatibility are FBX options.')
    if fmt != 'bvh' and args.bvh_profile is not None:
        raise ValueError('--bvh-profile requires --format bvh.')
    if fmt == 'blend' and args.overwrite:
        raise ValueError('Blender export writes a new bundle directory.')
    if ue5 and (args.rest_pose or args.prepend_tpose or args.place_at_origin or args.symmetric_rest or args.fbx_compatibility):
        raise ValueError('The UE5 Mannequin profile uses the stock rest pose and axis settings.')
    if not ue5 and (args.ue5_target or args.in_place or args.include_mesh):
        raise ValueError('Choose --fbx-profile ue5_mannequin to set mannequin options.')
    options = dict(format=fmt, fbx_profile=args.fbx_profile, export_face=bool(args.face),
                   face_format=args.face or 'txt', smooth_face=not args.raw_face,
                   bvh_profile=args.bvh_profile or 'generic', animate_camera=not args.no_camera_animation,
                   fbx_compatibility=args.fbx_compatibility,
                   prepend_tpose=args.prepend_tpose, place_at_origin=args.place_at_origin, symmetric_rest=args.symmetric_rest)
    if fmt == 'blend' and options['animate_camera']:
        options['background_refine'] = camera_refinement_enabled(root, args.model, args.refine_camera_rotation)
    if ue5:
        options.update(ue5_target=args.ue5_target or 'manny', ue5_in_place=args.in_place, ue5_include_mesh=args.include_mesh)
    else:
        options['fbx_rest_pose'] = args.rest_pose or 't'
    env = environment('auto')
    selected, errors = [], []
    archives = {}
    for ident, path, person in takes:
        if args.face:
            from .face_data import FaceArchive
            face_root = root_for(path)
            if face_root not in archives:
                archives[face_root] = FaceArchive(face_root)
            if person['id'] not in archives[face_root].index:
                raise ValueError(f"{person['name']}: run face track before exporting face.")
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
        plan = {'dry_run': True, 'project': str(root), 'outputs': [str(target) for target in targets]}
        if fmt == 'blend' and args.model in CAMERA_REFINEMENT_MODELS:
            plan['refine_camera_rotation'] = options.get('background_refine', False)
        return plan
    if group is not None and args.model != 'flowhmr':
        prepare_people_scene(root, env, report)
    plan = []
    for (path, person, kind, source), target in zip(selected, targets):
        row = export_row(path, kind, source, target, person['name'], options, people_root=root if group else None,
                         person_id=person['id'] if group else None)
        row['replace_existing'] = args.overwrite
        plan.append(row)
    stage = {'fbx':'save_fbx','bvh':'copy_bvh','blend':'save_blend'}[fmt]
    if args.whole_scene:
        kinds = {row['kind'] for row in plan}
        if len(kinds) != 1:
            raise ValueError('Choose a single motion result for the whole MAMMA scene.')
        stage = 'save_blend_scene'
        data = dict(root=str(root), kind=kinds.pop(), target=str(targets[0]),
                    people=list(dict.fromkeys(person['id'] for _, _, person in takes)),
                    takes=[path.relative_to(root).as_posix() for _, path, _ in takes],
                    **{key:options[key] for key in ('export_face','face_format','smooth_face')})
    else:
        data = dict(target=str(targets[0]), takes=plan) if combined else plan
    for target in targets:
        (target.parent.parent if fmt == 'blend' else target.parent).mkdir(parents=True, exist_ok=True)
    request = root / ('.cli-export-' + uuid.uuid4().hex + '.json')
    atomic_json(request, data)
    try:
        events = run_worker(stage, ['--project', str(root), '--output', str(request)], env, report)
    finally:
        from .blend_support import cleanup, partial
        for path in [request, *dict.fromkeys(partial(target.parent if fmt == 'blend' else target, request)
                                             for target in targets)]:
            try:
                if path.is_dir():
                    if (path / '.rgbmocap-owner').is_file():
                        cleanup(path, request)
                else:
                    path.unlink(missing_ok=True)
            except (OSError, ValueError) as error:
                report(f'Could not clean an export staging path: {error}')
    result = {'project': str(root), 'outputs': [str(target) for target in dict.fromkeys(targets)]}
    if args.face:
        result['face_outputs'] = [event['face_output'] for event in events if 'face_output' in event]
        errors = [event['face_error'] for event in events if 'face_error' in event]
        if not errors and not result['face_outputs']:
            errors = ['Face export did not publish any files.']
        if errors:
            result.update(failed=True, face_errors=errors)
    return result


def execute(args, report):
    if args.command == 'check':
        from .offline_jobs import DEDICATED
        model = args.model
        stages = [model + '_assets'] if model in DEDICATED else MOTION_TOOLS[model].stages(correction=True) + ['people']
        return {'runtime': 'verified', 'model': model, 'missing_models': missing_models(stages, tool=model)}
    if args.command in {'models','face','batch','contacts','prepare-preview'} or (args.command == 'project' and args.project_action == 'copy'):
        from .cli_tools import batch, contacts, copy_project, face_command, model_command, prepare_preview
        command = {'models': model_command, 'face': face_command, 'project': copy_project,
                   'batch': batch, 'contacts': contacts, 'prepare-preview': prepare_preview}
        return command[args.command](args, report)
    if args.command in {'stream','character','preview','reference-fbx'}:
        from .cli_media import character, preview, reference, stream
        return {'stream': stream, 'character': character,
                'preview': preview, 'reference-fbx': reference}[args.command](args, report)
    if args.command == 'project':
        return inspect(args)
    return {'reconstruct': reconstruct, 'correct': correct, 'export': export}[args.command](args, report)
