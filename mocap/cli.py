"""Headless command line for the single-video ExMR workflow.

Runs the worker stages the GUI runs for Reconstruct 3D, Apply Correction and
File > Export, with the same arguments, environment and request files, so a
project made here opens in the app with its caches and receipts intact. It
chooses no numerical setting the GUI does not also offer; defaults are the
GUI's built-in defaults.

Scope: ExMR, one automatically tracked person (no Find person), FBX export.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

PREFIX = '@@MOCAP@@'
STAGE_NAMES = {'prepare': 'Prepare video', 'track': 'Track person', 'exmr_batch': 'ExMR reconstruction',
               'correct': 'Motion Correction', 'export': 'Prepare motion', 'rig': 'Prepare target rig',
               'save_fbx': 'Export FBX'}


class StageFailed(RuntimeError):
    pass


class Console:
    def __init__(self, quiet=False, log=None):
        self.quiet = quiet
        self.log = open(log, 'a', encoding='utf-8') if log else None
        self._last = 0.

    def say(self, text):
        if not self.quiet:
            print(text, file=sys.stderr, flush=True)
        self.write(text)

    def write(self, text):
        if self.log:
            self.log.write(text + '\n')
            self.log.flush()

    def event(self, stage, event):
        kind = event.get('type')
        if kind == 'progress':
            now = time.monotonic()
            total, current = event.get('total') or 0, event.get('current') or 0
            if now - self._last < 2 and not (total and current >= total):
                return
            self._last = now
            percent = f' {100 * current / total:.0f}%' if total else ''
            self.say(f'  {STAGE_NAMES.get(stage, stage)}{percent} {event.get("message", "")}'.rstrip())
        elif kind in ('info', 'resource_mode', 'compute_plan'):
            self.say('  ' + str(event.get('message') or {k: v for k, v in event.items() if k != 'type'}))
        else:
            self.write(PREFIX + json.dumps(event))


def worker_environment(compute='auto', resource_safety=None, exmr_options=None):
    """The GUI's per-job inserts (gui.py next_job), over the launcher's environment."""
    from .exmr_options import options
    from .resource_safety import load
    env = dict(os.environ)
    env.update(PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1', RGB_MOCAP_COMPUTE_MODE=compute,
               RGB_MOCAP_RESOURCE_SAFETY='1' if (load() if resource_safety is None else resource_safety) else '0',
               RGB_MOCAP_FLOW_INTERVAL='0', RGB_MOCAP_FLOW_FAST_SAM='1', RGB_MOCAP_FLOW_BATCH4='1',
               RGB_MOCAP_FLOW_GPU_CROP='1', RGB_MOCAP_FLOW_HAND_STRIDE4='1', RGB_MOCAP_FLOW_HAMER='0',
               RGB_MOCAP_EXMR_OPTIONS=json.dumps(options(exmr_options)))
    return env


def terminate_tree(process):
    """Cancel like the GUI: the worker and its children (taskkill /T /F on Windows)."""
    if process.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(['taskkill.exe', '/PID', str(process.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        process.kill()
    process.wait()


def run_stage(stage, arguments, env, console):
    """Start one worker exactly as the GUI does and return its events."""
    from .paths import ROOT, worker_command
    request = None
    if stage == 'exmr_batch':
        from .queue_groups import write_request
        request = write_request(stage, arguments)
        project = arguments[arguments.index('--project') + 1]
        arguments = ['--request', str(request), '--project', str(project)]
    console.say(STAGE_NAMES.get(stage, stage) + '...')
    started = time.monotonic()
    events, failure = [], None
    try:
        command = worker_command(stage, *map(str, arguments))
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, encoding='utf-8', errors='replace', bufsize=1)
        try:
            for line in process.stdout:
                line = line.rstrip('\r\n')
                if not line.startswith(PREFIX):
                    if line:
                        console.write(line)
                    continue
                try:
                    event = json.loads(line[len(PREFIX):])
                except ValueError:
                    console.write(line)
                    continue
                events.append(event)
                console.event(stage, event)
                if event.get('type') in ('failed', 'blocked'):
                    failure = event.get('message') or event.get('type')
            code = process.wait()
        except BaseException:
            terminate_tree(process)
            raise
    finally:
        if request is not None:
            from .queue_groups import remove_request
            remove_request(request)
    if failure or code:
        raise StageFailed(f'{STAGE_NAMES.get(stage, stage)} failed: ' + (failure or f'worker exit code {code}'))
    console.say(f'  done in {time.monotonic() - started:.1f} s')
    return events


def video_range(video, start=None, end=None):
    """The GUI trims with OpenCV's frame count and rate (video_preview), not ffprobe."""
    import cv2
    capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
    try:
        if not capture.isOpened():
            raise ValueError('Cannot open the video: ' + str(video))
        frames, fps = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if frames <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError('Cannot determine the video length and frame rate.')
    duration = frames / fps
    first = 0.0 if start is None else float(start)
    last = duration if end is None else min(duration, float(end))
    if not 0 <= first < last:
        raise ValueError(f'Invalid trim {first}..{last} s for a {duration:.3f} s video.')
    return first, last


def correction_arguments(correction):
    args = ['--correction-preset', correction['preset'], '--correction-mode', correction['mode'],
            '--foot-lock', str(int(correction['foot_lock'])), '--hand-lock', str(int(correction['hand_lock'])),
            '--pelvis-offset', str(correction['pelvis_offset'])]
    return (['--floor', str(correction['floor'])] if not correction['auto_floor'] else []) + args


def missing_models(stages):
    from .models import missing
    names = [model.name for stage in stages for model in missing(stage)]
    return list(dict.fromkeys(names))


def reconstruct(video, *, start=None, end=None, project=None, correction, corrected=True, env, console):
    """Reconstruct 3D for one automatically tracked person (gui.py run_full/input_jobs)."""
    from .motion_tools import MOTION_TOOLS, result_kind
    from .queue_groups import join_single_pipeline
    kind = result_kind(correction=corrected)
    stages = MOTION_TOOLS['exmr'].stages(correction=corrected)
    unavailable = missing_models(stages)
    if unavailable:
        raise StageFailed('Prepare these models first (Model Manager): ' + ', '.join(unavailable))
    first, last = video_range(video, start, end)
    prepare = ['--video', str(Path(video).resolve()), '--start', str(first), '--end', str(last)]
    if project is not None:
        prepare += ['--project', str(project)]
    prepare += ['--motion-tool', 'exmr', '--result', kind]
    if corrected:
        prepare += correction_arguments(correction)
    events = run_stage('prepare', prepare, env, console)
    model_project = Path(next(e['project'] for e in events if e.get('type') == 'prepared'))
    # Same queue and argument filling as gui.py input_jobs/next_job.
    jobs = [(stage, ['--result', kind] if stage in ('correct', 'export') else None) for stage in stages]
    while jobs:
        jobs = join_single_pipeline(jobs, model_project)
        stage, arguments = jobs.pop(0)
        if arguments is None:
            arguments = ['--project', str(model_project)]
        elif '--project' not in arguments:
            arguments = ['--project', str(model_project), *arguments]
        run_stage(stage, arguments, env, console)
    return model_project, kind


def reconstruct_people(video, *, start=None, end=None, project=None, correction, corrected=True, exmr_options=None,
                       people=None, env, console):
    """Find person (Default tracker) then Reconstruct 3D for the checked people.

    Mirrors people_ui.find_people/input_person_jobs/people_event and gui.py run_full.
    Returns [(take project, person record)].
    """
    from .exmr_options import options
    from .motion_tools import MOTION_TOOLS, result_kind
    from .people_store import load_people, safe_child, save_people
    from .queue_groups import grouped_jobs
    from .scene_floor import ensure_config, save_config
    from .storage import atomic_json
    kind = result_kind(correction=corrected)
    stages = [s for s in MOTION_TOOLS['exmr'].stages(correction=corrected) if s != 'track']
    unavailable = missing_models(['people', *stages])
    if unavailable:
        raise StageFailed('Prepare these models first (Model Manager): ' + ', '.join(unavailable))
    first, last = video_range(video, start, end)
    prepare = ['--video', str(Path(video).resolve()), '--start', str(first), '--end', str(last)]
    if project is not None:
        prepare += ['--project', str(project)]
    events = run_stage('prepare', prepare + ['--motion-tool', 'exmr'], env, console)
    root = Path(next(e['project'] for e in events if e.get('type') == 'prepared'))
    events = run_stage('people', ['--project', str(root)], env, console)
    root = Path(next((e['project'] for e in events if e.get('type') == 'people_found'), root))
    state = load_people(root)
    found = state['people']
    console.say(f'  {len(found)} person(s) found: ' + ', '.join(f'{p["id"]}={p["name"]}' for p in found))
    if people is not None:
        wanted = set(people)
        unknown = wanted - {p['id'] for p in found}
        if unknown:
            raise StageFailed('No such person id: ' + ', '.join(map(str, sorted(unknown))))
        for person in found:
            person['checked'] = person['id'] in wanted
        save_people(root, state)
    selected = [p for p in found if p.get('checked')]
    if not selected:
        raise StageFailed('No person is checked. Pass --people with the ids to reconstruct.')
    ensure_config(root)
    manifest_path = root / 'project.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['exmr_options'] = options(exmr_options)
    atomic_json(manifest_path, manifest)
    applied = correction
    if corrected:
        # save_correction_options: shared floor plus the active (first checked) person's preview.
        save_config(root, applied)
        preview = safe_child(root, selected[0]['preview']) / 'project.json'
        if preview.is_file():
            value = json.loads(preview.read_text(encoding='utf-8'))
            value['correction'] = applied
            atomic_json(preview, value)
    fps = manifest['fps']
    begin, finish = round((first - manifest['start']) * fps), round((last - manifest['start']) * fps)
    takes = []
    for person in selected:
        args = ['--project', str(root), '--person-id', str(person['id']), '--first-frame', str(begin),
                '--end-frame', str(finish), '--result', kind, '--correction-preset', applied['preset'],
                '--correction-mode', applied['mode'], '--foot-lock', str(int(applied['foot_lock'])),
                '--hand-lock', str(int(applied['hand_lock'])), '--pelvis-offset', str(applied['pelvis_offset'])]
        if not applied['auto_floor']:
            args += ['--floor', str(applied['floor'])]
        for event in run_stage('person', args, env, console):
            if event.get('type') == 'person_prepared':
                takes += [(Path(path), person) for path in event['projects']]
    inference = [s for s in stages if s not in ('correct', 'export')]
    post = [s for s in stages if s in ('correct', 'export')]
    jobs = grouped_jobs([(s, ['--project', str(path), '--result', kind]) for path, _ in takes for s in inference], root)
    jobs += [(s, ['--project', str(path), '--result', kind]) for path, _ in takes for s in post]
    for stage, arguments in jobs:
        run_stage(stage, arguments, env, console)
    return root, kind, takes


def apply_correction(project, correction, env, console):
    """Apply Correction (gui.py apply_correction; people_ui.apply_scene_correction for a people group).

    `project` is a single take, a take of a people group, or the group folder itself. In a group the
    floor options are shared and every reconstructed take is corrected again, as in the app.
    """
    from .motion_tools import RESULTS, base_result, result_is_current, result_kind, result_sources
    from .people_store import load_people, root_for, safe_child
    from .storage import atomic_json
    project = Path(project).resolve()
    kind = result_kind(correction=True)
    group = load_people(project) if (project / 'people.json').is_file() else None
    root = project if group is not None else root_for(project)
    if root != project or group is not None:
        from .scene_floor import save_config
        save_config(root, correction)
    if group is None:
        path = project / 'project.json'
        state = json.loads(path.read_text(encoding='utf-8'))
        state['correction'] = correction
        atomic_json(path, state)
    if root == project and group is None:
        targets = [project]
    else:
        bases = result_sources(base_result(kind))
        targets = []
        for person in load_people(root)['people']:
            for number, take in enumerate(person.get('takes', []), 1):
                path = safe_child(root, take['project'])
                if all((path / RESULTS[base][0]).is_file() and result_is_current(path, base) for base in bases):
                    targets.append(path)
                elif person.get('checked'):
                    console.say(f"{person['name']}, Take {number}: skipped, its base motion is unavailable. Reconstruct it first.")
        if not targets:
            raise StageFailed('Reconstruct a person before applying the shared scene floor.')
    for path in targets:
        for stage in ('correct', 'export'):
            run_stage(stage, ['--project', str(path), '--result', kind], env, console)
    return kind


def take_context(project):
    """(people root, person) for a take of a people group, else (None, None)."""
    from .people_store import load_people, root_for, safe_child
    project = Path(project).resolve()
    root = root_for(project)
    if root == project:
        return None, None
    state = load_people(root)
    for person in (state or {}).get('people', []):
        if any(safe_child(root, take['project']) == project for take in person.get('takes', [])):
            return root, person
    raise ValueError('This take is not listed in its people group. Open the project in the app.')


def people_scene_stale(root):
    """The checks scene_data.SceneDataLoader.load makes before it prepares the shared scene cameras."""
    from .paths import MODELS
    from .people_store import load_people, media_receipt, safe_child
    from .scene_camera import VERSION, read_camera, take_projects
    from .scene_floor import current_resolution, read_config
    state = load_people(root)
    projects = take_projects(root, state) if state else []
    if not projects:
        return False
    path = root / 'people_scene.json'
    smplx = (MODELS / 'body_models/smplx/SMPLX_NEUTRAL.npz').exists()
    if (read_config(root) is not None and current_resolution(root) is None and smplx) or not path.exists() \
            or any(read_camera(project) is None for _, project in projects):
        return True
    reference = json.loads(path.read_text(encoding='utf-8'))
    if reference.get('version') != VERSION or reference.get('input') != media_receipt(root):
        return True
    anchor = read_camera(safe_child(root, reference['anchor']))
    return anchor is None or anchor['sources'] != reference.get('sources')


def prepare_people_scene(root, console, timeout=300):
    """Write people_scene.json and the scene cameras the way the GUI does when it opens a people scene.

    Motion Correction writes them itself; without it, only the GUI's scene loader did, and a people
    export needs them (scene_data.prepare_camera_process, same command and log file).
    """
    from .paths import ROOT
    root = Path(root).resolve()
    if not people_scene_stale(root):
        return
    console.say('Prepare shared scene cameras...')
    if getattr(sys, 'frozen', False) or os.environ.get('RGB_MOCAP_MANAGED') == '1':
        command = [str(ROOT / 'runtime/python.exe'), '-I', '-X', 'utf8', str(ROOT / 'scene_boot.py'), str(root)]
    else:
        command = [sys.executable, '-X', 'utf8', '-m', 'mocap.scene_camera', str(root)]
    log = root / '.scene_camera.log'
    with log.open('wb') as output:
        try:
            code = subprocess.run(command, stdout=output, stderr=output, cwd=str(ROOT), timeout=timeout,
                                  creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).returncode
        except subprocess.TimeoutExpired:
            raise StageFailed('Shared scene camera preparation timed out; retry after freeing RAM.') from None
    if code:
        tail = log.read_bytes()[-4096:].decode('utf-8', errors='replace')
        raise StageFailed('Cannot prepare shared cameras: ' + tail)
    if not (root / 'people_scene.json').is_file():
        raise StageFailed('Shared scene cameras are still unavailable after preparation; see ' + str(log))


def export_fbx(project, kind, target, options, env, console, *, person=None, people_root=None):
    """File > Export, FBX, one take (gui.py submit_fbx_exports)."""
    from .motion_tools import selected_export
    from .single_actor import preferences
    from .storage import atomic_json
    project = Path(project).resolve()
    target = Path(target).resolve()
    if not target.suffix:
        target = target.with_suffix('.fbx')
    source = selected_export(project, kind)
    if source is None:
        run_stage('rig', ['--project', str(project), '--result', kind], env, console)
        source = selected_export(project, kind)
    if source is None:
        raise StageFailed('The selected motion is unavailable or out of date. Reconstruct or apply correction first.')
    row = dict(project=str(project), kind=kind, source=str(source), target=str(target),
               rig_name=person['name'] if person else preferences(project)['name'],
               prepend_tpose=options.get('prepend_tpose', False),
               export_face=False, face_format='txt', smooth_face=False, bvh_profile='generic')
    if options['fbx_profile'] == 'ue5_mannequin':
        row.update(fbx_profile='ue5_mannequin', ue5_target=options['ue5_target'],
                   ue5_in_place=options['ue5_in_place'], ue5_include_mesh=options['ue5_include_mesh'],
                   prepend_tpose=False, place_at_origin=False, symmetric_rest=False)
    else:
        row.update(fbx_profile='generic', fbx_compatibility=options['fbx_compatibility'],
                   fbx_rest_pose=options['fbx_rest_pose'], place_at_origin=options['place_at_origin'],
                   symmetric_rest=options['symmetric_rest'])
        if not row['fbx_compatibility']:
            row.pop('fbx_compatibility')
    row.update(origin_group='person:' + str(person['id']) if person else str(project),
               animate_camera=True, background_refine=False, camera_translation_scale=1.)
    if people_root is not None:
        from .scene_export import context
        prepare_people_scene(people_root, console)
        row['scene'] = context(people_root, project)
    request = project / ('.fbx-export-' + uuid.uuid4().hex + '.json')
    atomic_json(request, [row])
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_stage('save_fbx', ['--project', str(project), '--output', str(request)], env, console)
    finally:
        request.unlink(missing_ok=True)
        staging = target.with_name('.' + target.name + '.rgbmocap-' + request.stem + '.partial')
        if staging.is_dir() and (staging / '.rgbmocap-owner').is_file():
            import shutil
            shutil.rmtree(staging, ignore_errors=True)
    return target


def correction_from(args):
    from .correction_settings import correction_options
    return correction_options(dict(auto_floor=args.floor is None, floor=args.floor or 0., preset=args.preset,
                                   pelvis_offset=args.pelvis_offset, mode=args.mode,
                                   foot_lock=args.foot_lock, hand_lock=args.hand_lock))


def fbx_options(args):
    return dict(fbx_profile=args.fbx_profile, ue5_target=args.ue5_target, ue5_in_place=args.in_place,
                ue5_include_mesh=args.include_mesh, fbx_compatibility=args.fbx_compatibility,
                fbx_rest_pose=args.rest_pose, place_at_origin=args.place_at_origin,
                symmetric_rest=args.symmetric_rest, prepend_tpose=args.prepend_tpose)


def add_correction_options(parser, with_toggle=True):
    from .correction_settings import PRESETS, MODES
    group = parser.add_argument_group('Motion Correction (GUI defaults)')
    if with_toggle:
        group.add_argument('--no-correction', action='store_true', help='Keep the initial motion (Enable Correction off).')
    group.add_argument('--preset', choices=list(PRESETS), default='normal')
    group.add_argument('--mode', choices=list(MODES), default='flat', help='flat = Flat Ground, multi_level = Multi Level')
    group.add_argument('--foot-lock', type=int, choices=(0, 1), default=1)
    group.add_argument('--hand-lock', type=int, choices=(0, 1), default=0)
    group.add_argument('--floor', type=float, help='Manual floor height (world Y, m); omitted = Auto Floor Height')
    group.add_argument('--pelvis-offset', type=float, default=0.)


def add_export_options(parser, required=False):
    group = parser.add_argument_group('FBX export (File > Export)')
    group.add_argument('--fbx', type=Path, required=required, help='Write the motion to this FBX file.')
    group.add_argument('--fbx-profile', choices=('generic', 'ue5_mannequin'), default='generic')
    group.add_argument('--ue5-target', choices=('manny', 'quinn'), default='manny')
    group.add_argument('--in-place', action='store_true', help='UE5 Mannequin: remove planar root travel.')
    group.add_argument('--include-mesh', action='store_true', help='UE5 Mannequin: include the mannequin mesh.')
    group.add_argument('--rest-pose', choices=('t', 'a'), default='t', help='Generic FBX reference pose.')
    group.add_argument('--fbx-compatibility', action='store_true')
    group.add_argument('--place-at-origin', action='store_true')
    group.add_argument('--symmetric-rest', action='store_true')
    group.add_argument('--prepend-tpose', action='store_true')


def people_ids(text):
    try:
        ids = sorted({int(part) for part in text.split(',') if part.strip()})
    except ValueError:
        raise argparse.ArgumentTypeError('Use comma-separated person ids, for example 1,2.') from None
    if not ids or ids[0] < 1:
        raise argparse.ArgumentTypeError('Person ids start at 1.')
    return ids


def fbx_targets(target, takes):
    """One take keeps the given name; several get the person name and take range appended."""
    from .person_names import export_stem
    target = Path(target)
    if len(takes) == 1:
        return [target]
    stem, suffix = target.stem, target.suffix or '.fbx'
    return [target.with_name(f'{stem}_{export_stem(person["name"])}_{path.name}{suffix}') for path, person in takes]


def add_global_options(parser, defaults=True):
    """Accepted before or after the command; a command's copy never overwrites a value given before it."""
    value = (lambda default: default) if defaults else (lambda default: argparse.SUPPRESS)
    parser.add_argument('--compute', default=value('auto'), help='Resource Mode, as in the app (default: auto = Automatic).')
    parser.add_argument('--log', type=Path, default=value(None), help='Append worker output to this file.')
    parser.add_argument('--quiet', action='store_true', default=value(False))
    parser.add_argument('--json', action='store_true', default=value(False), help='Print a JSON summary on stdout.')


def parser():
    root = argparse.ArgumentParser(prog='RGBMoCapCLI', description=__doc__.split('\n\n')[0])
    add_global_options(root)
    commands = root.add_subparsers(dest='command', required=True)

    rec = commands.add_parser('reconstruct', help='Video -> ExMR motion (Reconstruct 3D), optionally to FBX.')
    rec.add_argument('video', type=Path)
    rec.add_argument('--start', type=float, help='Trim start, seconds.')
    rec.add_argument('--end', type=float, help='Trim end, seconds.')
    rec.add_argument('--project', type=Path, help='Existing workspace folder to reuse (as reopening it in the app).')
    rec.add_argument('--moving-camera', action='store_true', help='ExMR Camera = Moving Camera (SimpleVO).')
    rec.add_argument('--no-smoothing', action='store_true', help='ExMR Smooth Motion off.')
    rec.add_argument('--find-person', action='store_true',
                     help='Run Find person (Default tracker) first and reconstruct the checked people.')
    rec.add_argument('--people', type=people_ids, metavar='IDS',
                     help='With --find-person: comma-separated person ids to reconstruct (default: as checked, Person 1).')
    add_correction_options(rec)
    add_export_options(rec)

    cor = commands.add_parser('correct', help='Apply Correction to an existing ExMR take.')
    cor.add_argument('project', type=Path, help='Model project folder (the one holding input.mp4).')
    add_correction_options(cor, with_toggle=False)
    add_export_options(cor)

    exp = commands.add_parser('export', help='Export an existing take to FBX.')
    exp.add_argument('project', type=Path)
    exp.add_argument('--result', default=None, help='Motion kind (default: the project pipeline result).')
    add_export_options(exp, required=True)
    for command in (rec, cor, exp):
        add_global_options(command, defaults=False)
    root.commands = dict(reconstruct=rec, correct=cor, export=exp)
    return root


def starting_settings():
    """Saved starting settings for new videos (Defaults... in the app), when present and valid.

    The GUI applies them at Open Video; options given on the command line still win.
    """
    from .correction_settings import PRESETS, MODES
    from .local_choices import saved
    start = saved('default_settings')
    if not isinstance(start, dict):
        return {}
    values = {}
    exmr = start.get('exmr_options') if isinstance(start.get('exmr_options'), dict) else {}
    if type(exmr.get('static_camera')) is bool:
        values['moving_camera'] = not exmr['static_camera']
    smoothing = exmr.get('smoothing')
    if type(smoothing) is bool or smoothing in ('global', 'none'):
        values['no_smoothing'] = smoothing in (False, 'none')
    correction = start.get('correction') if isinstance(start.get('correction'), dict) else {}
    if correction.get('mode') in MODES:
        values['mode'] = correction['mode']
    if correction.get('preset') in PRESETS:
        values['preset'] = correction['preset']
    for key in ('foot_lock', 'hand_lock'):
        if type(correction.get(key)) is bool:
            values[key] = int(correction[key])
    processing = start.get('processing_options') if isinstance(start.get('processing_options'), dict) else {}
    if isinstance(processing.get('resource_mode'), str):
        values['compute'] = processing['resource_mode']
    return values


def main(argv=None):
    cli = parser()
    defaults = starting_settings()
    if 'compute' in defaults:
        cli.set_defaults(compute=defaults.pop('compute'))
    for command in cli.commands.values():
        command.set_defaults(**defaults)
    args = cli.parse_args(argv)
    console = Console(args.quiet, args.log)
    summary = {}
    try:
        takes, people_root = None, None
        if args.command == 'reconstruct':
            if args.people and not args.find_person:
                raise ValueError('--people needs --find-person.')
            exmr = dict(static_camera=not args.moving_camera, smoothing='none' if args.no_smoothing else 'global')
            env = worker_environment(args.compute, exmr_options=exmr)
            common = dict(start=args.start, end=args.end, project=args.project, correction=correction_from(args),
                          corrected=not args.no_correction, env=env, console=console)
            if args.find_person:
                people_root, kind, takes = reconstruct_people(args.video, exmr_options=exmr, people=args.people, **common)
                project = people_root
            else:
                project, kind = reconstruct(args.video, **common)
        elif args.command == 'correct':
            env = worker_environment(args.compute)
            project = args.project.resolve()
            kind = apply_correction(project, correction_from(args), env, console)
        else:
            env = worker_environment(args.compute)
            project = args.project.resolve()
            if (project / 'people.json').is_file():
                from .people_store import load_people
                listed = [take['project'] for person in (load_people(project) or {}).get('people', [])
                          for take in person.get('takes', [])]
                raise ValueError('This is a people group folder; pass one of its takes:\n  ' +
                                 '\n  '.join(str(project / t) for t in listed or ['(no reconstructed takes yet)']))
            state = json.loads((project / 'project.json').read_text(encoding='utf-8'))
            kind = args.result or state.get('pipeline', {}).get('result') or 'initial'
        summary.update(project=str(project), result=kind)
        if takes is not None:
            summary['takes'] = [dict(project=str(path), person=person['id'], name=person['name']) for path, person in takes]
        if args.fbx:
            if takes is None:
                people_root, person = take_context(project)
                summary['fbx'] = [str(export_fbx(project, kind, args.fbx, fbx_options(args), env, console,
                                                 person=person, people_root=people_root))]
            else:
                summary['fbx'] = [str(export_fbx(path, kind, target, fbx_options(args), env, console,
                                                 person=person, people_root=people_root))
                                  for (path, person), target in zip(takes, fbx_targets(args.fbx, takes))]
    except (StageFailed, OSError, ValueError, KeyError) as error:
        console.say('Error: ' + str(error))
        if args.json:
            print(json.dumps(dict(summary, ok=False, error=str(error))))
        return 1
    except KeyboardInterrupt:
        console.say('Cancelled.')
        return 130
    console.say('Project: ' + summary['project'])
    for path in summary.get('fbx', []):
        console.say('FBX: ' + path)
    if args.json:
        print(json.dumps(dict(summary, ok=True)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
