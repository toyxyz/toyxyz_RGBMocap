"""Project, model, face and batch commands for the Windows CLI."""
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
import json

from .cli_runner import run_worker
from .paths import HOME
from .storage import atomic_json


@contextmanager
def request_file(data):
    folder = HOME / 'cache'
    folder.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='cli-', dir=folder) as temporary:
        path = Path(temporary) / 'request.json'
        atomic_json(path, data)
        yield path, Path(temporary)


def model_command(args, report):
    from . import model_manager as manager
    from .models import BY_KEY, MODELS_LIST, ready, GEMX_KEYS
    from .cli_workflows import environment

    def can_auto_download(model):
        return bool(model.drive_id or model.download_url or model.key in GEMX_KEYS)
    catalog = manager.workflow_catalog()
    consumers = manager.model_consumers(catalog)
    if args.action == 'list':
        rows = []
        for model in MODELS_LIST:
            if args.workflow and model.key not in catalog[args.workflow]:
                continue
            ready_now = ready(model)
            if not args.installed or ready_now:
                rows.append(dict(key=model.key, name=model.name, ready=ready_now,
                                 used_by=consumers[model.key], automatically_downloadable=can_auto_download(model)))
        return {'models': rows}
    if args.action == 'remove':
        unknown = set(args.keys) - BY_KEY.keys()
        if unknown:
            raise ValueError('Unknown model keys: ' + ', '.join(sorted(unknown)))
        if not args.dry_run:
            if not args.yes:
                raise ValueError('Pass --yes to remove files. Use --dry-run to review them.')
            manager.recover_removal()
        plan = manager.removal_plan(args.keys)
        summary = dict(models=[dict(key=key, used_by=consumers[key]) for key in args.keys],
                       files=[str(item.path) for item in plan], bytes=manager.physical_bytes(plan, reclaim=True))
        if args.dry_run:
            return {'dry_run': True, **summary}
        manager.remove_models(args.keys, plan)
        return summary
    model = BY_KEY.get(args.key)
    if model is None:
        raise ValueError('Unknown model key: ' + args.key)
    flags = ['--model', args.key]
    if args.action == 'download' and not can_auto_download(model):
        raise ValueError('This model requires a manual download. Import it with models import. ' + model.url)
    if args.action == 'import':
        if ready(model) and not args.replace:
            raise ValueError('The model is already ready. Pass --replace to import another copy.')
        flags += ['--video', str(args.file.resolve(strict=True))]
    if ready(model) and args.action == 'download':
        return {'key': model.key, 'path': str(model.path), 'cached': True}
    run_worker('download' if args.action == 'download' else 'import_model', flags, environment('auto'), report)
    if not ready(model):
        raise RuntimeError('The model did not pass readiness validation.')
    return {'key': model.key, 'path': str(model.path)}


def face_command(args, report):
    from .cli_workflows import environment, model_project
    from .face_data import FaceArchive, root_for
    from .face_neutral import neutral_frames, reset_neutral, set_neutral
    from .people_store import load_people

    root = root_for(model_project(args.project, args.model))
    if args.action == 'track':
        group = load_people(root)
        available = {p['id'] for p in group['people']} if group else {1}
        ids = sorted(set(args.person) if args.person else
                     ({p['id'] for p in group['people'] if p.get('checked')} if group else {1}))
        if not ids or not set(ids) <= available:
            raise ValueError('Choose a saved person ID from project inspect.')
        flags = ['--project', str(root)]
        for ident in ids:
            flags += ['--person', str(ident)]
        run_worker('face_track', flags, environment('auto'), report)
        archive = FaceArchive(root)
        return {'project': str(root), 'people': list(archive.ids)}
    archive = FaceArchive(root)
    if args.action == 'neutral':
        if args.frame < 1:
            raise ValueError('Capture frame numbers start at 1.')
        set_neutral(archive, args.person, args.frame - 1)
    elif args.action == 'reset-neutral':
        reset_neutral(root, args.person)
    coverage = [dict(person=person, tracked_frames=int(archive.valid[slot].sum()),
                     total_frames=archive.valid.shape[1]) for person, slot in archive.index.items()]
    return {'project': str(root), 'people': list(archive.ids),
            'coverage': coverage,
            'neutral_frames': {person: frame+1 for person, frame in neutral_frames(archive).items()}}


def copy_project(args, report):
    from . import project_workspace as workspace
    from .cli_workflows import environment, manifest, model_project, project_directory
    from .project_save import validate_destination
    from .project_storage import storage_root

    original = project_directory(args.project)
    if workspace.is_workspace(original):
        state = workspace.read(original)
        model = args.model or state['active_model']
        source, active = original, workspace.model_project(original, model, state)
        if not (active / 'project.json').is_file():
            active = None
    else:
        state = manifest(original)
        model = args.model or state.get('pipeline', {}).get('model', 'gvhmr')
        active = model_project(original, model)
        source = workspace.root_for(active, state) or storage_root(active, state) or active
    target = validate_destination(source, args.output)
    if args.dry_run:
        return {'dry_run': True, 'project': str(source), 'output': str(target)}
    with request_file(dict(source=str(source), active=str(active) if active is not None else None,
                           target=str(target), model=model)) as (path, _):
        run_worker('save_project', ['--request', str(path)], environment('auto'), report)
    return {'project': str(source), 'output': str(target)}


def batch(args, report):
    from .cli import parser
    from .cli_workflows import execute

    with args.file.open('rb') as source:
        raw = source.read(1024*1024+1)
    if len(raw) > 1024*1024:
        raise ValueError('The batch file exceeds 1 MiB.')
    data = json.loads(raw.decode('utf-8-sig'))
    jobs = data.get('jobs') if isinstance(data, dict) and data.get('version') == 1 else None
    if not isinstance(jobs,list) or not 1 <= len(jobs) <= 1024:
        raise ValueError('Use a version 1 batch file with 1–1024 jobs.')
    argument_parser = parser()
    queue = []
    for index, argv in enumerate(jobs, 1):
        if not isinstance(argv,list) or not argv or argv[0] not in ('reconstruct','correct','export') or not all(isinstance(v,str) for v in argv):
            raise ValueError(f'Batch job {index} must contain arguments for reconstruct, correct or export.')
        try:
            job = argument_parser.parse_args(argv)
        except SystemExit as error:
            raise ValueError(f'Invalid arguments in batch job {index}.') from error
        queue.append(job)

    results, prepared = [], []
    for index, job in enumerate(queue, 1):
        if args.dry_run:
            job.dry_run = True
        try:
            report(f'Batch job {index}/{len(queue)} · {job.command}')
            prior = None
            if job.command == 'export' and job.dry_run:
                project = job.project.resolve()
                prior = next((i for i, model, path in reversed(prepared) if model == job.model
                              and (project.is_relative_to(path) or path.is_relative_to(project))), None)
            if prior is not None:
                result = dict(dry_run=True, deferred=True, depends_on_job=prior,
                              project=str(job.project), output=str(job.output or job.output_dir))
            else:
                result = execute(job, report)
            candidate = result.get('project')
            if job.dry_run and job.command in ('reconstruct','correct') and candidate and not result.get('missing_models'):
                from .project_workspace import root_for
                path = Path(candidate).resolve()
                if (path/'project.json').is_file():
                    path = root_for(path) or path
                prepared.append((index, job.model, path))
            failed = bool(result.get('failed'))
            results.append(dict(index=index, ok=not failed, **result))
            if failed and not args.keep_going:
                break
        except (OSError, ValueError, RuntimeError) as error:
            results.append(dict(index=index, ok=False, error=str(error)))
            if not args.keep_going:
                break
    return {'jobs': results, 'failed': any(not job['ok'] for job in results)}


def contacts(args, report):
    from .cli_workflows import manifest, select_takes, selected_kind
    from .foot_contacts import read_edit, save_edit
    from .motion_tools import RESULTS, base_result, result_is_current
    from .result_artifacts import file_stat

    _, _, rows = select_takes(args)
    if len(rows) != 1:
        raise ValueError('Choose one saved take for contact editing.')
    _, path, _ = rows[0]
    base = base_result(selected_kind(path, args.result))
    state = manifest(path)
    if not result_is_current(path, base):
        raise ValueError('Reconstruct the base motion before editing contacts.')
    if args.model == 'gemx':
        from .gemx_correct import context
        receipt = context(path, state)
    else:
        names = dict.fromkeys((RESULTS[base][0], 'hmr4d_results.pt'))
        receipt = dict(frames=state['frames'], fps=state['fps'], sources={name:file_stat(path/name) for name in names})
    if args.action == 'list':
        ranges, revision, warning = read_edit(path, base, receipt)
        if args.model == 'gemx':
            from .gemx_correct import contact_track, selected
            selection = selected(path, ignore_contacts=True, ignore_options=True)
            track = contact_track(path, selection[1], state) if selection else None
        else:
            from .preview_data import load_contact_track
            track = load_contact_track(path, base)
        return {'project': str(path), 'ranges': ranges, 'revision': revision, 'warning': warning,
                'automatic': track['automatic'] if track else None,
                'frames': receipt['frames'], 'fps': receipt['fps']}
    ranges = None
    if args.action == 'set':
        with args.ranges.open('rb') as source:
            raw = source.read(1024*1024+1)
        if len(raw) > 1024*1024:
            raise ValueError('The contact edit file exceeds 1 MiB.')
        ranges = json.loads(raw.decode('utf-8-sig'))
        if not isinstance(ranges,list) or any(not isinstance(track,list) or
                any(not isinstance(pair,list) for pair in track) for track in ranges):
            raise ValueError('Use two or four arrays of zero-based [start,end) ranges.')
    record = save_edit(path, base, ranges, receipt)
    return {'project': str(path), 'revision': record['revision'], 'apply_correction': True}


def prepare_preview(args, report):
    from .cli_workflows import environment, prepare_people_scene, select_takes, selected_kind
    from .flow_placement import needs_update
    from .motion_tools import base_result, result_is_current, selected_export

    root, group, rows = select_takes(args)
    jobs = []
    for _, path, _ in rows:
        kind = selected_kind(path, args.result)
        placement_old = args.model == 'flowhmr' and needs_update(path)
        if not placement_old and not result_is_current(path, kind):
            raise ValueError(f'{path}: reconstruct or apply correction to prepare the chosen motion.')
        if args.model != 'gemx' and (placement_old or selected_export(path, kind) is None):
            jobs.append((path, kind))
    env = environment('auto')
    if not args.dry_run:
        for path, kind in jobs:
            run_worker('rig', ['--project', str(path), '--result', kind], env, report)
        if group and args.model != 'flowhmr':
            prepare_people_scene(root, env, report)
    return {'project': str(root), 'dry_run': args.dry_run, 'rig_jobs': [str(path) for path, _ in jobs]}
