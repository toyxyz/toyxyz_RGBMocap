"""Use existing media workers and stream transports from the command line."""
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
import json
import math
import os
import time
from uuid import uuid4

from .cli_runner import run_worker
from .cli_tools import request_file
from .paths import ROOT


def character(args, report):
    from .cli_workflows import environment, manifest, select_takes, selected_kind
    from .gemx_result import is_project
    from .motion_tools import selected_export
    from .retarget.runtime import blender
    from .scene_export import context
    from .workflow_jobs import publish_output

    _, group, rows = select_takes(args)
    if len(rows) != 1:
        raise ValueError('Select exactly one take for character export.')
    _, take, person = rows[0]
    asset = args.asset.resolve(strict=True)
    target = args.output.resolve()
    fmt = target.suffix.lower().removeprefix('.')
    if not asset.is_file() or asset.suffix.lower() not in ('.fbx','.vrm','.pmx'):
        raise ValueError('Choose an FBX, VRM or PMX character asset.')
    if fmt not in ('fbx','vmd') or target.exists() or not target.parent.is_dir():
        raise ValueError('Choose a new FBX or VMD in an existing directory.')
    if fmt == 'vmd' and asset.suffix.lower() != '.pmx':
        raise ValueError('VMD export requires a PMX character.')
    placement = args.vmd_placement
    root_bone = args.vmd_root_bone
    if fmt != 'vmd':
        if placement is not None:
            raise ValueError('--vmd-placement requires VMD output.')
        if root_bone is not None:
            raise ValueError('--vmd-root-bone requires VMD output.')
    kind = selected_kind(take, args.result)
    if selected_export(take, kind) is None:
        raise ValueError('Prepare the selected source motion before retargeting.')
    if args.dry_run:
        return {'dry_run': True, 'project': str(take), 'output': str(target), 'result': kind}
    blender()
    from .people_store import root_for
    transform = context(root_for(take), take)['transform'] if group else None
    temporary = target.with_name(f'.rgbmocap-{uuid4().hex}.{fmt}')
    env = environment('auto')
    try:
        with request_file({}) as (request, scratch):
            from .storage import atomic_json
            def process(job, extra=()):
                atomic_json(request, job)
                run_worker('character', ['--request', str(request), *extra], env, report)

            imported, animation, work = (scratch / name for name in ('imported','animation','export'))
            process(dict(mode='import', output=str(imported), file=str(asset)))
            source = dict(project=str(take), start=manifest(take)['start'], transform=transform)
            extra = () if is_project(take) else ('--project', str(take))
            process(dict(mode='retarget', output=str(animation), asset=str(imported),
                         result=kind, source_name=person['name'], takes=[source]), extra)
            result = json.loads((animation / 'complete.json').read_text(encoding='utf-8'))
            process(dict(mode='export', output=str(work), animation=str(animation), takes=result['takes'],
                         files=[str(temporary)], source_name=person['name'], format=fmt,
                         include_mesh=not args.no_mesh, vmd_placement=placement or 'original',
                         vmd_root_bone=root_bone or 'root'))
        publish_output(temporary, target, replace=False)
    finally:
        temporary.unlink(missing_ok=True)
    return {'project': str(take), 'output': str(target), 'result': kind}


def preview(args, report):
    from .cli_runner import run_process
    from .cli_workflows import camera_refinement_enabled, check_camera_refinement, environment, model_project
    from .workflow_jobs import publish_output

    check_camera_refinement(args)
    project = model_project(args.project, args.model)
    refine = camera_refinement_enabled(project, args.model, args.refine_camera_rotation)
    show_camera = refine or args.refine_camera_rotation is not None
    target = args.output.resolve()
    if target.suffix.lower() != '.mp4' or target.exists() or not target.parent.is_dir():
        raise ValueError('Choose a new MP4 in an existing directory.')
    if args.frames is not None and args.frames < 1:
        raise ValueError('Frame count must be positive.')
    output = target.with_name(f'.rgbmocap-{uuid4().hex}.mp4')
    env = environment('auto')
    try:
        command = [str(ROOT/'runtime/python.exe'), '-I', '-X', 'utf8', '-u', '-B',
                   str(ROOT/'cli_preview_boot.py'), str(project), str(output)]
        if args.frames is not None:
            command.extend(['--frames', str(args.frames)])
        if show_camera:
            command.extend(['--camera', 'refined' if refine else 'saved'])
        run_process(command, 'Render desktop preview', env, report, env['RGB_MOCAP_RESOURCE_SAFETY']=='1')
        publish_output(output, target, replace=False)
    finally:
        output.unlink(missing_ok=True)
    result = {'project': str(project), 'output': str(target)}
    if show_camera:
        result['refine_camera_rotation'] = refine
    return result


def stream_take(path, result, root, stack):
    import numpy as np
    from . import gemx_result
    from .bone_stream import StreamRig
    from .cli_workflows import manifest
    from .scene_export import context

    if gemx_result.is_project(path):
        rig = StreamRig.soma(gemx_result.load_rig(path))
        if result != 'initial':
            from .gemx_correct import selected
            folder = selected(path)[0]
        else:
            folder = path
        source = folder / gemx_result.POSE
    else:
        receipt = json.loads((path / f'{result}_mixamo.json').read_text(encoding='utf-8'))
        for field in ('asset_file','frames_file'):
            if Path(receipt[field]).name != receipt[field]:
                raise ValueError('Invalid stream artifact name.')
        from .export_options import recorded_auto_floor
        with np.load(path / receipt['asset_file'], allow_pickle=False) as asset:
            rig = StreamRig.mixamo(receipt['names'], asset, recorded_auto_floor(path, result))
        source = path / receipt['frames_file']
    frames = np.load(source, mmap_mode='r', allow_pickle=False)
    stack.callback(frames._mmap.close)
    if frames.ndim != 4 or frames.shape[1:] != (len(rig.bones),4,4):
        raise ValueError('Invalid stream pose array.')
    state = manifest(path)
    transform = np.asarray(context(root,path)['transform']) if root is not None else None
    return state, rig, frames, transform


def stream(args, report):
    from .bone_stream import BoneStream, PORT
    from .unreal_bake import BakeFace, BakeTake
    from .cli_workflows import select_takes, selected_kind
    from .motion_tools import selected_export

    if not math.isfinite(args.wait) or not 0 <= args.wait <= 120:
        raise ValueError('Choose a connection wait of 0–120 seconds.')
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        raise ValueError('--seconds must be positive and finite.')
    if args.port is not None and (args.target != 'blender' or not 1 <= args.port <= 65535):
        raise ValueError('The Blender port must be in the range 1–65535.')
    if args.face and args.target != 'unreal':
        raise ValueError('Face curves are available with the Unreal target.')
    root, group, rows = select_takes(args)
    archive = None
    if args.face:
        from .face_data import FaceArchive
        from .face_neutral import expression_values
        archive = FaceArchive(root)
        values = archive.values.copy()
        for slot in range(len(archive.ids)):
            values[slot] = expression_values(archive, slot, smooth=True)
        values.setflags(write=False)
        archive.values = values
    with ExitStack() as stack:
        takes = []
        for _, path, person in rows:
            kind = selected_kind(path, args.result)
            if selected_export(path, kind) is None:
                raise ValueError('Prepare the selected motion before streaming.')
            state, rig, frames, transform = stream_take(path, kind, root if group else None, stack)
            takes.append((path, person, state, rig, frames, transform))
        first = min(state['start'] for _,_,state,_,_,_ in takes)
        end = max(state['start'] + len(frames)/state['fps'] for _,_,state,_,frames,_ in takes)
        fps = max(state['fps'] for _,_,state,_,_,_ in takes)
        keys = {person['id']: sha256(f'{(root,args.model)!r}/{person["id"]}'.encode('utf-8')).hexdigest()
                for _,person,_,_,_,_ in takes}
        if args.target == 'unreal':
            from .unreal_stream import UnrealStream
            sender = UnrealStream(ROOT)
        else:
            sender = BoneStream(args.port or PORT)
        def stop_sender():
            sender.close()
            sender.thread.join(15)
        stack.callback(stop_sender)
        report(f'{args.target} streaming ready on localhost · {len(keys)} actor(s)')
        started = time.monotonic() + args.wait
        while time.monotonic() < started:
            if error := sender.error():
                raise RuntimeError(str(error))
            time.sleep(.02)
        ticks = 0
        received = sender.connected
        total = max(1, math.ceil((end-first)*fps - 1e-6))
        while (args.loop or ticks < total) and (args.seconds is None or ticks/fps < args.seconds):
            if error := sender.error():
                raise RuntimeError(str(error))
            due = started + ticks/fps
            wait = due-time.monotonic()
            if wait > 0:
                time.sleep(wait)
            now = first+(ticks % total)/fps
            actors, faces, bake_takes = {}, {}, {}
            received |= sender.connected
            for path, person, state, rig, frames, transform in takes:
                frame = math.floor((now-state['start'])*state['fps'] + 1e-6)
                if not 0 <= frame < len(frames):
                    continue
                ident = person['id']
                key = keys[ident]
                actors[key] = (key, person['name'], rig, frame, frames[frame].copy(), transform)
                if args.target == 'unreal':
                    bake_face = (BakeFace(archive, ident, float(state['start']), float(state['fps']))
                                 if archive is not None and ident in archive.index else None)
                    bake_takes[key] = BakeTake(Path(frames.filename), len(frames), state['fps'],
                                               rig, transform, bake_face)
                if archive is not None and ident in archive.index:
                    slot, sample, source = archive.resolve(path, frame, person=ident, state=state)
                    if sample >= 0:
                        faces[key] = dict(values=archive.values[slot,sample].tolist(), sample=sample,
                                          source_frame=source, tracked=bool(archive.valid[slot,source]))
            if args.target == 'unreal':
                sender.publish(tuple(actors.values()), faces=faces, takes=bake_takes)
            else:
                sender.publish(tuple(actors.values()), tuple(keys.values()))
                from queue import Empty
                try:
                    connection, command, actor, rig_key = sender.commands.get_nowait()
                except Empty:
                    pass
                else:
                    if connection == sender.connection and command != ord('A'):
                        sender.bake_reply = connection, bake_plan(actor, rig_key, keys, takes)
                        sender.changed.set()
            ticks += 1
        # Allow the sender to publish its final pose before closing the transport.
        time.sleep(.12)
        received |= sender.connected
    return {'target': args.target, 'frames': ticks, 'connection_seen': received,
            'people': list(keys)}


def reference(args, report):
    from .cli_workflows import environment, select_takes, selected_kind
    from .motion_tools import selected_export
    from .unreal_stream import rig_schema
    from .workflow_jobs import publish_output

    root, group, rows = select_takes(args)
    target = args.output.resolve()
    if (len(rows) != 1 or target.suffix.lower() != '.fbx' or target.exists()
            or target.is_relative_to(root) or not target.parent.is_dir()):
        raise ValueError('Choose one take and a new FBX outside the source project.')
    _, path, _ = rows[0]
    result = selected_kind(path, args.result)
    if selected_export(path, result) is None:
        raise ValueError('Prepare the selected motion before exporting the reference rig.')
    temporary = target.with_name(f'.rgbmocap-{uuid4().hex}.fbx')
    try:
        with ExitStack() as stack:
            _, rig, _, _ = stream_take(path, result, root if group else None, stack)
            with request_file(dict(target=str(temporary), schema=rig_schema(rig))) as (request, _):
                run_worker('stream_reference', ['--output', str(request)], environment('auto'), report)
        publish_output(temporary, target, replace=False)
    finally:
        temporary.unlink(missing_ok=True)
    return {'project': str(path), 'output': str(target)}


def bake_plan(actor, rig_key, keys, takes):
    from .bone_stream import MAX_BAKE_FRAMES
    spans = []
    for _, person, state, rig, frames, transform in takes:
        if keys[person['id']] != actor:
            continue
        if rig.key != rig_key:
            return 'The saved takes use different bind rigs. Choose one take for baking.'
        start, fps = state['start'], state['fps']
        spans.append((start, start+len(frames)/fps, start, fps, frames, rig, transform))
    if not spans:
        return 'Choose a streamed person before baking.'
    spans.sort(key=lambda row: row[0])
    if any(a[1] > b[0]+1e-7 for a,b in zip(spans,spans[1:])):
        return 'The selected takes overlap. Choose one take for baking.'
    starts = tuple(row[0] for row in spans)
    first, end = starts[0], spans[-1][1]
    fps = spans[0][3]
    n = math.ceil((end-first)*fps-1e-6)
    if not 1 <= n <= MAX_BAKE_FRAMES:
        return 'The selected bake is too long. Choose individual takes.'
    return actor, rig_key, fps, first, n, spans, starts, None
