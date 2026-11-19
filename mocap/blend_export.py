"""Portable Blender source animation, video and camera bundles."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .blend_support import Inputs, Bundle, read, copy_video, build, disk_check


class Take:
    def __init__(self, row):
        self.row = row
        self.project = Path(row['project']).resolve()
        self.kind = row['kind']
        self.source = Path(row['source']).resolve()
        self.scene = row.get('scene')
        self.inputs = Inputs()
        self.check_selection()
        self.state = read(self.inputs.track(self.project / 'project.json'))
        info = read(self.inputs.track(self.project / (self.kind + '_mixamo.json')))
        for key in ('asset_file', 'frames_file'):
            if Path(info[key]).name != info[key]:
                raise ValueError('Invalid committed animation path')
            self.inputs.track(self.project / info[key])
        self.inputs.track(self.source)
        self.frames = int(info['frames'])
        self.fps = float(info['fps'])
        self.start = float(self.state['start'])
        if (self.frames < 1 or not np.isfinite(self.fps) or self.fps <= 0 or not np.isfinite(self.start)
                or self.frames != int(self.state['frames']) or abs(self.fps - self.state['fps']) > 1e-6):
            raise ValueError('Video and selected animation timing differ')

    def check_selection(self):
        from .motion_tools import selected_export
        from .scene_export import context

        if selected_export(self.project, self.kind) != self.source:
            raise ValueError('Selected motion changed. Prepare Preview or Apply Correction first.')
        if self.scene is not None and context(self.scene['root'], self.project) != self.scene:
            raise ValueError('Shared scene placement changed')

    def verify(self):
        self.inputs.verify()
        self.check_selection()


def prepare_take(take, bundle, emit, camera_cache, videos, *, index=None, first=0, prepend=False):
    from .fbx_export import run as export_fbx
    from .camera_tracks import load, at
    from .camera_background_worker import refined, current
    from .storage import sha256
    from .video import probe

    row, work = take.row, bundle.work
    name_suffix = '' if index is None else f'-{index:04d}'
    group = take.scene['root'] if take.scene else None
    cameras = load(take.project, group)
    animate = row.get('animate_camera', True)
    if row.get('background_refine', False) and animate:
        cameras = refined(cameras, take.project, lambda: False,
                          lambda done, total: emit('progress', stage='save_blend', current=done, total=total,
                                                   message='Refining camera rotation'), camera_cache)
    for item in camera_cache.values():
        if not current(item['sources']):
            raise ValueError('Background camera inputs changed during export')
        for path, _ in item['sources']:
            take.inputs.track(path)
    if not current(list(cameras['camera_sources'].items())):
        raise ValueError('Camera sources changed during export')
    for path in cameras['camera_sources']:
        take.inputs.track(path)

    camera = cameras['cameras'][0]
    if abs(camera['K'][0, 1]) > 1e-6:
        raise ValueError('Blender export requires camera intrinsics without skew')
    scale = float(row.get('camera_translation_scale', 1.))
    if not np.isfinite(scale) or not 0 <= scale <= 100:
        raise ValueError('Invalid camera movement scale')
    times = take.start + np.arange(take.frames) / take.fps
    track = [at(camera, time, scale if animate else 0.) for time in times]
    if any(matrix is None for matrix in track):
        raise ValueError('Saved camera does not cover the selected motion')
    track = np.asarray(track)
    if not animate:
        track[:] = track[0]

    video = take.inputs.track(take.project / 'input.mp4')
    media = probe(video)
    if (media['frames'] != take.frames or abs(media['fps'] - take.fps) > .001
            or [media['width'], media['height']] != list(camera['size'])):
        raise ValueError('Source video timing or resolution differs from the saved camera')
    video_key = sha256(video) if index is not None else None
    video_name = videos.get(video_key)
    disk_check(bundle.folder.parent, (video.stat().st_size if video_name is None else 0) + take.source.stat().st_size * 4)
    if video_name is None:
        video_name = f'video{name_suffix}.mp4'
        copy_video(video, work / video_name, emit, 'save_blend')
        videos[video_key] = video_name

    fbx_name = f'animation{name_suffix}.fbx'
    nested = work / 'fbx-request.json'
    nested.write_text(json.dumps([{**row, 'target': str(work / fbx_name), 'place_at_origin': False,
                                  'symmetric_rest': False, 'prepend_tpose': prepend}]), encoding='utf-8')
    export_fbx(nested, emit)

    moving = animate and not np.array_equal(track, np.broadcast_to(track[0], track.shape))
    camera_track = f'camera{name_suffix}.npy' if moving else None
    if camera_track:
        np.save(work / camera_track, track, allow_pickle=False)
    job = dict(fbx=fbx_name, video=video_name, first=first, frames=take.frames, fps=take.fps,
               width=media['width'], height=media['height'], world_to_camera=track[0].tolist(),
               K=camera['K'].tolist(), prepend_tpose=prepend, camera_track=camera_track,
               camera_mode=cameras['note'])
    if index is not None:
        # The same take interval can appear under several people. Reuse its camera.
        job['camera_key'] = hashlib.sha256(track.tobytes()).hexdigest()
    report = dict(model=cameras['model'], result=take.kind, frames=take.frames, fps=take.fps,
                  video_start_frame=first + int(prepend), camera_mode=cameras['note'],
                  camera_warning=cameras.get('warning', ''), animate_camera=bool(animate),
                  camera_movement_scale=scale, background_refinement=cameras.get('background_report'),
                  camera_translation='Approximate display estimate, not measured metric camera motion')
    return job, report


def build_bundle(bundle, job, report, emit):
    work = bundle.work
    (work / 'blend-job.json').write_text(json.dumps({**job, 'target': bundle.target.name}), encoding='utf-8')
    emit('progress', stage='save_blend', current=0, total=0, message='Building Blender project')
    build(work, 'blend_export_scene.py', emit, 'save_blend')
    (work / 'camera_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    video_help = ('Keep the video files next to the .blend file when moving this folder.\n'
                  'All selected takes use the same timeline. Camera markers switch the reference video.\n'
                  if 'takes' in job else 'Keep video.mp4 beside the .blend file when moving this folder.\n')
    (work / 'README.txt').write_text(
        f'Open {bundle.target.name} in Blender 4.5 or newer.\n' + video_help
        + 'Numpad 0: camera view. Space: playback. The video is a viewport background.\n'
        'The source armatures are editable. Camera motion can include reconstruction error.\n'
        'See camera_report.json for the exported camera settings.\n', encoding='utf-8')
    for take in job.get('takes', [job]):
        (work / take['fbx']).unlink()
        if take['camera_track']:
            (work / take['camera_track']).unlink()
    for name in ('blend-job.json', 'blender.log'):
        (work / name).unlink(missing_ok=True)


def combined_job(takes, bundle, emit, camera_cache):
    takes.sort(key=lambda take: take.start)
    origin, fps = takes[0].start, takes[0].fps
    roots = {take.scene['root'] if take.scene else None for take in takes}
    if len(roots) > 1:
        raise ValueError('One Blender file requires takes from the same people scene.')
    if any(abs(take.fps - fps) > 1e-6 for take in takes):
        raise ValueError('One Blender file requires takes with the same frame rate.')
    lead = bool(takes[0].row.get('prepend_tpose'))
    offsets = [round((take.start - origin) * fps) for take in takes]
    end = max(offset + take.frames for offset, take in zip(offsets, takes)) + int(lead)
    if end - 1 > 1048574:
        raise ValueError('The combined timeline exceeds Blender’s frame range.')
    jobs, reports, videos = [], [], {}
    for index, (offset, take) in enumerate(zip(offsets, takes), 1):
        prepend = lead and offset == 0
        first = offset + int(lead and not prepend)
        job, report = prepare_take(take, bundle, emit, camera_cache, videos,
                                   index=index, first=first, prepend=prepend)
        job['name'] = take.row.get('rig_name', 'Person')
        jobs.append(job)
        reports.append(dict(name=job['name'], video=job['video'], **report))
    return dict(fps=fps, frame_end=end - 1, takes=jobs), dict(source_start=origin, fps=fps, takes=reports)


def run(request, emit=lambda *a, **k: None):
    from .gemx_result import is_project

    request = Path(request)
    bundles, take_checks, camera_cache = [], [], {}
    try:
        payload = read(request)
        combined = isinstance(payload, dict)
        rows = payload.get('takes') if combined else payload
        if not isinstance(rows, list) or not 1 <= len(rows) <= 4096:
            raise ValueError('Invalid Blender export request')
        if combined and any(row['target'] != payload['target'] for row in rows):
            raise ValueError('One Blender file requires a single export destination.')
        if any(is_project(row['project']) for row in rows):
            if len(rows) != 1:
                raise ValueError('Export one native SOMA result at a time.')
            from .gemx_blend import run as native_export
            return native_export(rows[0], request, emit)
        if combined and len(rows) > 1:
            take_checks = [Take(row) for row in rows]
            bundle = Bundle(payload['target'], request)
            bundles.append(bundle)
            job, report = combined_job(take_checks, bundle, emit, camera_cache)
            build_bundle(bundle, job, report, emit)
        else:
            folders = {str(Path(row['target']).resolve().parent).casefold() for row in rows}
            if len(folders) != len(rows):
                raise ValueError('Duplicate Blender bundle destination')
            for row in rows:
                take = Take(row)
                bundle = Bundle(row['target'], request)
                bundles.append(bundle)
                job, report = prepare_take(take, bundle, emit, camera_cache, {}, prepend=bool(row.get('prepend_tpose')))
                build_bundle(bundle, job, report, emit)
                take_checks.append(take)
        for take in take_checks:
            take.verify()
        for bundle in bundles:
            bundle.publish()
        return len(bundles)
    finally:
        for bundle in bundles:
            bundle.close()
        request.unlink(missing_ok=True)
