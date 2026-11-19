"""Portable Blender export for one committed native SOMA result."""

import json
from pathlib import Path
import struct

import numpy as np

from .blend_support import Bundle, Inputs, build, copy_video, disk_check
from .gemx_result import FILES, FPS, JOINT_COUNT, MESH, POSE, RECEIPT, VERTEX_COUNT, load_rig, manifest
from .motion_tools import selected_export

STAGE = 'save_blend'


def run(row, request, emit):
    project = Path(row['project']).resolve()
    source = Path(row['source']).resolve()
    kind = row['kind']
    if selected_export(project, kind) != source:
        raise ValueError('The selected native SOMA result changed.')
    inputs = Inputs()
    for name in ('project.json', RECEIPT, *FILES):
        inputs.track(project / name)
    from .gemx_correct import selected
    folder = project if kind == 'initial' else selected(project)[0]
    if folder != project:
        for path in (source, folder / MESH, folder / POSE):inputs.track(path)
    frames = manifest(project)['frames']
    rig = load_rig(project)
    video = project / 'input.mp4'
    disk_check(Path(row['target']).parent.parent, video.stat().st_size + frames * VERTEX_COUNT * 12)
    bundle = Bundle(row['target'], request)
    try:
        work = bundle.work
        copy_video(video, work / 'video.mp4', emit, STAGE)
        surface = np.load(folder / MESH, mmap_mode='r', allow_pickle=False)
        native = np.load(folder / POSE, mmap_mode='r', allow_pickle=False)
        try:
            if surface.shape != (frames, VERTEX_COUNT, 3) or native.shape != (frames, JOINT_COUNT, 4, 4):
                raise ValueError('Invalid SOMA animation dimensions.')
            first = surface[0][:, [0, 2, 1]].copy()
            first[:, 1] *= -1
            np.savez(work / 'rig.npz', **rig, first=first, pose=native)
            with (work / 'surface.pc2').open('xb') as output:
                output.write(struct.pack('<12siiffi', b'POINTCACHE2\0', 1, VERTEX_COUNT, 0., 1., frames))
                for start in range(0, frames, 32):
                    block = surface[start:start + 32][:, :, [0, 2, 1]].copy()
                    block[:, :, 1] *= -1
                    output.write(np.ascontiguousarray(block, dtype='<f4').tobytes())
                    emit('progress', stage=STAGE, current=min(frames, start+32), total=frames,
                         message='Baking the exact SOMA surface')
        finally:
            surface._mmap.close()
            native._mmap.close()
        state = manifest(project)
        (work / 'blend-job.json').write_text(json.dumps(dict(target=bundle.target.name, frames=frames, fps=FPS,
            width=state['width'], height=state['height'])), encoding='utf-8')
        build(work, 'gemx_blender_scene.py', emit, STAGE)
        for name in ('rig.npz', 'blend-job.json', 'blender.log'):
            (work / name).unlink()
        (work / 'README.txt').write_text(
            'Keep this .blend file, surface.pc2 and video.mp4 together.\n'
            'Open the SOMA Motion scene for the editable 77-bone rig and exact baked surface.\n'
            'The surface cache is independent of the rig. Switch to Input Video for the same frame range.\n',
            encoding='utf-8')
        inputs.verify()
        if selected_export(project, kind) != source:
            raise ValueError('Native motion changed during export.')
        bundle.publish()
        return 1
    finally:
        bundle.close()
