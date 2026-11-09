"""Fixed-camera GEM-X reconstruction. The optional worker never loads SMPL."""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

from .gemx_result import ASSET, FILES, FPS, JOINT_COUNT, MESH, PARAMETERS, POSE, RECEIPT, VERTEX_COUNT
from .result_artifacts import file_stat
from .storage import atomic_json, sha256


def emit(kind, **values):
    print('@@MOCAP@@' + json.dumps(dict(type=kind, **values)), flush=True)


def evaluate(engine, parameters, project, torch, np):
    def batch(first, stop, neutral=False):
        args = {}
        for name, value in parameters.items():
            value = value[first:stop]
            if neutral and name in ('global_orient', 'body_pose', 'transl'):
                value = torch.zeros_like(value)
            args[name] = value.to(device='cuda', dtype=torch.float32)
        with torch.inference_mode():
            result = engine.body_model(**args, pose2rot=True, apply_correctives=True, repose_to_bind_pose=True)
        return (result['transforms'][:, 1:].cpu().numpy(), result['vertices'].cpu().numpy())

    rest, neutral = batch(0, 1, neutral=True)
    rest = rest[0]
    layer = engine.body_model.runtime
    names = layer.public_joint_names
    parents = layer.output_joint_parent_ids[1:].cpu().numpy().astype(np.int32) - 1
    if len(names) != JOINT_COUNT + 1 or names[:2] != ('Root', 'Hips'):
        raise ValueError('Unsupported native SOMA hierarchy.')
    if neutral.shape != (1, VERTEX_COUNT, 3) or not np.isfinite(neutral).all():
        raise ValueError('Unsupported SOMA MID geometry.')
    rest[:, 1, 3] -= neutral[0, :, 1].min()
    ends = np.tile(np.array([0., .02, 0.], dtype=np.float32), (JOINT_COUNT, 1))
    for index, parent in enumerate(parents):
        if parent >= 0:
            ends[index] = .35 * (rest[index, :3, 3] - rest[parent, :3, 3])
    np.savez(project / ASSET, names=names[1:], parents=parents, rest=rest,
             faces=engine.faces.numpy(), ends=ends)
    n = len(parameters['global_orient'])
    pose = np.lib.format.open_memmap(project / POSE, 'w+', dtype='float32', shape=(n, JOINT_COUNT, 4, 4))
    mesh = None
    try:
        mesh = np.lib.format.open_memmap(project / MESH, 'w+', dtype='float32', shape=(n, VERTEX_COUNT, 3))
        lowest = np.empty(n, dtype=np.float32)
        for first in range(0, n, 16):
            stop = min(n, first + 16)
            world, vertices = batch(first, stop)
            rotation = world[..., :3, :3]
            if (not np.isfinite(world).all() or not np.isfinite(vertices).all()
                    or not np.allclose(rotation @ rotation.swapaxes(-1, -2), np.eye(3), atol=1e-4)
                    or not np.allclose(np.linalg.det(rotation), 1., atol=1e-4)):
                raise ValueError('GEM-X produced invalid native pose or geometry.')
            pose[first:stop] = world
            mesh[first:stop] = vertices
            lowest[first:stop] = vertices[:, :, 1].min(axis=1)
            emit('progress', current=stop, total=n, message='Evaluating native SOMA bones and surface')
        # GEM-X places the first pelvis at the origin. Use a single estimated
        # floor offset for the entire clip, preserving all relative motion.
        vertical_offset = -float(np.percentile(lowest, 5))
        pose[:, :, 1, 3] += vertical_offset
        for first in range(0, n, 16):
            mesh[first:first + 16, :, 1] += vertical_offset
        pose.flush()
        mesh.flush()
    finally:
        pose._mmap.close()
        if mesh is not None:
            mesh._mmap.close()
    return vertical_offset


def reconstruct(request, cfg):
    from .compute_policy import configure_worker_cpu, gpu_memory_fraction, retry_cuda
    from .memory_mode import effective_mode
    from .models import BY_KEY, missing
    from .paths import HOME
    from .project_workspace import create, media_path, read, set_active
    from .video import executable, probe

    threads = configure_worker_cpu('gemx_reconstruct')
    started = time.monotonic()
    if request.stat().st_size > 65536:
        raise ValueError('GEM-X request is too large.')
    job = json.loads(request.read_text(encoding='utf-8'))
    root = Path(job['root']).resolve()
    video = Path(job['video']).resolve(strict=True)
    start, end = float(job['start']), float(job['end'])
    source = probe(video, include_seek_offset=True)
    seek_offset = source.pop('seek_offset')
    offset = 0.
    if (root / 'project.json').is_file():
        workspace = read(root)
        if workspace.get('media') and media_path(root, workspace).resolve() == video:
            offset = float(workspace.get('time_offset', 0.))
    local_start = start - offset
    length = round((end-start)*FPS)
    if (job.get('version') != 1 or not math.isfinite(start) or not math.isfinite(end)
            or not math.isfinite(offset) or not 0 <= local_start < end-offset <= source['duration'] + .002 or not 4 <= length <= 1800
            or end-start > 60.001):
        raise ValueError('Select 4–1,800 frames at 30 fps, up to 60 seconds, within the video.')
    if missing('gemx_assets'):
        raise ValueError('Prepare the required GEM-X files in Model Manager.')
    model_keys = ('gemx_pose', 'gemx_observations', 'gemx_motion', 'gemx_stats', 'gemx_detector',
                  'gemx_soma_neutral', 'gemx_soma_rig', 'gemx_soma_transforms', 'gemx_soma_correctives')
    models = {key: BY_KEY[key].path.resolve() for key in model_keys}
    inputs = {path: file_stat(path) for path in [video, *models.values()]}
    for key, path in models.items():
        emit('progress', current=0, total=0, message='Verifying ' + BY_KEY[key].name)
        if sha256(path) != BY_KEY[key].expected_sha256:
            raise ValueError('Model changed. Reimport the verified file: ' + BY_KEY[key].name)
    parent = root / 'gemx/results'
    if not parent.resolve().is_relative_to(root):
        raise ValueError('GEM-X project directory leaves its workspace.')
    create(root, video, source, 'gemx')
    if shutil.disk_usage(root).free < 512*1024**2 + length*VERTEX_COUNT*12:
        raise OSError('Free more disk space for the SOMA surface cache.')
    project = parent / uuid.uuid4().hex
    project.mkdir(parents=True, exist_ok=False)
    finished = False
    try:
        clip = project / 'input.mp4'
        emit('progress', current=0, total=0, message='Preparing 30 fps GEM-X working video')
        # The timeline starts at the first video frame. FFmpeg seeks from the container start,
        # which can precede the video when another track begins earlier.
        seek = local_start + seek_offset
        command = [executable('ffmpeg'), '-v', 'error', '-nostdin', '-ss', f'{seek:.9f}', '-i', str(video),
                   '-map', '0:v:0', '-vf', 'fps=30:start_time=0,pad=ceil(iw/2)*2:ceil(ih/2)*2',
                   '-frames:v', str(length), '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
                   '-threads', '2', '-an', '-movflags', '+faststart', str(clip)]
        child = subprocess.run(command, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        if child.returncode:
            raise ValueError('Cannot prepare working video: ' + child.stderr[-1000:].decode('utf-8', 'replace'))
        media = probe(clip)
        frames = media['frames']
        if abs(media['fps'] - FPS) > .001 or not max(4, length - 1) <= frames <= length:
            raise ValueError(f'Could not prepare the 30 fps trim: expected {length} frames, got {frames} at {media["fps"]:.3f} fps.')
        if frames < length:
            emit('info', message=f'GEM-X working video has {frames} frames after 30 fps conversion (trim estimate: {length}).')
            length = frames
        cache = HOME / 'cache/gemx/warp'
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WARP_CACHE_PATH=str(cache))
        import numpy as np
        import torch
        from gem.inference.body_video import build_body_conditions, load_body_video_config
        from gem.inference.motion import load_predictor, run_motion
        from gem.inference.video import VideoFrames
        from gem.motion.geometry import MotionGeometry

        if torch.__version__ != cfg['torch'] or not torch.cuda.is_available():
            raise RuntimeError('The verified GEM-X CUDA runtime is unavailable. Run Setup_GEMX.cmd.')
        torch.set_num_threads(threads)
        free, total = torch.cuda.mem_get_info()
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction(free, total))
        config = load_body_video_config(cfg['engine'] / 'docs/configs/body_video_native77.yaml')
        for section, key in (('pose', 'gemx_pose'), ('observations', 'gemx_observations'),
                             ('detector', 'gemx_detector'), ('motion', 'gemx_motion')):
            config[section]['checkpoint'] = str(models[key])
        config['camera']['mode'] = 'static'
        config['motion']['stats'] = str(models['gemx_stats'])

        def progress(message):
            emit('progress', current=0, total=0, message=message)

        def infer(batch):
            for section in ('pose', 'observations'):
                config[section]['batch_size'] = batch
            emit('info', message=f'GEM-X · fixed camera · image batch {batch} · {length} frames')
            with VideoFrames(clip) as frames:
                sequence, _, _ = build_body_conditions(frames, config, device='cuda', progress=progress)
            progress('Predicting native temporal SOMA motion')
            predictor = load_predictor(config['motion']['checkpoint'], config['motion'],
                                       stats_path=config['motion']['stats'], device='cuda')
            result = run_motion(predictor, sequence, device='cuda')
            return {key: value.detach().cpu() for key, value in result['pred_soma_params_global'].items()}

        batch = 1 if effective_mode() == 'low_memory' or free < 5*1024**3 else 4
        parameters = retry_cuda(infer, batch, lambda value: progress(f'Retrying GEM-X with image batch {value}'))
        shapes = {'global_orient': 3, 'body_pose': 228, 'transl': 3,
                  'identity_coeffs': 128, 'scale_params': 57}
        if set(parameters) != set(shapes):
            raise ValueError('Unexpected native SOMA parameters.')
        for key, width in shapes.items():
            value = parameters[key]
            if key in ('identity_coeffs', 'scale_params') and value.shape == (1, width):
                value = value[:, None].expand(1, length, width)
            if value.shape != (1, length, width) or not torch.isfinite(value).all():
                raise ValueError('Invalid native SOMA parameter: ' + key)
            parameters[key] = value[0].contiguous()
        np.savez(project / PARAMETERS, **{key: value.numpy() for key, value in parameters.items()})
        gc.collect()
        torch.cuda.empty_cache()
        # Local asset aliases allow Model Manager's Use in Place for each file.
        from tempfile import TemporaryDirectory
        with TemporaryDirectory(prefix='soma-', dir=cache) as asset_dir:
            assets = Path(asset_dir)
            for key in ('gemx_soma_neutral', 'gemx_soma_rig', 'gemx_soma_transforms', 'gemx_soma_correctives'):
                destination = assets / Path(BY_KEY[key].relative).name
                try:
                    os.link(models[key], destination)
                except OSError:
                    shutil.copyfile(models[key], destination)
            engine = MotionGeometry(str(assets), device='cuda', chunk_size=16)
            vertical_offset = evaluate(engine, parameters, project, torch, np)
        if any(file_stat(path) != stamp for path, stamp in inputs.items()):
            raise ValueError('Video or model files changed during inference. Previous results were preserved.')
        state = dict(version=1, source=source, start=start, end=end, fps=FPS, frames=length,
                     width=media['width'], height=media['height'], camera='static', stages={},
                     timeline=dict(start=start, step=1/FPS, frames=length),
                     pipeline=dict(model='gemx', result='initial'), workspace_root='../../..')
        atomic_json(project / 'project.json', state)
        atomic_json(project / RECEIPT, dict(version=1, frames=length, fps=FPS,
                    skeleton='soma-native77', virtual_root='omitted',
                    units='metres', axes='native-y-up', transforms='joint-to-world',
                    geometry='soma-mid-correctives', camera='assumed-static',
                    vertical_offset_m=vertical_offset,
                    runtime=cfg['identity'], models={key: BY_KEY[key].expected_sha256 for key in model_keys},
                    outputs={name: file_stat(project / name) for name in FILES}))
        set_active(root, 'gemx', project)
        finished = True
        result = 'initial'
        if job.get('correct'):
            from .correction_settings import correction_options
            from .gemx_correct import run
            state['correction'] = correction_options(job.get('correction'))
            atomic_json(project / 'project.json', state)
            del engine
            gc.collect()
            torch.cuda.empty_cache()
            try:
                run(project, cfg, emit)
                result = 'corrected_initial'
                state['pipeline']['result'] = result
                atomic_json(project / 'project.json', state)
            except (OSError, ValueError, RuntimeError) as error:
                emit('info', message=f'GEM-X Motion Correction failed. Original motion was kept. {error}')
        emit('gemx_ready', project=str(project), root=str(root), result=result)
        emit('complete', stage='gemx_reconstruct', elapsed=time.monotonic()-started, output=str(project),
             peak_allocated_bytes=torch.cuda.max_memory_allocated())
    finally:
        if not finished and project.is_dir() and project.resolve().parent == parent.resolve():
            shutil.rmtree(project)


def main(argv, cfg):
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('gemx_reconstruct', 'gemx_correct'))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--request', type=Path)
    source.add_argument('--project', type=Path)
    parser.add_argument('--result', choices=('corrected_initial',))
    args = parser.parse_args(argv)
    try:
        if args.stage == 'gemx_correct' and args.project is not None:
            from .gemx_correct import run
            started = time.monotonic()
            output = run(args.project, cfg, emit)
            emit('gemx_corrected', project=str(args.project))
            emit('complete', stage=args.stage, output=str(output), elapsed=time.monotonic()-started)
        elif args.stage == 'gemx_reconstruct' and args.request is not None:
            reconstruct(args.request, cfg)
        else:
            parser.error('Invalid GEM-X stage arguments')
    except (OSError, ValueError, RuntimeError) as error:
        emit('failed', stage=args.stage, message=str(error))
        raise
    finally:
        if args.request is not None:
            args.request.unlink(missing_ok=True)
    return 0
