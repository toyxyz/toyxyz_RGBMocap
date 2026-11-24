"""Optional RIFE conversion on the same time grid used by offline motion models."""
from bisect import bisect_left, bisect_right
from contextlib import ExitStack, contextmanager
import importlib.util
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory, TemporaryFile

from .paths import ROOT

MODEL_KEY = 'rife49'
RECIPE = 'rife49-v1'
TARGET_FPS = 30
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
TIMESTAMP_EPS = 0.0000005


def recipe(enabled, fps):
    return RECIPE if enabled and 0 < fps < TARGET_FPS - .001 else ''


def _source_times(executable, source):
    value = json.loads(subprocess.check_output([
        executable('ffprobe'), '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height:stream_side_data=rotation:frame=best_effort_timestamp_time',
        '-show_streams', '-show_frames', '-of', 'json', str(source)], creationflags=NO_WINDOW))
    try:
        times = [float(frame['best_effort_timestamp_time']) for frame in value['frames']]
        info = value['streams'][0]
        if len(times) < 2 or not all(math.isfinite(t) for t in times):
            raise ValueError()
        if any(a > b for a, b in zip(times, times[1:])):
            raise ValueError()
    except (KeyError, IndexError, ValueError) as error:
        raise ValueError('RIFE needs at least two video frames with usable timestamps.') from error
    return [t - times[0] for t in times], info


def _layout(info, mode):
    w, h = info['width'], info['height']
    if any(abs(round(float(item.get('rotation', 0)))) % 180 == 90 for item in info.get('side_data_list', [])):
        w, h = h, w
    if mode == 'pad':
        w, h = w + w % 2, h + h % 2
        return w, h, f'pad={w}:{h}'
    if mode == 'fit':
        scale = min(1, 1920 / w, 1080 / h)
        w, h = int(w * scale), int(h * scale)
    w, h = max(2, w // 2 * 2), max(2, h // 2 * 2)
    return w, h, f'scale={w}:{h}'


@contextmanager
def _pipe(command, **kwargs):
    with TemporaryFile() as errors, subprocess.Popen(command, stderr=errors, creationflags=NO_WINDOW, **kwargs) as child:
        try:
            yield child
            if child.stdin:
                child.stdin.close()
            if child.wait():
                errors.seek(0)
                raise RuntimeError(errors.read(2048).decode('utf-8', 'replace').strip() or 'FFmpeg exited during RIFE conversion.')
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()


def _load_model(torch):
    from .models import BY_KEY, ready
    asset = BY_KEY[MODEL_KEY]
    if not ready(asset):
        raise RuntimeError('Download RIFE 4.9 in Model Manager or turn off RIFE Frame Interpolation.')
    payload = asset.path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != asset.expected_sha256:
        raise RuntimeError('RIFE 4.9 failed SHA-256 verification. Reimport the model in Model Manager.')
    if not torch.cuda.is_available():
        raise RuntimeError('RIFE Frame Interpolation requires a CUDA-capable GPU. Turn off RIFE to use standard conversion.')
    spec = importlib.util.spec_from_file_location('rgbmocap_rife', ROOT / 'third_party/RIFE/network.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.IFNet()
    state = torch.load(BytesIO(payload), map_location='cpu', weights_only=True)
    model.load_state_dict({key.removeprefix('module.'): value for key, value in state.items()}, strict=True)
    return model.eval().cuda()


def _frames(reader, writer, times, start, count, cuts, shape, model, torch, emit, cancelled):
    import numpy as np

    height, width = shape
    stride = height * width * 3

    def read():
        data = reader.read(stride)
        if len(data) != stride:
            raise ValueError('A decoded frame was missing during RIFE conversion.')
        return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)

    left, right = read(), None
    index = 0
    with torch.inference_mode():
        for frame in range(count):
            if cancelled and cancelled():
                from .project_save import SaveCancelled
                raise SaveCancelled('Video preparation cancelled.')
            target = start + frame / TARGET_FPS
            while index + 1 < len(times) and times[index + 1] <= target + TIMESTAMP_EPS:
                left, right = (right if right is not None else read()), None
                index += 1
            if index + 1 == len(times) or index + 1 in cuts or target <= times[index] + TIMESTAMP_EPS:
                image = left
            else:
                if right is None:
                    right = read()
                phase = (target - times[index]) / (times[index + 1] - times[index])
                a, b = (torch.from_numpy(image.copy()).permute(2, 0, 1)[None].to(
                    device='cuda', dtype=torch.float32).div_(255) for image in (left, right))
                image = model(a, b, phase)[0].permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().cpu().numpy()
            writer.write(image.tobytes())
            if emit and (frame % 15 == 0 or frame + 1 == count):
                emit('progress', current=frame + 1, total=count, message='RIFE interpolation to 30 fps')
    # Consume any remaining lookahead to let FFmpeg close its pipe normally.
    reader.read(stride)


def convert(source, target, start, count, *, layout='even', emit=None, cancelled=None):
    """Publish a video-only 30fps clip, retaining source time and shot boundaries."""
    import torch
    from .video import executable, probe

    if emit:
        emit('progress', current=0, total=count, message='Preparing RIFE interpolation')
    model = None
    try:
        model = _load_model(torch)
        times, info = _source_times(executable, source)
        first = max(0, bisect_right(times, start + TIMESTAMP_EPS) - 1)
        last = min(len(times) - 1, bisect_left(times, start + (count - 1) / TARGET_FPS))
        if last < first:
            last = first
        selection = f'select=between(n\\,{first}\\,{last})'
        width, height, geometry = _layout(info, layout)
        command = [executable('ffmpeg'), '-nostdin', '-hide_banner', '-v', 'error', '-i', str(source),
                   '-map', '0:v:0', '-an', '-frames:v', str(last - first + 1)]
        detect = selection + ',scdet=threshold=12,metadata=print:key=lavfi.scd.time:file=-'
        events = subprocess.check_output([*command, '-vf', detect, '-f', 'null', '-'], creationflags=NO_WINDOW)
        cuts = {int(line.split()[0].split(b':')[1]) for line in events.splitlines() if line.startswith(b'frame:')}
        raw = [*command, '-vf', selection + ',' + geometry, '-fps_mode', 'passthrough',
               '-pix_fmt', 'rgb24', '-c:v', 'rawvideo', '-threads', '1', '-f', 'rawvideo', '-']
        with TemporaryDirectory(prefix='.rife-', dir=target.parent) as temp:
            output = Path(temp) / 'working.mp4'
            encode = [executable('ffmpeg'), '-nostdin', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
                      '-video_size', f'{width}x{height}', '-framerate', '30', '-i', '-',
                      '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-threads', '2',
                      '-movflags', '+faststart', str(output)]
            with ExitStack() as stack:
                reader = stack.enter_context(_pipe(raw, stdout=subprocess.PIPE))
                writer = stack.enter_context(_pipe(encode, stdin=subprocess.PIPE))
                _frames(reader.stdout, writer.stdin, times[first:last + 1], start, count, cuts,
                        (height, width), model, torch, emit, cancelled)
            checked = probe(output)
            if checked['frames'] != count or abs(checked['fps'] - TARGET_FPS) > .001:
                raise ValueError('RIFE output frame count or frame rate does not match the requested trim.')
            output.replace(target)
    except torch.OutOfMemoryError as error:
        raise RuntimeError('RIFE ran out of GPU memory. Close other GPU apps or turn off RIFE to use standard conversion.') from error
    finally:
        del model
        torch.cuda.empty_cache()
