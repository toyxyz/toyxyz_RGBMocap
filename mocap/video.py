from __future__ import annotations

import json
import shutil
import subprocess
import os
from fractions import Fraction
from pathlib import Path

from .storage import atomic_json, project_key, sha256
from .paths import PROJECTS, ROOT


def executable(name: str) -> str:
    bundled = ROOT / "tools" / "bin" / f"{name}.exe"
    found = str(bundled) if bundled.exists() else shutil.which(name)
    if not found:
        raise RuntimeError("Video tools are missing. Check the app installation.")
    return found


def probe(path: Path, *, include_seek_offset: bool = False) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    proc = subprocess.run([executable("ffprobe"), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                          check=True, capture_output=True, encoding="utf-8", creationflags=0x08000000)
    raw = json.loads(proc.stdout)
    stream = next((s for s in raw["streams"] if s["codec_type"] == "video"), None)
    if stream is None:
        raise ValueError("The selected file has no video track.")
    rate = float(Fraction(stream["avg_frame_rate"]))
    duration = float(stream.get("duration", raw["format"].get("duration", 0)))
    info = {"path": str(path.resolve()), "width": int(stream["width"]), "height": int(stream["height"]),
            "fps": rate, "duration": duration, "frames": int(stream.get("nb_frames", round(duration * rate))),
            "codec": stream["codec_name"], "bytes": path.stat().st_size}
    if include_seek_offset:
        container_start = raw["format"].get("start_time")
        video_start = stream.get("start_time")
        info["seek_offset"] = float(video_start) - float(container_start) if video_start is not None and container_start is not None else 0.
    return info


def select_preview(project: Path) -> Path:
    source = project / "input.mp4"
    metadata = project / "preview.json"
    target = project / "preview.mp4"
    if target.exists() and metadata.exists():
        try:
            record = json.loads(metadata.read_text(encoding="utf-8"))
            stat = source.stat()
            if record.get("version") == 1 and record.get("source") == [stat.st_size, stat.st_mtime_ns]:
                return target
        except (OSError, ValueError):
            pass
    return source


def run_video_command(command, cancelled=None):
    if cancelled is None:
        return subprocess.run(command, check=True, capture_output=True, creationflags=0x08000000)
    import time
    # File-backed stderr bounds memory for a long encode and avoids pipe stalls.
    import tempfile
    with tempfile.TemporaryFile() as errors:
        with subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=errors, creationflags=0x08000000) as process:
            try:
                while process.poll() is None:
                    if cancelled():
                        from .project_save import SaveCancelled
                        raise SaveCancelled('Project save cancelled. The current project was kept.')
                    time.sleep(.05)
                if process.returncode:
                    errors.seek(0)
                    raise ValueError('Could not prepare the project video: '+errors.read(4096).decode('utf-8',errors='replace'))
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()


def ensure_preview(project: Path, cancelled=None) -> Path:
    """Prepare a short-GOP preview during explicit project preparation only.

    Playback itself never starts an encoder. Analysis still uses input.mp4.
    """
    source = project / "input.mp4"
    existing = select_preview(project)
    if existing != source:
        return existing
    info = probe(source)
    if info["width"] <= 1280 and info["height"] <= 720:
        return source
    temporary = project / f"preview.{os.getpid()}.partial.mp4"
    try:
        run_video_command([executable("ffmpeg"), "-hide_banner", "-loglevel", "error", "-i", str(source),
            "-map", "0:v:0", "-an", "-vf", "scale=1280:720:force_original_aspect_ratio=decrease:force_divisible_by=2",
            "-c:v", "libx264", "-preset", "veryfast", "-tune", "fastdecode", "-crf", "26",
            "-g", "12", "-keyint_min", "12", "-sc_threshold", "0", "-bf", "0", "-threads", "2",
            "-pix_fmt", "yuv420p", "-y", str(temporary)], cancelled)
        preview = probe(temporary)
        if preview["frames"] != info["frames"] or abs(preview["fps"] - info["fps"]) > .001:
            raise ValueError("Preview frame timing does not match the analysis video")
        temporary.replace(project / "preview.mp4")
        stat = source.stat()
        atomic_json(project / "preview.json", {"version": 1, "source": [stat.st_size, stat.st_mtime_ns],
            "frames": preview["frames"], "fps": preview["fps"], "width": preview["width"], "height": preview["height"],
            "keyframe_interval": 12, "analysis_video_unchanged": True})
    finally:
        temporary.unlink(missing_ok=True)
    return project / "preview.mp4"


def prepare(source: Path, start: float = 0.0, end: float | None = None, motion_tool='gvhmr', project=None, cancelled=None) -> Path:
    from .motion_tools import require_supported_tool
    tool=require_supported_tool(motion_tool)
    if tool.input_setup_only:
        raise ValueError('MAMMA input setup is available, but multi-view reconstruction is not integrated yet.')
    info = probe(source)
    from . import project_workspace as workspace
    if project is None:
        saved_root=PROJECTS/f'{source.stem}_{project_key(source,0,info["duration"])}_project'
        workspace.create(saved_root,source,info,motion_tool)
    elif not (Path(project)/'project.json').exists():
        saved_root=workspace.create(project,source,info,motion_tool)
    else:
        saved_root=workspace.root_for(project)
        if saved_root is None:raise ValueError('Open a model-folder project before reconstructing.')
    saved=workspace.read(saved_root)
    offset=float(saved.get('time_offset',0.))
    end = info['duration']+offset if end is None else min(end, info['duration']+offset)
    if start < offset or end <= start:
        raise ValueError("Check the start and end of the processing range.")
    root=workspace.model_project(saved_root,motion_tool,saved)
    if (root/'project.json').exists():
        from .people_store import root_for
        root=root_for(root)
        workspace.child(saved_root,root.relative_to(saved_root),motion_tool)
    if (root/'project.json').exists():
        old=json.loads((root/'project.json').read_text(encoding='utf-8'))
        if abs(old['start']-start)>1e-6 or abs(old['end']-end)>1e-6:
            root=saved_root/motion_tool/'clips'/workspace.range_key(start,end)
    root.mkdir(parents=True, exist_ok=True)
    video = root / "input.mp4"
    manifest_path = root / "project.json"
    if manifest_path.exists():
        from .motion_tools import require_supported_project
        if require_supported_project(root).key != motion_tool:
            raise ValueError('The existing project belongs to a different motion model. Prepare a new project.')
    if video.exists() and manifest_path.exists():
        ensure_preview(root, cancelled)
        workspace.set_active(saved_root,motion_tool,root)
        return root
    temporary = root / "input.partial.mp4"
    resample='fps=30:start_time=0' if motion_tool=='flowhmr' else 'fps=30'
    command = [executable("ffmpeg"), "-hide_banner", "-loglevel", "error", "-i", str(source), "-ss", str(start-offset),
               "-t", str(end - start), "-map", "0:v:0", "-an", "-vf", resample+",scale=trunc(iw/2)*2:trunc(ih/2)*2",
               "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-y", str(temporary)]
    run_video_command(command, cancelled)
    output = probe(temporary)
    if output["frames"] < 4:
        raise ValueError("Select a range containing at least four frames.")
    temporary.replace(video)
    timeline = {"start": start, "step": 1 / 30, "frames": output["frames"]}
    atomic_json(manifest_path, {"version": 1, "source": saved['source'], "source_sha256": sha256(source), "start": start,
        "end": end, "fps": 30, "frames": output["frames"], "width": output["width"], "height": output["height"],
        "camera": "static", "stages": {}, "pipeline": {"model": motion_tool}, "timeline": timeline,
        'workspace_root':os.path.relpath(saved_root,root),
        **({"original_times": [start + i / 30 for i in range(output["frames"])]} if output["frames"] <= 18000 else {})})
    ensure_preview(root, cancelled)
    workspace.set_active(saved_root,motion_tool,root)
    return root


def prepare_people_clip(project: Path, first: int, end: int) -> Path:
    """Slice the scene's prepared frame grid once, shared by every actor.

    Re-decoding the original at its native frame rate can drop a frame and
    desynchronise observations, camera slices and motion at take boundaries.
    """
    from .motion_tools import require_supported_project
    project = Path(project)
    tool = require_supported_project(project).key
    state = json.loads((project / 'project.json').read_text(encoding='utf-8'))
    if not 0 <= first < end <= state['frames']:
        raise ValueError('Invalid person take frame range')
    if first == 0 and end == state['frames']:
        return project
    source = project / 'input.mp4'
    stamp = source.stat()
    identity = dict(version=1, input=[stamp.st_size, stamp.st_mtime_ns],
                    first=first, end=end, model=tool, fps=state['fps'])
    target = project / 'people_media' / f'{first:09d}_{end:09d}'
    video = target / 'input.mp4'
    receipt = target / 'media.json'
    manifest = target / 'project.json'
    if manifest.exists():
        if require_supported_project(target).key != tool:
            raise ValueError('Person take media belongs to a different motion model; its files were preserved.')
    try:
        record = json.loads(receipt.read_text(encoding='utf-8'))
        output = video.stat()
        saved = json.loads(manifest.read_text(encoding='utf-8'))
        if (record['identity'] == identity and record['output'] == [output.st_size, output.st_mtime_ns]
                and saved['frames'] == end-first):
            ensure_preview(target)
            return target
    except (OSError, ValueError, KeyError):
        pass
    target.mkdir(parents=True, exist_ok=True)
    temporary = target / 'input.partial.mp4'
    try:
        subprocess.run([executable('ffmpeg'), '-hide_banner', '-loglevel', 'error', '-i', str(source),
            '-map', '0:v:0', '-an', '-vf', f'trim=start_frame={first}:end_frame={end},setpts=N/({state["fps"]}*TB)',
            '-frames:v', str(end-first), '-r', str(state['fps']), '-fps_mode', 'cfr', '-c:v', 'libx264', '-preset', 'fast',
            '-crf', '18', '-pix_fmt', 'yuv420p', '-threads', '2', '-y', str(temporary)],
            check=True, capture_output=True, creationflags=0x08000000)
        info = probe(temporary)
        if info['frames'] != end-first or abs(info['fps']-state['fps']) > .001:
            raise ValueError('Person take media does not match its scene frame range')
        temporary.replace(video)
        start = state['start']+first/state['fps']
        child = {**state, 'start': start, 'end': state['start']+end/state['fps'], 'frames': end-first,
                 'stages': {}, 'pipeline': {'model': tool},
                 'timeline': {'start': start, 'step': 1/state['fps'], 'frames': end-first}}
        for key in ('people_root', 'person_id', 'person_preview', 'person_frames'):
            child.pop(key, None)
        from .project_storage import inherit_storage
        inherit_storage(project,target,child)
        child.pop('original_times', None)
        if end-first <= 18000:
            child['original_times'] = [start+i/state['fps'] for i in range(end-first)]
        atomic_json(manifest, child)
        output = video.stat()
        atomic_json(receipt, dict(identity=identity, output=[output.st_size, output.st_mtime_ns]))
        ensure_preview(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
