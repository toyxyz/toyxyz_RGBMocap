"""Optional local playback measurement; never runs inference."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import psutil
from PySide6.QtCore import QTimer

from .process_memory import peak_ram_bytes, private_bytes


def schedule_check(window, seconds, output: Path, screenshot: Path | None, finish):
    samples = []
    state = {}
    mesh_samples = []
    process = psutil.Process()

    def cpu_time():
        cpu = process.cpu_times()
        return cpu.user + cpu.system

    def sample(index):
        if "started" in state and (not samples or samples[-1][1] != index):
            samples.append((time.perf_counter(), index, window.motion_view.index))

    signal = getattr(window.video_view, "frame_painted", window.slider.valueChanged)
    signal.connect(sample)
    def mesh_swapped():
        view = window.motion_view
        if "started" not in state or not window.timer.isActive():return
        if view.people_scene is not None:
            source_time=window.source_offset+window.video_view.index/window.info['fps']
            for project,frame in view.scene_presented.values():
                take=view.people_scene.by_project[Path(project)]
                expected=take.frame_at(source_time)
                if expected is not None:mesh_samples.append((frame,expected))
        elif view.mesh_presented_index is not None:
            mesh_samples.append((view.mesh_presented_index,window.video_view.index))
    if hasattr(window.motion_view,"frameSwapped"):
        window.motion_view.frameSwapped.connect(mesh_swapped)

    def write_report():
        report = state["report"]
        paused_elapsed = time.perf_counter() - state["paused_at"]
        report["paused_cpu_core_percent"] = (cpu_time() - state["paused_cpu"]) / paused_elapsed * 100
        view=window.motion_view
        renderers=[r for r in [view.mesh_renderer,*view.scene_renderers.values(),*view.target_renderers.values(),*view.bone_renderers.values()] if r]
        report["paused_mesh_compute_dispatches"] = sum(r.dispatches for r in renderers)-state.get('paused_dispatches',0)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report), flush=True)
        window.close()
        finish()

    def stop():
        elapsed = time.perf_counter() - state["started"]
        window.stop_playback()
        intervals = [b[0] - a[0] for a, b in zip(samples, samples[1:])]
        report = {"source_fps": window.info["fps"], "elapsed_seconds": elapsed,
                  "clip_in_frame": window.slider.clip_in, "clip_out_frame_exclusive": window.slider.clip_out,
                  "frames_outside_selected_clip": sum(not window.slider.clip_in <= i < window.slider.clip_out for _,i,_ in samples),
                  "camera_distance_m": float(window.motion_view.distance),
                  "presented_frames": len(samples), "effective_fps": len(samples) / elapsed,
                  "max_video_3d_frame_difference": max((abs(i - j) for _, i, j in samples), default=None),
                  "max_frame_interval_ms": max(intervals, default=0) * 1000,
                  "p95_frame_interval_ms": float(np.percentile(intervals, 95)) * 1000 if intervals else None,
                  "last_frame": samples[-1][1] if samples else None,
                  "cpu_core_percent": (cpu_time() - state["cpu"]) / elapsed * 100,
                  "logical_processors": psutil.cpu_count(),
                  "peak_process_ram_mib": peak_ram_bytes() / 1024**2,
                  "private_mib": private_bytes(process) / 1024**2,
                  "os_threads": process.num_threads(),
                  "inference_modules_loaded": "torch" in sys.modules}
        view = window.motion_view
        renderer = next(iter(view.target_renderers.values()),view.mesh_renderer) if view.native_vertices is not None else view.mesh_renderer
        renderers=[r for r in [view.mesh_renderer,*view.scene_renderers.values(),*view.target_renderers.values(),*view.bone_renderers.values()] if r]
        report.update(display_mode='raw',bone_overlay=view.show_bones,bone_render_error=view.bone_error,mesh_render_error=view.mesh_error,
            mesh_frames_observed=len(mesh_samples),
            max_mesh_video_frame_difference=max((abs(a-b) for a,b in mesh_samples),default=None),
            mesh_gpu_mib=sum(r.gpu_bytes for r in renderers)/1024**2,
            people_scene=view.people_scene is not None,visible_people=len(view.scene_presented),
            names_enabled=view.show_names,
            name_labels=[dict(name=name,x=rect.x(),y=rect.y(),width=rect.width(),height=rect.height())
                         for name,rect in view.name_labels_presented],
            mesh_vertices=renderer.vertex_count if renderer else 0,
            mesh_triangles=renderer.index_count//3 if renderer else 0)
        decoder = getattr(window.video_view, "decoder", None)
        if decoder:
            report.update(decoder_seek_count=decoder.seek_count, preview_cache_limit_mib=decoder.budget / 1024**2)
        if screenshot:
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            from .capture import capture_window
            capture_window(window).save(str(screenshot))
        state.update(report=report, paused_at=time.perf_counter(), paused_cpu=cpu_time(),paused_dispatches=sum(r.dispatches for r in renderers))
        QTimer.singleShot(1000, write_report)

    def begin():
        state.update(started=time.perf_counter(), cpu=cpu_time())
        window.toggle_play()
        QTimer.singleShot(round(seconds * 1000), stop)

    QTimer.singleShot(1200, begin)
