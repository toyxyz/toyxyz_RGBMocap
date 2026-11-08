from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

from .models import BY_KEY
from .paths import COMPAT, MODELS, UPSTREAM
from .storage import hard_link


def save_tensor(value, target):
    import torch
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, target)


def activate(name: str) -> Path:
    # Each engine has a separate process, so upstream's generic `utils` package
    # and timm version never leak into the GUI or another model.
    root = UPSTREAM / name
    for path in (root, COMPAT):
        if str(path) not in sys.path:sys.path.insert(0, str(path))
    os.chdir(root)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(MODELS / "yolo" / "settings"))
    import torch
    from .compute_policy import worker_threads
    torch.set_num_threads(worker_threads(os.environ.get('RGB_MOCAP_STAGE',name)))
    return root


def link_models(root: Path) -> None:
    """Hard links keep upstream relative paths working without copying GB files."""
    for key,model in BY_KEY.items():
        if key == 'person_reid' or key.startswith('phmr_'):
            continue  # Discovery-only weights are never used by motion backends.
        source = model.path
        if not source.exists():
            continue
        target = root / "inputs/checkpoints" / model.relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.samefile(source):
            continue
        if target.exists() or target.is_symlink():
            target.unlink()  # includes links left dangling by moved models
        try:
            hard_link(source, target)
        except OSError:
            # Cross-volume data folders use a link when possible. If linking is
            # disallowed, report it instead of unexpectedly copying huge models.
            try:
                target.symlink_to(source)
            except OSError as error:
                raise RuntimeError("Keep the app and models on the same drive, or allow model links.") from error


def track(project: Path, emit) -> Path:
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    import ultralytics.nn.tasks as yolo_tasks
    # This old upstream release predates PyTorch's weights_only default.
    # Limit full-object loading to our explicitly selected official detector.
    def load_official_detector(weight):
        selected = Path(weight).resolve()
        if selected != BY_KEY["yolo"].path.resolve():
            raise ValueError("Unexpected detector checkpoint path")
        return torch.load(selected, map_location="cpu", weights_only=False), str(selected)
    yolo_tasks.torch_safe_load = load_official_detector
    model = YOLO(str(BY_KEY["yolo"].path))
    video = cv2.VideoCapture(str(project / "input.mp4"))
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(video.get(3)), int(video.get(4))
    boxes, observed, previous = [], [], None
    for index in range(total):
        ok, frame = video.read()
        if not ok:
            raise RuntimeError(f"Cannot read video frame {index}.")
        prediction = model.predict(frame, device=0, conf=0.35, classes=0, verbose=False)[0]
        candidates = prediction.boxes.xyxy.detach().cpu().numpy()
        if len(candidates):
            areas = (candidates[:, 2] - candidates[:, 0]) * (candidates[:, 3] - candidates[:, 1])
            if previous is None:
                selected = int(areas.argmax())
            else:
                top_left = np.maximum(candidates[:, :2], previous[:2])
                bottom_right = np.minimum(candidates[:, 2:], previous[2:])
                intersection = np.maximum(bottom_right - top_left, 0).prod(1)
                old_area = np.prod(previous[2:] - previous[:2])
                iou = intersection / np.maximum(areas + old_area - intersection, 1)
                selected = int((iou + 0.1 * areas / areas.max()).argmax())
            previous = candidates[selected]
            boxes.append(previous.copy())
            observed.append(True)
        else:
            boxes.append(np.full(4, np.nan, dtype=np.float32))
            observed.append(False)
        if index % 10 == 0 or index == total - 1:
            emit("progress", stage="track", current=index + 1, total=total, message="Tracking the actor in the video")
    video.release()
    box_array = np.asarray(boxes, dtype=np.float32)
    visible = np.flatnonzero(observed)
    if not len(visible):
        raise RuntimeError("No actor was found. Choose a video with a visible full body.")
    for component in range(4):
        box_array[:, component] = np.interp(np.arange(total), visible, box_array[visible, component])
    focal = float(np.hypot(width, height))
    K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float32)
    output = project / "tracking.npz"
    np.savez_compressed(output, boxes=box_array, observed=np.asarray(observed), K=K)
    torch.save({"K": torch.from_numpy(K)}, project / "camera.pt")
    return output


def pva(project: Path, emit) -> Path:
    root = activate("HTD-Refine")
    import cv2
    import numpy as np
    import torch
    from pvanet.model_io import load_full_model
    from utils.bbox import crop_and_resize_np, expand_bbx_xyxy_to_ratio
    from scripts.demo_pvanet import run_pvanet_inference
    tracking = np.load(project / "tracking.npz")
    boxes = expand_bbx_xyxy_to_ratio(torch.from_numpy(tracking["boxes"]))
    from .inference_data import VideoCrops
    crops = VideoCrops(project / "input.mp4", boxes, crop_and_resize_np)
    # Retain only the active temporal window, including overlap, on CPU.
    sample = {"imgs": crops, "bbx_xyxy": boxes[None],
              "K_fullimg": torch.from_numpy(tracking["K"])[None, None].expand(1, len(boxes), 3, 3)}
    emit("progress", stage="pva", current=0, total=len(boxes), message="Loading the temporal refinement model")
    model = load_full_model(BY_KEY["pva"].path, device="cuda:0")
    free_vram, _ = torch.cuda.mem_get_info()
    # Image encoding has no time dependency. Smaller batches preserve all 120
    # temporal frames and FP32 precision on lower-memory GPUs.
    gib = 1024**3
    from .compute_policy import image_plan,retry_cuda
    import psutil
    spatial_batch = image_plan('pva',free_vram,psutil.virtual_memory().available)['image_batch']
    from .inference_data import bounded_spatial_encoder
    bounded_spatial_encoder(model, spatial_batch)
    from .storage import atomic_json
    atomic_json(project / "pva_compute_plan.json", {"spatial_batch":spatial_batch,"temporal_window":120,
        "overlap":30,"precision":"float32","free_vram_after_model_gib":free_vram/gib,"crop_cache_frames":120})

    class WindowRunner(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner, self.completed = inner, 0
            length = len(boxes)
            starts = [0] if length <= 120 else list(range(0, length - 120 + 1, 90))
            if length > 120 and starts[-1] != length - 120:
                starts.append(length - 120)
            self.ends = [min(start + 120, length) for start in starts]
            self.calls = 0

        def forward(self, images, boxes_xyxy, intrinsics):
            if isinstance(images, VideoCrops):
                images = images[:, :]
            self.completed = self.ends[self.calls]
            self.calls += 1
            nonlocal spatial_batch
            def reduced(batch):
                nonlocal spatial_batch
                spatial_batch=batch
                from pvanet.network.pvanet import PVANet
                from types import MethodType
                self.inner.encode_visual_features = MethodType(PVANet.encode_visual_features,self.inner)
                bounded_spatial_encoder(self.inner,spatial_batch)
                atomic_json(project / "pva_compute_plan.json", {"spatial_batch":spatial_batch,"temporal_window":120,
                    "overlap":30,"precision":"float32","reduced_after_oom":True,"crop_cache_frames":120})
            output=retry_cuda(lambda batch:self.inner(images.to('cuda:0'),boxes_xyxy.to('cuda:0'),intrinsics.to('cuda:0')),
                              spatial_batch,reduced)
            emit("progress", stage="pva", current=min(self.completed, len(boxes)), total=len(boxes), message="Analyzing joints and motion changes")
            return tuple(item.detach().cpu() for item in output)

    try:
        with torch.inference_mode():
            output = run_pvanet_inference(WindowRunner(model), sample, window_length=120, overlap=30, keep_heatmaps=False)
    finally:
        crops.close()
    keypoints, bbox_keypoints, confidence, velocities, accelerations, _, heatmap_shape, stats = output
    result = {"format": "pvanet_output_v1", "image_keypoints": keypoints,
              "keypoints_bbox01": bbox_keypoints, "keypoint_confidence": confidence,
              "camera_velocities": velocities, "camera_accelerations": accelerations,
              "K_fullimg": sample["K_fullimg"], "bbx_xyxy": boxes[None], "pvanet_window": stats,
              "meta": {"source": "HTD-Refine official checkpoint", "fps": 30, "world_pose": False}}
    target = project / "pva_results.pt"
    save_tensor(result, target)
    np.savez_compressed(project / "keypoints.npz", keypoints=keypoints[0].numpy(), confidence=confidence[0].numpy())
    return target


def features(project: Path, emit, phase=None) -> Path:
    import time
    started=time.monotonic()
    if phase not in (None,'pose','features'):raise ValueError('Invalid GVHMR feature phase')
    from .model_session import acquire,model_key,reusable_bytes
    root = activate("GVHMR")
    link_models(root)
    import numpy as np
    import torch
    from .gvhmr_preprocess import configure_precision,identity,cache_state,cached,mark_complete
    configure_precision()
    from hmr4d.utils.preproc.vitpose import VitPoseExtractor
    from hmr4d.utils.preproc.vitfeat_extractor import Extractor
    from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy
    from .inference_data import streamed_features
    from .compute_policy import image_plan,prefetch_plan
    import psutil
    with np.load(project / "tracking.npz") as tracking:boxes=tracking["boxes"].copy()
    bbx = get_bbx_xys_from_xyxy(torch.from_numpy(boxes), base_enlarge=1.2)
    plan=image_plan('gvhmr',torch.cuda.mem_get_info()[0]+reusable_bytes(),psutil.virtual_memory().available)
    memory=psutil.virtual_memory()
    plan['prefetch_batches']=prefetch_plan(memory.available,memory.total)
    plan.update(deterministic_conv=True,tf32=False)
    signature,settings=identity(project,plan['image_batch'])
    state=cache_state(project,signature,settings)
    from .storage import atomic_json
    atomic_json(project/'gvhmr_preprocess_plan.json',plan)
    def progress(message):
        return lambda current,total:emit('progress',stage='features',message=message,current=current,total=total)
    emit("progress", stage="gvhmr", message="Computing GVHMR joint observations", current=0, total=3)
    keypoint_path = project / "gvhmr_keypoints.pt"
    pose_cached=cached(project,state,keypoint_path.name)
    if pose_cached:
        kp = torch.load(keypoint_path, map_location="cpu", weights_only=False)
    else:
        pose = acquire(model_key("gv_pose",BY_KEY["vitpose"].path),VitPoseExtractor)
        kp = streamed_features(project,bbx,pose.extract,plan['image_batch'],plan['prefetch_batches'],
            progress('Computing GVHMR joint observations'))
        save_tensor(kp, keypoint_path)
        mark_complete(project,state,keypoint_path.name)
        del pose
        gc.collect()
        torch.cuda.empty_cache()
    if phase=='pose':
        emit('complete',stage='features',output=str(keypoint_path),cached=pose_cached,elapsed=time.monotonic()-started)
        return keypoint_path
    emit("progress", stage="gvhmr", message="Computing image features", current=1, total=3)
    feature_path = project / "features.pt"
    feature_cached=cached(project,state,feature_path.name)
    if feature_cached:
        features = torch.load(feature_path, map_location="cpu", weights_only=False)
    else:
        extractor = acquire(model_key("gv_features",BY_KEY["hmr2"].path),Extractor)
        features = streamed_features(project,bbx,lambda images,boxes: extractor.extract_video_features(images,boxes),
            plan['image_batch'],plan['prefetch_batches'],progress('Computing GVHMR image features'))
        save_tensor(features, feature_path)
        mark_complete(project,state,feature_path.name)
        del extractor
        gc.collect()
        torch.cuda.empty_cache()
    if phase=='features':emit('complete',stage='features',output=str(feature_path),cached=pose_cached and feature_cached,elapsed=time.monotonic()-started)
    return feature_path


def gvhmr(project: Path, emit) -> Path:
    features(project, emit)
    root = activate("GVHMR")
    link_models(root)
    import numpy as np
    import torch
    from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy
    import hmr4d.network.gvhmr.relative_transformer
    import hmr4d.model.gvhmr.utils.endecoder
    from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
    from hydra import compose, initialize_config_module
    with np.load(project / "tracking.npz") as tracking:tracking={key:tracking[key].copy() for key in tracking.files}
    bbx = get_bbx_xys_from_xyxy(torch.from_numpy(tracking["boxes"]), base_enlarge=1.2)
    kp = torch.load(project / "gvhmr_keypoints.pt", map_location="cpu", weights_only=False)
    features_tensor = torch.load(project / "features.pt", map_location="cpu", weights_only=False)
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        cfg = compose(config_name="demo")
    from .model_session import acquire,model_key
    def build():
        model = DemoPL(cfg.pipeline).eval().cuda()
        model.load_pretrained_model(BY_KEY["gvhmr"].path)
        return model
    model = acquire(model_key('gvhmr',BY_KEY['gvhmr'].path),build)
    data = {"length": torch.tensor(len(bbx)), "bbx_xys": bbx, "kp2d": kp,
            "K_fullimg": torch.from_numpy(tracking["K"])[None].expand(len(bbx), 3, 3),
            "cam_angvel": torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32)[None].expand(len(bbx), 6),
            "f_imgseq": features_tensor}
    emit("progress", stage="gvhmr", message="Reconstructing initial 3D motion", current=2, total=3)
    from .gvhmr_attention import predict
    result=predict(model,data,project,emit)
    from hmr4d.utils.net_utils import detach_to_cpu
    target = project / "hmr4d_results.pt"
    save_tensor(detach_to_cpu(result), target)
    return target


def refine(project: Path, emit) -> Path:
    root = activate("HTD-Refine")
    link_models(root)
    import argparse
    import torch
    from scripts.demo_optimization import load_gvhmr_result, load_pva_targets, align_inputs, run_optimization
    video = project / "input.mp4"
    args = argparse.Namespace(video=str(video), source="gvhmr", motion=str(project / "hmr4d_results.pt"),
                             static_cam=True, camera_traj=None, camera_intrinsics=str(project / "camera.pt"),
                             device="cuda:0", run_optimization=True, overwrite=True, vel_thr=None)
    emit("progress", stage="refine", current=0, total=0, message="Refining the full motion sequence")
    aligned = align_inputs(load_gvhmr_result(args.motion), load_pva_targets(project / "pva_results.pt"),
                           video, static_cam=True, camera_traj=None, camera_intrinsics=args.camera_intrinsics)
    run_optimization(aligned, args, project, "smplx", torch.device("cuda:0"))
    return project / "opt_results.pt"
