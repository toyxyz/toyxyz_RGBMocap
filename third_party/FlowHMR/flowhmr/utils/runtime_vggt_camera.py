from __future__ import annotations

"""VGGT-Omega camera extraction utilities.

Extract per-frame camera extrinsics from a video and save them as NPZ.
The saved ``camera_RT`` / ``RT`` keeps the raw VGGT/OpenCV convention. It is
converted to the V2M Y-up convention only when feeding the model.
"""

import os
import os.path as osp
import shutil
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import torch


_repo_root = osp.normpath(osp.join(osp.dirname(__file__), "..", ".."))
_vggt_repo_dir = osp.join(_repo_root, "third_party", "vggt-omega")
if _vggt_repo_dir not in sys.path:
    sys.path.insert(0, _vggt_repo_dir)

DEFAULT_VGGT_OMEGA_CKPT = osp.join(_repo_root, "ckpts", "vggt-omega", "vggt_omega_1b_512.pt")


@dataclass
class VGGTCameraResult:
    camera_RT: np.ndarray[Any, Any] # (T,4,4), raw VGGT/OpenCV cam <- world
    RT_vggt: np.ndarray[Any, Any] # (T,4,4), same as camera_RT, kept for clarity
    K: np.ndarray[Any, Any] # (T,3,3), intrinsics decoded in original-frame resolution
    image_size_hw: tuple[int, int]


_VGGT_MODEL_CACHE: dict[tuple[str, str, bool], Any] = {}


def _interpolate_camera_sequence(
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
    sample_indices: list[int],
    target_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate sampled camera predictions to every frame."""
    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}")
    if len(sample_indices) != int(extrinsic.shape[0]):
        raise ValueError(
            f"sample_indices length {len(sample_indices)} != extrinsic length {int(extrinsic.shape[0])}"
        )
    if len(sample_indices) == target_len and np.array_equal(
        np.asarray(sample_indices, dtype=np.int64), np.arange(target_len, dtype=np.int64)
    ):
        return extrinsic, intrinsic

    sample_x = np.asarray(sample_indices, dtype=np.float64)
    target_x = np.arange(target_len, dtype=np.float64)
    if sample_x[0] != 0 or sample_x[-1] != target_len - 1:
        raise ValueError(
            f"sample_indices must include first/last frame, got {sample_indices[:3]}...{sample_indices[-3:]}, "
            f"target_len={target_len}"
        )

    from scipy.spatial.transform import Rotation as R, Slerp

    device = extrinsic.device
    dtype = extrinsic.dtype
    ext_np = extrinsic.detach().float().cpu().numpy()
    int_np = intrinsic.detach().float().cpu().numpy()

    if len(sample_indices) == 1:
        full_ext = np.repeat(ext_np, target_len, axis=0)
        full_int = np.repeat(int_np, target_len, axis=0)
    else:
        full_ext = np.zeros((target_len, 3, 4), dtype=np.float32)
        rots = R.from_matrix(ext_np[:, :3, :3])
        full_ext[:, :3, :3] = Slerp(sample_x, rots)(target_x).as_matrix().astype(np.float32)
        for r in range(3):
            full_ext[:, r, 3] = np.interp(target_x, sample_x, ext_np[:, r, 3]).astype(np.float32)

        full_int = np.zeros((target_len, 3, 3), dtype=np.float32)
        for r in range(3):
            for c in range(3):
                full_int[:, r, c] = np.interp(target_x, sample_x, int_np[:, r, c]).astype(np.float32)

    return torch.from_numpy(full_ext).to(device=device, dtype=dtype), torch.from_numpy(full_int).to(device=device, dtype=dtype)


def _vggt_opencv_to_yup_rt(camera_rt: torch.Tensor) -> torch.Tensor:
    """Convert raw VGGT/OpenCV ``cam <- world`` RT to V2M ``cam <- y_up_world`` RT."""
    vggt_from_yup = torch.eye(4, dtype=camera_rt.dtype, device=camera_rt.device)
    vggt_from_yup[1, 1] = -1
    vggt_from_yup[2, 2] = -1
    return camera_rt @ vggt_from_yup


def _load_vggt_model(checkpoint_path: str, device: Any, camera_only: bool = False) -> Any:
    key = (osp.abspath(checkpoint_path), str(device), camera_only)
    if key in _VGGT_MODEL_CACHE:
        return _VGGT_MODEL_CACHE[key].to(device) # cpu -> device
    if not osp.isfile(checkpoint_path):
        raise FileNotFoundError(f"VGGT-Omega checkpoint not found: {checkpoint_path}")

    from vggt_omega.models import VGGTOmega
    with torch.device("meta"): 
        model = VGGTOmega(enable_depth=not camera_only)
    state_dict = torch.load(checkpoint_path, map_location="cpu", mmap=camera_only)
    if camera_only:
        state_dict = {k:v for k,v in state_dict.items() if not k.startswith("dense_head.")}
    model.load_state_dict(state_dict,assign=True)
    del state_dict
    
    model = model.eval()
    _VGGT_MODEL_CACHE[key] = model  # cache the model on CPU to avoid reloading from disk
    
    return model.to(device)


def _save_video_frames(
    video_path: str,
    frame_dir: str,
    *,
    frame_range: Optional[Tuple[int, int]] = None,
    max_frames: Optional[int] = None,
    frame_interval: int = 1,
) -> tuple[list[str], tuple[int, int], list[int]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")

    start, end = frame_range if frame_range is not None else (0, None)
    start = max(int(start), 0)
    if end is not None:
        end = int(end)
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    frame_interval = max(1, int(frame_interval))
    os.makedirs(frame_dir, exist_ok=True)
    paths: list[str] = []
    sample_indices: list[int] = []
    image_size_hw: tuple[int, int] | None = None
    frame_idx = start
    try:
        while True:
            if end is not None and frame_idx > end:
                break
            if max_frames is not None and max_frames > 0 and len(paths) >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if image_size_hw is None:
                h, w = frame.shape[:2]
                image_size_hw = (int(h), int(w))
            rel_idx = frame_idx - start
            should_save = (rel_idx % frame_interval == 0) or (end is not None and frame_idx == end)
            if should_save:
                out_path = osp.join(frame_dir, f"{len(paths):06d}.jpg")
                # RGB MoCap Windows patch: imwrite's narrow-path API silently
                # fails for Korean/emoji project folders. Encoding is unchanged.
                encoded_ok, encoded = cv2.imencode('.jpg', frame)
                if not encoded_ok:
                    raise IOError('Could not encode a VGGT camera frame')
                with open(out_path, 'wb') as output:
                    output.write(encoded.tobytes())
                paths.append(out_path)
                sample_indices.append(rel_idx)
            frame_idx += 1
    finally:
        cap.release()

    if not paths or image_size_hw is None:
        raise ValueError(f"VGGT-Omega read no frames from video: {video_path}, frame_range={frame_range}")
    return paths, image_size_hw, sample_indices


def extract_vggt_camera(
    video_path: str,
    *,
    output_npz: Optional[str] = None,
    frame_range: Optional[Tuple[int, int]] = None,
    expected_len: Optional[int] = None,
    checkpoint_path: str = DEFAULT_VGGT_OMEGA_CKPT,
    device: torch.device | str = "cuda",
    image_resolution: int = 512,
    preprocess_mode: str = "max_size",
    camera_only: bool = False,
    keep_frames: bool = False,
    frame_interval: int = 1,
) -> VGGTCameraResult:
    frame_interval = max(1, int(frame_interval))
    if output_npz and osp.exists(output_npz):
        data = np.load(output_npz)
        k_space = str(data["k_space"]) if "k_space" in data.files else ""
        # New cache: camera_RT/RT are raw VGGT OpenCV. Old cache may have
        # converted camera_RT, but still stores raw RT_vggt; prefer RT_vggt.
        camera_RT = (data["RT_vggt"] if "RT_vggt" in data.files else data["camera_RT"]).astype(np.float32)
        image_size_arr = data["image_size_hw"]
        image_size_hw = (int(image_size_arr[0]), int(image_size_arr[1]))
        cached_interval = int(data["camera_sampling_interval"]) if "camera_sampling_interval" in data.files else 1
        valid_interval = cached_interval == frame_interval
        valid_len = expected_len is None or camera_RT.shape[0] == int(expected_len)
        if k_space == "original_frame" and valid_len and valid_interval:
            print(f"  [vggt] skip cached camera: {output_npz}")
            return VGGTCameraResult(
                camera_RT=camera_RT,
                RT_vggt=camera_RT,
                K=data["K"].astype(np.float32),
                image_size_hw=image_size_hw,
            )
        if k_space != "original_frame":
            print(f"  [vggt] cached camera K is not original-frame space ({k_space!r}), recomputing")
        elif not valid_interval:
            print(f"  [vggt] cached camera interval mismatch: {cached_interval} != {frame_interval}, recomputing")
        else:
            print(f"  [vggt] cached camera length mismatch: {camera_RT.shape[0]} != {expected_len}, recomputing")

    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("VGGT-Omega camera extraction requires CUDA, but CUDA is not available")

    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    if output_npz:
        frame_dir = output_npz + ".frames"
        if osp.isdir(frame_dir):
            shutil.rmtree(frame_dir)
        tmp_ctx = None
        os.makedirs(frame_dir, exist_ok=True)
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="vggt_omega_frames_")
        frame_dir = tmp_ctx.name

    try:
        frame_paths, original_image_size_hw, sample_indices = _save_video_frames(
            video_path,
            frame_dir,
            frame_range=frame_range,
            max_frames=expected_len,
            frame_interval=frame_interval,
        )
        if expected_len is not None:
            target_len = int(expected_len)
            if sample_indices[-1] != target_len - 1:
                raise ValueError(
                    f"VGGT-Omega sampled frames do not cover the final frame: last_sample={sample_indices[-1]}, "
                    f"expected_last={target_len - 1}, video={video_path}, frame_range={frame_range}"
                )
        else:
            target_len = sample_indices[-1] + 1
        print(f"  [vggt] sampled {len(frame_paths)}/{target_len} frames (interval={frame_interval})")

        model = _load_vggt_model(checkpoint_path, device_obj, camera_only=camera_only)
        
        images = load_and_preprocess_images(
            frame_paths,
            mode=preprocess_mode,
            image_resolution=image_resolution,
        ).to(device_obj)

        with torch.inference_mode():
            with _model_ctx(model, device=device_obj,streaming_prefetch_count=1) as model:
                predictions = model(images)
                extrinsic, intrinsic = encoding_to_camera(
                    predictions["pose_enc"],
                    original_image_size_hw,
                )

        # Remove batch dimension from README-style outputs.
        extrinsic = extrinsic[0] if extrinsic.ndim == 4 else extrinsic
        intrinsic = intrinsic[0] if intrinsic.ndim == 4 else intrinsic
        if int(extrinsic.shape[0]) != len(sample_indices):
            raise ValueError(
                f"VGGT output length {int(extrinsic.shape[0])} != sampled frame count {len(sample_indices)}"
            )
        extrinsic, intrinsic = _interpolate_camera_sequence(
            extrinsic,
            intrinsic,
            sample_indices,
            target_len,
        )

        T = extrinsic.shape[0]
        RT_vggt = torch.eye(4, dtype=extrinsic.dtype, device=extrinsic.device).unsqueeze(0).repeat(T, 1, 1)
        RT_vggt[:, :3, :4] = extrinsic

        camera_RT_yup = _vggt_opencv_to_yup_rt(RT_vggt)

        vggt_hw = predictions["images"].shape[-2:]
        vggt_image_size_hw = (int(vggt_hw[0]), int(vggt_hw[1]))
        raw_rt_np = RT_vggt.detach().float().cpu().numpy().astype(np.float32)
        result = VGGTCameraResult(
            camera_RT=raw_rt_np,
            RT_vggt=raw_rt_np,
            K=intrinsic.detach().float().cpu().numpy().astype(np.float32),
            image_size_hw=original_image_size_hw,
        )

        if output_npz:
            os.makedirs(osp.dirname(output_npz), exist_ok=True)
            np.savez(
                output_npz,
                camera_RT=result.camera_RT,
                RT=result.camera_RT,
                RT_vggt=result.RT_vggt,
                camera_RT_yup=camera_RT_yup.detach().float().cpu().numpy().astype(np.float32),
                K=result.K,
                image_size_hw=np.array(result.image_size_hw, dtype=np.int64),
                vggt_image_size_hw=np.array(vggt_image_size_hw, dtype=np.int64),
                frame_range=np.array(frame_range if frame_range is not None else (0, T - 1), dtype=np.int64),
                checkpoint_path=checkpoint_path,
                image_resolution=image_resolution,
                preprocess_mode=preprocess_mode,
                camera_world="vggt_opencv_raw",
                k_space="original_frame",
                camera_sampling_interval=frame_interval,
                camera_sample_indices=np.array(sample_indices, dtype=np.int64),
            )
            print(
                f"  [vggt] camera saved → {output_npz}  "
                f"(T={T}, sampled={len(sample_indices)}, interval={frame_interval}, image_size={result.image_size_hw})"
            )

        return result
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
        elif not keep_frames and output_npz:
            shutil.rmtree(frame_dir, ignore_errors=True)


def load_or_extract_vggt_camera_rt(
    video_path: str,
    output_npz: str,
    *,
    frame_range: Optional[Tuple[int, int]],
    expected_len: int,
    device: torch.device | str,
    checkpoint_path: str = DEFAULT_VGGT_OMEGA_CKPT,
    image_resolution: int = 512,
    frame_interval: int = 1,
) -> torch.Tensor:
    result = extract_vggt_camera(
        video_path,
        output_npz=output_npz,
        frame_range=frame_range,
        expected_len=expected_len,
        checkpoint_path=checkpoint_path,
        device=device,
        image_resolution=image_resolution,
        frame_interval=frame_interval,
    )
    raw_rt = torch.from_numpy(result.camera_RT).to(torch.float32)
    return _vggt_opencv_to_yup_rt(raw_rt)




#####  Add streaming model wrapper for vggt-omega ,use for low Vram  #####

from contextlib import  contextmanager
import gc
import functools
import itertools
from collections.abc import Iterator
from torch import nn
from typing import TypeVar
_M = TypeVar("_M", bound=torch.nn.Module)
T = TypeVar("T")

def cleanup_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

class _SimpleLayerStore:

    def __init__(self, layers: nn.ModuleList, target_device: torch.device) -> None:
        self.target_device = target_device
        self.num_layers = len(layers)
        self._resident: set[int] = set()
        self._cpu_params: list[dict[str, torch.Tensor]] = []
        for layer in layers:
            cpu_copy = {}
            for name, tensor in itertools.chain(layer.named_parameters(), layer.named_buffers()):
                if tensor is None:
                    continue
                t = tensor.data
                if t.device.type != 'cpu':
                    t = t.to('cpu')
                cpu_copy[name] = t
            self._cpu_params.append(cpu_copy)

    def set_resident(self, resident: set[int]) -> None:
        self._resident = resident

    def is_resident(self, idx: int) -> bool:
        return idx in self._resident

    def load_layer_to_gpu(self, idx: int, layer: nn.Module, force: bool = False) -> None:
        if idx in self._resident and not force:
            return
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            if param is None:
                continue
            if name in self._cpu_params[idx]:
                param.data = self._cpu_params[idx][name].to(self.target_device)

    def unload_layer_from_gpu(self, idx: int, layer: nn.Module) -> None:
        if idx in self._resident:
            return
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            if param is None:
                continue
            if name in self._cpu_params[idx]:
                param.data = self._cpu_params[idx][name]  # 恢复为CPU副本（同一引用，不新建）

def _resolve_attr(module: nn.Module, dotted_path: str) -> nn.ModuleList:
    """Resolve a dotted attribute path like ``'model.language_model.layers'``."""
    obj: Any = module
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, nn.ModuleList):
        raise TypeError(f"Expected nn.ModuleList at '{dotted_path}', got {type(obj).__name__}")
    return obj

    
class SimpleLayerStreamingWrapper(nn.Module):
    
    def __init__(
        self,
        model: nn.Module,
        layers_attr: str,
        target_device: torch.device,
        active_count: int = 0,  
    ) -> None:
        super().__init__()
        self._model = model
        self._layers = _resolve_attr(model, layers_attr)
        self._target_device = target_device
        self._active_count = active_count
        self._store = _SimpleLayerStore(self._layers, self._target_device)
        self._non_layer_cpu_refs: dict[int, torch.Tensor] = {}

        n = len(self._layers)
        if active_count is not None and active_count >= 0:
            resident_count = min(active_count, n)
            self._resident = set(range(resident_count))
        else:
            self._resident = set()
        self._store.set_resident(self._resident)
        self._move_non_layer_params_to_gpu()
        for idx in sorted(self._resident):
            self._store.load_layer_to_gpu(idx, self._layers[idx], force=True)
        self._register_simple_hooks()
    
    def _move_non_layer_params_to_gpu(self) -> None:
        layer_tensor_ids = set()
        for layer in self._layers:
            for t in itertools.chain(layer.parameters(), layer.buffers()):
                layer_tensor_ids.add(id(t))

        def _ensure(param: torch.Tensor) -> None:
            pid = id(param)
            if pid not in self._non_layer_cpu_refs:
                self._non_layer_cpu_refs[pid] = param.data  
            param.data = self._non_layer_cpu_refs[pid].to(self._target_device)

        for p in self._model.parameters():
            if id(p) not in layer_tensor_ids:
                _ensure(p)
        for b in self._model.buffers():
            if id(b) not in layer_tensor_ids:
                _ensure(b)
    
    def _register_simple_hooks(self) -> None:
        idx_map = {id(layer): idx for idx, layer in enumerate(self._layers)}
        
        def _pre_hook(module: nn.Module, input, *, idx: int):
            self._store.load_layer_to_gpu(idx, module)
            for param in itertools.chain(module.parameters(), module.buffers()):
                param.data.record_stream(torch.cuda.current_stream(self._target_device))
        
        def _post_hook(module: nn.Module, input, output, *, idx: int):
            event = getattr(self, "_events", {}).get(idx)
            if event is not None:
                event.record(torch.cuda.current_stream(self._target_device))
                event.synchronize()
            self._store.unload_layer_from_gpu(idx, module)

            torch.cuda.empty_cache()
        
        for layer in self._layers:
            idx = idx_map[id(layer)]
            if idx in self._resident:
                continue  
            pre_hook = layer.register_forward_pre_hook(functools.partial(_pre_hook, idx=idx))
            post_hook = layer.register_forward_hook(functools.partial(_post_hook, idx=idx))
    
    def to(self, *args: Any, **kwargs: Any) -> "SimpleLayerStreamingWrapper":
        target = None
        if args:
            a = args[0]
            if isinstance(a, (str, torch.device)):
                target = torch.device(a)
        if target is not None and target.type == "cpu":
            for idx in range(len(self._layers)):
                if not self._store.is_resident(idx):
                    self._store.unload_layer_from_gpu(idx, self._layers[idx])
        else:
            self._move_non_layer_params_to_gpu()
        return self

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)
    
    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)
        
@contextmanager
def _streaming_model(
    model: _M,
    layers_attr: str,
    target_device: torch.device,
    prefetch_count: int,
    empty_cache: bool = False,
) -> Iterator[_M]:
    """Wrap *model* with :class:`LayerStreamingWrapper`, yield it, then tear down."""
    wrapped = SimpleLayerStreamingWrapper(
        model,
        layers_attr=layers_attr,
        target_device=target_device,
        active_count=prefetch_count,
    )
    try:
        yield wrapped  # type: ignore[misc]
    finally:
       
        wrapped.to("cpu")
        cleanup_memory()
        # Flush the host (pinned) memory cache so that freed pinned pages are
        # returned to the OS.  Without this, sequential streaming models
        # (e.g. text encoder then transformer) exhaust host memory because the
        # CachingHostAllocator keeps freed blocks cached indefinitely.
        if empty_cache:
            torch.cuda.synchronize(device=target_device)
            try:
                if hasattr(torch._C, "_host_emptyCache"):
                    torch._C._host_emptyCache()
            except Exception:
                print("Host empty cache cleanup failed; ignoring.", exc_info=True)

def _model_ctx(model,device=torch.device("cuda:0"),streaming_prefetch_count=None,) :
    if streaming_prefetch_count is not None:
        return _streaming_model(
            model,
            layers_attr="aggregator.frame_blocks",
            target_device=device,
            prefetch_count=streaming_prefetch_count,
        )

    return model
