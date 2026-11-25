"""Run pinned Meta video inference with portable CUDA and OpenCV operations."""

import functools
import sys
from collections import UserDict
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import patch


def compatible_builder():
    import cv2
    import numpy as np
    import torch

    # The native implementation documents an exact OpenCV equivalent. The
    # text path does not ask for corrective clicks during model propagation.
    edt = ModuleType('sam3.model.edt')
    edt.edt_triton = lambda data: torch.from_numpy(np.stack([
        cv2.distanceTransform(im, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        for im in data.cpu().numpy().astype(np.uint8)
    ])).to(data.device)
    sys.modules[edt.__name__] = edt
    from sam3 import model_builder
    from sam3.model import decoder
    from sam3.perflib import connected_components, nms
    from torch.nn.attention import SDPBackend

    def components(data):
        labels, sizes = [], []
        for image in data.cpu().numpy().reshape(-1, *data.shape[-2:]):
            _, label, stats, _ = cv2.connectedComponentsWithStats(image.astype(np.uint8), connectivity=8)
            counts = stats[:, cv2.CC_STAT_AREA].copy()
            counts[0] = 0
            labels.append(label)
            sizes.append(counts[label])
        return tuple(torch.from_numpy(np.stack(values)).reshape(data.shape).to(data.device)
                     for values in (labels, sizes))

    original_sdpa = decoder.sdpa_kernel
    decoder.sdpa_kernel = lambda _: original_sdpa(
        [SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
    nms.generic_nms = nms.generic_nms_cpu
    # SAM imports this function in other modules. Replace its fallback before
    # model construction by updating the implementation's module globals.
    connected_components.connected_components_cpu = components
    connected_components.HAS_CC_TORCH = True
    connected_components.get_connected_components = components
    return model_builder


def predictor(weights, folder):
    import torch
    from safetensors.torch import load_file

    model_builder = compatible_builder()
    with patch.object(model_builder, 'download_ckpt_from_hf', return_value=None):
        result = model_builder.build_sam3_multiplex_video_predictor(
            bpe_path=str(folder / 'sam3/assets/bpe_simple_vocab_16e6.txt.gz'),
            use_fa3=False, compile=False, warm_up=False, async_loading_frames=False)
    checkpoint = load_file(str(weights), device='cpu')
    state = result.model.load_state_dict(checkpoint, strict=False)
    names = dict(result.model.named_parameters(remove_duplicate=False))
    loaded = {id(names[k]) for k in checkpoint if k in names}
    missing = [k for k in state.missing_keys if k in names and id(names[k]) not in loaded]
    # The converted checkpoint omits CLIP's pooled output projection. SAM
    # uses the token embeddings only. Set the unused parameter explicitly.
    unused = 'detector.backbone.language_backbone.encoder.text_projection'
    if missing == [unused]:
        with torch.no_grad():
            names[unused].zero_()
        missing.clear()
    buffers = [k for k in state.missing_keys if k not in names and 'freqs_cis' not in k]
    if missing or buffers or state.unexpected_keys:
        raise ValueError('The SAM 3.1 checkpoint does not match the tracking model.')

    init = result.model.init_state

    @functools.wraps(init)
    def init_video(*args, offload_state_to_cpu=False, **kwargs):
        return init(*args, **kwargs)

    result.model.init_state = init_video
    result.model.batched_grounding_batch_size = 4
    result.model.postprocess_batch_size = 4
    return result


class _CpuMasks(UserDict):
    """Keep frame masks on the host and load masks individually on access."""

    def __init__(self, masks, device):
        self.device = device
        super().__init__(masks)

    def __setitem__(self, key, mask):
        super().__setitem__(key, mask.cpu())

    def __getitem__(self, key):
        return super().__getitem__(key).to(self.device)

    def copy(self):
        return dict(self.items())


class _CpuMaskCache(dict):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def __setitem__(self, frame, masks):
        super().__setitem__(frame, _CpuMasks(masks, self.device))


def _compatible_storage(operation):
    @functools.wraps(operation)
    def call(destination, source, key, source_key, *args, **kwargs):
        if key in destination and source[source_key].device != destination[key].device:
            source = {**source, source_key: source[source_key].to(destination[key].device)}
        return operation(destination, source, key, source_key, *args, **kwargs)
    return call


@contextmanager
def offload_tracking(predictor):
    import torch
    from sam3.model import video_tracking_multiplex as multiplex

    model = predictor.model
    tracker = model.tracker.model
    init_video = model.init_state
    reset_video = model.reset_state
    init_tracker = tracker.init_state
    device = next(model.parameters()).device

    @functools.wraps(init_video)
    def init_with_cpu_masks(*args, **kwargs):
        state = init_video(*args, **kwargs)
        state['cached_frame_outputs'] = _CpuMaskCache(device)
        return state

    @functools.wraps(reset_video)
    def reset_with_cpu_masks(state, *args, **kwargs):
        result = reset_video(state, *args, **kwargs)
        state['cached_frame_outputs'] = _CpuMaskCache(device)
        return result

    @functools.wraps(init_tracker)
    def init_with_cpu_state(*args, **kwargs):
        kwargs['offload_state_to_cpu'] = True
        return init_tracker(*args, **kwargs)

    def sync_cpu_copies(operation):
        @functools.wraps(operation)
        def call(inference_state, *args, **kwargs):
            result = operation(inference_state, *args, **kwargs)
            # The pinned tracker queues non-blocking copies to storage.
            # Complete those copies before its CPU masks are read or modified.
            torch.cuda.current_stream(device=inference_state['device']).synchronize()
            return result
        return call

    # Reconditioning and adding new objects mix resident and stored tensors.
    with (patch.multiple(model, init_state=init_with_cpu_masks, reset_state=reset_with_cpu_masks),
          patch.multiple(tracker, init_state=init_with_cpu_state,
                         _run_single_frame_inference=sync_cpu_copies(tracker._run_single_frame_inference),
                         _run_memory_encoder=sync_cpu_copies(tracker._run_memory_encoder)),
          patch.multiple(multiplex, _append=_compatible_storage(multiplex._append),
                         _merge=_compatible_storage(multiplex._merge))):
        yield
