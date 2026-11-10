"""Run pinned Meta video inference with Windows-compatible tensor operations."""

import functools
import sys
from types import ModuleType
from unittest.mock import patch


def windows_builder():
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

    model_builder = windows_builder()
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
