"""FlowHMR SAM path selection; each inference stage runs in its own process.

The original path retains original body/hand calls. The Fast adapter collects
the same conditioning slots from the validated merged left/right hand batch.
Upstream SAM/Fast source remains separately licensed in third_party.
"""
from pathlib import Path
import contextlib
import importlib
import os
import sys

FAST_COMMIT = 'd72aa36913a1673ace029d345f437561a85ec9e2'


def saved_options(project, state):
    """New processing defaults on; saved off is retained, including scene takes."""
    import json
    project = Path(project)
    options = state.get('flow_options', {})
    if state.get('people_root'):
        root = (project / state['people_root']).resolve()
        if project.resolve().is_relative_to(root) and (root / 'people.json').is_file():
            options = json.loads((root / 'project.json').read_text(encoding='utf-8')).get('flow_options', options)
    result = {}
    for key in ('fast_sam', 'sam_batch4', 'sam_gpu_crop', 'sam_hand_stride4', 'hamer'):
        value = options.get(key, key!='hamer')
        if not isinstance(value, bool):
            raise ValueError('Invalid FlowHMR setting: '+key+'; expected on or off')
        result[key] = value
    if result['hamer']:result.update(fast_sam=True,sam_hand_stride4=True)
    return result


def saved_fast_sam(project, state):
    return saved_options(project, state)['fast_sam']


def acceleration_options(project, state):
    options = saved_options(project, state)
    return {key: options['fast_sam'] and options[key] for key in ('sam_batch4', 'sam_gpu_crop', 'sam_hand_stride4')}


def source_root(fast):
    from .paths import UPSTREAM
    return UPSTREAM / 'FlowHMR/third_party' / ('fast-sam-3d-body' if fast else 'sam-3d-body')


def source_identity(fast):
    from .storage import sha256
    root = source_root(fast)
    if not (root / 'sam_3d_body/__init__.py').is_file():
        raise RuntimeError('The selected SAM runtime is missing. Repair or reinstall RGB MoCap.')
    # Include all executable fork sources, not only its meta-architecture.
    files = sorted((root / 'sam_3d_body').rglob('*.py')) if fast else [root / 'sam_3d_body/models/meta_arch/sam3d_body.py']
    return dict(mode='fast' if fast else 'original', commit=FAST_COMMIT if fast else None,
                files={p.relative_to(root).as_posix(): sha256(p) for p in files})


def configure_fast_environment():
    # Do not let unrelated shell settings opt this worker into research paths.
    for key in list(os.environ):
        if key.endswith('_INTERM_PRED_LAYERS') or key.endswith('_INTERM_PRED_INTERVAL') or key in (
                'INTERM_PRED_LAYERS', 'INTERM_PRED_INTERVAL', 'KEYPOINT_PROMPT_INTERM_LAYERS', 'MOMENTUM_ENABLED'):
            os.environ.pop(key, None)
    os.environ.update(USE_COMPILE='0', USE_COMPILE_BACKBONE='0', USE_TRT_BACKBONE='0',
        MHR_USE_CUDA_GRAPH='0', PARALLEL_DECODERS='0', GPU_HAND_PREP='0',
        DEBUG_NAN='0', DEBUG_HAND_PREP='0', DEBUG_BACKBONE_INPUT='0', INTERM_TIMING='0',
        LAYER_DTYPE='none', IMG_SIZE='0', USE_DALI='0', MHR_NO_CORRECTIVES='0',
        SKIP_KEYPOINT_PROMPT='1', COMPILE_WARMUP_BATCH_SIZES='',
        KEYPOINT_PROMPT_INTERM_INTERVAL='999')


def build_extractor(device, fast=True, low_memory=False):
    root = source_root(fast).resolve()
    if not (root / 'sam_3d_body/__init__.py').is_file():
        raise RuntimeError('The selected SAM runtime is missing. Repair or reinstall RGB MoCap.')
    loaded = sys.modules.get('sam_3d_body')
    if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(root):
        raise RuntimeError('SAM paths must run in separate worker processes.')
    if fast:
        configure_fast_environment()
    sys.path.insert(0, str(root))
    package = importlib.import_module('sam_3d_body')
    if not Path(package.__file__).resolve().is_relative_to(root):
        raise RuntimeError('The selected SAM package could not be activated.')
    # Preimport chooses the package before FlowHMR inserts its original path.
    from flowhmr.utils.runtime_sam_features import build_sam3d_extractor
    with open(os.devnull, 'w', encoding='utf-8') as quiet, contextlib.redirect_stdout(quiet):
        if not low_memory:
            extractor = build_sam3d_extractor(device)
        else:
            from flowhmr.utils import runtime_sam_features as runtime
            from .gpu_offload import BlockOffload, release
            model, config = package.load_sam_3d_body(checkpoint_path=runtime.SAM3D_CKPT_PATH,
                mhr_path=runtime.SAM3D_MHR_PATH, device='cpu')
            try:
                runtime._maybe_convert_backbone_to_fp16(model, device)
                model._mocap_offload = BlockOffload(model, device)
                extractor = runtime.Sam3DTokenExtractor(package.SAM3DBodyEstimator(
                    sam_3d_body_model=model, model_cfg=config))
            except BaseException:
                release(model)
                raise
    if fast:
        for hook in extractor._hooks:
            hook.remove()
        body, hands = [], []
        model = extractor.estimator.model
        extractor._hooks = [
            model.decoder.register_forward_hook(lambda m, i, out: body.append(out[0].detach())),
            model.decoder_hand.register_forward_hook(lambda m, i, out: hands.append(out[0].detach()))]
        extractor._flow_fast_collectors = body, hands
    return extractor


def extract_tokens(extractor, img, boxes, k, placement=False):
    collectors = getattr(extractor, '_flow_fast_collectors', None)
    if collectors is not None:
        import torch
        body, hands = collectors
        body.clear()
        hands.clear()
        try:
            with torch.no_grad(), open(os.devnull, 'w', encoding='utf-8') as quiet, contextlib.redirect_stdout(quiet):
                outputs = extractor.estimator.process_one_image(img, bboxes=boxes, cam_int=k, inference_type='full')
            if not body or len(hands) != 1 or hands[0].shape[0] != 2:
                raise ValueError('Fast SAM did not return the expected body and left/right hand tokens')
            token = torch.cat((body[0][0, 0], hands[0][0, 0], hands[0][1, 0])).cpu()
            return token, outputs
        finally:
            # Do not retain GPU activations between frames or after failure.
            body.clear()
            hands.clear()
    model = extractor.estimator.model
    previous = getattr(model, '_flow_tokens_only', False)
    model._flow_tokens_only = not placement
    try:
        return extractor.extract_frame(img, bboxes=boxes, cam_int=k, inference_type='full')
    finally:
        model._flow_tokens_only = previous

