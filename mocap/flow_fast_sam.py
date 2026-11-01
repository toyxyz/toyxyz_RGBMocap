"""Bounded frame batching and optional GPU crops for Fast SAM conditioning.

Each frame keeps its own body-derived hand boxes. Precision, resolution and
decoder layers are unchanged. GPU crops use upstream's bilinear sampler.
"""
import contextlib
import os


def merge_batches(batches):
    import torch
    person_keys = {'img', 'img_size', 'ori_img_size', 'bbox_center', 'bbox_scale',
                   'bbox', 'affine_trans', 'mask', 'mask_score', 'person_valid'}
    merged = {}
    for key in batches[0]:
        values = [batch[key] for batch in batches]
        if key in ('cam_int', 'img_ori'):
            merged[key] = values[0]
        elif torch.is_tensor(values[0]):
            merged[key] = torch.cat(values, dim=1 if key in person_keys else 0)
        else:
            merged[key] = values[0]
    return merged


def extract_batch(extractor, images, boxes, intrinsics, gpu_crops=False, placement=False):
    import numpy as np
    import torch
    from sam_3d_body.sam_3d_body_estimator import prepare_batch, recursive_to
    from sam_3d_body.models.meta_arch.sam3d_body import _prepare_hand_batches_gpu

    count = len(images)
    if not 1 <= count <= 16 or len(boxes) != count:
        raise ValueError('Fast SAM requires one tracked body per frame, in batches of at most sixteen')
    if any(image.shape != images[0].shape for image in images):
        raise ValueError('Fast SAM frame sizes must match')
    estimator = extractor.estimator
    model = estimator.model
    body, hands = extractor._flow_fast_collectors
    body.clear()
    hands.clear()

    def prepare_hands(image, left, right, camera):
        if gpu_crops:
            return _prepare_hand_batches_gpu(np.stack(images), left, right, camera,
                output_size=(model.cfg.MODEL.IMAGE_SIZE[1], model.cfg.MODEL.IMAGE_SIZE[0]), padding=.9,
                device=camera.device)
        flipped = left.copy()
        width = images[0].shape[1]
        flipped[:, 0] = width - left[:, 2] - 1
        flipped[:, 2] = width - left[:, 0] - 1
        left_batch = merge_batches([prepare_batch(frame[:, ::-1], estimator.transform_hand,
            flipped[i:i+1], cam_int=camera.clone()) for i, frame in enumerate(images)])
        right_batch = merge_batches([prepare_batch(frame, estimator.transform_hand,
            right[i:i+1], cam_int=camera.clone()) for i, frame in enumerate(images)])
        return recursive_to(left_batch, camera.device), recursive_to(right_batch, camera.device), flipped

    try:
        with torch.no_grad(), open(os.devnull, 'w', encoding='utf-8') as quiet, contextlib.redirect_stdout(quiet):
            batch = merge_batches([prepare_batch(frame, estimator.transform, boxes[i:i+1],
                cam_int=intrinsics) for i, frame in enumerate(images)])
            batch = recursive_to(batch, 'cuda')
            model._initialize_batch(batch)
            batch['cam_int'] = intrinsics.to(batch['img']).clone()
            output = model.run_inference(images[0], batch, inference_type='full',
                transform_hand=estimator.transform_hand, thresh_wrist_angle=estimator.thresh_wrist_angle,
                hand_batch_preparer=prepare_hands, features_only=not placement)
        if len(body) != 1 or len(hands) != 1 or len(body[0]) != count or len(hands[0]) != count * 2:
            raise ValueError('Fast SAM returned incomplete body/hand features')
        tokens = torch.stack([torch.cat((body[0][i, 0], hands[0][i, 0], hands[0][count+i, 0]))
                              for i in range(count)]).float().cpu()
        people = []
        if placement:
            pose = output[0]['mhr']
            joints = pose['pred_keypoints_3d'].detach().cpu().numpy()
            camera = pose['pred_cam_t'].detach().cpu().numpy()
            people = [dict(pred_keypoints_3d=joints[i], pred_cam_t=camera[i]) for i in range(count)]
        return tokens, people
    finally:
        body.clear()
        hands.clear()
