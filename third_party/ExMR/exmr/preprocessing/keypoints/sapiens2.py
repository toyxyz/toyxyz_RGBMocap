# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Crop preprocessing and UDP heatmap decoding adapted from Sapiens2
# (https://github.com/facebookresearch/sapiens2, sapiens/pose/benchmarks/fp32_pose_utils.py) and
# licensed under the Sapiens2 License, see licenses/Sapiens2_LICENSE.md.
"""Sapiens2-1B whole-body 2D keypoints (FP16, no flip test) -> COCO-63."""

import copy
from pathlib import Path

import cv2
import numpy as np
import sapiens.pose.models  # noqa: F401  (registers the pose models)
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from sapiens.engine.datasets import Compose
from sapiens.pose.datasets import UDPHeatmap
from sapiens.registry import MODELS
from mocap.batch_prefetch import batch_source
from tqdm import tqdm

from exmr.preprocessing.keypoints.mappings import SAPIENS2_TO_COCO63
from exmr.preprocessing.image_io import read_bgr_frame

# input resolution "height x width" -> batch size
RESOLUTIONS = {"1024x768": 1, "576x432": 2, "448x336": 4}

# inference part of Sapiens2's keypoints308 "shutterstock_goliath_3po" 1B config
MODEL_CFG = dict(
    type="PoseTopdownEstimator",
    backbone=dict(
        type="Sapiens2",
        arch="sapiens2_1b",
        img_size=(1024, 768),
        patch_size=16,
        final_norm=True,
        use_tokenizer=False,
        with_cls_token=True,
        out_type="featmap",
    ),
    decode_head=dict(
        type="PoseHeatmapHead",
        in_channels=1536,
        out_channels=308,
        deconv_out_channels=(1536, 1024),
        deconv_kernel_sizes=(4, 4),
        conv_out_channels=(768, 512, 256),
        conv_kernel_sizes=(1, 1, 1),
        loss_decode=dict(type="KeypointMSELoss", use_target_weight=True, loss_weight=10.0),
    ),
)
DATA_PREPROCESSOR_CFG = dict(
    type="ImagePreprocessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
)


class Sapiens2Keypoints:
    """2D keypoints of one person per frame, given its bounding boxes.

    Args:
        checkpoint: Sapiens2 1B pose weights (sapiens2_1b_pose.safetensors)
        resolution: crop resolution "height x width", one of RESOLUTIONS
    """

    def __init__(self, checkpoint, resolution="1024x768", num_workers=2, device="cuda:0", low_memory=False):
        if resolution not in RESOLUTIONS:
            raise ValueError(f"Sapiens2 resolution must be one of {list(RESOLUTIONS)}")
        self.batch_size = RESOLUTIONS[resolution]
        self.num_workers = num_workers
        h, w = map(int, resolution.split("x"))

        # Build and load weights on the CPU to avoid holding the model and
        # checkpoint tensors together in VRAM. Move the loaded model to the
        # requested device below; inference precision remains unchanged.
        with torch.device("cpu"):
            model = MODELS.build(copy.deepcopy(MODEL_CFG))
        model.load_state_dict(load_file(str(checkpoint), device="cpu"))
        model.data_preprocessor = MODELS.build(copy.deepcopy(DATA_PREPROCESSOR_CFG))
        if low_memory:
            # Same final FP16 layers, without a temporary full FP32 GPU copy.
            model.backbone.half()
            model.decode_head.half()
            model.to(device).eval()
        else:
            model.to(device).eval()
            model.backbone.half()
            model.decode_head.half()
        self.model = model
        self.pipeline = Compose(
            [
                dict(type="PoseGetBBoxCenterScale"),
                dict(type="PoseTopdownAffine", input_size=(w, h), use_udp=True),
                dict(type="PosePackInputs"),
            ]
        )
        self.codec = UDPHeatmap(input_size=(w, h), heatmap_size=(w // 4, h // 4), sigma=6)

    def __call__(self, frame_dir, bbx_xyxy):
        """frame_dir with %06d.jpg frames, bbx_xyxy (L, 4) -> (L, 63, 3) [x, y, raw score]"""
        boxes = bbx_xyxy.cpu().numpy()
        dataset = _FrameDataset(frame_dir, boxes, self.pipeline, self.model.data_preprocessor.cpu())
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=_collate,
            **({"prefetch_factor": 2} if self.num_workers else {}),
        )
        output = np.empty((len(boxes), 63, 3), dtype=np.float32)
        cursor = 0
        with batch_source(loader, batch_bytes=self.batch_size*3*1024*768*4) as batches, tqdm(total=len(boxes), desc="Sapiens2 keypoints", unit="frame") as pbar:
            for batch in batches:
                prediction = _predict(
                    batch["inputs"].cuda().half(), self.model, self.codec, batch["data_samples"]
                )
                output[cursor : cursor + len(prediction)] = prediction[:, SAPIENS2_TO_COCO63]
                cursor += len(prediction)
                pbar.update(len(prediction))
        if cursor != len(boxes) or not np.isfinite(output).all():
            raise RuntimeError("Incomplete or non-finite Sapiens2 output")
        return torch.from_numpy(output)


class _FrameDataset(torch.utils.data.Dataset):
    def __init__(self, frame_dir, boxes, pipeline, preprocessor):
        self.paths = [Path(frame_dir) / f"{i:06d}.jpg" for i in range(len(boxes))]
        self.boxes, self.pipeline, self.preprocessor = boxes, pipeline, preprocessor

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, index):
        image = read_bgr_frame(self.paths[index])
        if image is None:
            raise ValueError(f"Unreadable image: {self.paths[index]}")
        data = self.pipeline(
            dict(img=image, bbox=self.boxes[index][None], bbox_score=np.ones(1, dtype=np.float32))
        )
        data = self.preprocessor(data)
        return dict(inputs=data["inputs"][0], data_samples=data["data_samples"])


def _collate(batch):
    return dict(
        inputs=torch.stack([x["inputs"] for x in batch]),
        data_samples=[x["data_samples"] for x in batch],
    )


@torch.inference_mode()
def _predict(inputs, model, codec, metadata):
    """First 63 Sapiens2 keypoints in image coordinates, (N, 63, 3) [x, y, score]."""
    # the decoder runs in FP32 even though the model runs in FP16
    heatmaps = model(inputs)[:, :63].float()
    if heatmaps.device.type == "mps":
        heatmaps = heatmaps.cpu()  # the decoder needs float64, which MPS lacks
    keypoints, scores = _decode_udp(heatmaps, codec)
    width, height = codec.heatmap_size
    keypoints = keypoints / [width - 1, height - 1] * codec.input_size
    results = []
    for keypoint, score, data in zip(keypoints, scores, metadata):
        meta = data["meta"]
        keypoint = (
            keypoint / meta["input_size"] * meta["bbox_scale"]
            + meta["bbox_center"]
            - 0.5 * meta["bbox_scale"]
        )
        results.append(np.concatenate((keypoint, score[:, None]), axis=-1).astype(np.float32))
    return np.stack(results)


def _decode_udp(encoded, codec):
    """Vectorized GPU version of Sapiens2's OpenCV/NumPy UDP (DARK) heatmap decoder."""
    assert codec.heatmap_type == "gaussian"
    batch, channels, height, width = encoded.shape
    flat = encoded.flatten(2)
    scores, peak = flat.max(dim=-1)
    keypoints = torch.stack((peak % width, peak // width), dim=-1).to(torch.float32)
    keypoints[scores <= 0] = -1

    # OpenCV's sigma=0 rule for this kernel size, applied separably as it does
    kernel_size = codec.blur_kernel_size
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    coords = torch.arange(kernel_size, device=encoded.device, dtype=torch.float32)
    kernel = torch.exp(-((coords - (kernel_size - 1) * 0.5) ** 2) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    maps = encoded.reshape(batch * channels, 1, height, width)
    blurred = F.conv2d(maps, kernel.view(1, 1, 1, -1), padding=(0, kernel_size // 2))
    blurred = F.conv2d(blurred, kernel.view(1, 1, -1, 1), padding=(kernel_size // 2, 0))
    blurred = blurred.reshape(batch, channels, height, width)
    blurred *= encoded.amax(dim=(-2, -1), keepdim=True) / blurred.amax(dim=(-2, -1), keepdim=True)
    blurred = blurred.clamp_(1e-3, 50.0).log_()

    plane = (height + 2) * (width + 2)
    local = keypoints[..., 0].to(torch.int64) + 1
    local += (keypoints[..., 1].to(torch.int64) + 1) * (width + 2)
    local += torch.arange(channels, device=encoded.device).view(1, -1) * plane

    def gather(offset):
        # torch.remainder reproduces NumPy's negative flat-index wrapping
        index = torch.remainder(local + offset, channels * plane)
        channel = index // plane + torch.arange(batch, device=encoded.device).view(-1, 1) * channels
        spatial = index % plane
        y = (spatial // (width + 2) - 1).clamp(0, height - 1)
        x = (spatial % (width + 2) - 1).clamp(0, width - 1)
        return blurred.reshape(batch * channels, height, width)[channel, y, x]

    i_ = gather(0)
    ix1, iy1 = gather(1), gather(width + 2)
    ix1y1, ix1_y1_ = gather(width + 3), gather(-width - 3)
    ix1_, iy1_ = gather(-1), gather(-width - 2)
    dx, dy = 0.5 * (ix1 - ix1_), 0.5 * (iy1 - iy1_)
    derivative = torch.stack((dx, dy), dim=-1).unsqueeze(-1)
    dxx, dyy = ix1 - 2 * i_ + ix1_, iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    hessian = torch.stack((dxx, dxy, dxy, dyy), dim=-1).reshape(batch, channels, 2, 2)
    eye = torch.eye(2, device=encoded.device, dtype=torch.float64)
    inverse = torch.linalg.inv(hessian.to(torch.float64) + np.finfo(np.float32).eps * eye)
    correction = (inverse @ derivative.to(torch.float64)).squeeze(-1).to(torch.float32)
    keypoints.sub_(correction)
    return keypoints.cpu().numpy(), scores.cpu().numpy()
