"""Lossless, worker-scoped SAM input from MAMMA's existing video reader."""
from contextlib import contextmanager
import os


@contextmanager
def direct_video_frames(pipeline_type, video_source_type, sam_video, progress):
    """Replace only registered VideoSource inputs; restore upstream hooks on exit.

    The pinned SAM loader otherwise reads lossless PNGs into one CPU FP32 frame
    tensor. Keep its PIL resize, float64 division, FP32 assignment and in-place
    normalization exactly, but decode through the same reader used by YOLO/CLIP.
    Image-directory inputs retain the upstream loader. Readers belong to the
    upstream pipeline; this adapter owns no captures or background threads.
    """
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    original_resolve = pipeline_type._resolve_sam_video_source
    original_load = sam_video.load_video_frames
    registered = {}

    def key(path):
        return os.path.normcase(os.path.abspath(os.fspath(path)))

    def resolve(self, frames, cam_data=None):
        if not isinstance(frames, video_source_type):
            return original_resolve(self, frames, cam_data)
        # Cameras are processed sequentially. Do not retain every camera's reader
        # or create empty image files just to give SAM a list of frame indices.
        registered.clear()
        registered[key(frames.video_path)] = frames
        return str(frames.video_path)

    def load(video_path, image_size, offload_video_to_cpu,
             async_loading_frames=False, compute_device=torch.device('cuda'), **kwargs):
        frames = registered.get(key(video_path))
        if frames is None:
            return original_load(video_path=video_path, image_size=image_size,
                                 offload_video_to_cpu=offload_video_to_cpu,
                                 async_loading_frames=async_loading_frames,
                                 compute_device=compute_device, **kwargs)
        n = len(frames)
        if n < 1:
            raise ValueError('MAMMA SAM input has no video frames.')
        # Like the pinned eager loader: one normalized tensor plus one decoded
        # image at a time. This is not an additional whole-video RGB cache.
        images = torch.zeros(n, 3, image_size, image_size, dtype=torch.float32)
        height = width = None
        for i in range(n):
            if i % 30 == 0:
                progress('progress', current=i, total=n,
                         message='Prepare SAM input: ' + frames.cam_name)
            bgr = frames.reader.read_bgr(i)
            if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
                raise ValueError(f'Cannot decode MAMMA SAM frame {i} in {frames.cam_name}.')
            h, w = bgr.shape[:2]
            if height is not None and (h, w) != (height, width):
                raise ValueError('MAMMA camera resolution changed during decoding.')
            height, width = h, w
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with Image.fromarray(rgb) as image:
                with image.resize((image_size, image_size)) as resized:
                    pixels = np.array(resized) / 255.
            images[i] = torch.from_numpy(pixels).permute(2, 0, 1)
        mean = torch.tensor(kwargs.get('img_mean', (.485, .456, .406)), dtype=torch.float32)[:, None, None]
        std = torch.tensor(kwargs.get('img_std', (.229, .224, .225)), dtype=torch.float32)[:, None, None]
        if not offload_video_to_cpu:
            images = images.to(compute_device)
            mean = mean.to(compute_device)
            std = std.to(compute_device)
        images -= mean
        images /= std
        progress('progress', current=n, total=n,
                 message='SAM input ready without intermediate PNGs: ' + frames.cam_name)
        return images, height, width

    pipeline_type._resolve_sam_video_source = resolve
    sam_video.load_video_frames = load
    try:
        yield
    finally:
        pipeline_type._resolve_sam_video_source = original_resolve
        sam_video.load_video_frames = original_load
        registered.clear()
