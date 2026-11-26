"""Camera rotations of a moving camera: SimpleVO (default) or DPVO (optional).

Both return world-to-camera rotations (L, 3, 3), the model's camera-motion input
(see exmr.geometry.camera.cam_angvel_from_rotations).
"""

import contextlib
import ctypes
import ctypes.util
import queue
import threading
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch
from tqdm import tqdm

from exmr.geometry.rotation_conversions import quaternion_to_matrix
from exmr.preprocessing.video import frame_paths, frame_size, read_frames

DPVO_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "DPVO"


class SimpleVO:
    """SimpleVO (from GVHMR): relative camera poses from SIFT matches between every step-th
    frame (pycolmap two-view geometry), interpolated to all frames.

    Args:
        scale: frames are downscaled by this factor for matching
        step: frame stride between matched frames
        f_mm: assumed focal length in mm of a full-frame (36x24 mm) camera
        progress: show a progress bar
        frame_reader: optional image iterator. Receives file paths normally,
            or sampled integer indices when the caller supplies frame_count.
    """

    def __init__(self, scale=0.5, step=8, f_mm=24, progress=True, frame_reader=None):
        self.scale, self.step, self.f_mm, self.progress = scale, step, f_mm, progress
        self.sift = cv2.SIFT_create()
        # RGB MoCap: optional explicit pixel reader preserves the validated
        # canonical-video sampling without changing pose estimation.
        self.frame_reader = frame_reader

    def __call__(self, frame_dir=None, *, frame_count=None):
        """JPEG frame_dir, or frame_reader plus frame_count, -> (L, 3, 3) rotations.

        World-to-camera, float64, first one identity. The callback/count route
        can read a canonical video without extracting duplicate JPEG files.
        """
        with _opencv_multithreading():
            return torch.from_numpy(self.poses(frame_dir, frame_count=frame_count)[:, :3, :3])

    def poses(self, frame_dir=None, *, frame_count=None):
        """(L, 4, 4) world-to-camera transforms (float64), the first one identity."""
        _reset_random_state()
        # only the frames that are matched are read: every step-th and the last one
        # RGB MoCap's canonical-video reader accepts frame indices directly;
        # no duplicate JPEG extraction is needed solely to enumerate frames.
        if frame_count is None:
            paths = frame_paths(frame_dir)
            count = len(paths)
        else:
            if self.frame_reader is None or type(frame_count) is not int or frame_count < 1:
                raise ValueError("frame_count requires a frame_reader and a positive integer")
            paths = None
            count = frame_count
        sample_idxs = sorted({*range(0, count, self.step), count - 1})
        sources = [paths[i] for i in sample_idxs] if paths is not None else sample_idxs
        frames = [  # (H, W, 3) RGB
            cv2.cvtColor(
                cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2RGB,
            )
            for img in (self.frame_reader or read_frames)(sources)
        ]
        H, W = frames[0].shape[:2]

        # pinhole camera with the principal point at the center
        focal_length = (W**2 + H**2) ** 0.5 / (24**2 + 36**2) ** 0.5 * self.f_mm
        camera = pycolmap.Camera(
            camera_id=0, model="SIMPLE_PINHOLE", width=W, height=H,
            params=[focal_length, W / 2, H / 2],
        )  # fmt: skip
        options = pycolmap.TwoViewGeometryOptions(
            min_num_inliers=10,
            min_E_F_inlier_ratio=0.8,
            max_H_inlier_ratio=0.9,
            compute_relative_pose=True,
        )

        T_w2c = [np.eye(4)]  # T_w2c @ p_w = p_c
        features = self._features(frames[0])
        for frame in tqdm(frames[1:], desc="Visual odometry", disable=not self.progress):
            prev, features = features, self._features(frame)
            pts0, pts1 = _match(prev, features)
            if len(pts0) < options.min_num_inliers:
                T_w2c.append(T_w2c[-1].copy())
                continue
            matches = np.stack([np.arange(len(pts0)), np.arange(len(pts0))], axis=-1)
            answer = pycolmap.estimate_calibrated_two_view_geometry(
                camera, pts0.astype(np.float64), camera, pts1.astype(np.float64),
                matches=matches, options=options,
            )  # fmt: skip
            T_delta = np.eye(4)  # previous -> current camera
            T_delta[:3] = answer.cam2_from_cam1.matrix().astype(np.float32)
            T_w2c.append(T_delta @ T_w2c[-1])

        return _interpolate_poses(T_w2c, sample_idxs)

    def _features(self, img):
        """SIFT keypoints and descriptors of an RGB image."""
        # converting with the BGR weights (as in the evaluated setup) only changes the gray mix
        return self.sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)


def _match(features0, features1):
    """FLANN matches of two images' SIFT features with Lowe's ratio test -> (N, 2), (N, 2)
    float32 points."""
    (kp0, des0), (kp1, des1) = features0, features1
    if des0 is None or des1 is None or len(des0) < 2 or len(des1) < 2:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))  # KD-tree
    good = [
        pair[0]
        for pair in flann.knnMatch(des0, des1, k=2)
        if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance
    ]
    pts0 = np.float32([kp0[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 2)
    return pts0, pts1


@contextlib.contextmanager
def _opencv_multithreading():
    """OpenCV with its default number of threads. Ultralytics (the person tracker) turns
    OpenCV's multithreading off for the whole process; SimpleVO is up to ~2x faster with it, with
    identical results."""
    num_threads = cv2.getNumThreads()
    cv2.setNumThreads(-1)  # the default
    try:
        yield
    finally:
        cv2.setNumThreads(num_threads)


def _reset_random_state():
    """Reset the process-global random state SimpleVO depends on (C rand(), used by OpenCV's FLANN
    kd-trees; OpenCV's RNG; COLMAP's RANSAC) to that of a fresh process, so that results do not
    depend on what ran before in the same process. The order matters: COLMAP's seeding also
    calls srand()."""
    pycolmap.set_random_seed(0)
    # RGB MoCap Windows compatibility: the native C runtime is UCRT, not glibc.
    ctypes.CDLL(ctypes.util.find_library("c") or "ucrtbase.dll").srand(1)
    cv2.setRNGSeed(0)


class DPVOOdometry:
    """DPVO (Deep Patch Visual Odometry, https://github.com/princeton-vl/DPVO), the optional
    alternative to SimpleVO; needs DPVO built from third_party/DPVO (see docs/INSTALL.md).

    Unlike GVHMR's wrapper, frames are timestamped with their index (as in DPVO's own demo)
    instead of the wall-clock time, which DPVO's damped-linear motion model would otherwise turn
    into a dependence on processing speed. DPVO is not deterministic (random patch selection,
    atomic float accumulation on the GPU): repeated runs differ by ~0.1-0.4 degrees on average.

    Args:
        checkpoint: DPVO weights (dpvo.pth)
        resize: frames are downscaled by this factor, and further if their longer side would
            still exceed max_side pixels (4K videos: to 960 px, like 1080p ones; DPVO's time grows
            with the pixels, its rotations do not get better)
        max_side: see resize
        buffer: DPVO frame buffer size
    """

    def __init__(self, checkpoint, resize=0.5, max_side=960, buffer=4000):
        self.checkpoint, self.resize, self.max_side, self.buffer = (
            checkpoint, resize, max_side, buffer
        )  # fmt: skip

    def __call__(self, frame_dir, K):
        """frame_dir with %06d.jpg frames and their (3, 3) intrinsics -> (L, 3, 3) world-to-camera
        rotations (float32)"""
        from dpvo.config import cfg
        from dpvo.dpvo import DPVO

        intrinsics = torch.stack([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]).float()  # fx, fy, cx, cy
        paths = frame_paths(frame_dir)
        resize = min(self.resize, self.max_side / max(frame_size(frame_dir)))
        # frames are decoded in background threads (OpenCV releases the GIL), overlapping with
        # DPVO on the GPU. A thread rather than a process: importing dpvo switches the
        # multiprocessing start method to "spawn" for the whole interpreter.
        frames = queue.Queue(maxsize=8)
        stop = threading.Event()
        reader = threading.Thread(
            target=_dpvo_frame_stream,
            args=(frames, stop, paths, intrinsics, resize),
            daemon=True,
        )
        reader.start()

        slam = None
        pbar = tqdm(total=len(paths), desc="DPVO")
        # no_grad is essential: DPVO writes network outputs into persistent buffers, which
        # would otherwise keep every frame's autograd graph alive (memory grows ~0.25 GB/frame)
        try:
            with torch.no_grad():
                while (item := frames.get()) is not None:  # until the end marker
                    if isinstance(item, Exception):
                        raise item
                    t, image, frame_intrinsics = item
                    pbar.update()
                    image = torch.from_numpy(image).permute(2, 0, 1).cuda()
                    if slam is None:
                        cfg.merge_from_file(str(DPVO_ROOT / "config" / "default.yaml"))
                        cfg.BUFFER_SIZE = self.buffer
                        slam = DPVO(
                            cfg, str(self.checkpoint), ht=image.shape[1], wd=image.shape[2],
                            viz=False,
                        )  # fmt: skip
                    slam(t, image, frame_intrinsics.cuda())
                for _ in range(12):
                    slam.update()
                traj = _dpvo_terminate(slam)  # (L, 7) camera-to-world [t, q (x, y, z, w)]
        finally:
            pbar.close()
            stop.set()  # lets the reader exit even if it is blocked on the full queue
            reader.join()

        R_c2w = quaternion_to_matrix(torch.from_numpy(traj[:, [6, 3, 4, 5]]))
        return R_c2w.mT


def _dpvo_terminate(slam):
    """slam.terminate(), without its recursion: the (L, 7) camera-to-world poses.

    DPVO drops frames with little camera motion and gets their poses by chaining relative poses
    back to a kept frame, recursively with one call per frame. When the camera barely moves for
    a while, the chains exceed Python's recursion limit (~1000 frames) and cost quadratic time.
    Here each pose is computed once, from the pose of the earlier frame it refers to, with the
    same operations.
    """
    from dpvo import lietorch
    from dpvo.lietorch import SE3

    assert not (slam.cfg.LOOP_CLOSURE or slam.cfg.CLASSIC_LOOP_CLOSURE)  # not handled here
    for _ in range(12):
        slam.ran_global_ba[slam.n] = False
        slam.update()
    kept = {slam.pg.tstamps_[i]: slam.pg.poses_[i] for i in range(slam.n)}
    poses = []
    for t in range(slam.counter):
        if t in kept:
            poses.append(SE3(kept[t]))
        else:
            t0, dP = slam.pg.delta[t]  # t0 < t
            poses.append(dP * poses[t0])
    return lietorch.stack(poses, dim=0).inv().data.cpu().numpy()


def _dpvo_frame_stream(frames, stop, paths, intrinsics, resize):
    """Puts (frame index, downscaled BGR frame cropped to multiples of 16, intrinsics) into the
    frames queue, then None at the end (or the exception, if reading fails)."""

    def put(item):
        while not stop.is_set():
            try:
                frames.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    try:
        for t, frame in enumerate(read_frames(paths)):
            if stop.is_set():
                break
            image = cv2.resize(frame, None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA)
            h, w, _ = image.shape
            put((t, image[: h - h % 16, : w - w % 16], intrinsics * resize))
        put(None)
    except Exception as error:
        put(error)


def _interpolate_poses(T_w2c, sample_idxs):
    """Poses at sample_idxs -> all frames: linear translation, SLERP rotation."""
    sample_idxs = np.array(sample_idxs)
    translations = np.array([T[:3, 3] for T in T_w2c])
    quaternions = np.array([_rotmat_to_quat(T[:3, :3]) for T in T_w2c])

    out = []
    for i in range(sample_idxs[-1] + 1):
        if i in sample_idxs:
            out.append(T_w2c[np.where(sample_idxs == i)[0][0]])
            continue
        nxt = np.searchsorted(sample_idxs, i)
        prv = nxt - 1
        t = (i - sample_idxs[prv]) / (sample_idxs[nxt] - sample_idxs[prv])
        T = np.eye(4)
        T[:3, :3] = _quat_to_rotmat(_slerp(quaternions[prv], quaternions[nxt], t))
        T[:3, 3] = (1 - t) * translations[prv] + t * translations[nxt]
        out.append(T)
    return np.array(out)


def _rotmat_to_quat(R):
    """3x3 rotation matrix -> quaternion [w, x, y, z]."""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2  # 4 * qw
        return np.array([0.25 * S, (m21 - m12) / S, (m02 - m20) / S, (m10 - m01) / S])
    if (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2  # 4 * qx
        return np.array([(m21 - m12) / S, 0.25 * S, (m01 + m10) / S, (m02 + m20) / S])
    if m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2  # 4 * qy
        return np.array([(m02 - m20) / S, (m01 + m10) / S, 0.25 * S, (m12 + m21) / S])
    S = np.sqrt(1.0 + m22 - m00 - m11) * 2  # 4 * qz
    return np.array([(m10 - m01) / S, (m02 + m20) / S, (m12 + m21) / S, 0.25 * S])


def _quat_to_rotmat(q):
    """Quaternion [w, x, y, z] -> 3x3 rotation matrix."""
    qw, qx, qy, qz = q
    return np.array(
        [
            [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx**2 - 2 * qy**2],
        ]
    )


def _slerp(q0, q1, t):
    """Spherical linear interpolation between quaternions q0 and q1."""
    dot = np.dot(q0, q1)
    if dot < 0.0:  # shorter path
        q1 = -q1
        dot = -dot
    if dot > 0.9995:  # nearly identical: normalized linear interpolation
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    s0 = np.cos(theta) - dot * np.sin(theta) / np.sin(theta_0)
    s1 = np.sin(theta) / np.sin(theta_0)
    return (s0 * q0) + (s1 * q1)
