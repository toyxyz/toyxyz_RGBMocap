"""Run ViDiHand2 inference and commit a hand and camera take."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .motion import load


FIXED_CAMERA = np.array([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]])
MAX_VIDEO_BYTES = 2 * 1024**3


def progress(message):
    print("@@MOCAP@@"+json.dumps(dict(type="progress",message=message)),flush=True)


def decode_video(video):
    import imageio_ffmpeg
    import cv2

    cv2.setNumThreads(2)
    reader = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24", output_params=["-fps_mode","passthrough"])
    try:
        meta = next(reader)
        width,height = meta["size"]
        fps = float(meta["fps"])
        if not np.isfinite(fps) or fps <= 0 or min(width,height) < 16:
            raise ValueError("The input video does not provide valid dimensions and FPS.")
        scale = min(1., 960./max(width,height))
        size = (round(width*scale), round(height*scale))
        budget = MAX_VIDEO_BYTES // (size[0]*size[1]*3)
        if float(meta.get("duration") or 0.) * fps > budget + fps:
            raise ValueError("The decoded clip exceeds the memory budget. Trim it to a shorter take.")
        frames = []
        for raw in reader:
            if len(frames) >= budget:
                raise ValueError("The decoded clip exceeds the memory budget. Trim it to a shorter take.")
            image = np.frombuffer(raw,dtype=np.uint8).reshape(height,width,3)
            frames.append(cv2.resize(image,size,interpolation=cv2.INTER_AREA) if scale < 1. else image.copy())
    finally:
        reader.close()
    if len(frames) < 2:
        raise ValueError("ViDiHand2 requires at least two video frames.")
    result = np.stack(frames)
    return result, fps, (width,height)


def tilt_camera_track(rotation, center, degrees):
    """Pitch the world track about the first camera position, keeping relative motion."""
    if not -90. <= degrees <= 90.:
        raise ValueError("Camera tilt must be between -90 and 90 degrees.")
    if degrees == 0.:
        return rotation, center
    right = rotation[0, :, 0].copy()
    right[2] = 0.
    length = np.linalg.norm(right)
    if length < 1.e-6:
        raise ValueError("The first camera frame has no horizontal right axis for tilt.")
    world = Rotation.from_rotvec(right / length * np.radians(-degrees)).as_matrix()
    anchor = center[0]
    return world @ rotation, (center - anchor) @ world.T + anchor


def cameras(path, n, fps, src_size, size, start=0.):
    rotation = np.broadcast_to(FIXED_CAMERA,(n,3,3)).copy()
    center = np.zeros((n,3))
    width,height = size
    focal = .5*max(width,height) / np.tan(np.radians(37.5))
    key = np.broadcast_to(np.array([[focal,0.,(width-1)*.5],[0.,focal,(height-1)*.5],[0.,0.,1.]]),
                          (n,3,3)).copy()
    mode = "fixed_estimated"
    if path is None:
        return key,rotation,center,mode

    if path.suffix.lower()==".npz":
        data=np.load(path,allow_pickle=False)
        with data:
            if data["K"].shape!=(n,3,3) or data["camera_RT_yup"].shape!=(n,4,4):
                raise ValueError("Estimated camera frames do not match the prepared clip.")
            key=data["K"].astype(float)
            ext=data["camera_RT_yup"].astype(float)
        from .motion import Y_UP
        key[:,0]*=width/src_size[0]
        key[:,1]*=height/src_size[1]
        rotation=np.swapaxes(ext[:,:3,:3]@Y_UP,1,2)
        center=-np.einsum("tij,tj->ti",rotation,ext[:,:3,3])
        return key,rotation,center,"automatic"

    from utils.io import load_camera
    data = load_camera(path)
    if data["size"] != src_size:
        raise ValueError("Camera JSON image dimensions must match the input video.")
    ids=data["source_frame_ids"]
    if np.any(np.diff(ids)<=0):
        raise ValueError("Camera JSON must specify increasing source-frame numbers.")
    ids=ids-ids[0]
    indices=(start+np.arange(n)/fps)*data["fps"]
    if indices[0]<0 or indices[-1]>ids[-1]+1.01:
        raise ValueError("Camera JSON does not cover the prepared video range.")
    indices=np.minimum(indices,ids[-1])
    key = np.stack([np.interp(indices,ids,data["K"][:,i,j]) for i in range(3) for j in range(3)],
                   axis=-1).reshape(n,3,3)
    key[:,0]*=width/src_size[0]
    key[:,1]*=height/src_size[1]
    known = np.isfinite(data["E"]).all(axis=(1,2))
    if not known.any():
        return key,rotation,center,"fixed_calibrated"
    ext = data["E"][known]
    ext_frame = ids[known]
    orient = ext[:,:3,:3].transpose(0,2,1)
    if (not np.allclose(ext[:,3],[0,0,0,1],atol=1e-5) or
            not np.allclose(orient.transpose(0,2,1)@orient,np.eye(3),atol=1e-4) or
            not np.allclose(np.linalg.det(orient),1,atol=1e-4)):
        raise ValueError("Camera extrinsics must be rigid world-to-camera transforms.")
    anchor = -np.einsum("tij,tj->ti",orient,ext[:,:3,3])
    if len(ext_frame)==1:
        rotation[:] = orient[0]
        center[:] = anchor[0]
    else:
        rotation = Slerp(ext_frame, Rotation.from_matrix(orient))(
            np.clip(indices,ext_frame[0],ext_frame[-1])).as_matrix()
        center = np.stack([np.interp(indices,ext_frame,anchor[:,i]) for i in range(3)],axis=1)
    return key,rotation,center,"imported"


def checkpoint_paths(upstream, managed):
    from ..models import BY_KEY
    from ..storage import sha256
    from .assets import CHECKPOINTS

    if not (upstream / "vidihand_v2/inference.py").is_file():
        raise FileNotFoundError("ViDiHand2 runtime is missing vidihand_v2/inference.py. See docs/FIRST_PERSON_ARMS.md.")
    if (not isinstance(managed, dict) or managed.keys() - CHECKPOINTS.keys()
            or any(not isinstance(path, str) or not Path(path).is_absolute() for path in managed.values())):
        raise ValueError("Invalid managed ViDiHand2 model paths.")
    files, missing = {}, []
    for key, relative in CHECKPOINTS.items():
        model = BY_KEY[key]
        path = Path(managed[key]) if key in managed else upstream / "checkpoints" / relative
        if (not path.is_file() or path.stat().st_size != model.minimum_bytes
                or (key == "mano_right" and sha256(path) != model.expected_sha256)):
            missing.append(model.path.name)
        else:
            files[relative] = path
    if missing:
        raise FileNotFoundError("Open Model Manager > ViDiHand2 to download or import: " + ", ".join(missing))
    return files


def track_hands(files, video_name, rgb, key, fps):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ViDiHand2 requires a functioning CUDA runtime.")
    from utils import io as upstream_io

    # Set the resolver before the upstream modules import it. All three files
    # are already present and the isolated worker never downloads in inference.
    upstream_io.checkpoint = files.__getitem__
    from vidihand_v2.inference import load_model
    from .mano_preview import cache_template

    torch.set_num_threads(4)
    progress("Tracking both hands with ViDiHand2")
    tracker = load_model()
    sample = {"name":video_name,"video":rgb,"K":key,"fps":fps,"source_frame_ids":np.arange(len(rgb))}
    joint = tracker(sample,fit_mano=False)["joint"]
    try:
        cache_template(tracker.mano)
    except (OSError, ValueError) as error:
        print(f"Could not prepare the MANO preview: {error}", file=sys.stderr, flush=True)
    del tracker
    torch.cuda.empty_cache()
    return np.asarray(joint["joints3d"],dtype=float), np.asarray(joint["valid"],dtype=bool)


def run(args):
    refit = args.refit_from
    prior = load(refit) if refit is not None else None
    if refit is None:
        upstream = args.upstream.resolve()
        files = checkpoint_paths(upstream, args.managed_models)
        sys.path.insert(0,str(upstream))
    progress("Decoding the video")
    rgb,fps,source_size = decode_video(args.video)
    size = tuple(rgb.shape[2:0:-1])
    key,orient,center,mode = cameras(args.camera,len(rgb),fps,source_size,size,args.camera_start)
    tilt = args.camera_tilt if mode in ("automatic", "fixed_estimated") else 0.
    orient,center = tilt_camera_track(orient,center,tilt)
    if prior is None:
        hand,tracked = track_hands(files,args.video.stem,rgb,key,fps)
    else:
        if (mode not in ("automatic","fixed_estimated") or str(prior["camera_mode"])!=mode
                or abs(float(prior["fps"])-fps)>.1
                or tuple(prior["source_size"])!=source_size or tuple(prior["video_size"])!=size
                or not np.array_equal(prior["K"],key.astype(np.float32))):
            raise ValueError("Saved hand tracks do not match this video and camera. Reconstruct 3D first.")
        progress("Reusing the saved ViDiHand2 hand tracks")
        hand,tracked = prior["hand_camera"].astype(float),prior["tracked"]
    if (hand.shape != (len(rgb),2,21,3) or tracked.shape != (len(rgb),2)
            or not np.isfinite(hand[tracked]).all() or not tracked.any()):
        raise ValueError("ViDiHand2 did not return a valid hand track.")
    world = np.einsum("tij,tskj->tski", orient, hand) + center[:, None, None]
    f = NamedTemporaryFile(dir=args.output.parent,prefix=args.output.stem+"-",suffix=".part",delete=False)
    part = Path(f.name)
    try:
        with f:
            np.savez_compressed(f, version=np.int32(2), hand_world=world.astype(np.float32),
                                tracked=tracked, hand_camera=hand.astype(np.float32),
                                K=key.astype(np.float32),
                                camera_to_world=orient, camera_position=center,
                                fps=float(fps),source_size=np.asarray(source_size),
                                video_size=np.asarray(size),camera_mode=mode,camera_tilt=tilt)
        os.replace(part,args.output)
    finally:
        part.unlink(missing_ok=True)
    progress(f"Ready. Tracked {int(tracked.sum())} hand poses across {len(rgb)} frames.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for kind in ("upstream","video","output"):
        parser.add_argument("--"+kind,type=Path,required=True)
    parser.add_argument("--camera",type=Path)
    parser.add_argument("--camera-start",type=float,default=0.)
    parser.add_argument("--camera-tilt",type=float,default=0.)
    parser.add_argument("--refit-from",type=Path)
    parser.add_argument("--managed-models",type=json.loads,default={})
    parser.add_argument("--cancel",type=Path)
    args=parser.parse_args()
    started = time.monotonic()
    stage = "vidihand_refit" if args.refit_from is not None else "vidihand_track"
    if args.cancel is not None:
        import threading

        def watch():
            while not args.cancel.is_file():
                time.sleep(.2)
            os._exit(130)

        threading.Thread(target=watch,daemon=True).start()
    try:
        run(args)
    except (OSError,ValueError,RuntimeError,ImportError,KeyError) as error:
        print("@@MOCAP@@"+json.dumps(dict(type="failed",stage=stage,message=str(error))),flush=True)
        return 1
    print("@@MOCAP@@"+json.dumps(dict(type="complete",stage=stage,output=str(args.output),
                                     elapsed=time.monotonic()-started)),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
