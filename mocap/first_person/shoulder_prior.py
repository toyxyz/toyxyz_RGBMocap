"""Optional local weights for the camera and wrist shoulder prior."""
from zipfile import BadZipFile, ZipFile

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .runtime import ROOT

CACHE = ROOT / ".setup" / "vidihand2-shoulder-prior.npz"
SHAPES = {"mean": (85,), "sd": (85,), "0.weight": (96,85), "0.bias": (96,),
          "3.weight": (48,96), "3.bias": (48,), "5.weight": (3,48), "5.bias": (3,)}


def load_prior(path=CACHE):
    if not path.is_file():
        return None
    try:
        with ZipFile(path) as archive:
            if sum(item.file_size for item in archive.infolist()) > 256*1024:
                raise ValueError("Shoulder prior is too large.")
        with np.load(path, allow_pickle=False) as archive:
            weights = {name: archive[name] for name in SHAPES}
    except (BadZipFile, KeyError) as error:
        raise ValueError("Invalid ViDiHand2 shoulder prior.") from error
    for name, shape in SHAPES.items():
        values = weights[name]
        if values.shape != shape or values.dtype.kind != "f" or not np.isfinite(values).all():
            raise ValueError("Invalid ViDiHand2 shoulder prior.")
    if (weights["sd"] <= 0.).any():
        raise ValueError("Invalid ViDiHand2 shoulder prior normalization.")
    return weights


def predict(weights, camera, root, center, torso, wrist, tracked, fps):
    local = np.einsum("tji,tsj->tsi", torso, wrist-center[:,None])
    hands = np.where(tracked[...,None], local, 0.)
    forward = np.einsum("tji,tj->ti", torso, camera[:,:,2])
    right = np.einsum("tji,tj->ti", torso, camera[:,:,0])
    velocity = np.einsum("tji,tj->ti", torso, np.gradient(root,axis=0)*fps)
    features = gaussian_filter1d(np.c_[hands.reshape(len(camera),6),tracked,forward,right,velocity],
                                3.*fps/30., axis=0, mode="nearest")
    offsets = np.rint(np.array([-18,-9,0,9,18])*fps/30.).astype(int)
    indices = np.clip(np.arange(len(camera))[:,None]+offsets,0,len(camera)-1)
    x = (features[indices].reshape(len(camera),-1).astype(np.float32)-weights["mean"])/weights["sd"]
    for layer in ("0","3"):
        x = x @ weights[layer+".weight"].T + weights[layer+".bias"]
        x = x/(1.+np.exp(-np.clip(x,-60.,60.)))
    shift = x @ weights["5.weight"].T + weights["5.bias"]
    shift = np.clip(shift,[-.15,-.22,-.20],[.15,.12,.12])
    return gaussian_filter1d(shift,6.*fps/30.,axis=0,mode="nearest")
