"""Skin the locally installed MANO template to the ViDiHand2 preview bones."""
import os
from tempfile import NamedTemporaryFile
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np

from .runtime import ROOT


CACHE = ROOT / ".setup" / "vidihand2-mano-preview-v1.npz"
VERTEX_COUNT = 778
TEMPLATE_SHAPES = dict(vertices=(VERTEX_COUNT, 3), faces=(1538, 3),
                       joints=(21, 3), weights=(VERTEX_COUNT, 16))


def read_template(path=CACHE):
    try:
        with ZipFile(path) as archive:
            if sum(item.file_size for item in archive.infolist()) > 256 * 1024:
                return None
        with np.load(path, allow_pickle=False) as archive:
            template = {name: archive[name] for name in TEMPLATE_SHAPES}
    except (OSError, ValueError, KeyError, BadZipFile):
        return None
    for name, shape in TEMPLATE_SHAPES.items():
        array = template[name]
        if array.shape != shape or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            return None
    faces, weights = template["faces"], template["weights"]
    if (faces.dtype.kind not in "iu" or faces.min() < 0 or faces.max() >= VERTEX_COUNT
            or weights.min() < 0 or not np.allclose(weights.sum(axis=1), 1, atol=1.e-4)):
        return None
    return template


def cache_template(mano, path=CACHE):
    """Save numeric preview data locally while the existing worker owns MANO."""
    if read_template(path) is not None:
        return
    vertices = mano.layer.v_template.detach().cpu().numpy()
    joints = np.concatenate((mano.joint_template.detach().cpu().numpy(),
                             vertices[list(mano.tip_vertices)]))
    path.parent.mkdir(parents=True, exist_ok=True)
    file = NamedTemporaryFile(dir=path.parent, prefix=path.stem+"-", suffix=".part", delete=False)
    part = Path(file.name)
    try:
        with file:
            np.savez_compressed(file, vertices=vertices, joints=joints, faces=mano.faces,
                                weights=mano.layer.lbs_weights.detach().cpu().numpy())
        if read_template(part) is None:
            raise ValueError("Invalid MANO preview template.")
        os.replace(part, path)
    finally:
        part.unlink(missing_ok=True)


class ManoPreview:
    def __init__(self, template):
        from ..mesh_data import adjacency
        from .rig import HAND_HEADS, HAND_TAILS, frame_axes, hand_frames

        self.assets = []
        self.bind_coordinates = []
        generation = object()
        for side, sign in enumerate(([-1, 1, 1], [1, 1, 1])):
            joints = template["joints"] * sign
            vertices = template["vertices"] * sign
            directions = joints[HAND_TAILS] - joints[HAND_HEADS]
            roll_guides = hand_frames(joints[None])[0, HAND_HEADS, :3, 2]
            axes = frame_axes(directions, roll_guides)
            lengths = np.linalg.norm(directions, axis=1)
            if lengths.min() <= 1.e-6:
                raise ValueError("Invalid MANO preview bone lengths.")
            displacement = vertices[None] - joints[HAND_HEADS, None]
            local = (displacement @ axes) / lengths[:, None, None]
            weighted = np.concatenate((local, np.ones((*local.shape[:2], 1))), axis=-1)
            weighted *= template["weights"].T[..., None]
            self.bind_coordinates.append(weighted.astype(np.float32))
            # Mirroring changes the winding and requires its own normal topology.
            faces = np.ascontiguousarray(template["faces"][:, ::-1] if side == 0 else template["faces"],
                                         dtype=np.uint32)
            offsets, neighbors = adjacency(faces, VERTEX_COUNT)
            self.assets.append(dict(mode="raw", key=("mano", generation, side), faces=faces,
                                    vertex_count=VERTEX_COUNT, offsets=offsets, neighbors=neighbors))

    def vertices(self, side, pose):
        return np.einsum("jva,jba->vb", self.bind_coordinates[side], pose[:, :3, :])
