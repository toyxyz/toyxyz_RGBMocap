"""Draw held face landmarks in the same scene transform as the body surface."""
import numpy as np

from .face_data import EDGES, manifest
from .face_surface import FaceSurface


def draw_faces(view, actors, lines3d):
    archive=view.face_archive
    if archive is None:
        return
    if actors is None:
        if view.path is None or view.mesh_presented_index!=view.index:
            return
        native=view.native_vertices is not None
        asset=view.raw_asset.get('raw_asset') if native and view.raw_asset else view.mesh_asset
        frames=view.native_vertices if native else view.mesh_frames
        entries=[(None,view.path.parent,view.index,frames,asset,native,None)]
    else:
        entries=[]
        for take,frame in actors:
            if view.scene_presented.get(take.ident)!=(str(take.project),frame):
                continue
            native='native' in take.arrays
            asset=take.raw_asset.get('raw_asset') if native and take.raw_asset else take.asset
            entries.append((take.ident,take.project,frame,take.arrays.get('native' if native else 'mesh'),
                            asset,native,take.transform))
    for person,project,frame,frames,asset,native,transform in entries:
        if asset is None or frames is None or not project.is_relative_to(archive.root):
            continue
        try:
            if project not in view.face_manifests:
                view.face_manifests[project]=manifest(project)
            slot, sample, index=archive.resolve(project,frame,person,view.face_manifests[project])
            if sample<0:
                continue
            key=asset['key']
            if key not in view.face_surfaces:
                view.face_surfaces[key]=FaceSurface(archive,asset,native)
            points,front=view.face_surfaces[key].points(frames[frame],archive.offsets[slot,sample])
        except (OSError,KeyError,ValueError,IndexError):
            continue
        if transform is not None:
            points=points@transform[:3,:3].T+transform[:3,3]
            front=transform[:3,:3]@front
        eye,_=view._paint_camera
        if (eye-points.mean(0))@front>0:
            lines3d(points[EDGES], '#7de1ec' if sample==index else '#8bb5c0', 1.25)
