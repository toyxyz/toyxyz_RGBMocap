"""Preserve PromptHMR's shared world and constant reference-camera basis."""
from pathlib import Path
from .storage import atomic_json
from .people_store import load_people,media_receipt
from .scene_camera import read_camera,VERSION,take_projects


def prepare_cameras(root):
    root=Path(root);rows=take_projects(root,load_people(root))
    if not rows:raise ValueError('PromptHMR shared scene has no actors.')
    for _,project in rows:
        if read_camera(project) is None:raise ValueError('PromptHMR source or body changed. Reconstruct or prepare the selected source again.')
    ident,project=rows[0];camera=read_camera(project);target=root/'people_scene.json'
    atomic_json(target,dict(version=VERSION,input=media_receipt(root),anchor=project.relative_to(root).as_posix(),anchor_id=ident,
        world_to_camera=camera['world_to_camera'],sources=camera['sources'],floor=0.,K=camera['K'],camera_mode='moving' if camera.get('camera_track') else 'static',scale=1.))
    return target
