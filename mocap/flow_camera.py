"""FlowHMR camera policy; explicit static mode never guesses from actor motion."""
import math
import numpy as np

def static_camera(frames,width,height):
    if frames<1 or width<1 or height<1:raise ValueError('Invalid static camera dimensions')
    raw=np.repeat(np.eye(4,dtype=np.float32)[None],frames,axis=0)
    yup=raw.copy();yup[:,1,1]=-1;yup[:,2,2]=-1
    focal=math.hypot(width,height)
    k=np.array([[focal,0,width/2],[0,focal,height/2],[0,0,1]],np.float32)
    return dict(camera_RT=raw,RT=raw,RT_vggt=raw,camera_RT_yup=yup,K=np.repeat(k[None],frames,axis=0),
                image_size_hw=np.array([height,width]),camera_sampling_interval=-1,
                camera_sample_indices=np.array([],np.int64),camera_world='explicit_static_opencv',
                k_space='original_frame',camera_policy_version=1)

def relative_camera(rt):
    """Match official training: R(t) @ R(0).T; keep translation for placement."""
    value=rt.clone()
    value[:,:3,:3]=rt[:,:3,:3]@rt[0,:3,:3].T
    return value


def saved_camera_options(project,state):
    """People/takes inherit the scene camera policy rather than a row-local value."""
    import json
    from pathlib import Path
    project=Path(project);options=state.get('flow_options',{})
    if state.get('people_root'):
        root=(project/state['people_root']).resolve()
        if not project.resolve().is_relative_to(root) or not (root/'people.json').is_file():
            return options  # Detached baked takes remain viewable; processing validates the group.
        options=json.loads((root/'project.json').read_text(encoding='utf-8')).get('flow_options',options)
    return options


def saved_interval(project,state):
    return saved_camera_options(project,state).get('camera_interval',0)

