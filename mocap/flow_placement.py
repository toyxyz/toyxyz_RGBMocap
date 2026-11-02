"""Rigid, metric FlowHMR WV placement; original predictions stay immutable.

One scene WV basis, fixed actor XZ offsets and an equal-actor camera-height
estimate. This is not metric moving-camera translation or interaction fitting.
"""
from pathlib import Path
import json
import numpy as np
from .result_artifacts import file_stat
from .storage import atomic_json

VERSION=2


def wv_rotation(rotation):
    r=np.asarray(rotation,dtype=float)
    if r.shape!=(3,3) or not np.isfinite(r).all() or not np.allclose(r.T@r,np.eye(3),atol=2e-4) or not np.isclose(np.linalg.det(r),1,atol=2e-4):
        raise ValueError('Invalid FlowHMR camera rotation')
    z=r.T@np.array([0.,0.,1.]);x=np.cross([0.,1.,0.],z)
    if np.linalg.norm(x)<1e-5:
        raise ValueError('FlowHMR ground heading is undefined for a vertical camera; use a different take.')
    x/=np.linalg.norm(x);z=np.cross(x,[0.,1.,0.]);z/=np.linalg.norm(z)
    return np.stack((x,[0.,1.,0.],z))


def placement_fit(pelvis,anchors,cameras,scene_first):
    """Match official WV yaw, retain gravity, and fit only a constant offset."""
    pelvis=np.asarray(pelvis,float);cameras=np.asarray(cameras,float);a=np.asarray(anchors,float)
    if pelvis.ndim!=2 or pelvis.shape[1]!=3 or cameras.shape!=(len(pelvis),4,4) or not np.isfinite(pelvis).all():
        raise ValueError('Invalid FlowHMR placement frames')
    if a.ndim!=2 or a.shape[1]!=4 or not 1<=len(a)<=30 or not np.isfinite(a).all():
        raise ValueError('No valid SAM placement anchors. Reconstruct body features before preparing preview.')
    ids=a[:,0].astype(np.int64)
    if not np.array_equal(ids,a[:,0]) or len(np.unique(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=min(len(pelvis),30)) or np.any(a[:,3]<=.05):
        raise ValueError('Invalid SAM placement anchor indices or depth')
    shared=wv_rotation(scene_first);local=wv_rotation(cameras[0,:3,:3])
    q=shared@local.T
    # A later take has its OWN WV heading; transport it into the scene WV.
    rotated=pelvis@q.T
    cam=cameras[:,:3,:3]@shared.T
    for rotation in cameras[ids,:3,:3]:wv_rotation(rotation)
    difference=np.einsum('tji,tj->ti',cam[ids],a[:,1:])-rotated[ids]
    offset=np.median(difference,axis=0);height=-float(offset[1]);offset[1]=0.
    world=rotated+offset
    camera_points=np.einsum('tij,tj->ti',cam[ids],world[ids]-[0.,height,0.])
    residual=camera_points-a[:,1:]
    return dict(rotation=q,offset=offset,camera_rotation=cam,scene_basis=shared,camera_height=height,
                anchor_residual_m=dict(mean=float(np.linalg.norm(residual,axis=1).mean()),maximum=float(np.linalg.norm(residual,axis=1).max())))


def context(project,generated=False):
    from .flow_backend import scene_root
    from .models import BY_KEY
    project=Path(project);root=scene_root(project)
    state=json.loads((project/'project.json').read_text(encoding='utf-8'))
    inputs={name:file_stat(project/name) for name in ('flow_prediction.pt','flow_placement.json','flow_camera.npz','tracking.npz')}
    from .flow_hamer import prediction_name
    selected=prediction_name(project,generated=generated)
    if selected!='flow_prediction.pt':inputs[selected]=file_stat(project/selected)
    inputs['scene_camera']=file_stat(root/'flow_camera.npz')
    return dict(version=VERSION,inputs=inputs,frames=state['frames'],take=state.get('person_frames'),
                model=(file_stat(BY_KEY['smplh'].path) if BY_KEY['smplh'].path.exists() else json.loads((project/'flow_world.json').read_text(encoding='utf-8'))['context']['model']))


def current(project):
    try:
        project=Path(project);record=json.loads((project/'flow_world.json').read_text(encoding='utf-8'))
        return (np.isfinite(record['camera_height']) and np.asarray(record['offset']).shape==(3,) and np.isfinite(record['offset']).all()
                and not (project/'flow_world_pending.json').exists() and record['context']==context(project)
                and record['output']==file_stat(project/'hmr4d_results.pt'))
    except (OSError,ValueError,KeyError,TypeError):return False


def needs_update(project):
    project=Path(project)
    return (project/'flow_prediction.pt').is_file() and not current(project)


def prediction_current(project):
    """Validate the expensive generation separately from its replaceable placement."""
    from .flow_backend import identity
    project=Path(project)
    try:
        record=json.loads((project/'flowhmr_receipt.json').read_text(encoding='utf-8'))
        from .flow_hamer import prediction_name
        selected=prediction_name(project)
        return (not (project/'flowhmr_pending.json').exists() and record['identity']==identity(project,'flowhmr')
                and record['outputs']['flow_prediction.pt']==file_stat(project/'flow_prediction.pt')
                and (selected=='flow_prediction.pt' or record['outputs'][selected]==file_stat(project/selected)))
    except (OSError,ValueError,KeyError,TypeError):return False


def ensure(project,generated=False,body=None):
    import torch
    from .flow_backend import scene_root,load_tensor,save_tensor,validate_motion
    from .models import BY_KEY
    from flowhmr.core.bodymodels.smpl_skeleton import SMPLSkeleton
    from flowhmr.core.math.geometry import rot6d_to_rotation_matrix,rotation_matrix_to_rot6d
    project=Path(project)
    if current(project):return False
    if not generated and not prediction_current(project):
        raise ValueError('FlowHMR prediction no longer matches its camera/features/models. Reconstruct 3D before rebuilding placement.')
    before=context(project,generated=generated);root=scene_root(project)
    state=json.loads((project/'project.json').read_text(encoding='utf-8'));n=state['frames']
    with np.load(root/'flow_camera.npz',allow_pickle=False) as data:scene_first=data['camera_RT_yup'][0,:3,:3].copy()
    with np.load(project/'flow_camera.npz',allow_pickle=False) as data:cameras=data['camera_RT_yup'].copy()
    if root!=project:
        begin,end=state['person_frames']
        with np.load(root/'flow_camera.npz',allow_pickle=False) as data:
            if end-begin!=n or not np.array_equal(cameras,data['camera_RT_yup'][begin:end]):
                raise ValueError('Person take does not use the matching shared scene camera slice')
    from .flow_hamer import prediction_name
    raw=load_tensor(project/prediction_name(project,generated=generated));validate_motion(raw,n)
    body=body if body is not None else SMPLSkeleton(model_path=str(BY_KEY['smplh'].path))
    with torch.inference_mode():rest=body.compute_j_shaped(raw['shapes'].reshape(1,16))[0,0].numpy()
    pelvis=raw['trans'][0].numpy()+rest
    anchors=json.loads((project/'flow_placement.json').read_text(encoding='utf-8'))
    k=np.asarray(anchors['intrinsics'],float)
    if k.shape!=(3,3) or not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0 or not np.allclose(k[2],[0,0,1]):
        raise ValueError('Invalid original-frame SAM camera intrinsics')
    fit=placement_fit(pelvis,anchors['camera_pelvis'],cameras,scene_first)
    q=torch.from_numpy(fit['rotation'].astype(np.float32));offset=torch.from_numpy(fit['offset'].astype(np.float32))
    placed={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in raw.items()}
    placed['rot6d'][0,:,0]=rotation_matrix_to_rot6d(q@rot6d_to_rotation_matrix(raw['rot6d'][0,:,0]))
    placed['trans'][0]=(raw['trans'][0]+torch.from_numpy(rest))@q.T-torch.from_numpy(rest)+offset
    placed['root_rot6d']=placed['rot6d'][:,:,0];placed['body_rot6d']=placed['rot6d'][:,:,1:]
    # Upstream debug fields are in take WV, not the committed scene WV.
    for key in ('local_joints_positions','smooth_root_pos','pelvis_world'):placed.pop(key,None)
    placed.update(scene_offset=offset,source_representation='smplh',flow_version=1,placement_version=VERSION)
    validate_motion(placed,n)
    atomic_json(project/'flow_world_pending.json',dict(version=VERSION))
    if before!=context(project,generated=generated):raise ValueError('FlowHMR placement inputs changed; retry Prepare Preview.')
    save_tensor(project/'hmr4d_results.pt',placed)
    record=dict(context=before,output=file_stat(project/'hmr4d_results.pt'),camera_height=fit['camera_height'],
                scene_basis=fit['scene_basis'].tolist(),rotation=fit['rotation'].tolist(),offset=fit['offset'].tolist(),
                anchor_residual_m=fit['anchor_residual_m'],intrinsics=anchors['intrinsics'],
                method='Official WV yaw; shared scene basis; rigid XZ offset; native height retained')
    if before!=context(project,generated=generated):raise ValueError('FlowHMR inputs changed during placement commit; retry Prepare Preview.')
    atomic_json(project/'flow_world.json',record)
    sources={name:file_stat(project/name) for name in ('flow_prediction.pt','flow_tokens.pt','flow_camera.npz','flow_placement.json','tracking.npz','flow_world.json')}
    selected=prediction_name(project,generated=generated)
    if selected!='flow_prediction.pt':sources[selected]=file_stat(project/selected)
    atomic_json(project/'initial_sources.json',sources)
    # The original generator identity and immutable prediction remain reusable.
    receipt=project/'flowhmr_receipt.json'
    if receipt.exists():
        data=json.loads(receipt.read_text(encoding='utf-8'))
        for name in ('hmr4d_results.pt','initial_sources.json','flow_world.json'):data['outputs'][name]=file_stat(project/name)
        atomic_json(receipt,data)
    (project/'flow_world_pending.json').unlink(missing_ok=True)
    return True


def ensure_group(project):
    from .flow_backend import scene_root
    from .scene_camera import take_projects
    from .people_store import load_people
    root=scene_root(project);people=load_people(root) if (root/'people.json').exists() else None
    rows=take_projects(root,people) if people else [(0,Path(project))]
    changed=[];body=None
    for _,path in rows:
        if not needs_update(path):continue
        if body is None:
            from flowhmr.core.bodymodels.smpl_skeleton import SMPLSkeleton
            from .models import BY_KEY
            body=SMPLSkeleton(model_path=str(BY_KEY['smplh'].path))
        if ensure(path,body=body):changed.append(str(path))
    if people:
        from .scene_camera import prepare_cameras
        prepare_cameras(root)
    return changed
