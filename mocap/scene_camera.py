"""CPU-only camera receipts for a shared static-camera people scene.

Executed in a separate process. Never import Torch in the editor process.
"""
from pathlib import Path
import json
if __name__=='__main__':
    from .runtime_limits import configure_preview_cpu
    configure_preview_cpu()
import numpy as np
from .people_store import load_people,safe_child,media_receipt
from .storage import atomic_json

VERSION=2


def stamp(path):
    state=Path(path).stat()
    return [state.st_size,state.st_mtime_ns]


def validate_transform(value):
    value=np.asarray(value,dtype=float)
    if value.shape!=(4,4) or not np.isfinite(value).all() or not np.allclose(value[3],[0,0,0,1],atol=1e-6):
        raise ValueError('Invalid scene camera transform')
    rotation=value[:3,:3]
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=2e-4) or not np.isclose(np.linalg.det(rotation),1,atol=2e-4):
        raise ValueError('Scene camera must be rigid, without scale changes')
    return value


def camera_sources(project):
    if (project/'prompthmr_source.json').exists():
        return {name:stamp(project/name) for name in ('hmr4d_results.pt','input.mp4','prompthmr_source.json') if (project/name).exists()}
    if (project/'mamma_source.json').exists():
        return {name:stamp(project/name) for name in ('hmr4d_results.pt','input.mp4','mamma_source.json') if (project/name).exists()}
    if (project/'exmr_camera_sources.json').exists():
        # ExMR's paired native roots and registered shape determine the camera.
        # Publishing its first preview must not change the shared-floor identity.
        names=('hmr4d_results.pt','tracking.npz','input.mp4','exmr_camera_sources.json')
        return {name:stamp(project/name) for name in names if (project/name).exists()}
    names=('hmr4d_results.pt','tracking.npz','input.mp4','opt_results.pt','initial_joints.npz','initial_mesh.json')
    if (project/'flow_camera.npz').exists():names+=('flow_camera.npz',)
    if (project/'exmr_camera_sources.json').exists():names+=('exmr_camera_sources.json',)
    return {name:stamp(project/name) for name in names if (project/name).exists()}


def read_camera(project):
    path=project/'scene_camera.json'
    if not path.exists():return None
    record=json.loads(path.read_text(encoding='utf-8'))
    if record.get('version')!=VERSION or record.get('sources')!=camera_sources(project):return None
    from .models import BY_KEY
    if record.get('source_model')=='mamma':
        from .mamma_body import body_path
        model=body_path(project)
    else:model=BY_KEY['smplh' if record.get('source_model')=='flowhmr' else 'smplx'].path
    if model.exists() and record.get('model')!=stamp(model):return None
    validate_transform(record['world_to_camera'])
    intrinsic=np.asarray(record['K'],dtype=float)
    if intrinsic.shape!=(3,3) or not np.isfinite(intrinsic).all() or intrinsic[0,0]<=0 or intrinsic[1,1]<=0:
        raise ValueError('Invalid shared camera intrinsics')
    return record


def paired_camera(world_rotation,cam_rotation,world_translation,cam_translation,pelvis):
    """SMPL translation rotates about its shape-dependent rest pelvis."""
    result=np.eye(4)
    result[:3,:3]=cam_rotation@world_rotation.T
    result[:3,3]=cam_translation+pelvis-result[:3,:3]@(world_translation+pelvis)
    return validate_transform(result)


def take_projects(root,state):
    rows=[]
    for person in state['people']:
        for take in person.get('takes',[]):
            project=safe_child(root,take['project'])
            if (project/'hmr4d_results.pt').is_file():rows.append((person['id'],project))
    if len(rows)>4096:raise ValueError('Too many motion takes for one scene (maximum 4096). Trim the video.')
    return rows


def prepare_cameras(root):
    root=Path(root).resolve();state=load_people(root)
    if not state:raise ValueError('No people group')
    rows=take_projects(root,state)
    manifest=json.loads((root/'project.json').read_text(encoding='utf-8'))
    if manifest.get('pipeline',{}).get('model')=='prompthmr':
        from .prompthmr_scene import prepare_cameras as prepare_phmr
        return prepare_phmr(root)
    if manifest.get('pipeline',{}).get('model')=='mamma':
        from .mamma_scene import prepare_cameras as prepare_mamma
        return prepare_mamma(root)
    if manifest.get('pipeline',{}).get('model')=='flowhmr':
        from .flow_scene import prepare_cameras as prepare_flow
        return prepare_flow(root,rows)
    missing=[project for _,project in rows if read_camera(project) is None]
    if missing:
        import psutil
        if psutil.virtual_memory().available<1024**3:
            raise MemoryError('Free at least 1 GiB RAM to prepare shared scene cameras.')
        import torch
        torch.set_num_threads(1)
        from scipy.spatial.transform import Rotation
        from .models import BY_KEY
        model=BY_KEY['smplx'].path
        pelvis_template=pelvis_shapes=None
        if model.exists():
            with np.load(model,allow_pickle=False) as data:
                regressor=data['J_regressor'][0]
                pelvis_template=regressor@data['v_template']
                pelvis_shapes=np.einsum('v,vci->ci',regressor,data['shapedirs'][...,:10])
        for project in missing:
            sources=camera_sources(project)
            record=torch.load(project/'hmr4d_results.pt',map_location='cpu',weights_only=False,mmap=True)
            world,incam=record['smpl_params_global'],record['smpl_params_incam']
            if pelvis_template is not None:
                pelvis=pelvis_template+pelvis_shapes@world['betas'][0].numpy()
                model_stamp=stamp(model)
            else:
                # A committed original pelvis is sufficient for an already
                # baked project, even when its registered model is offline.
                from .result_artifacts import artifacts_current
                from .preview_data import mapped_array
                if not artifacts_current(project,'initial'):
                    raise FileNotFoundError('Import SMPLX_NEUTRAL.npz or restore the committed original preview to align scene cameras.')
                joints=mapped_array(project/'initial_joints.npz','joints')
                try:pelvis=joints[0,0]-world['transl'][0].numpy()
                finally:joints.close()
                info=json.loads((project/'initial_mesh.json').read_text(encoding='utf-8'))
                model_stamp=[info['model_size'],info['model_mtime_ns']]
            rotation=lambda params:Rotation.from_rotvec(params['global_orient'][0].numpy()).as_matrix()
            transform=paired_camera(rotation(world),rotation(incam),world['transl'][0].numpy(),incam['transl'][0].numpy(),pelvis)
            exmr=record.get('source_model')=='exmr'
            if exmr:
                shared=json.loads((project/'exmr_camera_sources.json').read_text(encoding='utf-8'))
                # All takes use the same full-scene camera orientation, including
                # takes beginning after frame zero; no independent person spacing.
                align=np.asarray(shared['reference_rotation'])@np.asarray(shared['first_rotation']).T
                transform[:3,:3]=align@transform[:3,:3]
                transform[:3,3]=align@transform[:3,3]
                validate_transform(transform)
            with np.load(project/'tracking.npz',allow_pickle=False) as tracking:K=tracking['K'].tolist()
            htd=True
            if (project/'opt_results.pt').exists():
                refined=torch.load(project/'opt_results.pt',map_location='cpu',weights_only=False,mmap=True)
                meta=refined.get('meta',{})
                trajectory=refined.get('T_w2c_optimized')
                htd=bool(meta.get('camera_locked')) and trajectory is not None and trajectory.shape==(len(world['transl']),4,4)
                if htd:
                    samples=np.linspace(0,len(trajectory)-1,min(4096,len(trajectory)),dtype=np.int64)
                    htd=np.allclose(trajectory[samples].numpy(),transform,atol=2e-4)
                del refined
            if camera_sources(project)!=sources:raise ValueError('Motion changed during scene camera preparation; retry.')
            atomic_json(project/'scene_camera.json',dict(version=VERSION,sources=sources,model=model_stamp,
                world_to_camera=transform.tolist(),K=K,htd_compatible=htd,source_model='exmr' if exmr else 'gvhmr',method='Paired root with SMPL-X pelvis pivot'+('; one scene camera' if exmr else '')))
            del record,world,incam
    anchor_path=root/'people_scene.json'
    previous=json.loads(anchor_path.read_text(encoding='utf-8')) if anchor_path.exists() else {}
    anchor=next(((ident,p) for ident,p in rows if p.relative_to(root).as_posix()==previous.get('anchor')),None)
    if anchor is None and rows:anchor=min(rows,key=lambda row:(row[0],str(row[1])))
    if anchor:
        ident,project=anchor;camera=read_camera(project)
        atomic_json(anchor_path,dict(version=VERSION,input=media_receipt(root),anchor=project.relative_to(root).as_posix(),
            anchor_id=ident,world_to_camera=camera['world_to_camera'],sources=camera['sources'],floor=0.,K=camera['K'],
            camera_mode='static',scale=1.))
    return anchor_path


def main():
    import sys
    from .compute_policy import configure_worker_cpu
    configure_worker_cpu('scene')
    try:
        root=Path(sys.argv[1]);prepare_cameras(root)
        from .scene_floor import read_config,resolve
        if read_config(root) is not None:
            try:resolve(root)
            except FileNotFoundError:pass  # Camera-only baked previews remain usable offline.
        return 0
    except Exception as error:
        print(str(error),file=sys.stderr,flush=True);return 1


if __name__=='__main__':raise SystemExit(main())
