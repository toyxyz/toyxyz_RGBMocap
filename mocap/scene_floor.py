"""One physical correction plane for a static-camera people group.

Settings/receipts stay lightweight. Torch and the body model are worker-only.
"""
from pathlib import Path
import hashlib,json,math
from .people_store import load_people,read_people_json,root_for,safe_child,media_receipt
from .storage import atomic_json
from .validation_snapshot import snapshot_value

VERSION=1


def scene_root(project,manifest=None):
    project=Path(project)
    if manifest is None:
        path=project/'project.json'
        if not path.exists():return None
        manifest=read_people_json(path)
    if 'people_root' not in manifest and not (project/'people.json').is_file():return None
    root=root_for(project,manifest)
    return root if (root/'people.json').is_file() else None


def floor_options(options):
    from .correction_settings import correction_options
    mode=correction_options(options)['mode']
    auto=bool(options.get('auto_floor',True))
    value=0. if auto else float(options.get('floor',0.))
    if not math.isfinite(value):raise ValueError('Shared floor height must be finite')
    # Preserve legacy flat-plane receipt signatures.
    return dict(auto_floor=auto,floor=value,**({'mode':mode} if mode!='flat' else {}))


def read_config(root):
    path=Path(root)/'scene_floor.json'
    if not path.exists():return None
    record=read_people_json(path)
    if record.get('version')!=VERSION:raise ValueError('Unsupported shared floor settings')
    return floor_options(record)


def save_config(root,options):
    options=floor_options(options)
    if read_config(root)!=options:
        atomic_json(Path(root)/'scene_floor.json',dict(version=VERSION,**options))
    return options


def ensure_config(root):
    root=Path(root);saved=read_config(root)
    if saved is not None:return saved
    # Migrate from the reference actor, never from whichever row was selected.
    reference=root/'people_scene.json';options={}
    if reference.exists():
        record=read_people_json(reference)
        manifest=safe_child(root,record['anchor'])/'project.json'
        if manifest.exists():options=read_people_json(manifest).get('correction',{})
    return save_config(root,options)


def inputs(root):
    from .scene_camera import stamp,take_projects,validate_transform
    from .models import BY_KEY
    root=Path(root).resolve();state=load_people(root)
    if state is None:raise ValueError('People scene unavailable')
    reference=read_people_json(root/'people_scene.json')
    anchor=safe_child(root,reference['anchor'])
    if reference['input']!=media_receipt(root) or reference['sources']['hmr4d_results.pt']!=stamp(anchor/'hmr4d_results.pt'):
        raise ValueError('Shared camera reference is out of date')
    validate_transform(reference['world_to_camera'])
    manifest=read_people_json(root/'project.json')
    mamma=manifest.get('pipeline',{}).get('model')=='mamma'
    flow=manifest.get('pipeline',{}).get('model')=='flowhmr'
    model=BY_KEY['smplh' if flow else 'smplx'].path
    model_stamp=stamp(model) if model.exists() and not mamma else None
    rows=[]
    for ident,project in take_projects(root,state):
        camera=read_people_json(project/'scene_camera.json')
        source=stamp(project/'hmr4d_results.pt')
        if camera['sources']['hmr4d_results.pt']!=source:raise ValueError('Actor camera is out of date')
        if mamma:
            from .mamma_body import body_path
            if camera['model']!=stamp(body_path(project)):raise ValueError('MAMMA actor body changed')
        elif model_stamp is not None and camera['model']!=model_stamp:raise ValueError('Scene body model changed')
        validate_transform(camera['world_to_camera'])
        rows.append(dict(person=ident,project=project.relative_to(root).as_posix(),source=source,
            world_to_camera=camera['world_to_camera'],**({'body':camera['model']} if mamma else {})))
        if model_stamp is None and not mamma:model_stamp=camera.get('model')
    return dict(version=VERSION,input=media_receipt(root),model=model_stamp,
        reference=reference['world_to_camera'],actors=rows)


@snapshot_value
def current_resolution(root):
    path=Path(root)/'scene_floor_resolved.json'
    if not path.exists():return None
    try:
        record=read_people_json(path)
        if record.get('version')!=VERSION or record['options']!=read_config(root) or record['inputs']!=inputs(root):return None
        if not math.isfinite(record['floor']):return None
        expected=signature(record['options'],record['inputs'],record['floor'])
        return record if record['signature']==expected else None
    except (OSError,ValueError,KeyError,TypeError):return None


def signature(options,context,height):
    return hashlib.sha256(json.dumps([VERSION,options,context,float(height)],sort_keys=True).encode()).hexdigest()


def sample_windows(frames,budget):
    """Adjacent windows retain velocity gates; total sampled frames is bounded."""
    if frames<=budget:return [(0,frames)]
    width=min(32,budget);count=max(1,budget//width)
    if count==1:return [(0,width)]
    return [(round(i*(frames-width)/(count-1)),round(i*(frames-width)/(count-1))+width) for i in range(count)]


def resolve(root,body=None):
    root=Path(root);options=ensure_config(root)
    cached=current_resolution(root)
    if cached is not None:return cached
    if read_people_json(root/'project.json').get('pipeline',{}).get('model')=='flowhmr':
        from .flow_floor import resolve
        return resolve(root,options)
    mamma=read_people_json(root/'project.json').get('pipeline',{}).get('model')=='mamma'
    context=inputs(root);height=options['floor'];samples=0;warning=''
    if options['auto_floor']:
        import numpy as np
        import torch
        from scipy.spatial.transform import Rotation
        from .correction import SoleGeometry,prepare_contacts
        from .models import BY_KEY
        if body is None and not mamma:
            import smplx
            model=BY_KEY['smplx'].path
            if not model.exists():raise FileNotFoundError('Import the body model to estimate the shared scene floor.')
            body=smplx.SMPLX(str(model),gender='neutral',num_pca_comps=12,flat_hand_mean=False,batch_size=1).eval()
        inverse=np.linalg.inv(np.asarray(context['reference']))
        estimates={};rows=context['actors']
        budget=max(2,min(512,8192//max(1,len(rows))))
        for row in rows:
            project=safe_child(root,row['project'])
            record=torch.load(project/'hmr4d_results.pt',map_location='cpu',weights_only=False,mmap=True)
            params=record['smpl_params_global'];length=len(params['body_pose'])
            if mamma:
                from .mamma_body import model_for
                actor_body=model_for(project)
            else:actor_body=body
            geometry=SoleGeometry(actor_body,params['betas'][0].numpy())
            transform=inverse@np.asarray(row['world_to_camera']);rotation=transform[:3,:3]
            probabilities=record['net_outputs']['static_conf_logits'][0,:,:4]
            if len(probabilities)!=length:raise ValueError('Scene contact and motion frame counts differ')
            fps=float(read_people_json(project/'project.json').get('fps',30))
            values=[]
            for first,end in sample_windows(length,budget):
                rv=torch.cat((params['global_orient'][first:end,None],params['body_pose'][first:end].reshape(-1,21,3)),1).numpy()
                matrices=Rotation.from_rotvec(rv.reshape(-1,3)).as_matrix().reshape(-1,22,3,3)
                matrices[:,0]=rotation@matrices[:,0]
                quat=Rotation.from_matrix(matrices.reshape(-1,3,3)).as_quat().reshape(-1,22,4)[...,[3,0,1,2]]
                roots=(params['transl'][first:end].numpy()+geometry.rest[0])@rotation.T+transform[:3,3]
                probes=geometry.sample(roots,quat)[0]
                active=prepare_contacts(probes,probabilities[first:end].sigmoid().numpy(),fps,preset='normal')[0]
                h=probes[...,1].reshape(-1,2,2).min(2)
                if active.any():values.extend(h[active].tolist())
                samples+=end-first
            if values:estimates.setdefault(row['person'],[]).append(float(np.quantile(values,.05)))
            del record,params,geometry,probabilities,actor_body
        # Equal actor weighting prevents a long/overlapping take dominating.
        height=float(np.median([np.median(v) for v in estimates.values()])) if estimates else 0.
        if not estimates:warning='No reliable ground samples. Shared reference floor 0 m is used; adjust it manually if needed.'
    if inputs(root)!=context or read_config(root)!=options:raise ValueError('Shared floor inputs changed; Apply Correction again.')
    record=dict(version=VERSION,options=options,inputs=context,floor=float(height),samples=samples,warning=warning,
        signature=signature(options,context,height),method='One common plane; equal-actor median of original motion sole estimates')
    atomic_json(root/'scene_floor_resolved.json',record)
    return record


def binding(project,manifest,body):
    root=scene_root(project,manifest)
    if root is None:return None
    from .scene_camera import prepare_cameras
    prepare_cameras(root)
    record=resolve(root,body)
    import numpy as np
    relative=Path(project).resolve().relative_to(root).as_posix()
    row=next(row for row in record['inputs']['actors'] if row['project']==relative)
    transform=np.linalg.inv(np.asarray(record['inputs']['reference']))@np.asarray(row['world_to_camera'])
    return dict(signature=record['signature'],floor=record['floor'],coordinate_frame='shared_world',
        local_to_shared=transform.tolist(),local_plane_normal=transform[1,:3].tolist(),
        local_plane_height=float(record['floor']-transform[1,3]))


def receipt_matches(project,manifest,receipt):
    root=scene_root(project,manifest)
    if root is None:return receipt is None
    if not receipt:return False
    current=current_resolution(root)
    return current is not None and receipt.get('signature')==current['signature'] and receipt.get('floor')==current['floor']
