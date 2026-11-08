"""Native SMPL-H adapter for RGB MoCap Motion Correction.

No inference, SMPL-X conversion, upstream smoothing or second target IK pass.
"""
from pathlib import Path
import json,time
import numpy as np
from scipy.spatial.transform import Rotation
from .result_artifacts import file_stat,begin_update,finish_update
from .storage import atomic_json,sha256

VERSION=1


def code_identity():
    from .paths import ROOT
    from .resource_cache import code_hash
    # Exact-byte aliases retain corrections after platform-only edits to these files.
    return {name:(code_hash if name in {'flow_correction.py','correction.py','scene_floor.py'} else sha256)(ROOT/'mocap'/name) for name in
        ('flow_correction.py','flow_floor.py','correction.py','multilevel_contacts.py','hand_lock.py','contact_trajectory.py','contact_arm_plan.py','palm_contact.py','contact_pose_plan.py','contact_temporal_root.py','anatomical_ik.py','correction_settings.py','foot_contacts.py','scene_floor.py')}


def contact_probabilities(params,n):
    # Official Flow representation is a regressed contact value, NOT logits.
    # ODE output may slightly overshoot [0,1]. Order: L ankle/toe, R ankle/toe.
    value=params.get('foot_contacts')
    if value is None or tuple(value.shape)!=(1,n,4):raise ValueError('FlowHMR contact predictions are missing or invalid. Reconstruct the motion again.')
    values=value[0].numpy()
    if not np.isfinite(values).all():raise ValueError('Non-finite FlowHMR contact predictions')
    return np.clip(values,0,1)


class SoleGeometry:
    """Exact upstream SMPL-H LBS (no pose blend), bounded foot-only surface."""
    def __init__(self,body,shape):
        import torch
        self.parents=body.parents.tolist()
        with torch.inference_mode():
            neutral=dict(rot_mats=torch.eye(3).expand(1,52,3,3),shapes=shape,trans=torch.zeros(1,3))
            shaped=body(neutral)['vertices'][0].numpy()
            self.rest=body.compute_j_shaped(shape)[0].numpy()
        weights=body.lbs_weights.numpy();self.groups=[];self.proxies=[]
        from .hand_lock import palm_offsets
        self.palm_offsets=palm_offsets(shaped,weights,self.rest)
        for ankle,toe in ((7,10),(8,11)):
            ids=np.flatnonzero(weights[:,ankle]+weights[:,toe]>.6);z=shaped[ids,2]
            if not len(ids):raise ValueError('SMPL-H foot skin is unavailable')
            for region,proxy in ((ids[z<=np.median(z)],ids[z<=np.quantile(z,.4)]),
                                 (ids[z>=np.median(z)],ids[z>=np.quantile(z,.65)])):
                selected=proxy[np.argsort(shaped[proxy,1])[:24]]
                self.groups.append(region);self.proxies.append(np.searchsorted(region,selected))
        self.ids=np.concatenate(self.groups);self.shaped=shaped[self.ids]
        if np.any(weights[self.ids,22:]>1e-7):raise ValueError('Unexpected hand influence on SMPL-H foot geometry')
        self.weights=weights[self.ids,:22]

    def sample(self,root,quat):
        from .correction import fk
        joints,world=fk(self.rest[:22],root,quat)
        translation=joints-np.einsum('tjik,jk->tji',world,self.rest[:22])
        blend=(self.weights@world.reshape(-1,22,9)).reshape(len(root),len(self.ids),3,3)
        vertices=np.einsum('tvij,vj->tvi',blend,self.shaped)+self.weights@translation
        probes=[];offset=0
        for ids,proxy in zip(self.groups,self.proxies):
            patch=vertices[:,offset:offset+len(ids)];offset+=len(ids)
            value=patch[:,proxy].mean(1);value[:,1]=patch[...,1].min(1);probes.append(value)
        return np.stack(probes,1),vertices


def motion_geometry(project,body=None):
    from .flow_backend import load_tensor,validate_motion
    from .models import BY_KEY
    from flowhmr.core.bodymodels.smpl_skeleton import SMPLMesh
    from flowhmr.core.math.geometry import rot6d_to_rotation_matrix
    from .flow_export import proper_rotations
    params=load_tensor(Path(project)/'hmr4d_results.pt');n=params['trans'].shape[1];validate_motion(params,n)
    body=body if body is not None else SMPLMesh(model_path=str(BY_KEY['smplh'].path))
    geometry=SoleGeometry(body,params['shapes'].reshape(1,16))
    local=proper_rotations(rot6d_to_rotation_matrix(params['rot6d'][0,:,:22]).numpy())
    quat=Rotation.from_matrix(local.reshape(-1,3,3)).as_quat().reshape(n,22,4)[...,[3,0,1,2]].astype(np.float32)
    roots=params['trans'][0].numpy()+geometry.rest[0]
    return params,geometry,roots,quat,contact_probabilities(params,n)


def run(project,emit=lambda *a,**k:None,result='corrected_initial'):
    import torch
    from .flow_backend import save_tensor,validate_motion
    from .motion_tools import result_is_current
    from .correction import correct_arrays,prepare_contacts,sole_metrics,contact_strength,backend
    from .correction_settings import effective_options
    from .scene_floor import binding,receipt_matches
    from .foot_contacts import read_edit,contact_revision
    project=Path(project);kind='corrected_initial';started=time.monotonic()
    if result!=kind:raise ValueError('FlowHMR supports original or corrected original motion only')
    if not result_is_current(project,'initial'):raise ValueError('FlowHMR source is incomplete or outdated. Reconstruct it first.')
    from .models import BY_KEY
    model=BY_KEY['smplh'].path;model_stamp=file_stat(model)
    sources={'hmr4d_results.pt':file_stat(project/'hmr4d_results.pt')}
    state=json.loads((project/'project.json').read_text(encoding='utf-8'));fps=float(state['fps'])
    options=effective_options(project,state)
    params,geometry,roots,quats,probabilities=motion_geometry(project);n=len(roots)
    shared=binding(project,state,None);floor=shared['floor'] if shared else None if options['auto_floor'] else options['floor']
    context=dict(frames=n,fps=fps,sources=sources)
    ranges,revision,warning=read_edit(project,'initial',context)
    hand_ranges=ranges[2:] if ranges is not None and len(ranges)==4 else None
    ranges=ranges[:2] if ranges is not None else None
    if not options['foot_lock']:ranges=[[],[]]
    probes=np.empty((n,4,3),np.float32)
    for first in range(0,n,128):probes[first:first+128]=geometry.sample(roots[first:first+128],quats[first:first+128])[0]
    # Mark before publishing any payload; interrupted generations cannot export.
    begin_update(project,kind,'correct')
    out_root,out_quat,report=correct_arrays(geometry.rest[:22],roots,quats,probes,probabilities,fps,floor,
        preset=options['preset'],contact_ranges=ranges,pelvis_offset=options['pelvis_offset'],mode=options['mode'],foot_enabled=options['foot_lock'],emit=emit)
    from .hand_lock import apply as apply_hands
    report['hand_lock']=apply_hands(geometry.rest[:22],roots,quats,out_root,out_quat,geometry.palm_offsets,fps,options['mode'],report['floor'],options['preset'],hand_ranges,options['hand_lock'],emit)
    report['warning']=' '.join(filter(None,(report['warning'],report['hand_lock']['warning'])))
    after=np.empty_like(probes)
    for first in range(0,n,128):after[first:first+128]=geometry.sample(out_root[first:first+128],out_quat[first:first+128])[0]
    active,anchors,*_=prepare_contacts(probes,probabilities,fps,floor,options['preset'],ranges,options['mode'])
    report.update(sole_metrics(probes,after,probabilities,active,anchors,report['segments'],fps,options['preset'],report['floor'],options['mode']))
    report['warning']=' '.join(filter(None,(report['warning'],warning)))
    if (report['full_lock_surface_error_max_m'] or 0)>.01 or report['contact_sole_penetration_over_1cm']:
        report['warning']=' '.join(filter(None,(report['warning'],'Some physical sole contacts remain unresolved; inspect floor height and timing.')))
    for label,values in (('before',probes),('after',after)):
        heights=values[...,1].reshape(n,2,2).min(2)
        levels=anchors[...,1] if options['mode']=='multi_level' else np.full_like(heights,report['floor'])
        report['contact_sole_height_'+label+'_m']=float(np.abs(heights[active]-levels[active]).mean()) if active.any() else 0.
    output=dict(params);output['rot6d']=params['rot6d'].clone();output['trans']=params['trans'].clone()
    matrices=Rotation.from_quat(out_quat[...,[1,2,3,0]].reshape(-1,4)).as_matrix().reshape(n,22,3,3)
    # Copy source trunk, fingers and predictions. Arms are published only when
    # enabled hand contacts were solved; their wrist world frames are retained.
    if n>1:
        for joint in ((1,2,4,5,7,8) if report['contacts'] else ()) + ((13,14,16,17,18,19,20,21) if report['hand_lock']['contacts'] else ()):
            output['rot6d'][0,:,joint]=torch.from_numpy(matrices[:,joint,:,:2].reshape(n,6).astype(np.float32))
        if report['contacts']:output['trans'][0]=torch.from_numpy((out_root-geometry.rest[0]).astype(np.float32))
    output['body_rot6d']=output['rot6d'][:,:,1:];output['root_rot6d']=output['rot6d'][:,:,0]
    # Derived upstream diagnostic fields are not valid after contact IK.
    for key in ('local_joints_positions','smooth_root_pos','pelvis_world'):output.pop(key,None)
    output['source']='initial';output['method']=report['method'];validate_motion(output,n)
    segments=report.pop('segments');automatic=report.pop('automatic_ranges')
    save_tensor(project/(kind+'.pt'),output)
    archive=project/(kind+'_contacts.npz')
    # NPZ is published atomically; pending remains until all receipts commit.
    temporary=archive.with_suffix('.tmp.npz')
    try:
        from .contact_trajectory import support_weights,diagnostic_plan
        np.savez_compressed(temporary,segments=np.asarray([[s['foot'],s['start'],s['end'],*s['anchor']] for s in segments],float).reshape(-1,6),
            automatic=np.asarray([[f,s,e] for f in range(2) for s,e in automatic[f]],np.int64).reshape(-1,3),
            **diagnostic_plan(anchors,support_weights(probes,active,fps),segments),
            planner_version=report['version'])
        temporary.replace(archive)
    finally:temporary.unlink(missing_ok=True)
    current=json.loads((project/'project.json').read_text(encoding='utf-8'))
    if (any(file_stat(project/k)!=v for k,v in sources.items()) or file_stat(model)!=model_stamp or
        contact_revision(project,'initial')!=revision or current['fps']!=fps or effective_options(project,current)!=options or not receipt_matches(project,current,shared)):
        raise ValueError('Correction inputs changed while processing. Apply Correction again.')
    report.update(source='initial',model_key='smplh',model_stat=model_stamp,options=options,contact_revision=revision,
        output_stat=file_stat(project/(kind+'.pt')),contacts_stat=file_stat(archive),segment_count=len(segments),
        adapter_version=VERSION,code=code_identity(),backend_sha256=sha256(Path(backend().__file__)),elapsed=time.monotonic()-started,
        contact_input='FlowHMR regressed L ankle/toe, R ankle/toe values clipped to [0,1]; not sigmoid logits')
    if shared:report['shared_floor']=shared
    atomic_json(project/(kind+'_sources.json'),sources);atomic_json(project/(kind+'_correction.json'),report)
    finish_update(project,kind,'correct')
    atomic_json(project/'correction_report.json',dict(version=VERSION,results={kind:report},elapsed=report['elapsed']))
    return project/(kind+'.pt')
