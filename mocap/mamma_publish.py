"""Publish native MAMMA actors through the app's source/export/contact paths."""
from pathlib import Path
import json
import os
import pickle
import shutil
import time
import numpy as np
from .storage import atomic_json,sha256
from .result_artifacts import file_stat
from .mamma_body import make_actor_body,model_for,convert_motion,contact_probabilities,WORLD_ROTATION


def media_link(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():return
    try:os.link(source,destination)
    except OSError:shutil.copy2(source,destination)


def publish(job,emit):
    import torch
    from scipy.spatial.transform import Rotation
    from .mamma_pipeline import work_for
    from .models import BY_KEY
    from .engines import save_tensor
    from .people_store import visible_runs,person_color,media_receipt
    from .project_workspace import read,set_active
    from .scene_camera import camera_sources,VERSION
    torch.set_num_threads(4)
    work=work_for(job);root=job['root'];n=job['frames']
    # Actor-only native data lives with the workspace, never in a global cache.
    group=root/'mamma'/'results'/work.name
    group.mkdir(parents=True,exist_ok=True)
    atomic_json(group/'mamma_pending.json',dict(work=str(work),stage='native publication'))
    entries=json.loads((work/'inputs.json').read_text(encoding='utf-8'))
    cameras=json.loads((work/'camera-world.json').read_text(encoding='utf-8'))
    primary=cameras[entries[0]['id']]
    transform=np.eye(4);transform[:3,:3]=np.asarray(primary['extrinsics_matrix'])[:,:3]@WORLD_ROTATION.T
    transform[:3,3]=np.asarray(primary['extrinsics_matrix'])[:,3]
    source={**job['inputs'][0]['metadata'],**job['inputs'][0]};source['path']=str(Path(source['path']).resolve())
    media_link(Path(entries[0]['path']),group/'input.mp4')
    selected='corrected_initial' if job['correction_enabled'] else 'initial'
    common=dict(version=1,source=source,start=job['start'],end=job['start']+n/30,fps=30,frames=n,
        width=entries[0]['size'][0],height=entries[0]['size'][1],camera='static',stages={},
        timeline=dict(start=job['start'],step=1/30,frames=n),pipeline=dict(model='mamma',result=selected),
        correction=job['correction'],mamma_options=job['options'],mamma_work=os.path.relpath(work,group),
        workspace_root=os.path.relpath(root,group))
    atomic_json(group/'project.json',common)
    with BY_KEY['mamma_vertices'].path.open('rb') as f:D=pickle.load(f).cpu().numpy()
    prior={}
    try:prior={p['id']:p for p in json.loads((group/'people.json').read_text(encoding='utf-8'))['people']}
    except (OSError,ValueError,KeyError):pass
    people=[];projects=[]
    fit=work/'fit/motion'
    filenames=sorted(fit.glob('smplx_params_body_id-*.npz'))
    if not filenames or len(filenames)>8:raise ValueError('MAMMA returned no actors or more than eight actors for this experimental integration.')
    for number,path in enumerate(filenames,1):
        raw_id=int(path.stem.rsplit('-',1)[1])
        with np.load(path,allow_pickle=False) as archive:
            pose=archive['smplx_pose'];beta=archive['smplx_betas'];trans=archive['smplx_translation'];contacts=archive['smplx_floor_contact']
        if pose.shape!=(n,165) or np.asarray(beta).shape!=(1,16) or contacts.shape!=(n,512):
            raise ValueError('MAMMA result dimensions do not match the selected common trim.')
        seen=np.zeros(n,bool)
        for e in entries:
            with np.load(work/'landmarks/motion'/(e['id']+'.npz'),allow_pickle=False) as z:
                if raw_id>=z['visibilities'].shape[1]:raise ValueError('MAMMA actor identity does not match its landmark/mask ID.')
                seen|=(z['visibilities'][:,raw_id].max(axis=1)>.01)
        runs=visible_runs(np.flatnonzero(seen),30,0,n)
        if not runs:continue
        person=dict(id=number,name=prior.get(number,{}).get('name',f'Person {number}'),
            color=prior.get(number,{}).get('color',person_color(number)),checked=prior.get(number,{}).get('checked',True),
            observations=int(seen.sum()),first=int(np.flatnonzero(seen)[0]),last=int(np.flatnonzero(seen)[-1]),takes=[])
        for take,(first,end) in enumerate(runs):
            project=group/'actors'/f'person_{number:04}'/f'take_{take:03}'
            project.mkdir(parents=True,exist_ok=True)
            from . import mamma_body
            native=dict(raw_stat=file_stat(path),raw_source=path.relative_to(root).as_posix(),range=[first,end],
                body_source=file_stat(BY_KEY['mamma_body'].path),adapter=sha256(Path(mamma_body.__file__)))
            receipt=project/'mamma_source.json'
            reuse=False
            try:
                reuse=json.loads(receipt.read_text(encoding='utf-8'))['native']==native and file_stat(project/'hmr4d_results.pt')==json.loads(receipt.read_text(encoding='utf-8'))['output']
                if reuse:
                    from .mamma_body import body_path
                    body_path(project)
            except (OSError,ValueError,KeyError):reuse=False
            state=dict(common,frames=end-first,start=job['start']+first/30,end=job['start']+end/30,
                people_root=os.path.relpath(group,project),person_id=number,person_frames=[first,end],
                workspace_root=os.path.relpath(root,project),
                timeline=dict(start=job['start']+first/30,step=1/30,frames=end-first),
                mamma_preview_videos=[dict(id=e['id'],path=os.path.relpath(e['path'],project),first=first) for e in entries])
            if reuse:
                old=json.loads((project/'project.json').read_text(encoding='utf-8'))
                state['mamma_body']=old['mamma_body']
                atomic_json(project/'project.json',state)
            else:
                entry=make_actor_body(BY_KEY['mamma_body'].path,beta,project)
                state['mamma_body']=entry;atomic_json(project/'project.json',state)
                body=model_for(project)
                params=convert_motion(pose[first:end],beta,trans[first:end],body)
                prob=contact_probabilities(body,beta[0],D,contacts[first:end])
                confidence=torch.logit(torch.from_numpy(np.clip(prob,1e-6,1-1e-6)))
                incam={k:v.clone() for k,v in params.items()}
                r=transform[:3,:3];t=transform[:3,3]
                root_r=Rotation.from_rotvec(params['global_orient'].numpy()).as_matrix()
                incam['global_orient']=torch.from_numpy(Rotation.from_matrix(r@root_r).as_rotvec().astype('float32'))
                shaped=body.v_template.numpy()+np.einsum('vci,i->vc',body.shapedirs.numpy(),beta[0,:10])
                pelvis=body.J_regressor.numpy()[0]@shaped
                incam['transl']=torch.from_numpy(((params['transl'].numpy()+pelvis)@r.T+t-pelvis).astype('float32'))
                save_tensor(dict(smpl_params_global=params,smpl_params_incam=incam,
                    net_outputs=dict(static_conf_logits=confidence[None]),source_model='mamma'),project/'hmr4d_results.pt')
                atomic_json(receipt,dict(version=1,native=native,output=file_stat(project/'hmr4d_results.pt')))
                del body,params,incam
            # Local clips make source/take timing unambiguous in the generic UI.
            from .video import executable
            if not (project/'input.mp4').exists():
                if first==0 and end==n:media_link(group/'input.mp4',project/'input.mp4')
                else:
                    import subprocess
                    subprocess.run([executable('ffmpeg'),'-v','error','-i',str(group/'input.mp4'),'-vf',f'trim=start_frame={first}:end_frame={end},setpts=PTS-STARTPTS',
                        '-an','-c:v','libx264','-crf','18','-preset','fast','-threads','2',str(project/'input.mp4')],check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            atomic_json(project/'scene_camera.json',dict(version=VERSION,sources=camera_sources(project),model=file_stat(project/state['mamma_body']['file']),
                world_to_camera=transform.tolist(),K=primary['intrinsic_matrix'],htd_compatible=False,source_model='mamma',method='One calibrated MAMMA world; fixed camera, exact shaped pelvis'))
            relative=project.relative_to(group).as_posix()
            person['takes'].append(dict(first=first,end=end,project=relative))
            projects.append(project)
        person['preview']=person['takes'][0]['project'];people.append(person)
        emit('progress',current=number,total=len(filenames),message=f'Prepare MAMMA source actor {number}')
    if not projects:raise ValueError('No sufficiently visible MAMMA actor takes could be published.')
    atomic_json(group/'people.json',dict(version=1,input=media_receipt(group),people=people,active=people[0]['id']))
    from .mamma_scene import prepare_cameras
    prepare_cameras(group)
    from .scene_floor import save_config,resolve
    save_config(group,job['correction']);resolve(group)
    from .export import export_project
    from .correction import run as correct
    for index,project in enumerate(projects):
        emit('progress',current=index,total=len(projects),message=f'Prepare MAMMA motion, contacts and export bones {index+1}/{len(projects)}')
        export_project(project,'initial')
        if job['correction_enabled']:
            correct(project,emit,selected)
            export_project(project,selected)
    set_active(root,'mamma',projects[0])
    from .mamma_pipeline import validate_inputs
    validate_inputs(job)
    state=read(root);state['mamma_inputs']=job['inputs'];state['mamma_active_camera']=job.get('active_camera',0)
    state['models']['mamma']['settings']=dict(result=selected,correction=job['correction'],mamma_options=job['options'])
    atomic_json(root/'project.json',state)
    (group/'mamma_pending.json').unlink(missing_ok=True)
    emit('mamma_ready',project=str(projects[0]),group=str(group),root=str(root),result=selected)
