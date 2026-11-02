"""Publish PromptHMR's joint scene without independent actor placement/refitting."""
from pathlib import Path
import json,os,uuid
import numpy as np
from .storage import atomic_json,sha256
from .result_artifacts import file_stat


def hand_policy(person):
    """Zero matrices in the video path are missing hands, not rotations."""
    rotations=np.asarray(person['smplx_cam']['rotmat'])
    if rotations.shape!=(len(person['frames']),55,3,3):raise ValueError('Invalid PromptHMR rotation dimensions.')
    hands=rotations[:,25:]
    absent=np.all(hands==0,axis=(-1,-2))
    if absent.all():return 'registered-neutral-untracked-v1'
    if absent.any() or not np.isfinite(hands).all():raise ValueError('Inconsistent PromptHMR hand rotations.')
    if not np.allclose(np.linalg.det(hands),1.,atol=1e-3) or not np.allclose(hands@hands.swapaxes(-1,-2),np.eye(3),atol=1e-3):
        raise ValueError('Invalid PromptHMR hand rotations.')
    return 'native'


def convert(params,body,indices,*,neutral_hands=False):
    """Rebase valid native hands; use registered neutral for missing video hands."""
    import torch
    pose=np.asarray(params['pose'],dtype=np.float32)[indices]
    betas=np.asarray(params['shape'],dtype=np.float32)
    if betas.shape[0]==1:betas=np.repeat(betas,len(params['pose']),axis=0)
    trans=np.asarray(params['trans'],dtype=np.float32)[indices]
    if pose.shape!=(len(indices),165) or betas.shape[1]!=10 or trans.shape!=(len(indices),3):raise ValueError('Unsupported PromptHMR body dimensions.')
    result=dict(global_orient=torch.from_numpy(pose[:,:3].copy()),body_pose=torch.from_numpy(pose[:,3:66].copy()),
        left_hand_pose=torch.from_numpy(pose[:,75:120].copy())-body.left_hand_mean.detach().cpu(),
        right_hand_pose=torch.from_numpy(pose[:,120:165].copy())-body.right_hand_mean.detach().cpu(),
        betas=torch.from_numpy(betas[indices].copy()),transl=torch.from_numpy(trans.copy()))
    if neutral_hands:
        # The shared evaluator adds the registered relaxed hand mean. Supplying
        # zero coefficients here gives a fixed neutral hand without inventing
        # tracked motion or converting the upstream zero matrices into angles.
        result['left_hand_pose'].zero_();result['right_hand_pose'].zero_()
    if not all(torch.isfinite(v).all() for v in result.values()):raise ValueError('Non-finite PromptHMR pose.')
    return result


def publish(job,emit):
    import torch,joblib,smplx,subprocess
    from scipy.spatial.transform import Rotation
    from .prompthmr_pipeline import work_for,validate_inputs
    from .models import BY_KEY
    from .people_store import visible_runs,person_color,media_receipt
    from .scene_camera import VERSION,camera_sources
    from .engines import save_tensor
    from .mamma_publish import media_link
    from .video import executable
    from .project_workspace import set_active,read
    torch.set_num_threads(4);work=work_for(job);root=job['root'];n=job['frames']
    done=json.loads((work/'complete.json').read_text(encoding='utf8'))
    if done['identity']!=job['identity'] or any(file_stat(work/name)!=stamp for name,stamp in done['outputs'].items()):
        raise ValueError('PromptHMR inference generation is incomplete or changed. Reconstruct again; previous work is preserved.')
    group=root/'prompthmr/results'/work.name;group.mkdir(parents=True,exist_ok=True)
    atomic_json(group/'prompthmr_pending.json',dict(work=work.name))
    record=joblib.load(work/'results.pkl');media=json.loads((work/'media.json').read_text(encoding='utf8'))
    if not 1<=len(record['people'])<=8:raise ValueError('PromptHMR returned an unsupported actor count.')
    selected='corrected_initial' if job['correction_enabled'] else 'initial'
    common=dict(version=1,source=job['source'],start=job['start'],end=job['start']+n/30,fps=30,frames=n,
        width=media['width'],height=media['height'],camera=job['options'].get('camera','static'),stages={},timeline=dict(start=job['start'],step=1/30,frames=n),
        pipeline=dict(model='prompthmr',result=selected),correction=job['correction'],prompthmr_options=job['options'],
        workspace_root=os.path.relpath(root,group))
    atomic_json(group/'project.json',common);media_link(work/'input.mp4',group/'input.mp4')
    prior={}
    try:prior={p['id']:p for p in json.loads((group/'people.json').read_text(encoding='utf8'))['people']}
    except (OSError,ValueError,KeyError):pass
    body=smplx.SMPLX(str(BY_KEY['smplx'].path),use_pca=False,flat_hand_mean=False).eval()
    camera=record['camera_world'];transform=np.eye(4);transform[:3,:3]=np.asarray(camera['Rcw'])[0];transform[:3,3]=np.asarray(camera['Tcw'])[0]
    from .scene_camera import validate_transform
    validate_transform(transform)
    # Upstream uses an 896-pixel-tall image. K is scaled back to the actual
    # working video dimensions; world distances are never scaled per actor.
    tracks=list(record['people'].items());height=896 if media['height']>896 else media['height']
    model_width=int(media['width']*(height/media['height']));model_width-=model_width%2
    center=np.asarray(camera['img_center']);focal=float(camera['img_focal'])
    sx=media['width']/model_width;sy=media['height']/height
    K=np.array([[focal*sx,0,center[0]*sx],[0,focal*sy,center[1]*sy],[0,0,1.]])
    moving=job['options'].get('camera')=='moving';camera_track=None
    if moving:
        from .camera_tracks import valid_rotations
        rr=np.asarray(camera['Rcw'],dtype=float);tt=np.asarray(camera['Tcw'],dtype=float)
        valid_rotations(rr)
        if rr.shape!=(n,3,3) or tt.shape!=(n,3) or not np.isfinite(tt).all():raise ValueError('PromptHMR camera track does not cover the selected video.')
        matrices=np.tile(np.eye(4),(n,1,1));matrices[:,:3,:3]=rr;matrices[:,:3,3]=tt
        target=group/'camera_track.npz';temp=target.with_name('camera_track-'+__import__('uuid').uuid4().hex+'.tmp')
        try:
            with temp.open('wb') as stream:np.savez(stream,world_to_camera=matrices)
            os.replace(temp,target)
        finally:temp.unlink(missing_ok=True)
        camera_track=dict(path='camera_track.npz',stat=file_stat(target),start=job['start'],fps=30)
    people=[];projects=[]
    for number,(track_id,person) in enumerate(tracks,1):
        hands=hand_policy(person)
        frames=np.asarray(person['frames'],dtype=np.int64);observed=np.asarray(person.get('detected',np.ones(len(frames))),dtype=bool)
        if frames.shape!=(len(person['smplx_world']['pose']),) or observed.shape!=frames.shape or np.any(np.diff(frames)<=0) or (frames<0).any() or (frames>=n).any():
            raise ValueError('Invalid PromptHMR actor frame mapping.')
        runs=visible_runs(frames[observed],30,0,n)
        row=dict(id=number,name=prior.get(number,{}).get('name',f'Person {number}'),color=prior.get(number,{}).get('color',person_color(number)),
            checked=prior.get(number,{}).get('checked',True),observations=int(observed.sum()),first=int(frames[0]),last=int(frames[-1]),takes=[])
        for take,(first,end) in enumerate(runs):
            positions=np.searchsorted(frames,np.arange(first,end))
            if (positions>=len(frames)).any() or not np.array_equal(frames[positions],np.arange(first,end)):
                # Do not fill an unobserved discontinuity with another actor.
                raise ValueError('PromptHMR returned a non-contiguous track. Use a shorter trim around this take.')
            project=group/'actors'/f'person_{number:04}'/f'take_{take:03}';project.mkdir(parents=True,exist_ok=True)
            state=dict(common,frames=end-first,start=job['start']+first/30,end=job['start']+end/30,
                timeline=dict(start=job['start']+first/30,step=1/30,frames=end-first),people_root=os.path.relpath(group,project),
                person_id=number,person_frames=[first,end],workspace_root=os.path.relpath(root,project))
            native=dict(raw=file_stat(work/'results.pkl'),range=[first,end],track_id=str(track_id),body=file_stat(BY_KEY['smplx'].path),adapter=sha256(Path(__file__)),hand_pose_policy=hands)
            receipt=project/'prompthmr_source.json';reuse=False
            try:
                old=json.loads(receipt.read_text(encoding='utf8'));reuse=old['native']==native and old['output']==file_stat(project/'hmr4d_results.pt')
            except (OSError,ValueError,KeyError):pass
            atomic_json(project/'project.json',state)
            if not reuse:
                params=convert(person['smplx_world'],body,positions,neutral_hands=hands!='native');incam={k:v.clone() for k,v in params.items()}
                pelvis=body.J_regressor.numpy()[0]@body.v_template.numpy()+np.einsum('v,vci,bi->bc',body.J_regressor.numpy()[0],body.shapedirs.numpy(),params['betas'].numpy())
                r=rr[first:end] if moving else transform[:3,:3];t=tt[first:end] if moving else transform[:3,3]
                incam['global_orient']=torch.from_numpy(Rotation.from_matrix(r@Rotation.from_rotvec(params['global_orient'].numpy()).as_matrix()).as_rotvec().astype('float32'))
                root_positions=params['transl'].numpy()+pelvis
                incam['transl']=torch.from_numpy(((np.einsum('tij,tj->ti',r,root_positions) if moving else root_positions@r.T)+t-pelvis).astype('float32'))
                logits=np.asarray(person['smplx_cam']['static_conf_logits'],dtype=np.float32)
                if logits.shape!=(len(frames),6) or not np.isfinite(logits).all():raise ValueError('Missing or invalid native PromptHMR contact confidence.')
                save_tensor(dict(smpl_params_global=params,smpl_params_incam=incam,
                    net_outputs=dict(static_conf_logits=torch.from_numpy(logits[positions].copy())[None]),source_model='prompthmr',hand_pose_policy=hands),project/'hmr4d_results.pt')
                atomic_json(receipt,dict(version=1,native=native,output=file_stat(project/'hmr4d_results.pt')))
            if not (project/'input.mp4').exists():
                if first==0 and end==n:media_link(group/'input.mp4',project/'input.mp4')
                else:
                    subprocess.run([executable('ffmpeg'),'-v','error','-i',str(group/'input.mp4'),'-vf',f'trim=start_frame={first}:end_frame={end},setpts=PTS-STARTPTS',
                        '-an','-c:v','libx264','-crf','18','-preset','fast','-threads','2',str(project/'input.mp4')],check=True,creationflags=0x08000000)
            cam_record=dict(version=VERSION,sources=camera_sources(project),model=file_stat(BY_KEY['smplx'].path),
                world_to_camera=transform.tolist(),K=K.tolist(),htd_compatible=False,source_model='prompthmr',method='Native shared PromptHMR world; no independent actor alignment')
            if moving:cam_record['camera_track']={**camera_track,'path':os.path.relpath(group/'camera_track.npz',project)}
            atomic_json(project/'scene_camera.json',cam_record)
            row['takes'].append(dict(first=first,end=end,project=project.relative_to(group).as_posix()));projects.append(project)
        if row['takes']:row['preview']=row['takes'][0]['project'];people.append(row)
        emit('progress',current=number,total=len(tracks),message=f'Prepare PromptHMR source actor {number}')
    if not projects:raise ValueError('No sufficiently observed PromptHMR actor takes could be published.')
    atomic_json(group/'people.json',dict(version=1,input=media_receipt(group),people=people,active=people[0]['id']))
    from .prompthmr_scene import prepare_cameras
    prepare_cameras(group)
    from .scene_floor import save_config,resolve
    save_config(group,job['correction']);resolve(group)
    from .export import export_project
    from .correction import run as correct
    for index,project in enumerate(projects):
        emit('progress',current=index,total=len(projects),message=f'Prepare PromptHMR contacts and export bones {index+1}/{len(projects)}')
        export_project(project,'initial')
        if job['correction_enabled']:correct(project,emit,selected);export_project(project,selected)
    validate_inputs(job)
    (group/'prompthmr_pending.json').unlink(missing_ok=True)
    set_active(root,'prompthmr',projects[0]);state=read(root)
    state['models']['prompthmr']['settings']=dict(result=selected,correction=job['correction'],prompthmr_options=job['options'])
    atomic_json(root/'project.json',state)
    emit('prompthmr_ready',project=str(projects[0]),group=str(group),root=str(root),result=selected)
