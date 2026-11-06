"""Transactional finger-only HandFlow results; importing this module is GUI-safe."""
from pathlib import Path
import json,os,uuid
from .storage import atomic_json,sha256
from .result_artifacts import file_stat
from .motion_tools import RESULTS,result_sources,base_result,body_result,has_handflow

VERSION=1


def code_path():
    # Frozen GUI modules live in PYZ; the worker sidecar is the shared identity.
    from .paths import ROOT
    return ROOT/'mocap/handflow.py'


def result_current(project,kind):
    project=Path(project)
    if (project/'handflow_pending.json').exists():return False
    try:
        record=json.loads((project/(kind+'_handflow.json')).read_text(encoding='utf-8'))
        if record['version']!=VERSION or file_stat(project/RESULTS[kind][0])!=record['output_stat']:return False
        if record['code_stat']!=file_stat(code_path()):return False
        if not (project/(kind+'_sources.json')).is_file():return False
        from .motion_tools import result_is_current
        if not result_is_current(project,body_result(kind)):return False
        state=json.loads((project/'project.json').read_text(encoding='utf-8'))
        if record['fps']!=state.get('fps',30) or record['frames']!=state['frames']:return False
        from .models import BY_KEY
        if any(BY_KEY[k].path.is_file() and file_stat(BY_KEY[k].path)!=v for k,v in record['models'].items()):return False
        return True
    except (OSError,ValueError,KeyError,TypeError):return False


def input_signature(project):
    import hashlib
    from importlib.metadata import version
    from .models import BY_KEY,STAGE_REQUIREMENTS
    from .paths import UPSTREAM
    from .resource_cache import code_hash  # Exact-byte aliases retain results after platform-only edits.
    names=('input.mp4','tracking.npz','gvhmr_keypoints.pt','camera.pt')
    payload=dict(version=VERSION,inputs={n:file_stat(project/n) for n in names},
        models={k:file_stat(BY_KEY[k].path) for k in STAGE_REQUIREMENTS['handflow']},
        runtime={k:version(k) for k in ('torch','ultralytics','timm')},
        code=[code_hash(Path(__file__).with_name('handflow_engine.py')),
              sha256(UPSTREAM/'HandFlow/utils/inference_utils.py')])
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),payload


def run(project,emit,result):
    """Infer once from this actor's observations, merge onto every selected body base."""
    import numpy as np,torch
    from .engines import save_tensor
    from .models import BY_KEY
    from .motion_tools import result_is_current
    project=Path(project);selected=base_result(result)
    if not has_handflow(selected):raise ValueError('Select a HandFlow result for finger inference.')
    state=json.loads((project/'project.json').read_text(encoding='utf-8'));fps=float(state.get('fps',30));n=state['frames']
    signature,payload=input_signature(project)
    cache=project/'handflow_prediction.json';record={}
    try:record=json.loads(cache.read_text(encoding='utf-8'))
    except (OSError,ValueError):pass
    cached=False
    if record.get('signature')==signature:
        try:
            name=record['file'];assert Path(name).name==name
            cached=file_stat(project/name)==record['stat']
            if cached:
                with np.load(project/name,allow_pickle=False) as data:
                    rotations=data['rotations'];valid=data['valid']
                cached=bool(rotations.shape==(n,2,15,3,3) and np.isfinite(rotations).all() and valid.shape==(n,2)
                    and np.max(np.abs(rotations.swapaxes(-1,-2)@rotations-np.eye(3)))<2e-4 and np.min(np.linalg.det(rotations))>.9998)
        except (OSError,ValueError,KeyError,AssertionError):cached=False
    atomic_json(project/'handflow_pending.json',dict(generation=uuid.uuid4().hex))
    if not cached:
        from .handflow_engine import infer
        rotations,valid,details=infer(project,emit,n)
        if input_signature(project)[0]!=signature:raise ValueError('HandFlow inputs changed during inference. Run Reconstruct 3D again.')
        name='handflow-prediction-'+uuid.uuid4().hex[:12]+'.npz'
        temporary=project/(name+'.partial.npz');np.savez(temporary,rotations=rotations,valid=valid);os.replace(temporary,project/name)
        record=dict(signature=signature,file=name,stat=file_stat(project/name),details=details)
        atomic_json(cache,record)
    from scipy.spatial.transform import Rotation
    import smplx
    model=smplx.SMPLX(str(BY_KEY['smplx'].path),use_pca=False,flat_hand_mean=False).eval()
    means=np.stack([model.left_hand_mean.numpy(),model.right_hand_mean.numpy()])
    angles=Rotation.from_matrix(rotations.reshape(-1,3,3)).as_rotvec().reshape(n,2,45).astype(np.float32)
    warnings=[]
    for side,label in enumerate(('Left','Right')):
        if not valid[:,side].any():
            angles[:,side]=means[side]
            warnings.append(label+' hand was not detected; source default fingers are retained.')
        elif valid[:,side].mean()<.3:warnings.append(label+' hand detection coverage is low; inspect the finger motion.')
    written=[]
    for kind in result_sources(selected):
        if not has_handflow(kind):continue
        source_kind=body_result(kind)
        if not result_is_current(project,source_kind):raise ValueError('Selected body result is stale; reconstruct it first.')
        filename,key=RESULTS[source_kind];source_stamp=file_stat(project/filename)
        source=torch.load(project/filename,map_location='cpu',weights_only=False)[key]
        if len(source['body_pose'])!=n:raise ValueError('HandFlow body/video frame counts differ.')
        params=dict(source)
        # SMPL-X adds its mean after PCA expansion. Store all 45 residuals,
        # never project the physical predictions down to 12 PCA coefficients.
        for side,field in enumerate(('left_hand_pose','right_hand_pose')):
            params[field]=torch.from_numpy(angles[:,side]-means[side])
        if not all(torch.equal(params[k],v) for k,v in source.items() if k not in ('left_hand_pose','right_hand_pose')):
            raise ValueError('HandFlow changed body motion.')
        target=project/RESULTS[kind][0]
        save_tensor({'smpl_params_global':params,'source':source_kind,'method':'HandFlow finger-only; body and wrists unchanged'},target)
        if file_stat(project/filename)!=source_stamp:raise ValueError('Body source changed during HandFlow merge.')
        sources={filename:source_stamp,record['file']:record['stat'],'handflow_prediction.json':file_stat(cache),
                 **{k:v for k,v in payload['inputs'].items()}}
        atomic_json(project/(kind+'_sources.json'),sources)
        atomic_json(project/(kind+'_handflow.json'),dict(version=VERSION,output_stat=file_stat(target),code_stat=file_stat(code_path()),
            models=payload['models'],fps=fps,frames=n,body_and_wrists_unchanged=True,physical_finger_rotations=True,
            detection_coverage=valid.mean(0).tolist(),prediction_cached=cached,warning=' '.join(warnings)))
        # Existing manual foot timings remain meaningful: all body tensors above
        # are bit-identical. Copy valid edits only for a brand-new hand variant.
        from .foot_contacts import state_path,read_edit,save_edit
        if not state_path(project,kind).exists():
            context=dict(frames=n,fps=fps,sources={filename:source_stamp,'hmr4d_results.pt':file_stat(project/'hmr4d_results.pt')})
            ranges,_,_=read_edit(project,source_kind,context)
            if ranges is not None:
                save_edit(project,kind,ranges,dict(frames=n,fps=fps,sources={target.name:file_stat(target),'hmr4d_results.pt':file_stat(project/'hmr4d_results.pt')}))
        written.append(kind)
    if input_signature(project)[0]!=signature:raise ValueError('HandFlow inputs changed during merge.')
    atomic_json(project/'handflow_report.json',dict(results=written,cached=cached,frames=n,body_and_wrists_unchanged=True,
        detection_coverage=valid.mean(0).tolist(),details=record.get('details',{})))
    (project/'handflow_pending.json').unlink()
    # Small inference caches retain one committed generation. Never touch user inputs.
    for old in project.glob('handflow-prediction-*.npz'):
        if old.name!=record['file']:
            try:old.unlink()
            except OSError:pass
    return project/RESULTS[selected][0]
