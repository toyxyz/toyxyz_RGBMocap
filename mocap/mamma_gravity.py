"""Camera-balanced GeoCalib gravity and isolated Flow-runtime bridge for MAMMA."""
import numpy as np

def unit(x):
    x=np.asarray(x,float);n=np.linalg.norm(x,axis=-1,keepdims=True)
    if not np.isfinite(x).all() or np.any(n<1e-8):raise ValueError('Nonfinite or zero gravity vector')
    return x/n

def angular(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a,b),axis=-1),np.sum(a*b,axis=-1)))

def spherical(values,weights=None):
    values=unit(values);weights=np.ones(len(values)) if weights is None else np.asarray(weights,float)
    u=unit(np.sum(weights[:,None]*values,axis=0))
    for _ in range(12):
        residual=np.radians(angular(values,u));robust=np.minimum(1.,np.radians(2.)/np.maximum(residual,1e-8))
        nxt=unit(np.sum((weights*robust)[:,None]*values,axis=0))
        if angular(u,nxt)<1e-8:break
        u=nxt
    return u

def estimate(observations,expected_cameras,check_holdouts=True):
    if len(set(expected_cameras))!=len(expected_cameras):raise ValueError('Duplicate expected camera identity')
    pairs=[(r['camera'],r['frame']) for r in observations]
    if len(set(pairs))!=len(pairs):raise ValueError('Duplicate camera-time observation')
    if any(r['camera'] not in expected_cameras for r in observations):raise ValueError('Unexpected camera identity')
    rows=[];rejected=[]
    for row in observations:
        sigma=row.get('uncertainty_radians',{}).get('gravity_uncertainty',np.nan)
        bg=row.get('background_fraction',0.)
        if not np.isfinite(sigma) or sigma<0 or sigma>np.radians(10) or not np.isfinite(bg) or bg<.25:
            rejected.append((row['camera'],row['frame']));continue
        try:u=unit(row['up_saved_world'])
        except ValueError:rejected.append((row['camera'],row['frame']));continue
        rows.append((row,u))
    cameras=sorted(set(row['camera'] for row,_ in rows));centres=[];names=[];kept=[]
    for c in cameras:
        group=[(row,u) for row,u in rows if row['camera']==c]
        if len(group)<2:continue
        v=np.array([u for row,u in group]);dist=angular(v[:,None],v[None,:]);medoid=v[np.argmin(np.sum(np.minimum(dist,6),axis=1))]
        group=[x for x in group if angular(x[1],medoid)<=6]
        if len(group)<2:continue
        centres.append(spherical([u for row,u in group]));names.append(c);kept.extend(group)
    need=max(2,int(np.ceil(len(expected_cameras)*.75)))
    if len(centres)<need:raise ValueError('Too few independent cameras support gravity')
    centres=np.array(centres);dist=angular(centres[:,None],centres[None,:]);counts=(dist<=6).sum(1)
    best=int(counts.max());candidates=np.flatnonzero(counts==best)
    if any(dist[i,j]>8 for i in candidates for j in candidates):raise ValueError('Competing gravity modes')
    seed=candidates[np.argmin(np.sum(np.minimum(dist[candidates],6),axis=1))];good=dist[seed]<=6
    if good.sum()<need:raise ValueError('Camera gravity estimates disagree')
    names=[c for c,g in zip(names,good) if g];centres=centres[good];up=spherical(centres)
    residual=angular(centres,up)
    accepted=[(r,u) for r,u in kept if r['camera'] in names and angular(u,up)<=6]
    if len(accepted)<int(np.ceil(len(observations)*.75)):raise ValueError('Insufficient accepted camera-time samples')
    # Equal total camera influence; do not multiply identical static views' influence by sample count.
    sample_residual=np.array([angular(u,up) for _,u in accepted]);rms=float(np.sqrt(np.mean(residual**2)))
    if rms>2 or np.percentile(sample_residual,90)>3:raise ValueError('Gravity angular dispersion too large')
    result=dict(up=up.tolist(),cameras=names,accepted_samples=len(accepted),total_samples=len(observations),camera_rms_deg=rms,sample_residual_p90_deg=float(np.percentile(sample_residual,90)),rejected_samples=rejected,estimator='equal-camera bounded angular Huber / spherical mean',thresholds_are_provisional=True)
    if check_holdouts:
        loo=[angular(spherical(np.delete(centres,k,axis=0)),up) for k in range(len(centres))] if len(centres)>2 else [float(residual.max())]
        lot=[]
        for t in sorted(set(row['frame'] for row,_ in accepted)):
            vals=[]
            for c in names:
                v=[u for row,u in accepted if row['frame']!=t and row['camera']==c]
                if v:vals.append(spherical(v))
            if len(vals)>=need:lot.append(angular(spherical(vals),up))
        if max(loo)>1.5 or (lot and max(lot)>1):raise ValueError('Gravity estimate changes under camera/time holdout')
        result.update(camera_holdout_max_deg=float(max(loo)),time_holdout_max_deg=float(max(lot)) if lot else None)
    return result

def support_identity():
    from .pear.alignment_support import configuration,identity
    from .runtime_registry import PROFILES
    from .storage import sha256
    folder=configuration()
    return dict(identity=identity(),provenance=sha256(folder/'PROVENANCE.json'),
                profile=PROFILES['flow'],python=PROFILES['shared']['python'])


def world_basis(up,reference_rotation):
    up=unit(up);r=np.asarray(reference_rotation,float)
    forward=r.T@np.array([0.,0.,1.]);forward-=up*np.dot(forward,up)
    if np.linalg.norm(forward)<1e-6:
        # Looking straight down/up has no observable projected forward yaw.
        right=r.T@np.array([1.,0.,0.]);right-=up*np.dot(right,up)
        forward=np.cross(up,unit(right))
    forward=unit(forward);right=unit(np.cross(forward,up))
    return np.stack([right,forward,up])


def infer(work,entries,frames,rotations,intrinsics,emit):
    import json,subprocess,uuid
    from .paths import ROOT
    from .storage import atomic_json,sha256
    support=support_identity();token=uuid.uuid4().hex
    request=work/'gravity-request.json';output=work/'gravity-result.json'
    atomic_json(request,dict(version=1,token=token,support=support,frames=list(frames),
        cameras=[dict(id=e['id'],rotation=r.tolist(),intrinsics=k.tolist())
                 for e,r,k in zip(entries,rotations,intrinsics)]))
    digest=sha256(request)
    process=subprocess.Popen([str(ROOT/'runtime/python.exe'),'-I','-X','utf8','-B','-u',
        str(ROOT/'worker_boot.py'),'flow_mamma_gravity',str(request)],stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',creationflags=0x08000000)
    tail=[]
    try:
        while True:
            line=process.stdout.readline(65537)
            if not line:break
            if len(line)>65536:raise RuntimeError('Invalid GeoCalib worker output.')
            tail.append(line.rstrip());tail=tail[-20:]
            if line.startswith('@@MOCAP@@'):
                event=json.loads(line[len('@@MOCAP@@'):]);kind=event.pop('type')
                emit(kind,**event)
        if process.wait():raise RuntimeError('MAMMA GeoCalib failed. '+ '\n'.join(tail)[-3500:])
        if sha256(request)!=digest or output.stat().st_size>2*1024**2:
            raise ValueError('MAMMA gravity inputs changed during processing.')
        result=json.loads(output.read_text('utf8'))
        if result['token']!=token or result['request_sha256']!=digest or result['support']!=support_identity():
            raise ValueError('Stale or changed MAMMA gravity result.')
        checked=estimate(result['observations'],[e['id'] for e in entries])
        atomic_json(work/'gravity-estimation.json',{**result,'consensus':checked})
        return unit(checked['up'])
    finally:
        if process.poll() is None:
            process.kill();process.wait(timeout=10)
        process.stdout.close()
