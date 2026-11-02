"""PR #18-derived optional background rotation refinement of display cameras.

Preserve the original camera centers and source motion; no AI or project writes.
"""
import json
from pathlib import Path
import numpy as np
from .result_artifacts import file_stat


def fit_rotation(a,b,k,saved,size):
    """Fit on half of robust matches; accept only on held-out matches."""
    import cv2
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    a=np.asarray(a,float).reshape(-1,2);b=np.asarray(b,float).reshape(-1,2)
    if len(a)<60:return None,{'reason':'too few background matches'}
    _,mask=cv2.findHomography(a,b,cv2.RANSAC,1.5)
    if mask is None:return None,{'reason':'no background consensus'}
    keep=mask.ravel().astype(bool);a=a[keep];b=b[keep]
    if len(a)<60:return None,{'reason':'too few background inliers'}
    cells=np.floor(a/np.array(size)*[4,3]).astype(int)
    if len(np.unique(cells,axis=0))<6:return None,{'reason':'insufficient background coverage'}
    order=np.random.default_rng(42).permutation(len(a));train=order[::2];test=order[1::2]
    rays=np.c_[a,np.ones(len(a))]@np.linalg.inv(k).T
    def project(r):
        q=rays@r.T@k.T
        return q[:,:2]/np.maximum(q[:,2:],1e-8)
    def matrix(v):return Rotation.from_rotvec(v).as_matrix()@saved
    solution=least_squares(lambda v:(project(matrix(v))[train]-b[train]).ravel(),np.zeros(3),loss='soft_l1',f_scale=1,max_nfev=35)
    fitted=matrix(solution.x)
    before=np.linalg.norm(project(saved)[test]-b[test],axis=1)
    after=np.linalg.norm(project(fitted)[test]-b[test],axis=1)
    report=dict(matches=len(a),before_px=float(np.median(before)),after_px=float(np.median(after)),p90_px=float(np.percentile(after,90)),correction_degrees=float(np.rad2deg(np.linalg.norm(solution.x))))
    accepted=(solution.success and report['correction_degrees']<=2 and report['after_px']<2.5 and report['p90_px']<5 and report['after_px']<report['before_px']*.8 and report['before_px']-report['after_px']>.2)
    report['reason']='accepted' if accepted else 'parallax, small benefit or unreliable rotation fit'
    return (fitted if accepted else None),report


def refine(project,track,intrinsics,cancelled=lambda:False,progress=lambda *args:None):
    import cv2
    from scipy.spatial.transform import Rotation,Slerp
    from .video import select_preview
    project=Path(project);track=np.asarray(track,float)
    sources=[]
    def read(path):
        sources.append((path,file_stat(path)))
        return json.loads(path.read_text(encoding='utf-8'))
    state=read(project/'project.json');video=select_preview(project);sources.append((video,file_stat(video)))
    for name in ('input.mp4','preview.json'):
        path=project/name
        if path.exists():sources.append((path,file_stat(path)))
    n=len(track);fps=float(state['fps']);width=960;height=round(state['height']*width/state['width'])
    if (n<1 or n*16*8>32*1024**2 or height<1 or width*height>2*1024**2
            or not np.isfinite(fps) or fps<=0):raise ValueError('Background refinement exceeds its bounded preview size')
    if int(state['frames'])!=n:raise ValueError('Background camera timing differs from the selected take')
    model=state.get('pipeline',{}).get('model')
    if model not in ('exmr','flowhmr'):raise ValueError('Background refinement is unavailable for this model')
    if track.shape!=(n,4,4) or not np.isfinite(track).all():raise ValueError('Invalid camera track')
    k=np.array(intrinsics,float);k[:2]*=width/state['width']
    # Mask every known actor, not just the exported actor.
    box_tracks=[];root=(project/state.get('people_root','.')).resolve()
    if not project.resolve().is_relative_to(root):raise ValueError('Invalid people root')
    people=root/'people.json';begin=int(state.get('person_frames',[0])[0])
    candidates=[(project,begin)]
    if people.is_file():
        group=read(people)
        for person in group['people']:
            for take in person.get('takes',[]):
                path=(root/take['project']).resolve()
                if not path.is_relative_to(root):raise ValueError('Invalid take path')
                candidates.append((path,int(take['first'])))
    seen=set()
    for path,start in candidates:
        path=path/'tracking.npz'
        if path in seen or not path.exists():continue
        seen.add(path);sources.append((path,file_stat(path)))
        import zipfile
        with zipfile.ZipFile(path) as archive:
            if archive.getinfo('boxes.npy').file_size>32*1024**2:raise ValueError('Background actor mask exceeds its size limit')
        with np.load(path,allow_pickle=False) as data:boxes=np.array(data['boxes'],float)
        if boxes.ndim!=2 or boxes.shape[1]!=4:raise ValueError('Invalid background actor boxes')
        box_tracks.append((start,boxes))
    if not box_tracks:raise ValueError('Actor tracking is unavailable for background masking')
    def check():
        if cancelled():raise InterruptedError('Background camera refinement cancelled')
    def foreground_mask(index):
        mask=np.full((height,width),255,np.uint8)
        for start,boxes in box_tracks:
            j=begin+index-start
            if 0<=j<len(boxes) and np.isfinite(boxes[j]).all():
                x1,y1,x2,y2=boxes[j]*width/state['width'];pad=max(10,(x2-x1)*.08)
                cv2.rectangle(mask,(max(0,int(x1-pad)),max(0,int(y1-pad))),(min(width-1,int(x2+pad)),min(height-1,int(y2+pad))),0,-1)
        return mask
    report=dict(method='Background rotation refinement with held-out validation; fixed camera centers',reference_frame=0,measurement_width=width,samples=[],accepted=0)
    result=track.copy();cap=cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():raise ValueError('Cannot read camera reference video')
        if abs(cap.get(cv2.CAP_PROP_FPS)-fps)>.01 or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))!=n:raise ValueError('Camera video timing does not match motion')
        def frame(index):
            check();cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,img=cap.read();check()
            if not ok:raise ValueError('Cannot decode camera reference frame '+str(index))
            if abs(img.shape[1]/img.shape[0]-width/height)>.01:raise ValueError('Camera video aspect ratio does not match motion')
            return cv2.resize(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),(width,height))
        check();progress(0,0);first=frame(0);points=cv2.goodFeaturesToTrack(first,1200,.01,10,mask=foreground_mask(0))
        if points is None or len(points)<60:
            report['reason']='Insufficient visible background; original camera retained'
            if any(file_stat(p)!=s for p,s in sources):raise ValueError('Background tracking inputs changed; retry')
            return result,sources,report
        step=max(1,round(fps/5),int(np.ceil(n/600)))
        ids=np.unique(np.r_[np.arange(0,n,step),n-1]);corrections=[np.eye(3)]
        cv2.setRNGSeed(42)
        for sample,index in enumerate(ids[1:],1):
            check();progress(sample-1,len(ids)-1)
            gray=frame(index);q,status,_=cv2.calcOpticalFlowPyrLK(first,gray,points,None,winSize=(31,31),maxLevel=4)
            if q is None:
                corrections.append(np.eye(3));continue
            back,back_status,_=cv2.calcOpticalFlowPyrLK(gray,first,q,None,winSize=(31,31),maxLevel=4)
            if back is None:
                corrections.append(np.eye(3));continue
            a=points[:,0];b=q[:,0];xy=np.rint(b).astype(int);inside=(xy[:,0]>=0)&(xy[:,0]<width)&(xy[:,1]>=0)&(xy[:,1]<height)
            keep=(status.ravel()==1)&(back_status.ravel()==1)&(np.linalg.norm(back[:,0]-a,axis=1)<1)&inside
            mask=foreground_mask(index);keep[inside]&=mask[xy[inside,1],xy[inside,0]]>0
            saved=track[index,:3,:3]@track[0,:3,:3].T
            fitted,detail=fit_rotation(a[keep],b[keep],k,saved,(width,height))
            detail['frame']=int(index);report['samples'].append(detail)
            corrections.append(fitted@saved.T if fitted is not None else np.eye(3))
            report['accepted']+=int(fitted is not None)
        check();progress(len(ids)-1,len(ids)-1)
        # Interpolate only short, validated intervals; failures retain the saved
        # camera with a taper to identity at the rejected sample, never bridge gaps.
        smooth=Slerp(ids,Rotation.from_matrix(corrections))(np.arange(n)).as_matrix() if n>1 else np.eye(3)[None]
        result[:,:3,:]=smooth@track[:,:3,:]
        report['reason']='Applied validated rotation corrections' if report['accepted'] else 'No reliable improvement; original camera retained'
    finally:cap.release()
    check()
    if any(file_stat(p)!=s for p,s in sources):raise ValueError('Background tracking inputs changed; retry')
    return result,sources,report
