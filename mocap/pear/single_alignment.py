"""Single-view scale estimate, not a triangulation or measured camera baseline."""
import numpy as np
from .multiview import BODY_IDS


def metric_scale(samples,K,depth,confidence):
    relatives=samples.get('camera_relative')
    if relatives is None or relatives.shape!=(2,1,55,3):raise ValueError('Single-camera samples need a new Camera Align.')
    ratios=[]
    for s in range(2):
        uv=samples['uv'][s,0,BODY_IDS];j=relatives[s,0,BODY_IDS]
        if not np.isfinite(uv).all() or not np.isfinite(j).all():continue
        rays=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(K).T
        A=np.zeros((len(j),2,3));A[:,0,0]=1;A[:,1,1]=1;A[:,:,2]=-rays[:,:2]
        b=rays[:,:2]*j[:,2,None]-j[:,:2]
        translation,_,rank,_=np.linalg.lstsq(A.reshape(-1,3),b.reshape(-1),rcond=None)
        if rank!=3 or translation[2]<=.1:continue
        predicted=j+translation
        reprojection=predicted@K.T;reprojection=reprojection[:,:2]/reprojection[:,2,None]
        if np.percentile(np.linalg.norm(reprojection-uv,axis=1),95)>max(samples['sizes'][0])*.02:continue
        h,w=depth[s].shape;size=samples['sizes'][0]
        pixels=np.rint(uv*np.array([w,h])/size).astype(int)
        valid=(pixels[:,0]>=0)&(pixels[:,0]<w)&(pixels[:,1]>=0)&(pixels[:,1]<h)
        pixels=pixels[valid];metric=predicted[valid,2]
        d=depth[s,pixels[:,1],pixels[:,0]];c=confidence[s,pixels[:,1],pixels[:,0]]
        good=np.isfinite(d)&(d>1e-5)&(metric>.1)&np.isfinite(c)
        if np.count_nonzero(good)<6:continue
        # VGGT confidence varies with scene content. Easy walls/floors must not
        # disqualify every person sample; rank only the depth evidence used here.
        good&=c>np.percentile(c[good],20)
        ratio=metric[good]/d[good]
        if len(ratio)>=6:ratios.extend(ratio)
    if len(ratios)<12:raise ValueError('Single-camera scale has insufficient valid person-depth samples. Try another calibration time or add a camera.')
    ratios=np.asarray(ratios);median=np.median(ratios)
    if np.percentile(abs(ratios-median),90)>median*.45:raise ValueError('Single-camera depth scale is inconsistent. Choose a clearer calibration time.')
    return ratios.tolist()
