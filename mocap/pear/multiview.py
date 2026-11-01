"""Fixed-rig, causal multiview fusion. No mesh solve or per-frame calibration."""
import numpy as np
from scipy.spatial.transform import Rotation
from .bone_roll import JOINTS,TIPS
from .temporal_filter import ParameterFilter,AdaptiveFilter,strength_scale

CV=np.diag([-1.,-1.,1.])
DISPLAY=np.array([[1.,0,0],[0,0,1.],[0,-1.,0]])
BODY_IDS=np.array([16,17,18,19,20,21,1,2,4,5,7,8])

def local_matrices(body):
    return np.concatenate([body['global_pose'].reshape(1,3,3),body['body_pose'].reshape(21,3,3),
        np.tile(np.eye(3),(3,1,1)),body['left_hand_pose'].reshape(15,3,3),body['right_hand_pose'].reshape(15,3,3)])

def fk(local,rig):
    rotations=local.copy();joints=np.zeros((145,3));rest=rig.rest
    for k in range(1,55):
        p=int(rig.native_parents[k]);rotations[k]=rotations[p]@local[k]
        joints[k]=joints[p]+rotations[p]@(rest[k]-rest[p])
    for k,offset in rig.tip_offsets.items():joints[TIPS[k]]=joints[k]+rotations[k]@offset
    return joints,rotations

def observations(packet,rig):
    joints,_=fk(local_matrices(packet['body']),rig);joints+=rig.rest[0]
    cam=packet['camera'];q=joints@cam[:3,:3].T+cam[:3,3];crop=packet['crop_bbox']
    if np.any(q[:55,2]<=.01):raise ValueError('Invalid PEAR crop-camera depth')
    return crop[:2]+crop[2:]/2-12*crop[2:]*q[:,:2]/q[:,2:]

class Fusion:
    def __init__(self,rig,cameras):
        self.rig=rig;self.K=np.asarray(cameras['intrinsics']);self.E=np.asarray(cameras['extrinsics'])
        n=len(self.K)
        if self.K.shape!=(n,3,3) or self.E.shape!=(n,3,4) or not 1<=n<=32 or not np.isfinite(self.K).all() or not np.isfinite(self.E).all():raise ValueError('Invalid camera alignment arrays')
        rr=self.E[:,:,:3]
        if np.any(self.K[:,[0,1],[0,1]]<=0) or np.max(abs(rr.swapaxes(-1,-2)@rr-np.eye(3)))>1e-4 or np.min(np.linalg.det(rr))<.999:raise ValueError('Invalid camera alignment rotations or intrinsics')
        self.P=self.K@self.E;self.children=[np.flatnonzero(rig.native_parents==i) for i in range(55)]
        self.filter=ParameterFilter();self.root_filter=AdaptiveFilter(1.2,4.,vector=True);self.reset()
    def reset(self):
        self.filter.reset();self.root_filter.reset();self.weights=None;self.time=None
    def quality(self,c,j,uv,packet):
        q=np.empty(55);rr=self.E[c,:,:3];box=packet['bbox'];size=packet['image_size']
        scale=np.clip(np.sqrt(max(box[3]-box[1],1)/420),.45,1.5)
        cam=j[:55]@rr.T;torso=(uv[3]+uv[9])/2;radius=max(12,np.linalg.norm(uv[16]-uv[17])*.38)
        for k in range(55):
            child=self.children[k];delta=j[child].mean(0)-j[k] if len(child) else j[k]-j[max(0,int(self.rig.native_parents[k]))]
            d=rr@delta;observable=np.sqrt(max(0.,1-(d[2]/max(np.linalg.norm(d),1e-6))**2))
            inside=np.clip(np.minimum(np.minimum(uv[k],size-uv[k]),40).min()/40,.05,1.)
            occluded=k in (18,19,20,21,4,5,7,8) and np.linalg.norm(uv[k]-torso)<radius and cam[k,2]>cam[9,2]+.08
            q[k]=scale*(.15+.85*observable)*inside*(.25 if occluded else 1.)
        q[25:40]=q[20];q[40:55]=q[21]
        return q
    def run(self,packets,timestamp,smoothing=True,strength=50,smooth_rotations=True,smooth_position=True):
        available=[i for i,p in enumerate(packets) if p is not None]
        if not available:self.reset();return None
        obs={};quality=np.zeros((len(packets),55));locals=[]
        for c in available:
            p=packets[c];L=local_matrices(p['body']);obs[c]=observations(p,self.rig)
            L[0]=self.E[c,:,:3].T@CV@p['camera'][:3,:3]@L[0]
            j,_=fk(L,self.rig);locals.append(L);quality[c]=self.quality(c,j,obs[c],p)
        weights=quality/np.maximum(quality.sum(0),1e-8);dt=0 if self.time is None else timestamp-self.time
        if self.weights is not None and 0<dt<=.25:weights=self.weights+(1-np.exp(-dt/.1))*(weights-self.weights)
        weights[[i for i in range(len(packets)) if i not in available]]=0
        weights/=np.maximum(weights.sum(0),1e-8);self.weights=weights;self.time=timestamp
        L=locals[0].copy();A=np.asarray(locals)
        for _ in range(3):
            delta=Rotation.from_matrix((L.swapaxes(-1,-2)[None]@A).reshape(-1,3,3)).as_rotvec().reshape(len(available),55,3)
            L=L@Rotation.from_rotvec((weights[available,:,None]*delta).sum(0)).as_matrix()
        if smoothing and smooth_rotations:
            self.filter.set_strength(strength)
            b=dict(global_pose=L[:1][None],body_pose=L[1:22][None],left_hand_pose=L[25:40][None],right_hand_pose=L[40:55][None])
            b,_=self.filter.update(b,{},timestamp);L=local_matrices(b)
        else:self.filter.reset()
        j,r=fk(L,self.rig);aa=[];bb=[]
        for c in available:
            for k in BODY_IDS:
                u,v=obs[c][k];p=self.P[c];mat=np.stack([p[0,:3]-u*p[2,:3],p[1,:3]-v*p[2,:3]])
                w=np.sqrt(max(.02,quality[c,k]));aa.append(mat*w)
                bb.append((-mat@j[k]-[p[0,3]-u*p[2,3],p[1,3]-v*p[2,3]])*w)
        a=np.concatenate(aa);b=np.concatenate(bb);root,_,rank,_=np.linalg.lstsq(a,b,rcond=None)
        if rank<3 or not np.isfinite(root).all() or np.linalg.norm(root)>1000:raise ValueError('Camera geometry cannot determine a stable position. Run Camera Align again.')
        if smoothing and smooth_position:
            scale=strength_scale(strength);self.root_filter.min_cutoff=1.2/scale;self.root_filter.beta=4/scale;self.root_filter.max_cutoff=8/scale
            root=self.root_filter.update(root,dt)
        else:self.root_filter.reset()
        j+=root
        return dict(joints=np.ascontiguousarray(j@DISPLAY.T,dtype='float32'),
            bounds=np.ascontiguousarray(j[np.r_[JOINTS,list(TIPS.values())]]@DISPLAY.T,dtype='float32'),
            rotations=np.ascontiguousarray(DISPLAY@r,dtype='float32'),
            bone_facing=np.ascontiguousarray(DISPLAY@(r[JOINTS]@self.rig.basis),dtype='float32'))
