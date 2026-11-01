"""Causal adaptive smoothing; rotation matrices remain on SO(3).

Speed-adaptive cutoff follows the 1 Euro principle (Casiez et al., CHI 2012).
Rotations use geodesic interpolation instead of averaging matrix entries.
No future frames, queues, global trajectory solve or contact constraints.
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation

def alpha(cutoff,dt):return 1/(1+1/(2*math.pi*np.asarray(cutoff)*dt))

def strength_scale(strength):
    value=float(strength)
    if not math.isfinite(value):raise ValueError('Invalid smoothing strength')
    return 2**((min(100.,max(1.,value))-50.)/25.)

class AdaptiveFilter:
    def __init__(self,min_cutoff=1.2,beta=.65,rotation=False,vector=False,max_cutoff=8.):
        self.min_cutoff=min_cutoff;self.beta=beta;self.rotation=rotation
        self.vector=vector;self.max_cutoff=max_cutoff;self.reset()
    def reset(self):self.raw=None;self.value=None;self.derivative=None
    def update(self,value,dt):
        x=np.asarray(value,dtype=np.float64)
        # A paused frame/control refresh has no elapsed source time. Retain the
        # unaffected channel instead of reseeding it from an unsmoothed sample.
        if self.raw is not None and dt==0 and self.raw.shape==x.shape:return self.value.astype(np.float32)
        if self.raw is None or not 0<dt<=.25 or self.raw.shape!=x.shape:
            self.raw=x.copy();self.value=x.copy()
            self.derivative=np.zeros(x.shape[:-2]+(3,)) if self.rotation else np.zeros_like(x)
            return self.value.astype(np.float32)
        if self.rotation:
            # Compare angular velocities in the same (parent) coordinate frame,
            # rather than averaging vectors expressed in changing local frames.
            delta=Rotation.from_matrix((x@self.raw.swapaxes(-1,-2)).reshape(-1,3,3)).as_rotvec().reshape(self.derivative.shape)/dt
        else:delta=(x-self.raw)/dt
        self.derivative+=alpha(1.,dt)*(delta-self.derivative)
        speed=np.linalg.norm(self.derivative,axis=-1,keepdims=True) if self.rotation or self.vector else np.abs(self.derivative)
        # Quiet poses get stronger smoothing. Cap responsiveness so noisy
        # derivative spikes cannot effectively disable filtering.
        a=alpha(np.minimum(self.max_cutoff,self.min_cutoff+self.beta*speed),dt)
        if self.rotation:
            residual=Rotation.from_matrix((self.value.swapaxes(-1,-2)@x).reshape(-1,3,3)).as_rotvec().reshape(self.derivative.shape)
            self.value=self.value@Rotation.from_rotvec((a*residual).reshape(-1,3)).as_matrix().reshape(x.shape)
        else:self.value+=a*(x-self.value)
        self.raw=x.copy()
        return self.value.astype(np.float32)

class ParameterFilter:
    """One Euro state in two bounded batches: rotations and scalar parameters."""
    def __init__(self,strength=50.):self.strength=float(strength);self.reset()
    def reset(self):self.filters={};self.last_time=None;self.schema=None
    def set_strength(self,strength):
        scale=strength_scale(strength);self.strength=float(strength)
        for rotation,f in self.filters.items():
            f.min_cutoff=(1.2 if rotation else np.where(self.shape,.35,1.2))/scale
            f.beta=(.65 if rotation else np.where(self.shape,.02,.65))/scale
            f.max_cutoff=8./scale
    def update(self,body,flame,timestamp):
        dt=timestamp-self.last_time if self.last_time is not None else 0.
        entries=[]
        for prefix,values in (('body',body),('flame',flame)):
            for key in sorted(values):
                x=np.asarray(values[key]);original=x.shape
                axis_angle=prefix=='flame' and key in ('eye_pose_params','pose_params','jaw_params')
                if axis_angle:x=Rotation.from_rotvec(x.reshape(-1,3)).as_matrix()
                entries.append((prefix,key,x,original,axis_angle,x.shape[-2:]==(3,3)))
        schema=tuple((p,k,x.shape,shape,aa,rot) for p,k,x,shape,aa,rot in entries)
        if not 0<=dt<=.25 or schema!=self.schema:self.filters={};dt=0.
        self.last_time=timestamp;self.schema=schema
        result={'body':{},'flame':{}}
        for rotation in (True,False):
            items=[e for e in entries if e[-1]==rotation]
            if not items:continue
            values=np.concatenate([x.reshape(-1,3,3) if rotation else x.reshape(-1) for _,_,x,_,_,_ in items])
            if rotation not in self.filters:
                if rotation:self.filters[rotation]=AdaptiveFilter(rotation=True)
                else:
                    self.shape=np.concatenate([np.full(x.size,k in ('shape','shape_params','hand_scale','head_scale')) for _,k,x,_,_,_ in items])
                    self.filters[rotation]=AdaptiveFilter()
                self.set_strength(self.strength)
            filtered=self.filters[rotation].update(values,dt);offset=0
            for prefix,key,x,original,axis_angle,_ in items:
                count=x.size//9 if rotation else x.size
                value=filtered[offset:offset+count].reshape(x.shape);offset+=count
                if axis_angle:value=Rotation.from_matrix(value).as_rotvec().reshape(original).astype(np.float32)
                result[prefix][key]=value
        return result['body'],result['flame']
