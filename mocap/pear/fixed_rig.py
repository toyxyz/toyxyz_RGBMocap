"""One-shot CPU anatomy calibration, then fixed-length FK without live skinning."""
import numpy as np
from collections import OrderedDict
from .bone_roll import JOINTS,TIPS,SkeletonLayout,source_basis,unit


def forward_toe_tips(rest,joints):
    """Centre display toe tips on each foot's sagittal plane, keeping length/pitch."""
    result=np.asarray(joints).copy();up=unit(rest[3]-rest[0])
    for ankle,toe in ((7,10),(8,11)):
        tip=TIPS[toe];offset=result[tip]-rest[toe];length=np.linalg.norm(offset)
        forward=rest[toe]-rest[ankle];forward=unit(forward-up*np.dot(forward,up))
        height=float(np.clip(np.dot(offset,up),-length,length))
        result[tip]=rest[toe]+forward*np.sqrt(max(0.,length*length-height*height))+up*height
    return result


class CalibrationFrames:
    """Small parameter-only history for an exact displayed-frame Reset Bones."""
    def __init__(self):self.frames=OrderedDict()
    def clear(self):self.frames.clear()
    def remember(self,key,body,flame):
        self.frames[key]=tuple({k:np.asarray(v).copy() for k,v in group.items()} for group in (body,flame))
        while len(self.frames)>32:self.frames.popitem(last=False)
    def get(self,key):return self.frames.get(key)


class FixedRig(SkeletonLayout):
    def __init__(self,calibrator):
        self.calibrator=calibrator
        self.native_parents=calibrator.smplx.parents.detach().cpu().numpy().astype('int32')
        super().__init__(self.native_parents)
        self.revision=0;self.clear()

    def clear(self):
        self.rest=None;self.tip_offsets=None;self.basis=None;self.sole_offsets=None

    def calibrate(self,body,flame):
        """Publish a complete replacement only after validating the current pose's anatomy."""
        import torch
        from ..source_surface import layout,endpoints
        b={k:torch.from_numpy(np.asarray(v,dtype='float32').copy()) for k,v in body.items()}
        f={k:torch.from_numpy(np.asarray(v,dtype='float32').copy()) for k,v in flame.items()}
        for key in ('global_pose','body_pose','left_hand_pose','right_hand_pose'):
            b[key]=torch.eye(3).expand_as(b[key]).clone()
        with torch.inference_mode():
            neutral=self.calibrator(b,f,pose_type='aa')
            vertices=neutral['vertices']
            # Identity body rotations leave the shaped template unchanged. The
            # physical FK pivots precede EHM's optional mesh-joint replacement.
            rest=torch.einsum('jv,bvc->bjc',self.calibrator.smplx.J_regressor,vertices)
            if 'joints_offset' in b:rest=rest+b['joints_offset']
            rest=rest[0].numpy().copy();v=vertices[0].numpy();j=neutral['joints'][0].numpy().copy()
        lengths=np.linalg.norm(rest[1:]-rest[self.native_parents[1:]],axis=1)
        if rest.shape!=(55,3) or not np.isfinite(rest).all() or not np.isfinite(v).all() or np.any(lengths<1e-5) or np.any(lengths>5):
            raise ValueError('Cannot calibrate bones from this pose. Choose a clearly visible person and try Reset Bones again.')
        ids,lookup=layout('smplx');surface=endpoints(v[None,ids],lookup)[0]
        basis=source_basis(rest[JOINTS],self.parents,self.children,surface)
        j[:55]=rest;j=forward_toe_tips(rest,j)
        p,ends=self.landmarks(j);y=unit(self.directions(p,ends));x=basis[:,:,0]
        x=unit(x-y*np.sum(x*y,-1,keepdims=True));basis=np.stack([x,y,unit(np.cross(x,y))],-1).astype('float32')
        offsets={native:(j[tip]-rest[native]).astype('float32') for native,tip in TIPS.items()}
        if not np.isfinite(basis).all() or np.max(abs(basis.swapaxes(-1,-2)@basis-np.eye(3)))>1e-4:
            raise ValueError('Invalid bone calibration. Try Reset Bones on another frame.')
        # One neutral rigid sole proxy per foot; no skin evaluation at runtime.
        weights=self.calibrator.smplx.lbs_weights.detach().cpu().numpy()
        sole=[];up=unit(rest[3]-rest[0])
        for ankle,toe in ((7,10),(8,11)):
            patch=v[(weights[:,ankle]+weights[:,toe])>.65]
            if len(patch)<8:raise ValueError('Cannot calibrate the foot sole from this body model.')
            height=patch@up;low=patch[height<=np.percentile(height,8)]
            sole.append(np.median(low,axis=0)-rest[ankle])
        self.sole_offsets=np.asarray(sole,dtype='float32')
        self.rest=rest.astype('float32');self.tip_offsets=offsets;self.basis=basis;self.revision+=1

    def pose(self,body,camera):
        if self.rest is None:raise RuntimeError('PEAR bones have not been calibrated.')
        local=np.concatenate([np.asarray(body['global_pose']).reshape(1,3,3),np.asarray(body['body_pose']).reshape(21,3,3),
            np.tile(np.eye(3),(3,1,1)),np.asarray(body['left_hand_pose']).reshape(15,3,3),
            np.asarray(body['right_hand_pose']).reshape(15,3,3)]).astype('float32')
        rotations=local.copy();joints=np.zeros((145,3),dtype='float32');joints[:55]=self.rest
        for i in range(1,55):
            parent=int(self.native_parents[i]);rotations[i]=rotations[parent]@local[i]
            joints[i]=joints[parent]+rotations[parent]@(self.rest[i]-self.rest[parent])
        for native,tip in TIPS.items():joints[tip]=joints[native]+rotations[native]@self.tip_offsets[native]
        rotations=np.asarray(camera,dtype='float32')[:3,:3]@rotations
        facing=rotations[JOINTS]@self.basis
        bounds=joints[np.r_[JOINTS,list(TIPS.values())]]
        return bounds,joints,rotations,facing
