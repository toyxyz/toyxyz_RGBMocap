"""Image evidence for the forearm direction at a tracked wrist."""
import cv2
import numpy as np

ANGLES = np.linspace(-np.pi, np.pi, 90, endpoint=False)
UV_AXIS = np.stack([np.cos(ANGLES), np.sin(ANGLES)], axis=1)
TUBE = np.array([-1.55,-1.20,-.74,-.38,0,.38,.74,1.20,1.55])
INNER = slice(2,7)
OUTER = [0,1,7,8]
STEP = np.arange(18.,148.,8.)


def uv(points,k):
    q=points@k.T
    return q[...,:2]/np.maximum(q[...,2:3],.035)


def visible_score(frame, joints, key, elbows):
    """Score arm-like image bands extending behind a detected wrist."""
    h,w=frame.shape[:2]
    anchor=uv(joints[0],key)
    if not (8<anchor[0]<w-8 and 8<anchor[1]<h-8):
        return np.zeros(len(ANGLES)),0.0
    elbow2=uv(elbows,key)
    aim=elbow2-anchor
    distance=np.linalg.norm(aim,axis=1)
    axis=aim/np.maximum(distance[:,None],1e-6)
    normal=np.stack((-axis[:,1],axis[:,0]),axis=1)
    palm_coords=uv(joints[[0,1,4,7,10,13]],key)
    support=palm_coords[1:].mean(axis=0)-anchor
    palm_dir=support/max(np.linalg.norm(support),1e-6)
    # Estimate appearance from the centre of the tracked wrist and proximal palm.
    seed=anchor[None]+np.array([-.08,.0,.10,.22,.34,.46])[:,None]*support[None]
    offsets=np.array([(a,b) for a in (-2,0,2) for b in (-2,0,2)])
    seed=np.rint(seed[:,None]+offsets[None]).astype(int).reshape(-1,2)
    seed[:,0]=np.clip(seed[:,0],0,w-1)
    seed[:,1]=np.clip(seed[:,1],0,h-1)
    lab=cv2.cvtColor(frame,cv2.COLOR_BGR2LAB).astype(np.float32)
    med=np.median(lab[seed[:,1],seed[:,0]],axis=0)
    # Chromaticity gets more weight than lightness, to allow local shading.
    delta=(lab-med)/np.array([35.,9.,12.],dtype=np.float32)
    likelihood=np.exp(-0.5*np.sum(delta*delta,axis=-1)).astype(np.float32)
    radius=np.clip(key[0,0]*.019/joints[0,2],8.,26.)
    locations=anchor+axis[:,None,None,:]*STEP[None,:,None,None] + normal[:,None,None,:]*TUBE[None,None,:,None]*radius
    x=locations[...,0]
    y=locations[...,1]
    poly_mask=((x>=0)&(x<w-1)&(y>=0)&(y<h-1))
    flat=(len(ANGLES),-1)
    scores=cv2.remap(likelihood,x.reshape(flat).astype(np.float32),y.reshape(flat).astype(np.float32),
                     cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).reshape(x.shape)
    core=scores[:,:,INNER].mean(-1)
    borders=scores[:,:,OUTER].mean(-1)
    center=scores[:,:,4]
    support=poly_mask[:,:,4] & (distance[:,None]>STEP[None,:]+5)
    spatial=np.clip((1.27-STEP/230),0.2,1.)
    band=.56*center + .45*core + .48*np.maximum(core-borders,-.30)
    # Reject early-only matches to a small palm or an occluding object's edge.
    weights=support*spatial
    length=np.sum(weights,axis=-1)
    response=np.sum(weights*band,axis=-1)/np.maximum(length,1e-6)
    response*=np.clip((support.sum(-1)-1)/6,0.0,1.)
    response-=0.32*np.maximum(axis@palm_dir-.12,0.)*min(np.linalg.norm(palm_coords[1:].mean(0)-anchor)/37,1.)
    response[distance<30]=-0.25
    top=int(np.argmax(response))
    separated=(axis@axis[top])<np.cos(np.radians(22))
    runner=max(float(response[separated].max()) if np.any(separated) else float(response[top]),0.)
    margin=float(response[top])-runner
    reliability=float(np.clip((response[top]-.30)/.4,0,1)*np.clip((margin+.055)/.18,0,1))
    if not np.isfinite(reliability):
        return np.zeros(len(ANGLES)),0.
    return response,reliability


def image_strip(frame, points, key):
    """Wrist-anchored skin and parallel contour scores, including sleeves."""
    h,w=frame.shape[:2]
    wrist=points[0]
    root=uv(wrist,key)
    if not (5<root[0]<w-5 and 5<root[1]<h-5 and wrist[2]>.045):
        return np.zeros(len(ANGLES)),0.
    end=root+230.*UV_AXIS
    z=wrist[2]
    arbitrary=np.column_stack([(end[:,0]-key[0,2])*z/key[0,0],
                               (end[:,1]-key[1,2])*z/key[1,1],np.full(len(end),z)])
    skin,skin_conf=visible_score(frame,points,key,arbitrary)
    lab=cv2.GaussianBlur(cv2.cvtColor(frame,cv2.COLOR_BGR2LAB).astype(np.float32),(5,5),0.)
    arm_radius=np.clip(key[0,0]*.028/z,8.,26.)
    steps=np.arange(15.,150.,7.)
    normal=np.stack((-UV_AXIS[:,1],UV_AXIS[:,0]),axis=1)
    center=root+steps[None,:,None]*UV_AXIS[:,None,:]
    contour=[]
    for width in (.72,1.,1.35):
        scalars=arm_radius*width*np.array([-1.38,-1.06,-.72,-.36,0.,.36,.72,1.06,1.38])
        xy=(center[:,:,None]+normal[:,None,None,:]*scalars[None,None,:,None]).astype(np.float32)
        samples=cv2.remap(lab,xy[...,0].reshape(len(ANGLES),-1),xy[...,1].reshape(len(ANGLES),-1),
                          cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).reshape(*xy.shape[:-1],3)
        in_frame=((xy[...,0]>0)&(xy[...,0]<w-1)&(xy[...,1]>0)&(xy[...,1]<h-1)).all(-1)
        # A forearm/sleeve is a strip with two side edges and a coherent interior.
        factor=np.array([19.,9.,10.])
        left=np.linalg.norm((samples[:,:,0]-samples[:,:,2])/factor,axis=-1)
        right=np.linalg.norm((samples[:,:,8]-samples[:,:,6])/factor,axis=-1)
        inside=np.linalg.norm((samples[:,:,3]-samples[:,:,5])/factor,axis=-1)
        bilateral=np.minimum(np.minimum(left,right),2.3)*(1.-.38*np.clip(inside,0.,1.))
        weight=in_frame*np.clip(1.2-steps[None]/230.,.35,1.1)
        response=np.sum(bilateral*weight,axis=1)/np.maximum(weight.sum(1),1.e-8)
        response*=np.clip((in_frame.sum(1)-2)/7,0.,1.)
        contour.append(response)
    contour=np.max(contour,axis=0)
    toward_hand = uv(np.mean(points[[1,4,7,10]],axis=0),key)-root
    hand_direction = toward_hand / max(np.linalg.norm(toward_hand), 1.e-8)
    score=.70*skin+.30*contour-.23*np.maximum(UV_AXIS@hand_direction-.05,0.)
    top=int(np.argmax(score))
    outside=UV_AXIS@UV_AXIS[top]<np.cos(np.deg2rad(22.))
    margin=score[top]-np.max(score[outside])
    conf=float(np.clip((margin+.025)/.13,0.,1.)*np.clip((score[top]-.21)/.42,0.,1.))
    return score,max(.75*skin_conf,conf)


def image_cost(elbows,camera_wrist,camera,center,key,score,confidence):
    # Use a short step on the ray for direction when the elbow itself is behind the camera.
    projected=(elbows-center)@camera
    direction = projected-camera_wrist
    step = camera_wrist + .04*direction/np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1.e-8)
    bearing=uv(step,key)-uv(camera_wrist,key)
    angles=np.arctan2(bearing[:,1],bearing[:,0])
    response=np.interp(angles,ANGLES,score,period=2.*np.pi)
    strength=np.clip(np.linalg.norm(bearing,axis=1)/20.,0.,1.)
    return -(response-np.median(score))*(.12+1.08*confidence)*strength
