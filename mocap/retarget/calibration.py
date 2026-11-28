"""Explicit source bind removal and a virtual common leg reference.

The character's skin bind, hip pivots, segment lengths and source caches remain
untouched. A virtual reference rotates the leg segments to parallel down axes;
the pelvis width is still the actual character's width.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
CHAINS=[(1,4,7),(2,5,8)]
LEGS=[1,2,4,5]

def unit(v):return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-12)

def frame(direction,front=np.array([0.,0.,1.])):
    y=unit(np.asarray(direction,float));z=unit(front-y*np.dot(y,front));x=unit(np.cross(y,z))
    return np.stack([x,y,np.cross(x,y)],axis=-1)

def reference(rest):
    world=np.tile(np.eye(3),(22,1,1));canonical=frame([0.,-1.,0.])
    for h,k,a in CHAINS:
        world[h]=canonical@frame(rest[k]-rest[h]).T
        world[k]=canonical@frame(rest[a]-rest[k]).T
    return world

def source_neutral(asset):
    return asset['fbx_rest_frames'][:22]@asset['fbx_display_frames'][:22].swapaxes(-1,-2)

def desired(source_world,neutral,rest):
    common=reference(rest);out=source_world.copy()
    for j in LEGS:out[:,j]=source_world[:,j]@neutral[j].T@common[j]
    return out

def clavicle_rest_lift(source_rest, target_rest):
    """Align the clavicle reference for a target with lower rest arms."""
    clavicles, upper_arms, elbows = [13, 14], [16, 17], [18, 19]
    source_collar = source_rest[upper_arms] - source_rest[clavicles]
    target_collar = target_rest[upper_arms] - target_rest[clavicles]
    source_arm = source_rest[elbows] - source_rest[upper_arms]
    target_arm = target_rest[elbows] - target_rest[upper_arms]

    def elevation(direction):
        return np.arctan2(direction[:, 1], np.linalg.norm(direction[:, [0, 2]], axis=1))

    # Keep the Unreal receiver's rest-pose blend. Near-horizontal targets
    # retain their reference. A clear A-pose gets the full positive gap.
    drop = -elevation(target_arm) - np.maximum(np.radians(10), -elevation(source_arm))
    blend = np.clip(drop / np.radians(30), 0., 1.)
    weight = blend * blend * (3. - 2. * blend)
    gap = np.maximum(0., elevation(source_collar) - elevation(target_collar))
    axis = unit(np.cross(target_collar, [0., 1., 0.]))
    valid = np.logical_and.reduce([np.linalg.norm(v, axis=1) > 1e-6
                                   for v in (source_collar, target_collar, source_arm, target_arm)])
    return R.from_rotvec(axis * (weight * gap * valid)[:, None]).as_matrix()


def clavicle_rotations(source_world, neutral, target_chest, source_rest, target_rest):
    """Transfer the captured chest-relative change onto the target bind axes."""
    chest, clavicles = 9, [13, 14]
    source_local = source_world[:, chest, None].swapaxes(-1, -2) @ source_world[:, clavicles]
    bind_local = neutral[chest].T @ neutral[clavicles]
    delta = source_local @ bind_local.swapaxes(-1, -2)
    return target_chest[:, None] @ delta @ clavicle_rest_lift(source_rest, target_rest)


def reorient_pose(data,source_world,neutral,align_many,fk,local_from_world):
    """Change outgoing roll while keeping every existing joint centre fixed."""
    world=data['world'].copy();want=desired(source_world,neutral,data['rest'])
    for h,k,a in CHAINS:
        for j,ch in [(h,k),(k,a)]:
            segment=data['rest'][ch]-data['rest'][j]
            predicted=want[:,j]@segment
            actual=data['points'][:,ch]-data['points'][:,j]
            world[:,j]=align_many(predicted,actual)@want[:,j]
    local=local_from_world(world);points,world=fk(local,data['rest'],data['root'])
    assert np.max(np.abs(points-data['points']))<1e-7
    return dict(data,local=local,points=points,world=world)
