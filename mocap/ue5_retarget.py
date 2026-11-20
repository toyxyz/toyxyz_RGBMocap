"""Retarget prepared offline poses to the actual Unreal mannequin reference rigs."""
from dataclasses import dataclass
import re

import numpy as np
from scipy.spatial.transform import Rotation


SIDES = (('Left', 'l'), ('Right', 'r'))
FINGERS = ('Thumb', 'Index', 'Middle', 'Ring', 'Pinky')
TARGET_SPINE = tuple(f'spine_{index:02}' for index in range(1, 6))
TARGET_NECK = ('neck_01', 'neck_02', 'head')
LEG = ('thigh', 'calf', 'foot', 'ball')
ARM = ('clavicle', 'upperarm', 'lowerarm', 'hand')
# Native capture is in right-handed Y-up metres. Mirror into Unreal's LH, Z-up
# centimetres, then match the rigs' horizontal hip directions.
SOURCE_TO_UNREAL = np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])


@dataclass(frozen=True)
class Rig:
    names: tuple
    parents: np.ndarray
    positions: np.ndarray
    rotations: np.ndarray

    @property
    def indices(self):
        return {name:i for i,name in enumerate(self.names)}


def target_rig(rows):
    names = tuple(row['name'] for row in rows)
    parents = np.asarray([row['parent'] for row in rows], dtype=int)
    positions = np.asarray([row['position'] for row in rows], dtype=float)
    quats = np.asarray([row['rotation'] for row in rows], dtype=float)
    if (not 64 <= len(names) <= 256 or names[0] != 'root' or len(set(names)) != len(names)
            or any(not re.fullmatch(r'[a-z][a-z0-9_]*', name) for name in names)
            or parents[0] != -1 or any(not 0 <= p < j for j,p in enumerate(parents[1:],1))
            or positions.shape != (len(names),3) or quats.shape != (len(names),4)
            or not np.isfinite(positions).all() or not np.isfinite(quats).all()
            or not np.allclose(np.linalg.norm(quats,axis=-1),1.,atol=1e-5)):
        raise ValueError('Invalid UE5 mannequin reference hierarchy or transforms.')
    required = ['root', 'pelvis', *TARGET_SPINE, *TARGET_NECK]
    required += [bone+'_'+side for _,side in SIDES for bone in (*LEG,*ARM)]
    required += [f'{finger.lower()}_{segment:02}_{side}' for _,side in SIDES
                 for finger in FINGERS for segment in range(1,4)]
    if not set(required).issubset(names) or parents[names.index('pelvis')] != 0:
        raise ValueError('The Unreal reference must be a stock Manny or Quinn mannequin.')
    return Rig(names, parents, positions, Rotation.from_quat(quats).as_matrix())


def _world(rig):
    positions = rig.positions.copy()
    rotations = rig.rotations.copy()
    for i,p in enumerate(rig.parents):
        if p >= 0:
            positions[i] = positions[p] + rotations[p] @ positions[i]
            rotations[i] = rotations[p] @ rotations[i]
    return positions, rotations


def _transpose(rotation):
    return np.swapaxes(rotation,-1,-2)


def _apply(rotation, vectors):
    return np.einsum('...ij,...j->...i',rotation,vectors)


def _unit(vector, default=(0.,0.,-1.)):
    norm = np.linalg.norm(vector,axis=-1,keepdims=True)
    return np.where(norm>1e-8,vector/np.maximum(norm,1e-8),default)


def _aim(before,after):
    """Shortest swing, with a deterministic axis at 180 degrees."""
    a,b = _unit(before),_unit(after)
    cross = np.cross(a,b)
    cosine = np.clip(np.sum(a*b,axis=-1),-1.,1.)
    sine = np.linalg.norm(cross,axis=-1)
    fallback = _unit(np.cross(a,np.eye(3)[np.argmin(abs(a),axis=-1)]))
    axis = np.where((sine>1e-8)[...,None],_unit(cross),fallback)
    return Rotation.from_rotvec(axis*np.arctan2(sine,cosine)[...,None]).as_matrix()


def _distance(positions,chain):
    return np.r_[0.,np.cumsum(np.linalg.norm(np.diff(positions[chain],axis=0),axis=-1))]


def _chain(deltas, source_positions, target_positions, source, target):
    """Distribute chest/neck orientation by normalized distance along each rest chain."""
    src = _distance(source_positions,source)
    dst = _distance(target_positions,target)
    if min(src[-1],dst[-1]) <= 1e-8:
        raise ValueError('Invalid torso or neck length for the UE5 mannequin retarget.')
    result = {}
    for j,distance in zip(target[1:],dst[1:]):
        at = distance/dst[-1]*src[-1]
        segment = max(1,min(len(source)-1,np.searchsorted(src,at)))
        a,b = deltas[:,source[segment-1]],deltas[:,source[segment]]
        t = np.clip((at-src[segment-1])/max(src[segment]-src[segment-1],1e-8),0.,1.)
        steps = Rotation.from_matrix(_transpose(a)@b).as_rotvec()
        result[j] = a @ Rotation.from_rotvec(steps*t).as_matrix()
    return result


def _direction_frame(forward, normal):
    forward = _unit(forward)
    other = np.eye(3)[np.argmin(abs(forward), axis=-1)]
    across = _unit(np.cross(normal, forward), _unit(np.cross(other, forward)))
    return np.stack((forward, across, np.cross(forward, across)), axis=-1)


def _palm(points, wrist, bases):
    index, middle, pinky = (points[..., joint, :] for joint in bases)
    forward = middle - points[..., wrist, :]
    return _direction_frame(forward, np.cross(forward, index - pinky))


def _align_elbow(delta, rest_forward, rest_plane, forward, plane, confidence):
    # Aim at the observed endpoint first. Resolve roll from the bend plane
    # when there is enough flexion to identify it.
    axis = _unit(forward)
    delta = _aim(_apply(delta, rest_forward), axis) @ delta
    normal = _apply(delta, rest_plane)
    turn = np.arctan2(np.sum(np.cross(normal, plane) * axis, axis=-1),
                      np.sum(normal * plane, axis=-1))
    return Rotation.from_rotvec(axis * (turn * confidence)[:, None]).as_matrix() @ delta


def _arm_targets(source, target, source_rest, points, reference_p, reference_r,
                 deltas, rest_tips, prefix, side):
    source_arm = tuple(source[prefix + part] for part in ('Shoulder', 'Arm', 'ForeArm', 'Hand'))
    arm = tuple(target[part + '_' + side] for part in ARM)
    clavicle, upper, forearm, hand = arm
    rest = np.diff(reference_p[list(arm)], axis=0)
    tracked = np.diff(points[:, source_arm], axis=1)
    desired = {}
    collar = deltas[:, source_arm[0]]
    desired[clavicle] = _aim(_apply(collar, rest[0]), tracked[:, 0]) @ collar

    bend = np.cross(_unit(tracked[:, 1]), _unit(tracked[:, 2]))
    strength = np.linalg.norm(bend, axis=-1)
    confidence = np.clip(strength / np.sin(np.radians(25.)), 0., 1.)
    confidence = confidence**2 * (3. - 2. * confidence)
    rest_plane = _unit(np.cross(rest[1], rest[2]))
    for part, joint in ((1, upper), (2, forearm)):
        desired[joint] = _align_elbow(deltas[:, source_arm[part]], rest[part], rest_plane,
                                      tracked[:, part], _unit(bend), confidence)

    anchors = ('Index', 'Middle', 'Pinky')
    source_bases = tuple(source.get(f'{prefix}Hand{digit}1') for digit in anchors)
    target_bases = tuple(target[f'{digit.lower()}_01_{side}'] for digit in anchors)
    if None in source_bases:
        desired[hand] = deltas[:, source_arm[-1]]
        return desired
    source_palm = _palm(source_rest, source_arm[-1], source_bases)
    target_palm = _palm(reference_p, hand, target_bases)
    live_palm = _palm(points, source_arm[-1], source_bases)
    desired[hand] = live_palm @ target_palm.T

    # Phalange axes are independent of the wrist roll and mirror differently
    # on each side. Calibrate every physical segment against its palm plane.
    for digit in FINGERS:
        source_digit = tuple(source.get(f'{prefix}Hand{digit}{i}') for i in range(1, 4))
        if None in source_digit:
            continue
        target_digit = tuple(target[f'{digit.lower()}_{i:02}_{side}'] for i in range(1, 4))
        source_directions = np.diff(source_rest[list(source_digit)], axis=0)
        target_directions = np.diff(reference_p[list(target_digit)], axis=0)
        tip = source_directions[-1]
        if rest_tips is not None and np.linalg.norm(rest_tips[source_digit[-1]]) > 1e-5:
            tip = rest_tips[source_digit[-1]]
        target_tip = reference_r[target_digit[-1], :, 0].copy()
        if target_tip @ target_directions[-1] < 0.:
            target_tip *= -1.
        for j, (src, dst) in enumerate(zip(source_digit, target_digit)):
            source_forward = source_directions[j] if j < 2 else tip
            target_forward = target_directions[j] if j < 2 else target_tip
            source_frame = _direction_frame(source_forward, source_palm[:, 2])
            target_frame = _direction_frame(target_forward, target_palm[:, 2])
            desired[dst] = deltas[:, src] @ source_frame @ target_frame.T

    # Distribute wrist pronation over the stock forearm twist bones.
    axis = _unit(tracked[:, 2])
    residual = desired[hand] @ _transpose(desired[forearm])
    quaternion = Rotation.from_matrix(residual).as_quat()
    twist = 2. * np.arctan2(np.sum(quaternion[:, :3] * axis, axis=-1), quaternion[:, 3])
    twist = np.unwrap((twist + np.pi) % (2. * np.pi) - np.pi)
    for segment in (1, 2):
        helper = target.get(f'lowerarm_twist_{segment:02}_{side}')
        if helper is not None:
            offset = reference_p[helper] - reference_p[forearm]
            weight = np.clip((offset @ rest[2]) / (rest[2] @ rest[2]), 0., 1.)
            desired[helper] = Rotation.from_rotvec(axis * (twist * weight)[:, None]).as_matrix() @ desired[forearm]
    return desired


def retarget(source_names, rest_positions, rest_rotations, positions, rotations, target, *,
             in_place=False, source_tip_local=None):
    """Return parent-local tracks in Unreal centimetres, preserving target bone lengths."""
    bone_count = len(source_names)
    frame_count = len(positions)
    if (frame_count < 2 or positions.shape != (frame_count, bone_count, 3)
            or rotations.shape != (frame_count, bone_count, 3, 3)
            or rest_positions.shape != (bone_count, 3)
            or rest_rotations.shape != (bone_count, 3, 3)):
        raise ValueError('UE5 mannequin export requires a complete source rig and at least two motion frames.')
    if any(not np.isfinite(values).all() for values in (positions, rotations, rest_positions, rest_rotations)):
        raise ValueError('The source rig or motion contains non-finite transforms.')
    sk = {name:i for i,name in enumerate(source_names)}
    tk = target.indices
    native = 'LeftShin' in sk
    source_leg = ('Leg','Shin','Foot','ToeBase') if native else ('UpLeg','Leg','Foot','ToeBase')
    source_spine = ('Hips','Spine1','Spine2','Chest') if native else ('Hips','Spine','Spine1','Spine2')
    source_neck = (source_spine[-1],'Neck1','Neck2','Head') if native else (source_spine[-1],'Neck','Head')
    source_arm = ('Shoulder','Arm','ForeArm','Hand')
    try:
        legs = [(tuple(sk[prefix+n] for n in source_leg),tuple(tk[n+'_'+side] for n in LEG))
                for prefix,side in SIDES]
        for prefix, _ in SIDES:
            for name in source_arm:
                sk[prefix + name]
        source_spine = tuple(sk[name] for name in source_spine)
        source_neck = tuple(sk[name] for name in source_neck)
    except KeyError as error:
        raise ValueError('The prepared source is missing a required UE5 retarget bone: '+str(error)) from error

    ref_p,ref_r = _world(target)
    src_rest = _apply(SOURCE_TO_UNREAL,rest_positions)*100.
    left,right = legs
    src_right = src_rest[right[0][0]]-src_rest[left[0][0]]
    dst_right = ref_p[right[1][0]]-ref_p[left[1][0]]
    angle = np.arctan2(dst_right[1],dst_right[0])-np.arctan2(src_right[1],src_right[0])
    align = Rotation.from_rotvec([0.,0.,angle]).as_matrix() @ SOURCE_TO_UNREAL
    src_rest = _apply(align,rest_positions)*100.
    src_p = _apply(align,positions)*100.
    deltas = align @ rotations @ _transpose(rest_rotations) @ align.T
    rest_tips = None
    if source_tip_local is not None:
        if source_tip_local.shape != (bone_count, 3) or not np.isfinite(source_tip_local).all():
            raise ValueError('Invalid source bone tip offsets.')
        rest_tips = _apply(align @ rest_rotations, source_tip_local) * 100.

    leg_lengths = [np.diff(_distance(ref_p,list(dst[:3]))) for _,dst in legs]
    source_span = np.mean([_distance(src_rest,list(src[:3]))[-1] for src,_ in legs])
    if source_span <= 1e-8 or min(map(np.min,leg_lengths)) < 1.:
        raise ValueError('Invalid leg lengths for the UE5 mannequin retarget.')
    ratio = np.mean([np.sum(lengths) for lengths in leg_lengths])/source_span

    desired = {tk['pelvis']:deltas[:,sk['Hips']]}
    desired.update(_chain(deltas,src_rest,ref_p,list(source_spine),
                          [tk['pelvis'],*(tk[name] for name in TARGET_SPINE)]))
    desired.update(_chain(deltas,src_rest,ref_p,list(source_neck),
                          [tk['spine_05'],*(tk[name] for name in TARGET_NECK)]))
    for src,dst in legs:
        for j,(i,k) in enumerate(zip(src,dst)):
            delta = deltas[:,i]
            if j < 2:
                before = _apply(delta,ref_p[dst[j+1]]-ref_p[k])
                after = src_p[:,src[j+1]]-src_p[:,i]
                delta = _aim(before,after) @ delta
            desired[k] = delta

    for prefix, side in SIDES:
        desired.update(_arm_targets(sk, tk, src_rest, src_p, ref_p, ref_r,
                                    deltas, rest_tips, prefix, side))

    planar = ratio*(src_p[:,sk['Hips'],:2]-src_p[0,sk['Hips'],:2])
    center = ratio*src_p[0,sk['Hips'],:2]
    local_p = np.broadcast_to(target.positions,(len(src_p),*target.positions.shape)).copy()
    local_r = np.broadcast_to(target.rotations,(len(src_p),*target.rotations.shape)).copy()
    wp,wr = local_p.copy(),local_r.copy()
    displacement = ratio*(src_p[:,sk['Hips']]-src_rest[sk['Hips']])
    displacement[:,:2] -= center
    if in_place:
        displacement[:,:2] -= planar
    else:
        local_p[:,0,:2] += planar
    pelvis = tk['pelvis']

    def compose():
        for j,parent in enumerate(target.parents):
            if parent < 0:
                wp[:,j],wr[:,j] = local_p[:,j],local_r[:,j]
            else:
                wp[:,j] = wp[:,parent]+_apply(wr[:,parent],local_p[:,j])
                wr[:,j] = wr[:,parent]@local_r[:,j]

    for j,parent in enumerate(target.parents):
        if parent < 0:
            wp[:,j],wr[:,j] = local_p[:,j],local_r[:,j]
            continue
        if j in desired:
            local_r[:,j] = _transpose(wr[:,parent])@desired[j]@ref_r[j]
        if j == pelvis:
            local_p[:,j] = _apply(_transpose(wr[:,parent]),ref_p[j]+displacement-wp[:,parent])
        wp[:,j] = wp[:,parent]+_apply(wr[:,parent],local_p[:,j])
        wr[:,j] = wr[:,parent]@local_r[:,j]

    goals = []
    lower = np.full(len(src_p),-15.)
    upper = np.full(len(src_p),15.)
    foot_rotations = []
    # Use one horizontal anchor for both feet. Separate rest offsets would
    # change the captured stance width by a constant on every frame.
    horizontal_offset = np.mean(
        [ref_p[dst[2], :2] - ratio * src_rest[src[2], :2] for src, dst in legs], axis=0)
    for (src,dst),lengths in zip(legs,leg_lengths):
        source_ankle,target_ankle = src[2],dst[2]
        goal = ref_p[target_ankle]+ratio*(src_p[:,source_ankle]-src_rest[source_ankle])
        goal[:, :2] = ratio * src_p[:, source_ankle, :2] + horizontal_offset - center
        if in_place:
            goal[:,:2] -= planar
        goals.append(goal)
        foot_rotations.append(wr[:,target_ankle].copy())
        gap = goal-wp[:,dst[0]]
        max_reach = lengths.sum()-.02
        span = max_reach**2-np.sum(gap[:,:2]**2,axis=-1)
        height = np.sqrt(np.maximum(span,0.))
        lower = np.maximum(lower,np.where(span>=0.,gap[:,2]-height,-np.inf))
        upper = np.minimum(upper,np.where(span>=0.,gap[:,2]+height,np.inf))
    adjustment = np.where(lower<=upper,np.clip(0.,lower,upper),np.clip((lower+upper)/2.,-15.,15.))
    local_p[:,pelvis] += _apply(_transpose(wr[:,0]),np.column_stack((np.zeros((len(src_p),2)),adjustment)))
    compose()

    # Fixed-length two-bone reach with a pole derived from the source knee.
    for (src,dst),lengths,goal,foot_rotation in zip(legs,leg_lengths,goals,foot_rotations):
        h,k,a,_ = dst
        hip = wp[:,h].copy()
        gap = goal-hip
        axis = _unit(gap)
        l1,l2 = lengths
        dist = np.clip(np.linalg.norm(gap,axis=-1),abs(l1-l2)+.02,l1+l2-.02)
        source_axis = _unit(src_p[:, src[2]] - src_p[:, src[0]])
        source_pole = src_p[:, src[1]] - src_p[:, src[0]]
        source_pole -= np.sum(source_pole * source_axis, axis=-1)[:, None] * source_axis
        pole = _apply(_aim(source_axis, axis), source_pole)
        toe_forward = ref_p[dst[3]] - ref_p[a]
        toe_forward[2] = 0.
        fallback = _apply(desired[pelvis], _unit(toe_forward))
        fallback -= np.sum(fallback * axis, axis=-1)[:, None] * axis
        # A nearly straight source leg has no reliable knee plane.
        confidence = np.clip(np.linalg.norm(pole, axis=-1) / 3., 0., 1.)[:, None]
        bend = _unit(confidence * _unit(pole) + (1. - confidence) * _unit(fallback))
        bend = _unit(bend - np.sum(bend * axis, axis=-1)[:, None] * axis,
                     _unit(np.cross(axis, np.eye(3)[np.argmin(abs(axis), axis=-1)])))
        along = (l1*l1-l2*l2+dist*dist)/(2.*dist)
        knee = hip+axis*along[:,None]+bend*np.sqrt(np.maximum(0.,l1*l1-along*along))[:,None]
        ankle = hip+axis*dist[:,None]
        upper_world = _aim(wp[:,k]-hip,knee-hip)@wr[:,h]
        lower_world = _aim(wp[:,a]-wp[:,k],ankle-knee)@wr[:,k]
        local_r[:,h] = _transpose(wr[:,target.parents[h]])@upper_world
        local_r[:,k] = _transpose(upper_world)@lower_world
        local_r[:,a] = _transpose(lower_world)@foot_rotation
    compose()

    # Match IK helpers to the baked extremities, including the stock rest offsets.
    guides = {'ik_foot_l':'foot_l','ik_foot_r':'foot_r','ik_hand_gun':'hand_r',
              'ik_hand_l':'hand_l','ik_hand_r':'hand_r'}
    for helper,bone in guides.items():
        if helper not in tk:
            continue
        j,k = tk[helper],tk[bone]
        p = target.parents[j]
        offset = ref_r[k].T@(ref_p[j]-ref_p[k])
        wp[:,j] = wp[:,k]+_apply(wr[:,k],offset)
        wr[:,j] = wr[:,k]@ref_r[k].T@ref_r[j]
        local_p[:,j] = _apply(_transpose(wr[:,p]),wp[:,j]-wp[:,p])
        local_r[:,j] = _transpose(wr[:,p])@wr[:,j]

    return local_p,local_r
