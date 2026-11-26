"""Camera-rooted, fixed-length arm IK for a ViDiHand2 take."""
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

from .arm_image import image_cost
from .motion import ARM_LENGTHS

WIDTH = .177
SHOULDERS = np.array([[-WIDTH,0.,0.],[WIDTH,0.,0.]])
FLEX_MIN, FLEX_MAX = np.deg2rad([3., 145.])
ROLL_MAX = np.deg2rad(110.)
N = 120
PHASE = np.linspace(-np.pi,np.pi,N,endpoint=False)
COS_SIN = np.stack([np.cos(PHASE),np.sin(PHASE)],axis=1)
UP = np.array([0.,0.,1.])
TINY = 1.e-8


def project_plane(vector, normal):
    return vector - np.sum(vector*normal,axis=-1,keepdims=True)*normal


def normalize(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),TINY)


def camera_body(camera, center, up, fps):
    # Use the calibrated camera right vector after applying the stored image orientation.
    # Re-project onto the measured gravity plane so head pitch/roll stays out of the torso.
    raw_right=project_plane(camera[:,:,0],up)
    raw_forward=project_plane(camera[:,:,2],up)
    side_fallback=normalize(np.cross(raw_forward,up))
    room=np.linalg.norm(raw_right,axis=1)
    weight=np.clip((room-.20)/.30,0.,1.)[:,None]
    side=normalize(weight*normalize(raw_right)+(1.-weight)*side_fallback)
    if np.any(np.linalg.norm(side,axis=1)<.9):
        good=np.linalg.norm(side,axis=1)>.9
        if not good.any():
            raise ValueError('Camera heading is undefined')
        ix=np.arange(len(side))
        side[~good]=side[good][np.argmin(abs(ix[~good,None]-ix[good][None]),axis=1)]
    side=normalize(gaussian_filter1d(side,4.2*fps/30.,axis=0,mode='nearest'))
    forward=np.cross(up,side)
    flat_center=gaussian_filter1d(center,2.0*fps/30.,axis=0,mode='nearest')
    # Looking down often puts the camera ahead of the chest. Keep the
    # correction bilateral and smooth rather than chasing individual wrists.
    pitch=np.clip(-camera[:,:,2]@up,0.,1.)
    offset=.088+.085*gaussian_filter1d(pitch,4.2*fps/30.,mode='nearest')
    shoulder_center=flat_center-.225*up-offset[:,None]*forward
    orientation=np.stack([side,forward,np.broadcast_to(up,side.shape)],axis=-1)
    return shoulder_center,orientation


def neutral_bend(upper):
    # Transport anterior (+Y) from a hanging humerus (-Z) by its shortest rotation.
    hang=np.array([0.,0.,-1.])
    anterior=np.array([0.,1.,0.])
    cross=np.cross(hang,upper)
    dot=upper@hang
    moved=anterior+np.cross(cross,anterior)+np.cross(cross,np.cross(cross,anterior))/np.maximum(1.+dot[:,None],1.e-7)
    overhead=dot<-.979
    moved[overhead]=project_plane(anterior,upper[overhead])
    return normalize(project_plane(moved,upper))


def segment_core_distance(a,b):
    """Exact closest distance to the torso proxy ellipsoid, expressed in body coordinates."""
    center=np.array([0.,-.016,-.22])
    axes=np.array([.139,.107,.225])
    start=(a-center)/axes
    delta=(b-a)/axes
    amount=np.clip(-np.sum(start*delta,axis=-1)/np.maximum(np.sum(delta*delta,axis=-1),TINY),0.,1.)
    return np.linalg.norm(start+amount[...,None]*delta,axis=-1)


def anatomical(shoulder, elbows, wrist, side, palm):
    # All vectors are evaluated in the same body frame. Palm is a soft cue
    # because wrist orientation can be underconstrained by occluded fingers.
    upper=normalize(elbows-shoulder)
    fore=normalize(wrist-elbows)
    bend=normalize(project_plane(fore,upper))
    neutral=neutral_bend(upper)
    roll=np.arctan2(np.sum(np.cross(neutral,bend)*upper,axis=1),np.sum(neutral*bend,axis=1))
    flex=np.arccos(np.clip(np.sum(upper*fore,axis=1),-1.,1.))
    shoulder_safe=(upper[:,1]>=-np.sin(np.deg2rad(50.))) & (side*upper[:,0]>=-np.sin(np.deg2rad(52.)))
    hinge_safe=(abs(roll)<=ROLL_MAX+1.e-8) & (flex>=FLEX_MIN-1.e-8)&(flex<=FLEX_MAX+1.e-8)
    core_a=segment_core_distance(shoulder+.060*upper,elbows)
    core_b=segment_core_distance(elbows,wrist)
    core=np.minimum(core_a,core_b)
    valid=shoulder_safe & hinge_safe & (core>=1.)
    palm_angle=np.arccos(np.clip(fore@palm,-1.,1.))
    return valid,roll,flex,core,palm_angle


def pick_path(elbows,swivels,allowed,cost,wrist,valid):
    path=np.full(len(valid),-1,int)
    ids=np.flatnonzero(valid)
    for run in np.split(ids,np.flatnonzero(np.diff(ids)!=1)+1):
        if not len(run):
            continue
        local=np.where(allowed[run],cost[run],np.inf)
        loss=local[0]
        trace=[]
        for p,f,local_cost in zip(run,run[1:],local[1:]):
            residual=elbows[f,None]-elbows[p,:,None]
            wrist_step=np.linalg.norm(wrist[f]-wrist[p])
            d=np.linalg.norm(residual,axis=-1)
            transition=36.*d**2+320.*np.maximum(0.,d-.027-.35*wrist_step)**2
            transition+=.085*np.maximum(0.,1.-swivels[p]@swivels[f].T)
            v=loss[:,None]+transition
            back=np.argmin(v,axis=0)
            trace.append(back)
            loss=v[back,np.arange(N)]+local_cost
        choice=int(np.argmin(loss))
        path[run[-1]]=choice
        for f,back in zip(run[-2::-1],trace[::-1]):
            choice=int(back[choice])
            path[f]=choice
    return path


def world_to_camera(points, rotation, center):
    trailing = (1,) * (points.ndim - 2)
    return np.einsum("tji,t...j->t...i", rotation, points - center.reshape((-1, *trailing, 3)))


def reach_bounds(lengths):
    upper, lower = lengths
    return np.sqrt(upper**2+lower**2+2*upper*lower*np.cos([FLEX_MAX,FLEX_MIN]))


def elbow_candidates(shoulder, wrist, side, palm, lengths):
    upper, lower = lengths
    delta = wrist - shoulder
    distance = np.linalg.norm(delta)
    low, high = reach_bounds(lengths)
    if not low <= distance <= high:
        return None
    axis = delta / distance
    for guide in ([side*.68, -.11, -1.], [side*.92, 0., -.10], [0., 1., 0.]):
        pole = project_plane(np.array(guide), axis)
        if np.linalg.norm(pole) >= .15:
            break
    first = normalize(pole)
    second = np.cross(axis, first)
    swivels = COS_SIN @ np.stack([first, second])
    along = (upper**2 + distance**2 - lower**2) / (2. * distance)
    radius = np.sqrt(max(0., upper**2 - along**2))
    elbows = shoulder + along * axis + radius * swivels
    allowed, roll, flex, core, palm_angle = anatomical(shoulder, elbows, wrist, side, palm)
    cost = .07*(1.-np.cos(PHASE)) + .045*(roll/ROLL_MAX)**2
    cost += .18*np.maximum(0., (palm_angle-np.deg2rad(38.))/np.deg2rad(65.))**2
    cost += .09*np.maximum(0., normalize(elbows-shoulder)[:, 2]+.10)**2
    cost += .10*np.maximum(0., 1.22-core)**2
    if not allowed.any():
        return None
    return elbows, swivels, allowed, cost


def placement_score(data, body, center, evidence, lengths, frames):
    """Compare shoulder placements using the same viable bends as the full solve."""
    local = world_to_camera(data["hand_world"][frames],body[frames],center[frames])
    palm = normalize(local[:,:,[1,4,7,10]].mean(axis=2)-local[:,:,0])
    camera, root, intrinsics = (data[key] for key in ("camera_to_world","camera_position","K"))
    tracked = data["tracked"][frames]
    feasible = 0
    total = 0.
    for index,side in zip(*np.nonzero(tracked)):
        candidate = elbow_candidates(SHOULDERS[side],local[index,side,0],
                                     side*2-1,palm[index,side],lengths)
        if candidate is None:
            continue
        feasible += 1
        elbows,_,allowed,rest = candidate
        frame = frames[index]
        elbows = elbows[allowed] @ body[frame].T + center[frame]
        image = image_cost(elbows,data["hand_camera"][frame,side,0],
                           camera[frame],root[frame],intrinsics[frame],*evidence[frame,side])
        total += .10*np.clip(-np.min(image+rest[allowed]),0.,1.)
    count = max(int(tracked.sum()),1)
    return feasible/count,total/count


def wrist_distances(data, body, center):
    shoulders = np.einsum("tij,sj->tsi",body,SHOULDERS) + center[:,None]
    return np.linalg.norm(data["hand_world"][:,:,0]-shoulders,axis=-1)


def estimate_lengths(distances, tracked):
    reaches = []
    for distance,seen in zip(distances.T,tracked.T):
        frames = np.flatnonzero(seen)
        if len(frames)<8:
            continue
        reaches.append(np.percentile(distance[frames],94 if len(frames)>=20 else 75))
        # Include sustained reaches anywhere in the take, even when too rare
        # to affect the percentile. Require nearby frames to support a peak.
        for run in np.split(frames,np.flatnonzero(np.diff(frames)!=1)+1):
            if len(run)<5:
                continue
            samples = distance[run]
            trend = median_filter(samples,size=5,mode="mirror")
            reaches.append(np.max(samples[samples<=trend+.035],initial=0.))
    reach = max(reaches) if reaches else np.median(distances[tracked])
    reference = sum(ARM_LENGTHS)
    total = np.clip(1.025*reach,.80*reference,.88)
    # Wrist positions constrain total reach. Keep the existing segment ratio.
    return tuple(round(total*part/reference,3) for part in ARM_LENGTHS)


def fit_arms(data, body, center, evidence, lengths):
    world = data["hand_world"]
    n = len(world)
    tracked = data["tracked"]
    wrist = world[:, :, 0]
    hand_local = world_to_camera(world, body, center)
    palm = normalize(hand_local[:, :, [1, 4, 7, 10]].mean(2) - hand_local[:, :, 0])
    shoulders = np.einsum("tij,sj->tsi", body, SHOULDERS) + center[:, None]
    arms = np.full((n, 2, 3, 3), np.nan)
    for side in range(2):
        samples = (n, len(PHASE))
        elbows = np.zeros((*samples, 3))
        swivels = np.zeros_like(elbows)
        allowed = np.zeros(samples, dtype=bool)
        cost = np.zeros(samples)
        for frame in np.flatnonzero(tracked[:, side]):
            cand = elbow_candidates(SHOULDERS[side], hand_local[frame, side, 0],
                                    side*2-1, palm[frame, side], lengths)
            if cand is None:
                continue
            local, sw, allowed[frame], cost[frame] = cand
            elbows[frame] = local @ body[frame].T + center[frame]
            swivels[frame] = sw @ body[frame].T
            image_score, confidence = evidence[frame, side]
            cost[frame] += image_cost(elbows[frame], data["hand_camera"][frame, side, 0],
                                           data["camera_to_world"][frame], data["camera_position"][frame],
                                           data["K"][frame], image_score, confidence)
        valid = allowed.any(axis=1)
        path = pick_path(elbows, swivels, allowed, cost, wrist[:, side], valid)
        chosen = np.flatnonzero(path >= 0)
        arms[chosen, side, 0] = shoulders[chosen, side]
        arms[chosen, side, 1] = elbows[chosen, path[chosen]]
        arms[chosen, side, 2] = wrist[chosen, side]
        for run in np.split(chosen, np.flatnonzero(np.diff(chosen) != 1)+1):
            if len(run):
                arms[run, side] = smooth_run(arms[run, side], 2.*float(data["fps"])/30.,
                                             body[run], center[run], palm[run, side], side*2-1, lengths)
    return arms.astype(np.float32)


def transport(axis, initial):
    """Carry a radial axis along the shoulder-wrist directions with minimal twist."""
    basis = np.empty_like(axis)
    basis[0] = initial
    for index in range(1, len(axis)):
        previous, current = axis[index-1:index+1]
        last = basis[index-1]
        turn = np.cross(previous, current)
        similarity = previous @ current
        if similarity > -.99999:
            last = last + np.cross(turn,last) + np.cross(turn,np.cross(turn,last))/(1.+similarity)
        else:
            # At an antipodal turn the shortest rotation is ambiguous.
            last = last - (last @ current)*current
        basis[index] = normalize(last - (last @ current)*current)
    return basis


def constrain(shoulder, elbow, wrist, body, center, palm, side):
    # Torso and shoulder limit tests use the exact original body frame.
    shoulder_local = (shoulder-center) @ body
    local = (elbow-center) @ body
    wrist_local = (wrist-center) @ body
    return anatomical(shoulder_local, local, wrist_local, side, palm)


def smooth_run(original, sigma, body, center, palm, side, lengths):
    count = len(original)
    if count < 3:
        return original
    shoulder, elbow, wrist = np.moveaxis(original, 1, 0)
    axis = normalize(wrist-shoulder)
    reach = np.linalg.norm(wrist-shoulder,axis=1)
    upper, forearm = lengths
    along = (upper**2+reach**2-forearm**2)/(2.*reach)
    radius = np.sqrt(np.maximum(upper**2-along**2,0.))
    circle_center = shoulder + axis*along[:,None]
    pole = normalize(elbow-circle_center)
    base = transport(axis, pole[0])
    perpendicular = np.cross(axis,base)
    original_angle = np.unwrap(np.arctan2(np.einsum("ij,ij->i",perpendicular,pole),
                                           np.einsum("ij,ij->i",base,pole)))
    filter_sigma = min(sigma,(count-1)/3.)
    target_angle = gaussian_filter1d(original_angle,filter_sigma,mode="nearest")
    result = original.copy()
    weights = np.array([1.,.75,.5,.25,.125,0.])
    for i in range(count):
        angles = original_angle[i] + weights*(target_angle[i]-original_angle[i])
        poles = base[i]*np.cos(angles[:,None]) + perpendicular[i]*np.sin(angles[:,None])
        candidates = circle_center[i]+radius[i]*poles
        allowed,*_ = constrain(shoulder[i], candidates, wrist[i], body[i], center[i], palm[i],side)
        choices = np.flatnonzero(allowed)
        if len(choices):
            result[i,1] = candidates[choices[0]]
    return result
