"""Track two independent ViDiHand2 hands and a camera-driven head."""
from pathlib import Path
import json

import numpy as np

from .motion import HAND_PARENTS, TAKE_NAME, Y_UP, arm_lengths, load

FINGERS = ("Index", "Middle", "Pinky", "Ring", "Thumb")
SIDES = ("Left", "Right")
PARENTS = np.asarray(HAND_PARENTS)
HAND_HEADS = np.arange(16)
HAND_TAILS = np.array([4 if i == 0 else np.flatnonzero(PARENTS == i)[0] for i in HAND_HEADS])
HEAD_LENGTH = .18
THUMB_ROOT = 13
LABELS = (
    "Wrist",
    *(f"{finger}{joint}" for finger in FINGERS for joint in (1, 2, 3)),
    "ThumbTip", "IndexTip", "MiddleTip", "RingTip", "PinkyTip",
)


def is_project(project):
    try:
        state = json.loads((Path(project)/"project.json").read_text(encoding="utf-8"))
        return state.get("pipeline", {}).get("model") == "vidihand2"
    except (OSError, ValueError, AttributeError):
        return False


def overlay(data):
    projected = np.einsum("tij,tskj->tski", data["K"], data["hand_camera"])
    pixels = projected[..., :2] / np.maximum(projected[..., 2:3], .025) / data["video_size"]
    visible = data["tracked"][:, :, None] & (projected[..., 2] >= .025)
    return np.where(visible[..., None], pixels, np.nan).astype(np.float32)


def hold(points, valid):
    visible = np.flatnonzero(valid)
    indices = np.searchsorted(visible, np.arange(len(points)), side="right") - 1
    return points[visible[np.maximum(indices, 0)]]


def sampled(project):
    data = load(project/TAKE_NAME)
    state = json.loads((project/"project.json").read_text(encoding="utf-8"))
    if len(data["tracked"]) != state["frames"] or abs(float(data["fps"]) - state["fps"]) > .1:
        raise ValueError("Hand tracks do not match the prepared clip.")
    return data


def open_preview(result, cancelled):
    from ..preview_data import LoadCancelled
    from .mano_preview import ManoPreview, read_template

    if not (result.project/TAKE_NAME).is_file():
        return
    data = sampled(result.project)
    points = data["hand_world"] @ Y_UP.T
    visible = data["tracked"]
    held = np.full_like(points, np.nan)
    bones = np.full((len(points),2,len(HAND_HEADS),4,4), np.nan, dtype=np.float32)
    wrists = []
    for side, seen in enumerate(visible.T):
        if not seen.any():
            continue
        first = int(np.argmax(seen))
        hand = tracked_hand_frames(points[:, side], seen)[first:]
        held[first:, side] = hand[..., :3, 3]
        wrists.append(hold(points[:, side, 0], seen))
        frames = hand[:, HAND_HEADS]
        lengths = np.linalg.norm(hand[..., :3, 3][:, HAND_TAILS]-frames[..., :3, 3], axis=-1)
        frames[..., :3, :3] *= lengths[..., None, None]
        bones[first:, side] = frames
    anchor = np.mean(wrists, axis=0)
    view = points[visible].reshape(-1,3)
    head = head_frames(data)
    if head is not None:
        anchor = head[:, 0, :3, 3]
        view = np.concatenate((view, head[:, 0, :3, 3], head[:, 1, :3, 3]))
        head = head[:, :1].copy()
        head[..., :3, :3] *= HEAD_LENGTH
        result.arrays["first_person_head"] = head
    arm_count = 0
    lengths = arm_lengths(data)
    arms = arm_frames(data)
    if arms is not None:
        frame, side = np.nonzero(np.isfinite(arms).all(axis=(2,3,4)))
        if len(frame):
            arms[..., :3, :3] *= np.asarray(lengths)[None,None,:,None,None]
            result.arrays["first_person_arms"] = arms
            arm_count = len(frame)
            view = np.concatenate((view,arms[frame,side,:,:3,3].reshape(-1,3)))
    low, high = view.min(axis=0), view.max(axis=0)
    if cancelled():
        raise LoadCancelled()
    template = read_template()
    if template is not None:
        try:
            result.target_assets["first_person_mano"] = ManoPreview(template)
        except ValueError:
            pass  # A broken local mesh cache must leave saved bones usable.
    result.arrays.update(first_person_hands=held, first_person_bones=bones,
                         first_person_anchor=anchor[:, None], first_person_points=overlay(data))
    result.summaries["first_person_anchor"] = dict(
        floor=float(low[1]), height=float(high[1]-low[1]), center=(low+high)/2,
        span=max(.5, float(np.max(high-low))), arm_count=arm_count, hand_count=int(visible.sum()),
        arm_lengths=lengths, arm_auto_lengths=bool(data.get("arm_ik_auto_lengths",False)))


def frame_axes(direction, guide):
    """Build a proper frame with +Y following the bone and +Z near its roll guide."""
    from ..bone_overlay import reference_frame

    valid = np.linalg.norm(direction, axis=-1) >= 1.e-6
    if not valid.any():
        return np.broadcast_to(np.eye(3), (len(direction),3,3))
    y = unit(direction)
    x = np.cross(y, guide)
    guided = np.linalg.norm(x, axis=-1) >= .05
    x = unit(x)
    axes = np.stack((x, y, np.cross(x, y)), axis=-1)
    for i in np.flatnonzero(valid & ~guided):
        axes[i] = reference_frame(y[i])
    return hold(axes, valid)


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1.e-8)


def hand_frames(points):
    from scipy.spatial.transform import Rotation

    frames = np.broadcast_to(np.eye(4), (len(points), len(PARENTS), 4, 4)).copy()
    frames[..., :3, 3] = points
    palm = unit(np.cross(points[:, 1]-points[:, 0], points[:, 7]-points[:, 0]))
    wrist_direction = points[:, HAND_TAILS[0]] - points[:, 0]
    frames[:, 0, :3, :3] = frame_axes(wrist_direction, palm)
    basis = np.eye(3)
    for joint, tail in zip(HAND_HEADS[1:], HAND_TAILS[1:]):
        parent = PARENTS[joint]
        parent_frame = frames[:, parent, :3, :3]
        direction = points[:, tail] - points[:, joint]
        local = np.einsum("tji,tj->ti", parent_frame, direction)
        root = parent == 0
        # With +Y along the phalanx, X flexes the joint and Z spreads the root.
        profile = np.hypot(local[:, 0], local[:, 1]) if root else local[:, 1]
        flexion = np.arctan2(local[:, 2], profile)
        axes = Rotation.from_rotvec(flexion[:, None] * basis[0]).as_matrix()
        if root:
            spread = np.arctan2(-local[:, 0], local[:, 1])
            axes = Rotation.from_rotvec(spread[:, None] * basis[2]).as_matrix() @ axes
        world = parent_frame @ axes
        if joint == THUMB_ROOT:
            world = thumb_opposition(points, direction, world, tail)
        frames[:, joint, :3, :3] = world
        # Carry the constrained direction down the chain with the observed length.
        # The wrist and all five knuckle positions stay anchored to the model.
        frames[:, tail, :3, 3] = frames[:, joint, :3, 3] + world[:, :, 1] * np.linalg.norm(direction, axis=-1)[:, None]
    tips = np.arange(len(HAND_HEADS), len(PARENTS))
    frames[:, tips, :3, :3] = frames[:, PARENTS[tips], :3, :3]
    return frames


def thumb_opposition(points, direction, frame, inner):
    """Fit a continuous thumb base twist to the observed bend plane."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial.transform import Rotation

    next_segment = points[:, HAND_TAILS[inner]] - points[:, inner]
    normal = np.cross(unit(direction), unit(next_segment))
    bend = np.linalg.norm(normal, axis=-1)
    across = np.einsum("ti,ti->t", unit(normal), frame[:, :, 0])
    upright = -np.einsum("ti,ti->t", unit(normal), frame[:, :, 2])
    confidence = np.clip((bend - .10) / .25, 0., 1.)

    angles = np.deg2rad(np.arange(-80., 81., 5.))
    alignment = across[:, None] * np.cos(angles) + upright[:, None] * np.sin(angles)
    # The plane alone cannot distinguish opposite normals. Favor small changes
    # across the clip, and approach neutral when the thumb is straight.
    cost = confidence[:, None] * (1. - alignment**2) + .10 * (angles/angles[-1])**2
    delta = angles[:, None] - angles
    transition = np.where(np.abs(delta) <= np.deg2rad(15.1), 8. * delta**2, np.inf)
    history = np.zeros(cost.shape, dtype=np.intp)
    accumulated = cost[0]
    columns = np.arange(len(angles))
    for i in range(1, len(cost)):
        options = accumulated[:, None] + transition
        history[i] = np.argmin(options, axis=0)
        accumulated = cost[i] + options[history[i], columns]
    path = np.empty(len(cost), dtype=np.intp)
    path[-1] = np.argmin(accumulated)
    for i in range(len(cost)-1, 0, -1):
        path[i-1] = history[i, path[i]]
    roll = gaussian_filter1d(angles[path], 1., mode="nearest")
    return frame @ Rotation.from_rotvec(roll[:, None] * [0., 1., 0.]).as_matrix()


def tracked_hand_frames(points, valid):
    frames = np.empty((len(points), len(PARENTS), 4, 4))
    frames[valid] = hand_frames(points[valid])
    return hold(frames, valid)


def head_frames(data):
    if "camera_to_world" not in data or "camera_position" not in data:
        return None
    from ..camera_tracks import valid_rotations

    camera = np.asarray(data["camera_to_world"], dtype=float)
    center = np.asarray(data["camera_position"], dtype=float)
    count = len(data["tracked"])
    if camera.shape != (count,3,3) or center.shape != (count,3) or not np.isfinite(center).all():
        raise ValueError("The saved first-person head camera is invalid.")
    valid_rotations(camera)
    axes = Y_UP @ camera @ np.diag([-1.,-1.,1.])
    origin = center @ Y_UP.T + axes @ np.array([0.,-.07,-.06])
    frames = np.broadcast_to(np.eye(4), (count,2,4,4)).copy()
    frames[..., :3, :3] = axes[:, None]
    frames[:, 0, :3, 3] = origin
    frames[:, 1, :3, 3] = origin + axes[:, :, 1]*HEAD_LENGTH
    return frames


def arm_frames(data):
    """Build upper-arm and forearm frames from the same joints used for export."""
    if "arm_ik_world" not in data:
        return None
    points = data["arm_ik_world"] @ Y_UP.T
    valid = np.isfinite(points).all(axis=(2,3))
    output = np.full((*points.shape[:2],2,4,4), np.nan)
    joints = points[valid]
    if len(joints):
        frames = np.broadcast_to(np.eye(4),(len(joints),2,4,4)).copy()
        segments = np.diff(joints,axis=1)
        guide = unit(np.cross(segments[:,0],segments[:,1]))
        for index in range(2):
            frames[:,index,:3,:3] = frame_axes(segments[:,index],guide)
        frames[..., :3,3] = joints[:,:2]
        output[valid] = frames
    return output


def export_rig(data):
    """Export native hands, head, and any saved two-bone arm chains."""
    names, parents, tracks = [], [], []
    world = data["hand_world"] @ Y_UP.T
    head = head_frames(data)
    arms = arm_frames(data)
    camera_parent = -1
    if arms is not None and np.isfinite(arms).any():
        root = head[:,0:1].copy()
        root[:,0,:3,3] = data["camera_position"] @ Y_UP.T
        tracks.append(root)
        names.append("CameraRoot")
        parents.append(-1)
        camera_parent = 0
    for s, (side, valid) in enumerate(zip(SIDES, data["tracked"].T)):
        if not valid.any():
            continue
        wrist_parent = camera_parent
        if camera_parent >= 0:
            solved = np.isfinite(arms[:,s]).all(axis=(1,2,3))
            if solved.any():
                # FBX requires transform keys on every frame. Hold arm joints
                # through gaps and keep the hand roots at their tracked positions.
                tracks.append(hold(arms[:,s],solved))
                parents.extend((camera_parent,len(names)))
                wrist_parent = len(names)+1
                names.extend((side+"UpperArm",side+"Forearm"))
        tracks.append(tracked_hand_frames(world[:, s], valid))
        parents.extend(np.where(PARENTS < 0, wrist_parent, PARENTS+len(names)))
        names.extend(side+label for label in LABELS)
    if head is not None:
        parents.extend((camera_parent,len(names)))
        names.extend(("Head","HeadTip"))
        tracks.append(head)
    return names, np.asarray(parents), np.concatenate(tracks, axis=1), float(data["fps"])
