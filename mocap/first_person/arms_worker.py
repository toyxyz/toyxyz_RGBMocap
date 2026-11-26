"""Estimate optional arms from the saved ViDiHand2 hand and camera tracks."""
import argparse
from heapq import nlargest
from itertools import chain
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..camera_tracks import valid_rotations
from ..result_artifacts import file_stat
from .arm_ik import UP, camera_body, estimate_lengths, fit_arms, placement_score, reach_bounds, wrist_distances
from .arm_image import image_strip
from .motion import ARM_LENGTHS, TAKE_NAME, arm_lengths, save_take, validate_arm_lengths
from .rig import is_project, sampled
from .shoulder_prior import load_prior, predict

STAGE = "vidihand_arms"


def evidence(video, data, emit):
    size = tuple(int(value) for value in data["video_size"])
    capture = cv2.VideoCapture(str(video))
    tracked = data["tracked"]
    result = {}
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open the prepared video for ViDiHand2 arm estimation.")
        for frame, visible in enumerate(tracked):
            ok, image = capture.read()
            if not ok:
                raise ValueError("Prepared video ended before the saved ViDiHand2 track.")
            sides = np.flatnonzero(visible)
            if len(sides) and image.shape[1::-1] != size:
                image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            for side in sides:
                result[frame, side] = image_strip(image,data["hand_camera"][frame,side],data["K"][frame])
            if frame % 15 == 0 or frame+1 == len(tracked):
                emit("progress", stage=STAGE, current=frame+1, total=len(tracked),
                     message="Reading forearm directions from the video")
    finally:
        capture.release()
    return result


def arm_placement(data, weights, up):
    camera,root,tracked = (data[key] for key in ("camera_to_world","camera_position","tracked"))
    fps = float(data["fps"])
    prior,body = camera_body(camera,root,up,fps)
    # Use the training basis for features and the exact camera root for IK.
    center = prior + root - gaussian_filter1d(root,2.*fps/30.,axis=0,mode="nearest")
    if weights is not None:
        shift = predict(weights,camera,root,prior,body,data["hand_world"][:,:,0],tracked,fps)
        center += np.einsum("tij,tj->ti",body,shift)
    return body,center


def pitch_placements(data, weights):
    """Candidate torso pitches for a camera whose world pose was estimated."""
    if str(data["camera_mode"]) not in ("automatic","fixed_estimated"):
        return
    forward = data["camera_to_world"][:,:,2].mean(axis=0)
    forward -= (forward @ UP)*UP
    magnitude = np.linalg.norm(forward)
    if magnitude < .1:
        return
    forward /= magnitude
    for amount in np.arange(15.,76.,7.5):
        for degrees in (amount,-amount):
            tilt = np.deg2rad(degrees)
            up = np.cos(tilt)*UP-np.sin(tilt)*forward
            try:
                body,center = arm_placement(data,weights,up)
            except ValueError:
                continue
            yield degrees,body,center


def sampled_arm_frames(data):
    frames = np.flatnonzero(data["tracked"].any(axis=1))
    return frames[np.linspace(0,len(frames)-1,min(len(frames),64),dtype=int)]


def fallback_placement(data, weights, cues, lengths):
    """Try a shared torso pitch on estimated cameras when every arm fails."""
    frames = sampled_arm_frames(data)
    best,selected = 0.,None
    for degrees,body,center in pitch_placements(data,weights):
        coverage, guidance = placement_score(data,body,center,cues,lengths,frames)
        score = coverage + guidance - .02*abs(degrees)/75.
        if score > best:
            best,selected = score,(degrees,body,center)
    return selected


def automatic_candidates(data, weights, cues, body, center):
    frames = sampled_arm_frames(data)
    tracked = data["tracked"]
    reference = sum(ARM_LENGTHS)
    for degrees,axes,origin in chain(((0.,body,center),),pitch_placements(data,weights)):
        distances = wrist_distances(data,axes,origin)
        lengths = estimate_lengths(distances,tracked)
        low,high = reach_bounds(lengths)
        visible = distances[tracked]
        reachable = (visible>=low)&(visible<=high)
        coverage, guidance = placement_score(data,axes,origin,cues,lengths,frames)
        # Rank by valid elbows first. Arm length and image alignment
        # break ties between candidates with the same sampled coverage.
        score = guidance + .4*reachable.mean() - .045*abs(degrees)/75. - .75*max(0.,sum(lengths)-reference)
        yield (bool(reachable.all()),coverage,score),degrees,axes,origin,lengths


def fit_automatic_arms(data, weights, cues, body, center):
    candidates = automatic_candidates(data,weights,cues,body,center)
    best = None
    for _,degrees,axes,origin,lengths in nlargest(3,candidates,key=lambda candidate:candidate[0]):
        arms = fit_arms(data,axes,origin,cues,lengths)
        valid = np.isfinite(arms).all(axis=(2,3))
        missing = data["tracked"] & ~valid
        if not missing.any():
            return arms,degrees,lengths
        longest = 0
        for side in missing.T:
            ends = np.flatnonzero(np.diff(np.r_[False,side,False]))
            longest = max(longest,int(np.max(ends[1::2]-ends[::2],initial=0)))
        score = int(valid.sum()),-longest
        if best is None or score>best:
            best,result = score,(arms,degrees,lengths)
    return result


def estimate(project, emit, lengths=None, *, automatic=False):
    if not is_project(project):
        raise ValueError("Open a reconstructed ViDiHand2 project to estimate arms.")
    take = project/TAKE_NAME
    video = project/"input.mp4"
    stamp = file_stat(take)
    video_stamp = file_stat(video)
    data = sampled(project)
    lengths = arm_lengths(data) if lengths is None else validate_arm_lengths(lengths)
    camera, root = data.get("camera_to_world"), data.get("camera_position")
    count = len(data["tracked"])
    if camera is None or root is None or camera.shape != (count,3,3) or root.shape != (count,3) or not np.isfinite(root).all():
        raise ValueError("Arm estimation needs a saved ViDiHand2 camera track.")
    valid_rotations(camera)
    weights = load_prior()
    body,center = arm_placement(data,weights,UP)
    cv2.setNumThreads(2)
    cues = evidence(video, data, emit)
    degrees = 0.
    if automatic:
        emit("progress",stage=STAGE,message="Estimating a shared arm length from both wrist tracks")
        arms,degrees,lengths = fit_automatic_arms(data,weights,cues,body,center)
    else:
        emit("progress", stage=STAGE, message="Solving and smoothing both elbows")
        arms = fit_arms(data,body,center,cues,lengths)
    solved = int(np.isfinite(arms).all(axis=(2,3)).sum())
    if not solved and not automatic:
        emit("progress",stage=STAGE,message="Checking alternative arm orientations")
        fallback = fallback_placement(data,weights,cues,lengths)
        if fallback is not None:
            degrees,body,center = fallback
            arms = fit_arms(data,body,center,cues,lengths)
            solved = int(np.isfinite(arms).all(axis=(2,3)).sum())
    if not solved:
        upper, forearm = (length*100. for length in lengths)
        retry = ("Turn off Auto Arm Length to enter lengths, or align the camera and apply again."
                 if automatic else "Adjust the lengths or align the camera, then apply again.")
        raise ValueError(f"Arm IK found no valid pose at {upper:g} cm upper arm and {forearm:g} cm forearm. "
                         + retry)
    if file_stat(video) != video_stamp:
        raise ValueError("The prepared video changed during arm estimation.")
    data["arm_ik_world"] = arms
    data["arm_ik_lengths"] = np.asarray(lengths)
    data["arm_ik_auto_lengths"] = np.array(automatic,dtype=bool)
    save_take(take, data, stamp)
    if automatic:
        upper,forearm = (value*100. for value in lengths)
        emit("progress",stage=STAGE,message=f"Arm IK estimated {upper:g} cm upper arms and {forearm:g} cm forearms.")
    if degrees:
        emit("progress",stage=STAGE,message=f"Arm IK used a {degrees:g} degree torso pitch correction.")
    emit("progress",stage=STAGE,message=f"Arm IK solved {solved}/{int(data['tracked'].sum())} tracked hand poses.")


def main(arguments, emit):
    parser = argparse.ArgumentParser()
    parser.add_argument("--project",type=Path,required=True)
    arm_lengths_group = parser.add_mutually_exclusive_group()
    arm_lengths_group.add_argument("--arm-lengths",type=float,nargs=2,metavar=("UPPER_M","FOREARM_M"))
    arm_lengths_group.add_argument("--auto-arm-lengths",action="store_true")
    args = parser.parse_args(arguments)
    started = time.monotonic()
    try:
        estimate(args.project,emit,args.arm_lengths,automatic=args.auto_arm_lengths)
    except (OSError,ValueError,RuntimeError,KeyError,ImportError,cv2.error) as error:
        emit("failed",stage=STAGE,message=str(error))
        return 1
    emit("complete",stage=STAGE,elapsed=time.monotonic()-started,output=str(args.project))
    return 0
