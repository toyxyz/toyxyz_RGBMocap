"""RGB MoCap CPU contact correction. Imported by workers, never by playback.

Sole-aware goals, global contact anchors and exact ankle orientation retention
adapt the pinned NVIDIA solver to SMPL-X without retargeting the character.
"""
from pathlib import Path
import importlib.util
import json
import math
import tempfile
import time
import os
import shutil
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d
from scipy.linalg import solve_banded
from .correction_settings import PRESETS

PARENTS = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]
VERSION = 9
CORE = 1024
HALO_SECONDS = 4.0


def backend():
    from .paths import CORRECTION_BIN,NATIVE_SUFFIX
    files=list(CORRECTION_BIN.glob('_motion_correction*'+NATIVE_SUFFIX))
    if not files:raise RuntimeError('Motion Correction runtime is missing. Reinstall or rebuild the app.')
    spec=importlib.util.spec_from_file_location('_motion_correction',files[0])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def fk(rest, root, quat):
    local=Rotation.from_quat(quat[..., [1,2,3,0]].reshape(-1,4)).as_matrix().reshape(-1,22,3,3)
    rotations=local.copy();positions=np.empty((len(root),22,3))
    positions[:,0]=root
    for j in range(1,22):
        p=PARENTS[j]
        positions[:,j]=positions[:,p]+np.einsum('tij,j->ti',rotations[:,p],rest[j]-rest[p])
        rotations[:,j]=rotations[:,p]@local[:,j]
    return positions,rotations


def intervals(active):
    edges=np.diff(np.r_[False,active,False].astype(np.int8))
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))


def reachable_pelvis_drop(hips, targets, lengths, active):
    """Minimum downward pelvis move for both support legs, with extension margin.

    Use calibrated physical reach with a small singularity margin. Anatomical
    source IK does not need the legacy native 1.5% extension/fade workaround.
    Do not stretch bones or shift stance anchors to hide unreachable goals.
    """
    radius=np.asarray(lengths)*.9995
    horizontal=np.sum((hips[...,[0,2]]-targets[...,[0,2]])**2,axis=-1)
    vertical=hips[...,1]-targets[...,1]
    limit=np.sqrt(np.maximum(radius**2-horizontal,0))
    possible=(horizontal<radius**2)&(vertical>0)&active
    needed=np.where(possible,np.maximum(vertical-limit,0),0).max(1)
    # Bad manual timings must not collapse the actor to satisfy arbitrary pins.
    return np.minimum(needed,.25)


def reachable_pelvis_lift(hips,targets,minimum,maximum,active):
    """Small upward reach compensation for deep crouches, without stretching.

    Lowering alone can make a target too close when knee flexion is limited.
    Respect the other leg's maximum reach and never force incompatible pins.
    This is root planning only, not a second anatomical leg solve.
    """
    horizontal=np.sum((hips[...,[0,2]]-targets[...,[0,2]])**2,axis=-1)
    vertical=hips[...,1]-targets[...,1]
    lower=np.sqrt(np.maximum((minimum*1.0005)**2-horizontal,0))-vertical
    upper=np.sqrt(np.maximum((maximum*.9995)**2-horizontal,0))-vertical
    eligible=active&(vertical>0)
    needed=np.maximum(np.where(eligible,lower,0).max(1),0)
    permitted=np.where(eligible,upper,np.inf).min(1)
    return np.minimum(np.minimum(needed,np.maximum(permitted,0)),.12)


def contact_strength(segments,n,fps,preset):
    config=PRESETS[preset]
    strength=np.zeros((n,2),np.float32)
    for segment in segments:
        s,e,foot=segment['start'],segment['end'],segment['foot']
        # Transitions now live outside the immutable stance interval. Fading
        # inside a contact created large skating errors and stronger release
        # jumps. Weak still deliberately retains its partial hold.
        strength[s:e,foot]=config['strength']
    return strength


def support_points(probes,probabilities,active,reference=None,fps=30.):
    from .contact_trajectory import support_weights,support_samples
    weights=support_weights(probes if reference is None else reference,active,fps)
    return support_samples(probes,weights)


def airborne_excursions(probes,fps):
    """Reject bounded two-foot flight arcs, without grounding away their apex.

    A continuously floating whole take is ambiguous and remains groundable.
    Only arcs with observed lower samples on BOTH sides are rejected. This
    geometric gate is not a pressure classifier or a substitute for video QA.
    """
    n=len(probes);flight=np.zeros(n,bool)
    if n<5:return flight
    h=probes[...,1].min(1)
    # Slow-motion flips can remain airborne for several video seconds. Use
    # local lower envelopes so slow world-height drift in a long take is not
    # confused with a jump; require an observed rise AND landing afterwards.
    reach=min(n,max(2,round(fps*4.)))
    baseline=minimum_filter1d(h,size=2*reach+1,mode='nearest')
    raised=h-baseline
    smoothed=gaussian_filter1d(h,max(.5,fps*.03),mode='nearest')
    velocity=np.gradient(smoothed)*fps
    for first,end in intervals(raised>.15):
        peak=first+int(np.argmax(smoothed[first:end]));lo=max(0,peak-reach);hi=min(n,peak+reach+1)
        if peak==lo or peak+1==hi:continue
        left=float(smoothed[lo:peak].min());right=float(smoothed[peak+1:hi].min());top=float(smoothed[peak])
        if top-max(left,right)<.15:continue
        before=np.flatnonzero(smoothed[lo:peak]<left+.06)
        after=np.flatnonzero(smoothed[peak+1:hi]<right+.06)
        if not len(before) or not len(after):continue
        start=lo+before[-1]+1;stop=peak+1+after[0]
        # Landing can return to a slightly DIFFERENT inferred world height.
        # Detect the end of rapid descent rather than requiring old floor Y.
        descent=velocity[peak+1:stop+1]
        landed=np.flatnonzero((np.minimum.accumulate(descent)<-.8)&(descent>-.35)&
                              (top-smoothed[peak+1:stop+1]>.15))
        if len(landed):stop=peak+1+int(landed[0])
        if stop-start>8.*fps:continue
        flight[start:stop]=True
    return flight


def sole_metrics(before,after,probabilities,active,anchors,segments,fps,preset,floor,mode='flat'):
    """Separate representative-point locks from BOTH physical sole patches."""
    from .contact_trajectory import handoff_mask
    handoffs=handoff_mask(segments,len(before))
    locked=(contact_strength(segments,len(before),fps,preset)>=.999)&~handoffs
    point=support_points(after,probabilities,active,reference=before,fps=fps)
    level=anchors[...,1] if mode=='multi_level' else floor
    penetration=np.maximum(level-after[...,1].reshape(-1,2,2).min(2),0)
    blended=active&~locked
    modes=[];difference=before[...,1].reshape(-1,2,2)
    for foot in (0,1):
        delta=difference[:,foot,0]-difference[:,foot,1]
        for name,mask in (('heel',delta<-.01),('flat',np.abs(delta)<=.01),('forefoot',delta>.01)):
            modes.append(dict(foot=foot,mode=name,frames=int((mask&active[:,foot]).sum())))
    from .contact_trajectory import support_weights,material_slip
    slip=material_slip(before,after,active,support_weights(before,active,fps),fps)
    if mode=='multi_level':
        weights=support_weights(before,active,fps)
        w=.5*(weights[1:]+weights[:-1]);pairs=locked[1:]&locked[:-1]
        for label,probes in (('before',before),('after',after)):
            delta=np.diff(np.asarray(probes).reshape(-1,2,2,3),axis=0)
            speed=np.linalg.norm(delta[:,:,0]*(1-w[...,None])+delta[:,:,1]*w[...,None],axis=-1)[pairs]*fps
            slip['support_material_3d_speed_'+label+'_max_mps']=float(speed.max()) if len(speed) else None
            slip['support_material_3d_speed_'+label+'_p95_mps']=float(np.quantile(speed,.95)) if len(speed) else None
    return dict(**slip,automatic_handoff_frames=int(handoffs.sum()),full_lock_frames=int(locked.sum()),blended_contact_frames=int(blended.sum()),
        full_lock_surface_error_max_m=float(np.linalg.norm(point-anchors,axis=-1)[locked].max()) if locked.any() else None,
        full_lock_height_error_max_m=float(np.abs(point[...,1]-level)[locked].max()) if locked.any() else None,
        full_lock_sole_penetration_max_m=float(penetration[locked].max()) if locked.any() else None,
        blended_sole_penetration_max_m=float(penetration[blended].max()) if blended.any() else None,
        noncontact_sole_penetration_max_m=float(penetration[~active].max()) if mode=='flat' and (~active).any() else None,
        contact_sole_penetration_over_1cm=int(((penetration>.01)&active).sum()),
        geometric_support_modes=modes,
        short_contact_segments=int(sum(s['end']-s['start']<max(2,round(fps*.1)) for s in segments)))


def prepare_contacts(probes, probabilities, fps, floor=None, preset='normal', contact_ranges=None,mode='flat'):
    """Resolve contact phases and grounded sole anchors once, before windowing.

Static confidence is not ground truth. Reject raised/moving feet, preserve
airborne intervals, and split anchors when the source intentionally travels.
"""
    if preset not in PRESETS:raise ValueError('Unknown Motion Correction preset: '+str(preset))
    config=PRESETS[preset]
    n=len(probes)
    if probabilities.shape!=(n,4) or probes.shape!=(n,4,3):raise ValueError('Invalid sole/contact dimensions')
    if not math.isfinite(fps) or fps<=0 or not np.isfinite(probes).all() or not np.isfinite(probabilities).all():
        raise ValueError('Invalid contacts, positions or frame rate')
    if np.any(probabilities<0) or np.any(probabilities>1):raise ValueError('Contact probabilities must be in [0,1]')
    if mode=='multi_level':
        from .multilevel_contacts import prepare
        return prepare(probes,probabilities,fps,preset,contact_ranges)
    if mode!='flat':raise ValueError('Unknown correction mode: '+str(mode))
    height=probes[...,1].reshape(n,2,2).min(2)
    points=probes.reshape(n,2,2,3).mean(2)
    speed=np.zeros((n,2));vertical=np.zeros((n,2))
    if n>1:
        speed[1:]=np.linalg.norm(np.diff(points[...,[0,2]],axis=0),axis=-1)*fps
        vertical[1:]=np.abs(np.diff(height,axis=0))*fps
        speed[0]=speed[1];vertical[0]=vertical[1]
    confidence=probabilities.reshape(n,2,2).max(2)
    geometry=(speed<.7)&(vertical<.8)&((height-height.min(1,keepdims=True))<.12)
    active=(confidence>config['confidence'])&geometry
    if config['gap']:
        # Extend only seeded stances, close to the floor and moving slowly.
        # A new step must still pass the original high-confidence entry gate.
        support_height=height[active]
        reference=float(np.quantile(support_height,.05) if floor is None and len(support_height)
                        else (height.min() if floor is None else floor))
        near=np.abs(height-reference)<.04
        holding=(confidence>config['release'])&geometry&near&(speed<.25)
        for foot in range(2):
            for first,end in intervals(holding[:,foot]|active[:,foot]):
                seeds=np.flatnonzero(active[first:end,foot])
                if len(seeds):active[first+seeds[0]:end,foot]=True
            # Close a brief dropout only between two existing stances. Do not
            # bridge a lifted foot, a sizeable step or an unbounded gap.
            for first,end in intervals(~active[:,foot]):
                if first==0 or end==n or end-first>round(fps*config['gap']):continue
                if not np.all(geometry[first:end,foot]&near[first:end,foot]):continue
                if np.linalg.norm(points[end,foot,[0,2]]-points[first-1,foot,[0,2]])>.04:continue
                active[first:end,foot]=True
    minimum=max(2,round(fps*config['minimum']))
    for foot in range(2):
        for first,end in intervals(active[:,foot]):
            if end-first<minimum:active[first:end,foot]=False
    active[airborne_excursions(probes,fps)]=False
    if contact_ranges is not None:
        from .foot_contacts import normalize_ranges
        contact_ranges=normalize_ranges(contact_ranges,n)
        active[:]=False
        for foot in range(2):
            for first,end in contact_ranges[foot]:active[first:end,foot]=True
    values=height[active]
    floor=float(np.quantile(values,.05) if floor is None and len(values) else (height.min() if floor is None else floor))
    if not math.isfinite(floor):raise ValueError('Floor height must be finite')
    # Correct only support intervals; in flight interpolate the offset, keeping
    # the original jump trajectory instead of snapping either foot down.
    support=active.any(1)
    target=np.zeros(n)
    if support.any():
        support_height=np.where(active,height,np.inf).min(1)
        desired=np.clip(floor-support_height[support],-.75,.75)
        target=np.interp(np.arange(n),np.flatnonzero(support),desired)
        if n>=3:
            weight=np.where(support,1.,.02)
            smooth=4.*(fps/30.)**4
            diagonal=weight+smooth*np.r_[1.,5.,np.full(max(0,n-4),6.),5.,1.] if n>=4 else weight+smooth*np.array([1.,4.,1.])
            first=-2.*smooth*np.r_[1.,np.full(max(0,n-3),2.),1.]
            bands=np.zeros((5,n));bands[2]=diagonal
            bands[1,1:]=first;bands[3,:-1]=first
            bands[0,2:]=smooth;bands[4,:-2]=smooth
            target=solve_banded((2,2),bands,weight*target,check_finite=False)
    from .contact_trajectory import support_weights,rolling_anchor
    weights=support_weights(probes,active,fps)
    offsets=support_points(probes,probabilities,active,fps=fps);anchors=offsets.copy()
    segments=[]
    for foot in range(2):
        # Heel/forefoot choice follows source roll. Both contacts use a central
        # sole point; its height is the lower surface, not the ankle joint.
        point=offsets[:,foot]
        for first,end in intervals(active[:,foot]):
            start=first
            while start<end:
                stop=start+1
                while stop<end and (contact_ranges is not None or np.linalg.norm(point[stop,[0,2]]-point[start,[0,2]])<config['anchor_drift']):
                    stop+=1
                # A one/two-sample tail is not a new stance. Starting an
                # independent anchor there caused a snap while the timeline
                # still showed one continuous contact. Keep timing unchanged.
                if contact_ranges is None and end-stop<minimum:stop=end
                path=rolling_anchor(probes[start:stop,2*foot:2*foot+2],weights[start:stop,foot],floor)
                anchors[start:stop,foot]=path
                segments.append({'foot':foot,'start':int(start),'end':int(stop),'anchor':path[0].tolist(),
                                 'anchor_end':path[-1].tolist()})
                start=stop
    from .contact_trajectory import couple_anchors,smooth_auto_handoffs
    # Only explicit user stances need cross-gap position coupling. Applying
    # this to auto travel splits can accumulate a false long-duration pin.
    if contact_ranges is not None:couple_anchors(anchors,offsets,segments,fps)
    else:smooth_auto_handoffs(anchors,probes,weights,segments,fps)
    shift=np.zeros((n,3),np.float32);shift[:,1]=np.clip(target,-.75,.75)
    if support.any():
        drift=np.sum((anchors-offsets)*active[...,None],axis=1)/np.maximum(active.sum(1,keepdims=True),1)
        for axis in (0,2):
            values=np.interp(np.arange(n),np.flatnonzero(support),drift[support,axis])
            shift[:,axis]=gaussian_filter1d(values,max(1.,fps*.06),mode='nearest')
        norms=np.linalg.norm(shift[:,[0,2]],axis=1)
        shift[:,[0,2]]*=np.minimum(1.,.12/np.maximum(norms,1e-8))[:,None]
    return active,anchors,offsets,shift,floor,segments


def correct_arrays(rest, root, quat, probes, probabilities, fps=30., floor=None, core=CORE, output=None, emit=lambda *a,**k:None, preset='normal',contact_ranges=None,pelvis_offset=0.,mode='flat',foot_enabled=True):
    from .correction_settings import correction_options
    pelvis_offset=correction_options({'pelvis_offset':pelvis_offset})['pelvis_offset']
    n=len(root)
    if rest.shape!=(22,3) or root.shape!=(n,3) or quat.shape!=(n,22,4):raise ValueError('Expected a 22-joint SMPL-X motion')
    if not n:raise ValueError('Motion contains no frames')
    if not isinstance(core,int) or core<1:raise ValueError('Solver window size must be a positive integer')
    if not np.isfinite(rest).all() or not np.isfinite(root).all() or not np.isfinite(quat).all():raise ValueError('Motion contains non-finite values')
    if np.any(np.linalg.norm(quat,axis=-1)<1e-8):raise ValueError('Zero-length rotation quaternion')
    if any(np.linalg.norm(rest[b]-rest[a])<1e-6 for a,b in ((1,4),(4,7),(2,5),(5,8))):
        raise ValueError('Leg bone lengths must be nonzero')
    from .foot_contacts import segment_ranges
    plan=prepare_contacts(probes,probabilities,fps,floor,preset,mode=mode)
    automatic_ranges=segment_ranges(plan[5]) if foot_enabled else [[],[]]
    if not foot_enabled:contact_ranges=[[],[]]
    if contact_ranges is not None:
        del plan
        plan=prepare_contacts(probes,probabilities,fps,floor,preset,contact_ranges,mode)
    active,anchors,sole_points,shift,floor,segments=plan
    out_root,out_quat=output if output is not None else (np.empty_like(root),np.empty_like(quat))
    if n==1:
        out_root[:]=root;out_quat[:]=quat
        return out_root,out_quat,{'version':VERSION,'preset':preset,'mode':mode,'frames':1,'fps':fps,'floor':floor,
            'segments':[],'automatic_ranges':automatic_ranges,'contacts':0,'core_frames':core,'halo_frames':0,
            'native_seconds':0.,'max_root_y_offset_m':0.,'max_root_xz_offset_m':0.,'max_reach_pelvis_drop_m':0.,
            'max_contact_goal_error_m':0.,'max_ankle_global_change_deg':0.,'rejected_knee_frames':0,'straight_knee_support_frames':0,
            'method':'Original one-frame pose retained','bypassed':'fewer than two frames',
            'warning':'One-frame clip: original pose retained. Temporal foot locking needs at least two frames.'}
    native=backend()
    if getattr(native,'rgb_mocap_anatomical_version',0)!=1:
        raise RuntimeError('Motion Correction runtime is outdated. Reinstall the updated app.')
    offsets=rest.copy()
    for j in range(1,22):offsets[j]-=rest[PARENTS[j]]
    ref=offsets.astype(float).tolist()
    from .mixamo_rig import align
    axes=np.ascontiguousarray([align([0,-1,0],rest[knee]-rest[hip])@np.array([1.,0,0])
                              for hip,knee in ((1,4),(2,5))],np.float64).reshape(-1)
    halo=max(2,round(fps*HALO_SECONDS))
    strength=contact_strength(segments,n,fps,preset)
    # A rise followed by descent may be a staircase, not a ballistic flight.
    # Multi-level contact evidence is local 3D foot motion, independent of Y=0.
    if mode=='flat':flight=airborne_excursions(probes,fps)
    else:
        from .multilevel_contacts import flight_mask
        flight=flight_mask(probes,fps,active) if contact_ranges is None else np.zeros(n,bool)
    target_shift=shift.copy()
    from .contact_trajectory import contact_offsets
    # Manual timing owns its finite approach/release even when the source's
    # automatic flight guess disagrees. Farther flight stays unmodified.
    transition_flight=flight if contact_ranges is None else np.zeros_like(flight)
    goal_offsets,transition=contact_offsets(probes,active,anchors,sole_points,target_shift,segments,fps,strength,transition_flight)
    drop=np.zeros(n,np.float32)
    # Resolve globally before bounded native windows, including transitions.
    # Targets retain the original grounding shift; pelvis lowering must not
    # lower the foot goals along with the body.
    for first in range(0,n,core):
        last=min(n,first+core)
        j,g=fk(rest,root[first:last],quat[first:last])
        from .anatomical_ik import reach_lengths
        lengths=reach_lengths(rest,g)
        goals=j[:,[7,8]]+target_shift[first:last,None]
        targets=goals+goal_offsets[first:last]
        drop[first:last]=reachable_pelvis_drop(j[:,[1,2]]+shift[first:last,None],targets,lengths,
                                             active[first:last]|transition[first:last])
    if np.any(drop):
        envelope=maximum_filter1d(drop,size=2*max(1,round(fps*.1))+1,mode='nearest')
        drop=np.maximum(drop,gaussian_filter1d(envelope,max(.5,fps*.03),mode='nearest'))
        shift[:,1]-=drop
    reach_raise=np.zeros(n,np.float32)
    for first in range(0,n,core):
        last=min(n,first+core);j,g=fk(rest,root[first:last],quat[first:last])
        from .anatomical_ik import reach_lengths,minimum_reach_lengths
        targets=j[:,[7,8]]+target_shift[first:last,None]+goal_offsets[first:last]
        reach_raise[first:last]=reachable_pelvis_lift(j[:,[1,2]]+shift[first:last,None],targets,
            minimum_reach_lengths(rest,g),reach_lengths(rest,g),active[first:last]|transition[first:last])
    shift[:,1]+=reach_raise
    # Raise only as far as the automatic reach compensation lowered the body.
    # Negative adjustments fade with contacts; never offset an airborne clip.
    # Keep target_shift untouched so this cannot move the floor or foot goals.
    automatic_drop=np.maximum(drop-reach_raise,0)
    adjustment=(np.minimum(automatic_drop,pelvis_offset) if pelvis_offset>=0 else
                pelvis_offset*strength.max(axis=1))
    shift[:,1]+=adjustment
    # Interpolating between two grounding offsets must not put an airborne
    # actor through the plane near takeoff/landing. Lift the WHOLE body only
    # as needed; keep the flight pose/foot rotations and never create a pin.
    # A wholly contact-free take, and explicit manual pins, stay untouched.
    air_clearance=np.where(flight&(~active.any(1))&active.any()&(mode=='flat'),
        np.minimum(np.maximum(floor-probes[...,1].min(1)-shift[:,1],0),.12),0.)
    shift[:,1]+=air_clearance
    from .contact_pose_plan import optimise
    planner_flight=flight if contact_ranges is None else flight&~transition.any(1)
    shift,temporal_report=optimise(rest,root,quat,probes,shift,target_shift,goal_offsets,active,transition,planner_flight,floor,fps,mode=mode,emit=emit)
    max_goal_error=0.;transition_error=0.;knee_reversals=0;rotation_error=0.;native_seconds=0.;straight_knees=0
    anatomical_error=0.;anatomical_delta=0.;anatomical_flex=0.;swing_lift=0.;swing_frames=0
    for start in range(0,n,core):
        end=min(n,start+core);left=max(0,start-halo);right=min(n,end+halo)
        original_q=np.array(quat[left:right],np.float32,copy=True)
        original_root=np.array(root[left:right],np.float32,copy=True)
        positions,original_global=fk(rest,original_root,original_q)
        r=original_root+shift[left:right]
        original_goals=positions[:,[7,8]]+target_shift[left:right,None]
        targets=original_goals+goal_offsets[left:right]
        # A free foot may be lifted by its own leg in a grounded take. Never
        # lower a swing foot, create a lock, or touch detected flight. Brief
        # confidence dropouts must not disable collision with the known plane.
        # Keep a 1cm grazing tolerance; bound bad input/floor corrections.
        support_height=probes[left:right,...,1].reshape(-1,2,2).min(2)
        planned=active[left:right]|transition[left:right]
        free_height=support_height+np.where(planned,targets[...,1]-positions[:,[7,8],1],shift[left:right,1,None])
        guard=(~active[left:right])&active.any()&(~flight[left:right,None])&(free_height<floor-.01)&(mode=='flat')
        lift=np.where(guard,np.minimum(np.maximum(floor-free_height,0),.12),0.)
        free_goals=np.where(planned[...,None],targets,positions[:,[7,8]]+shift[left:right,None])
        free_goals[...,1]+=lift
        targets=np.where(guard[...,None],free_goals,targets)
        # Boundary blends and Weak remain partial XZ holds, but may not push
        # an active sole below the physical floor while blending toward it.
        expected=support_height+targets[...,1]-positions[:,[7,8],1]
        floor_lift=np.where(active[left:right]&(mode=='flat'),np.minimum(np.maximum(floor-expected,0),.12),0.)
        targets[...,1]+=floor_lift
        solve_active=planned|guard
        keep=slice(start-left,end-left)
        swing_frames+=int(guard[keep].sum());swing_lift=max(swing_lift,float(lift[keep].max()))
        q=original_q.copy();zero=np.zeros(len(r),np.float32)
        c=np.zeros((len(r),4),np.float32);c[:,[0,2]]=solve_active
        begin=time.monotonic()
        anatomy=native.correct_motion(r.reshape(-1),q.reshape(-1),r.copy().reshape(-1),q.copy().reshape(-1),
            zero,zero,zero,zero,zero,zero,c.reshape(-1),PARENTS,ref,[[0.,0.,0.,1.] for _ in PARENTS],
            20,21,7,8,.5,.04,False,np.ascontiguousarray(targets,np.float32).reshape(-1),axes)
        native_seconds+=time.monotonic()-begin
        # The native backend now returns the FINAL calibrated leg solution.
        # Independently inspect its quantized output, not only backend metrics.
        out_local=Rotation.from_quat(q[...,[1,2,3,0]].reshape(-1,4)).as_matrix().reshape(-1,22,3,3)
        src_local=Rotation.from_quat(original_q[...,[1,2,3,0]].reshape(-1,4)).as_matrix().reshape(-1,22,3,3)
        for foot,knee in enumerate((4,5)):
            delta=Rotation.from_matrix(out_local[:,knee]@src_local[:,knee].swapaxes(-1,-2)).as_rotvec()
            axis=axes.reshape(2,3)[foot];off=delta-(delta@axis)[:,None]*axis
            anatomical_error=max(anatomical_error,float(np.rad2deg(np.linalg.norm(off[keep],axis=1).max())))
        if anatomical_error>.0001:raise ValueError('Native correction exceeded the anatomical knee axis')
        anatomical_error=max(anatomical_error,anatomy['max_added_knee_off_axis_deg'])
        anatomical_delta=max(anatomical_delta,anatomy['max_added_knee_rotation_deg'])
        anatomical_flex=max(anatomical_flex,anatomy['max_corrected_knee_flex_deg'])
        act=active[start:end]
        for foot,(hip,knee,ankle) in enumerate(((1,4,7),(2,5,8))):
            a=np.cross(positions[:,knee]-positions[:,hip],positions[:,ankle]-positions[:,knee])
            # Match the native straight-leg singularity threshold. Its input
            # plane has no reliable sign, so it cannot serve as a flip guard.
            native_area_limit=.02*np.linalg.norm(rest[knee]-rest[hip])*np.linalg.norm(rest[ankle]-rest[knee])
            source_area=np.linalg.norm(a,axis=1)
            straight_knees+=int(((source_area<native_area_limit)[keep]&act[:,foot]).sum())
        final_points,final_global=fk(rest,r,q)
        # Measure the published pose, after anatomical axis validation and IK.
        if act.any():max_goal_error=max(max_goal_error,float(np.linalg.norm(final_points[keep][:,[7,8]]-targets[keep],axis=-1)[act].max()))
        if transition[start:end].any():
            transition_error=max(transition_error,float(np.linalg.norm(final_points[keep][:,[7,8]]-targets[keep],axis=-1)[transition[start:end]].max()))
        delta=original_global[:,[7,8]].transpose(0,1,3,2)@final_global[:,[7,8]]
        rotation_error=max(rotation_error,float(np.rad2deg(Rotation.from_matrix(delta[keep].reshape(-1,3,3)).magnitude()).max()))
        out_root[start:end]=r[keep];out_quat[start:end]=q[keep]
        emit('progress',stage='correct',current=end,total=n,message='Motion Correction: stabilizing ground contacts')
    report={'version':VERSION,'preset':preset,'mode':mode,'automatic_ranges':automatic_ranges,'frames':n,'fps':fps,'floor':floor,'core_frames':core,'halo_frames':halo,
        'native_seconds':native_seconds,'contacts':int(active.sum()),'segments':segments,
        'max_root_y_offset_m':float(np.abs(shift[:,1]).max()),'max_root_xz_offset_m':float(np.linalg.norm(shift[:,[0,2]],axis=1).max()),'max_contact_goal_error_m':max_goal_error,
        'max_reach_pelvis_drop_m':float(automatic_drop.max()),
        'max_reach_pelvis_raise_m':float(np.maximum(reach_raise-drop,0).max()),
        'max_transition_goal_error_m':transition_error,
        'pelvis_offset_m':pelvis_offset,
        'max_applied_pelvis_lift_m':float(np.maximum(adjustment,0).max()),
        'max_applied_pelvis_lower_m':float(np.maximum(-adjustment,0).max()),
        'max_ankle_global_change_deg':rotation_error,'rejected_knee_frames':knee_reversals,
        'straight_knee_support_frames':straight_knees,
        'max_added_knee_off_axis_deg':anatomical_error,'max_added_knee_rotation_deg':anatomical_delta,
        'max_corrected_knee_flex_deg':anatomical_flex,
        'swing_clearance_frames':swing_frames,'max_swing_clearance_lift_m':swing_lift,
        'air_clearance_frames':int((air_clearance>0).sum()),'max_air_clearance_lift_m':float(air_clearance.max()),
        'airborne_rejected_frames':int(flight.sum()) if contact_ranges is None else 0,
        'transition_frames':int(transition.sum()),
        'support_model':'geometry-driven heel/forefoot material handoff; physical minimum Y',
        'method':'RGB MoCap Motion Correction; native anatomical contacts; continuous sole support; source global ankle rotations retained',
        'warning':'No confident foot contacts; foot correction was skipped.' if foot_enabled and not active.any() else
                  'Some foot locks remain unresolved; inspect the result.' if max_goal_error>.01 else ''}
    adjacent=(transition[1:]|transition[:-1])&(~flight[1:,None])&(~flight[:-1,None])
    speed=np.linalg.norm(np.diff(goal_offsets[...,[0,2]],axis=0),axis=-1)*fps
    report['max_transition_offset_speed_mps']=float(speed[adjacent].max()) if adjacent.any() else 0.
    report['fast_transition_steps']=int(((speed>2.)&adjacent).sum())
    if report['fast_transition_steps']:
        report['warning']=' '.join(filter(None,(report['warning'],
            'Some short contact gaps still require rapid foot motion; inspect contact timing.')))
    if transition_error>.01:
        report['warning']=' '.join(filter(None,(report['warning'],
            'Some contact transitions exceed anatomical reach; inspect floor height and timing.')))
    if mode=='multi_level':
        report['support_model']='Independent source-relative 3D stance heights; no global ground plane'
        report['method']='Multi Level foot contacts; native anatomical leg correction; source elevation and ankle orientation retained'
        report['terrain_collision_evaluated']=False
        report['warning']=' '.join(filter(None,(report['warning'],
            'Multi Level estimates contacts from motion. Terrain and object collisions are not solved.')))
    report.update(temporal_report)
    plans = ([temporal_report['contact_chain_plan']] if 'contact_chain_plan' in temporal_report else
             [tile['report']['contact_chain_plan'] for tile in
              temporal_report.get('bounded_contact_plan', {}).get('windows', [])
              if 'contact_chain_plan' in tile['report']])
    report['contact_plan_limit_windows'] = sum(not plan['success'] for plan in plans)
    if report['contact_plan_limit_windows']:
        report['warning'] = ' '.join(filter(None, (report['warning'],
            'Some contact-transition windows reached the planning limit; inspect motion and contact timing.')))
    return out_root,out_quat,report


def cleanup_orphans(project):
    import psutil
    project=Path(project).resolve()
    for folder in project.glob('.correction-*'):
        resolved=folder.resolve()
        if not resolved.is_relative_to(project) or resolved.is_symlink() or not resolved.is_dir():continue
        try:
            owner=json.loads((resolved/'owner.json').read_text())
            try:alive=abs(psutil.Process(int(owner['pid'])).create_time()-float(owner['created']))<.01
            except (psutil.NoSuchProcess,ValueError):alive=False
            if not alive:shutil.rmtree(resolved)
        except (OSError,ValueError,KeyError,psutil.AccessDenied):pass


class SoleGeometry:
    """Exact LBS over all foot-dominant vertices, with stable sole XZ probes.

    Cover the arch/side edges too: the lowest neutral 24 points do not contain
    the lowest point after FootMR roll. Time windows and vertex count remain
    bounded by the registered body topology, independent of recording length.
    """
    def __init__(self,body,betas):
        self.parents=body.parents.numpy().tolist()
        self.mean=body.pose_mean.numpy().reshape(55,3)
        shaped=body.v_template.numpy()+np.einsum('vci,i->vc',body.shapedirs.numpy(),betas)
        self.rest=body.J_regressor.numpy()@shaped
        weights=body.lbs_weights.numpy()
        from .hand_lock import palm_offsets
        self.palm_offsets=palm_offsets(shaped,weights,self.rest)
        groups=[];mean_groups=[]
        for ankle,toe in ((7,10),(8,11)):
            ids=np.flatnonzero(weights[:,ankle]+weights[:,toe]>.6)
            z=shaped[ids,2]
            middle=np.median(z)
            for region,proxy in ((ids[z<=middle],ids[z<=np.quantile(z,.4)]),
                                 (ids[z>=middle],ids[z>=np.quantile(z,.65)])):
                # Keep the legacy XZ material points; expand only the region
                # searched for its physical minimum Y.
                selected=proxy[np.argsort(shaped[proxy,1])[:24]]
                groups.append(region);mean_groups.append(np.searchsorted(region,selected))
        self.groups=groups;self.mean_groups=mean_groups;self.ids=np.concatenate(groups)
        self.shaped=shaped[self.ids];self.weights=weights[self.ids]
        self.posedirs=body.posedirs.numpy().reshape(486,-1,3)[:,self.ids].reshape(486,-1)

    def sample(self,root,quat):
        n=len(root)
        rotations=np.tile(Rotation.from_rotvec(self.mean).as_matrix(),(n,1,1,1))
        rotations[:,:22]=Rotation.from_quat(quat[..., [1,2,3,0]].reshape(-1,4)).as_matrix().reshape(n,22,3,3)
        posed=self.shaped+((rotations[:,1:]-np.eye(3)).reshape(n,486)@self.posedirs).reshape(n,-1,3)
        world_r=rotations.copy();joints=np.empty((n,55,3));joints[:,0]=root
        for j in range(1,55):
            p=self.parents[j];joints[:,j]=joints[:,p]+np.einsum('tij,j->ti',world_r[:,p],self.rest[j]-self.rest[p])
            world_r[:,j]=world_r[:,p]@rotations[:,j]
        translations=joints-np.einsum('tjik,jk->tji',world_r,self.rest)
        blend=(self.weights@world_r.reshape(n,55,9)).reshape(n,len(self.ids),3,3)
        vertices=np.einsum('tvij,tvj->tvi',blend,posed)+self.weights@translations
        probes=[];offset=0
        for ids,mean_ids in zip(self.groups,self.mean_groups):
            patch=vertices[:,offset:offset+len(ids)];offset+=len(ids)
            value=patch[:,mean_ids].mean(1);value[:,1]=patch[...,1].min(1);probes.append(value)
        return np.stack(probes,1),vertices


def run(project,emit,result):
    import torch
    import smplx
    import psutil
    from .paths import MODELS
    from .motion_tools import RESULTS, result_sources, base_result
    from .engines import save_tensor
    from .storage import atomic_json,sha256
    from .compute_policy import worker_threads
    from .result_artifacts import begin_update,finish_update,file_stat
    torch.set_num_threads(worker_threads('correct'))
    project=Path(project)
    cleanup_orphans(project)
    from .mamma_body import body_path
    model=body_path(project)
    model_stamp=file_stat(model)
    body=smplx.SMPLX(str(model),gender='neutral',num_pca_comps=12,flat_hand_mean=False,batch_size=1).eval()
    manifest=json.loads((project/'project.json').read_text(encoding='utf-8'))
    from .correction_settings import effective_options
    from .scene_floor import scene_root,ensure_config,binding,receipt_matches
    group=scene_root(project,manifest)
    if group is not None:ensure_config(group)
    options=effective_options(project,manifest)
    shared=binding(project,manifest,body)
    transform=np.asarray(shared['local_to_shared']) if shared else np.eye(4)
    shared_rotation,shared_translation=transform[:3,:3],transform[:3,3]
    preset=options['preset']
    floor=None if options.get('auto_floor',True) else float(options.get('floor',0))
    if shared:floor=float(shared['floor'])
    fps=float(manifest.get('fps',30));reports={}
    contact_source_stamp=file_stat(project/'hmr4d_results.pt')
    source=torch.load(project/'hmr4d_results.pt',map_location='cpu',weights_only=False,mmap=True)
    logits=source['net_outputs']['static_conf_logits'][0,:,:4]
    probabilities=logits.sigmoid().numpy()
    started=time.monotonic()
    for kind in result_sources(result):
        if not kind.startswith('corrected_'):continue
        base=base_result(kind);filename,key=RESULTS[base]
        sources={name:file_stat(project/name) for name in dict.fromkeys((filename,'hmr4d_results.pt'))}
        if sources['hmr4d_results.pt']!=contact_source_stamp:raise ValueError('Source motion changed during correction. Apply Correction again.')
        record=torch.load(project/filename,map_location='cpu',weights_only=False,mmap=True)
        params=record[key];n=len(params['body_pose'])
        if not n:raise ValueError('Motion contains no frames')
        from .foot_contacts import read_edit
        context={'frames':n,'fps':fps,'sources':sources}
        contact_ranges,contact_revision,contact_warning=read_edit(project,base,context)
        hand_ranges=contact_ranges[2:] if contact_ranges is not None and len(contact_ranges)==4 else None
        contact_ranges=contact_ranges[:2] if contact_ranges is not None else None
        if not options['foot_lock']:contact_ranges=[[],[]]
        if n!=len(probabilities):raise ValueError('Contact and motion frame counts differ')
        if not torch.allclose(params['betas'],params['betas'][:1].expand_as(params['betas']),atol=1e-5,rtol=0):
            raise ValueError('Motion Correction requires one constant actor shape')
        geometry=SoleGeometry(body,params['betas'][0].numpy())
        arrays=[]
        with tempfile.TemporaryDirectory(prefix='.correction-',dir=project) as temporary:
            Path(temporary,'owner.json').write_text(json.dumps({'pid':os.getpid(),'created':psutil.Process().create_time()}))
            try:
                def array(name,shape):
                    a=np.lib.format.open_memmap(Path(temporary)/(name+'.npy'),mode='w+',dtype=np.float32,shape=shape);arrays.append(a);return a
                roots=array('root',(n,3));quats=array('quat',(n,22,4));probes=array('sole',(n,4,3))
                out_roots=array('output-root',(n,3));out_quats=array('output-quat',(n,22,4))
                for start in range(0,n,128):
                    end=min(n,start+128)
                    rv=torch.cat((params['global_orient'][start:end,None],params['body_pose'][start:end].reshape(-1,21,3)),1).numpy()
                    if shared:
                        matrices=Rotation.from_rotvec(rv.reshape(-1,3)).as_matrix().reshape(-1,22,3,3)
                        matrices[:,0]=shared_rotation@matrices[:,0]
                        quats[start:end]=Rotation.from_matrix(matrices.reshape(-1,3,3)).as_quat().reshape(-1,22,4)[...,[3,0,1,2]]
                    else:quats[start:end]=Rotation.from_rotvec(rv.reshape(-1,3)).as_quat().reshape(-1,22,4)[...,[3,0,1,2]]
                    roots[start:end]=(params['transl'][start:end].numpy()+geometry.rest[0])@shared_rotation.T+shared_translation
                    probes[start:end]=geometry.sample(roots[start:end],quats[start:end])[0]
                _,_,report=correct_arrays(geometry.rest[:22],roots,quats,probes,probabilities,fps,floor,
                    output=(out_roots,out_quats),emit=emit,preset=preset,contact_ranges=contact_ranges,
                    pelvis_offset=options['pelvis_offset'],mode=options['mode'],foot_enabled=options['foot_lock'])
                from .hand_lock import apply as apply_hands
                report['hand_lock']=apply_hands(geometry.rest[:22],roots,quats,out_roots,out_quats,geometry.palm_offsets,fps,options['mode'],report['floor'],preset,hand_ranges,options['hand_lock'],emit)
                report['warning']=' '.join(filter(None,(report['warning'],report['hand_lock']['warning'])))
                report['contact_revision']=contact_revision
                if shared:report['shared_floor']=shared
                if contact_warning:report['warning']=' '.join(filter(None,(report['warning'],contact_warning)))
                pose=array('pose',(n,22,3));translation=array('translation',(n,3))
                after=array('after-sole',(n,4,3))
                for start in range(0,n,128):
                    end=min(n,start+128)
                    pose[start:end]=Rotation.from_quat(out_quats[start:end,...,[1,2,3,0]].reshape(-1,4)).as_rotvec().reshape(-1,22,3)
                    translation[start:end]=(out_roots[start:end]-shared_translation)@shared_rotation-geometry.rest[0]
                    after[start:end]=geometry.sample(out_roots[start:end],out_quats[start:end])[0]
                before=probes
                contact_plan=prepare_contacts(probes,probabilities,fps,floor,preset,contact_ranges,options['mode'])
                active,anchors=contact_plan[:2]
                from .contact_trajectory import support_weights
                applied_weights=support_weights(probes,active,fps)
                for prefix,values in (('before',before),('after',after)):
                    h=values[...,1].reshape(n,2,2).min(2)
                    levels=anchors[...,1] if options['mode']=='multi_level' else np.full_like(h,report['floor'])
                    report['contact_sole_height_'+prefix+'_m']=float(np.abs(h[active]-levels[active]).mean()) if active.any() else 0.
                report['sole_below_floor_before_m']=float(np.maximum(report['floor']-before[...,1],0).mean()) if options['mode']=='flat' else None
                report['sole_below_floor_after_m']=float(np.maximum(report['floor']-after[...,1],0).mean()) if options['mode']=='flat' else None
                # A mean across all contacts concealed centimetre-scale lifts.
                # Inspect the actual skinned support surface at full-strength
                # frames separately from intentional preset boundary blends.
                report.update(sole_metrics(before,after,probabilities,active,anchors,report['segments'],fps,preset,report['floor'],options['mode']))
                locked=contact_strength(report['segments'],n,fps,preset)>=.999
                if locked.any() and report['full_lock_surface_error_max_m']>.01:
                    message='Some foot locks remain unresolved; inspect the result.'
                    if message not in report['warning']:report['warning']=' '.join(filter(None,(report['warning'],message)))
                if options['pelvis_offset'] and locked.any() and report['full_lock_surface_error_max_m']>.01:
                    report['warning']+=' Pelvis adjustment: maximum full-lock sole error %.1f cm; reduce the height offset to improve contact.' % (100*report['full_lock_surface_error_max_m'])
                if report['contact_sole_penetration_over_1cm']:
                    report['warning']=' '.join(filter(None,(report['warning'],
                        'Some contact sole patches penetrate the correction floor by over 1 cm; inspect the result.')))
                if (report['noncontact_sole_penetration_max_m'] or 0)>.01:
                    report['warning']=' '.join(filter(None,(report['warning'],
                        'A non-contact foot penetrates the correction floor; check floor height and contact timing.')))
                output=dict(params)
                output['body_pose']=params['body_pose'].clone()
                for joint in (1,2,4,5,7,8) + ((13,14,16,17,18,19,20,21) if report['hand_lock']['contacts'] else ()):
                    output['body_pose'][:,3*(joint-1):3*joint]=torch.from_numpy(np.asarray(pose[:,joint]).copy())
                output['transl']=torch.from_numpy(np.asarray(translation).copy())
                begin_update(project,kind,'correct')
                save_tensor({'smpl_params_global':output,'source':base,'method':report['method']},project/RESULTS[kind][0])
                report['output_stat']=file_stat(project/RESULTS[kind][0])
                del output
            finally:
                for a in arrays:a.flush();a._mmap.close()
        segments=report.pop('segments',[])
        automatic=report.pop('automatic_ranges',[[],[]])
        from .contact_trajectory import diagnostic_plan
        np.savez_compressed(project/f'{kind}_contacts.npz',segments=np.asarray([
            [s['foot'],s['start'],s['end'],*s['anchor']] for s in segments],dtype=np.float64).reshape(-1,6),
            automatic=np.asarray([[foot,s,e] for foot in range(2) for s,e in automatic[foot]],dtype=np.int64).reshape(-1,3),
            **diagnostic_plan(anchors,applied_weights,segments),planner_version=VERSION)
        report['segment_count']=len(segments)
        report['contacts_stat']=file_stat(project/f'{kind}_contacts.npz')
        from .foot_contacts import contact_revision as saved_contact_revision
        current=json.loads((project/'project.json').read_text(encoding='utf-8'))
        if (any(file_stat(project/name)!=stamp for name,stamp in sources.items()) or file_stat(model)!=model_stamp
                or saved_contact_revision(project,base)!=contact_revision or current.get('fps',30)!=fps
                or effective_options(project,current)!=options or not receipt_matches(project,current,shared)):
            raise ValueError('Correction inputs changed while processing. Apply Correction again.')
        atomic_json(project/f'{kind}_sources.json',sources)
        report['source']=base;report['options']=options
        report['model_stat']=model_stamp
        report['backend_sha256']=sha256(Path(backend().__file__))
        atomic_json(project/f'{kind}_correction.json',report)
        finish_update(project,kind,'correct')
        reports[kind]=report
    atomic_json(project/'correction_report.json',{'results':reports,'elapsed':time.monotonic()-started,'version':VERSION})
    return project/RESULTS[result][0]
