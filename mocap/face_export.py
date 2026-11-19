"""Write face animation sidecars beside exported motion."""
import csv
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np

from .face_data import FaceArchive, manifest, root_for
from .face_smoothing import smooth_face_values
from .face_neutral import neutral_frames, expression_values
from .people_store import load_people, safe_child


# iOS Live Link Face ARKit recording column order.
LIVE_LINK_CHANNELS = '''eyeBlinkLeft eyeLookDownLeft eyeLookInLeft eyeLookOutLeft
eyeLookUpLeft eyeSquintLeft eyeWideLeft eyeBlinkRight eyeLookDownRight eyeLookInRight
eyeLookOutRight eyeLookUpRight eyeSquintRight eyeWideRight jawForward jawRight jawLeft
jawOpen mouthClose mouthFunnel mouthPucker mouthRight mouthLeft mouthSmileLeft
mouthSmileRight mouthFrownLeft mouthFrownRight mouthDimpleLeft mouthDimpleRight
mouthStretchLeft mouthStretchRight mouthRollLower mouthRollUpper mouthShrugLower
mouthShrugUpper mouthPressLeft mouthPressRight mouthLowerDownLeft mouthLowerDownRight
mouthUpperUpLeft mouthUpperUpRight browDownLeft browDownRight browInnerUp browOuterUpLeft
browOuterUpRight cheekPuff cheekSquintLeft cheekSquintRight noseSneerLeft noseSneerRight
tongueOut'''.split()
LIVE_LINK_ROTATIONS = ('HeadYaw', 'HeadPitch', 'HeadRoll', 'LeftEyeYaw', 'LeftEyePitch',
                       'LeftEyeRoll', 'RightEyeYaw', 'RightEyePitch', 'RightEyeRoll')
LIVE_LINK_HEADER = ('Timecode', 'BlendShapeCount',
                    *(name[0].upper() + name[1:] for name in LIVE_LINK_CHANNELS),
                    *LIVE_LINK_ROTATIONS)


def plan(request, stage):
    from .blend_support import read
    payload=read(request)
    if stage=='save_blend_scene':
        if not payload.get('export_face'):
            return []
        root=Path(payload['root']).resolve()
        from .scene_export import scene_take_rows
        state=load_people(root)
        rows=[dict(project=str(safe_child(root,take['project'])),target=payload['target'],
                   rig_name=person['name'],export_face=True,
                   face_format=payload.get('face_format','txt'),smooth_face=payload.get('smooth_face',False))
              for person,take in scene_take_rows(state,payload['people'],payload.get('takes'))]
        origin=float(manifest(root)['start'])
        leading=False
    elif isinstance(payload,dict):
        rows=payload['takes']
        if not any(row.get('export_face') for row in rows):
            return []
        timed=sorted(((float(manifest(Path(row['project']))['start']),row) for row in rows),
                     key=lambda entry:entry[0])
        origin=timed[0][0]
        rows=[row for _,row in timed]
        leading=bool(rows[0].get('prepend_tpose'))
    else:
        rows=payload
        origin,leading=None,False
    return [(row,origin,leading) for row in rows if row.get('export_face')]


def expression_frames(archive, project, state, filtered):
    for frame in range(int(state['frames'])):
        slot, sample, source = archive.resolve(project, frame, state=state)
        weights = None
        if sample >= 0:
            weights = archive.values[slot, sample] if filtered is None else filtered[sample]
        yield frame, sample, source, weights


def write_txt(stream, archive, project, state, offset, leading, name, filtered):
    fps=float(state['fps'])
    stream.write('# rgbmocap_face_v1\n')
    stream.write('# channels=ARKit52; tongueOut=unsupported_zero\n')
    stream.write('# frame_origin=0; timestamps=seconds; missing_values=empty\n')
    stream.write('# state=tracked,held,missing,reference\n')
    stream.write('# person='+json.dumps(name,ensure_ascii=False)+'\n')
    stream.write(f'# fps={fps:.9f}; capture_fps={archive.info["fps"]}; capture_start={archive.info["start"]:.9f}\n')
    writer=csv.writer(stream,delimiter='\t',lineterminator='\n')
    writer.writerow(['export_frame','export_seconds','source_frame','source_seconds',
                     'state','last_detected_source_frame',*archive.channels])
    if leading:
        index=offset-1
        writer.writerow([index,f'{index/fps:.9f}','','','reference','',*([0.]*52)])
    for frame, sample, source, weights in expression_frames(archive, project, state, filtered):
        label='missing' if sample<0 else 'tracked' if sample==source else 'held'
        coeff=['']*52 if weights is None else [f'{v:.7f}' for v in weights]
        writer.writerow([frame+offset,f'{(frame+offset)/fps:.9f}',source,
            f'{state["start"]+frame/fps:.9f}',label,sample if sample>=0 else '',*coeff])


def live_link_timecode(frame, fps):
    # Unreal infers 30 or 60 fps from the frame field. Express export times
    # on a 30 fps clock, with subframes so other export rates keep their timing.
    seconds, subframes = divmod(round(frame * 30_000 / fps), 30_000)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    frames, fraction = divmod(subframes, 1000)
    return f'{hours:02}:{minutes:02}:{seconds:02}:{frames:02}.{fraction:03}'


def write_live_link_csv(stream, archive, project, state, offset, leading, filtered):
    fps = float(state['fps'])
    channels = [archive.channels.index(name) for name in LIVE_LINK_CHANNELS]
    empty = (0,) * len(channels)
    rotations = (0,) * len(LIVE_LINK_ROTATIONS)
    count = len(channels) + len(rotations)
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(LIVE_LINK_HEADER)
    if leading:
        writer.writerow([live_link_timecode(offset - 1, fps), count, *empty, *rotations])
    for frame, _, _, weights in expression_frames(archive, project, state, filtered):
        values = empty if weights is None else [f'{weights[index]:.7f}' for index in channels]
        writer.writerow([live_link_timecode(frame + offset, fps), count, *values, *rotations])


def write_sidecar(destination, archive, project, state, offset, leading, name, *, filtered=None):
    temp=destination.with_name('.'+uuid4().hex+'.partial')
    try:
        with temp.open('w',encoding='utf8',newline='') as stream:
            if destination.suffix == '.csv':
                write_live_link_csv(stream, archive, project, state, offset, leading, filtered)
            else:
                write_txt(stream, archive, project, state, offset, leading, name, filtered)
        for version in range(1,10000):
            target=destination if version==1 else destination.with_name(f'{destination.stem}_{version}{destination.suffix}')
            try:
                if os.name=='nt':
                    temp.rename(target)
                else:
                    os.link(temp,target)
                return target
            except FileExistsError:
                continue
        raise OSError('Too many existing face export files.')
    finally:
        temp.unlink(missing_ok=True)


def publish(rows, emit):
    archives={}
    smoothed={}
    for ordinal,(row,origin,combined_lead) in enumerate(rows,1):
        project=Path(row['project'])
        target=Path(row['target'])
        if not target.is_file():
            continue
        try:
            state=manifest(project)
            root=root_for(project,state)
            if root not in archives:
                archives[root]=FaceArchive(root)
            archive=archives[root]
            person=state.get('person_id',1)
            if person not in archive.index:
                raise ValueError('Run Track Face for this person first.')
            leading=bool(row.get('prepend_tpose')) if origin is None else combined_lead and abs(state['start']-origin)<1e-7
            offset=int(leading) if origin is None else round((state['start']-origin)*state['fps'])+int(combined_lead)
            suffix='_face' if origin is None or len(rows)==1 else f'_face_{ordinal:02d}'
            extension = '.csv' if row.get('face_format') == 'livelink_csv' else '.txt'
            destination = target.with_name(target.stem + suffix + extension)
            calibration = neutral_frames(archive)
            filtered = None
            smoothing = bool(row.get('smooth_face'))
            if smoothing or person in calibration:
                key = root, person, smoothing
                if key not in smoothed:
                    smoothed[key] = expression_values(archive, archive.index[person],
                                                       smooth=smoothing, frames=calibration)
                filtered = smoothed[key]
            written = write_sidecar(destination, archive, project, state, offset, leading,
                                    row.get('rig_name','Person '+str(person)), filtered=filtered)
            emit('info',message=f'Exported face {extension[1:].upper()}: {written}')
        except (OSError,ValueError,KeyError,IndexError) as error:
            emit('info',message=f'Face export for {row.get("rig_name","selected person")}: {error}')
