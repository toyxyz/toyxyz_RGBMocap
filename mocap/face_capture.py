"""Track expressions in a prepared clip, separately for each existing person."""
import argparse
from contextlib import ExitStack
from pathlib import Path
from time import monotonic

from .face_data import CHANNELS, FaceArchive, POINTS, capture_inputs, fingerprint, write


def crop(image, box, cv2, np):
    if box is None:
        return image, (0,0,image.shape[1],image.shape[0])
    x1,y1,x2,y2 = box
    height = y2-y1
    side = round(max(64, .44*height, min(.9*(x2-x1), .59*height)))
    left,top = np.rint(np.array(((x1+x2)/2,y1+.132*height))-side/2).astype(int)
    h,w = image.shape[:2]
    a,b,c,d = max(0,left),max(0,top),min(w,left+side),min(h,top+side)
    if c<=a or d<=b:
        return None, None
    patch = cv2.copyMakeBorder(image[b:d,a:c],b-top,top+side-d,a-left,left+side-c,cv2.BORDER_REPLICATE)
    return patch, (left,top,side,side)


def read_face(patch, placement, detect, mp, cv2, np):
    if patch is None:
        return None
    ratio=384/max(patch.shape[:2])
    size=(max(1,round(patch.shape[1]*ratio)),max(1,round(patch.shape[0]*ratio)))
    rgb=cv2.cvtColor(cv2.resize(patch,size),cv2.COLOR_BGR2RGB)
    result=detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
    if not result.face_landmarks or not result.face_blendshapes:
        return None
    raw=np.asarray([(p.x,p.y,p.z) for p in result.face_landmarks[0]],dtype=np.float32)
    if raw.shape!=(478,3) or not np.isfinite(raw).all():
        return None
    x,y,w,h=placement
    return raw*[w,h,w]+[x,y,0], result.face_blendshapes[0]


def square_at(image, center, side, np):
    h,w=image.shape[:2]
    side=min(int(side),h,w)
    x,y=np.clip(np.rint(np.asarray(center)-side/2).astype(int),[0,0],[w-side,h-side])
    return image[y:y+side,x:x+side],(x,y,side,side)


def find_unboxed(image, previous, stream, scout, timestamp, mp, cv2, np):
    h,w=image.shape[:2]
    length=.49*min(h,w)
    if previous is not None:
        center, size=previous
        patch,place=square_at(image,center,max(115,3.4*size),np)
        result=read_face(patch,place,lambda image:stream.detect_for_video(image,timestamp),mp,cv2,np)
        if result is not None and np.linalg.norm(result[0][1,:2]-center)<max(55,1.3*size):
            return result
    # Older single-person projects have no saved body boxes. Search overlapping
    # source crops to recover small faces after a cut, then follow the face crop.
    tiles=[(fx*w,fy*h) for fy in (.18,.36,.55,.74,.9) for fx in (.22,.50,.78)]
    focus=previous[0] if previous is not None else np.array([w*.5,h*.4])
    tiles.sort(key=lambda xy:np.linalg.norm(focus-xy))
    found=[]
    for center in tiles:
        patch,place=square_at(image,center,length,np)
        result=read_face(patch,place,scout.detect,mp,cv2,np)
        if result is None:
            continue
        nose=result[0][1,:2]
        width=np.ptp(result[0][:468,0])
        distance=np.linalg.norm(focus-nose)
        if previous is None or distance<max(length*.8,previous[1]*2.5):
            found.append((width/(1+distance/max(length,1)),result))
        if previous is not None and found:
            break
    return max(found,key=lambda pair:pair[0])[1] if found else None


def belongs(nose, own, others, np):
    if own is None:
        return True
    x,y = nose[:2]
    x1,y1,x2,y2 = own
    width,height = x2-x1,y2-y1
    if not (x1-.08*width<=x<=x2+.08*width and y1-.1*height<=y<=y1+.38*height):
        return False

    def distance(box):
        a,b,c,d=box
        return np.hypot(x-(a+c)/2,y-(b+.11*(d-b)))
    return all(distance(own)<=distance(other)+.07*height for other in others)


def duplicates(landmarks, np):
    from itertools import combinations
    rejected = set()
    for (a_id,a),(b_id,b) in combinations(landmarks.items(),2):
        a,b = a[:468,:2],b[:468,:2]
        l1,l2 = a.min(0),b.min(0)
        h1,h2 = a.max(0),b.max(0)
        overlap = np.prod(np.maximum(0,np.minimum(h1,h2)-np.maximum(l1,l2)))
        union = np.prod(h1-l1)+np.prod(h2-l2)-overlap
        if overlap/max(union,1)>=.55 and np.linalg.norm(a[1]-b[1])<=.22*max(h1[0]-l1[0],h2[0]-l2[0]):
            rejected.update((a_id,b_id))
    return rejected


def capture(project, requested, emit):
    import numpy as np

    from . import face_runtime
    from .face_surface import expression_offsets, read_canonical, surface_binding

    project = project.resolve()
    media,info,tracks = capture_inputs(project)
    identity = fingerprint(project,media,info,tracks)
    wanted = [key for key in tracks if not requested or key in requested]
    if not wanted:
        raise ValueError('Check a person in the People list to track their face.')
    try:
        previous = FaceArchive(project)
    except (ValueError,OSError,KeyError):
        previous = None
    if previous is not None and set(wanted) <= set(previous.ids):
        return len(wanted), True

    count, fps = info['frames'], info['fps']
    # Analyze the whole checked overlapping group together if a face is
    # missing. Independently labeling the same face in separate runs can mix IDs.
    actors = sorted(set(wanted) | (set(previous.ids) if previous else set()))
    boxes = {}
    # Unchecked neighbors still constrain identity when their faces are nearby.
    for actor, source in tracks.items():
        if isinstance(source, Path):
            with np.load(source, allow_pickle=False) as track:
                boxes[actor] = (track['boxes'], track['observed'])
        elif isinstance(source, tuple):
            from .face_tracks import read_projected
            boxes[actor] = read_projected(source, info)
        if actor in boxes and (boxes[actor][0].shape != (count, 4) or boxes[actor][1].shape != (count,)):
            raise ValueError('Person tracking has a different frame count.')
    if all(actor in boxes for actor in wanted) and not any(boxes[actor][1].any() for actor in wanted):
        raise ValueError('No saved face search regions for the checked people. Prepare or reconstruct their motion first.')
    folder = face_runtime.setup(emit)
    import cv2
    import mediapipe as mp
    cv2.setNumThreads(2)
    canonical = read_canonical(folder/'canonical.obj')
    binding, indices, weights = surface_binding(info, canonical)
    task = mp.tasks.vision
    options = task.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(folder/'face_landmarker.task'),
                                         delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=task.RunningMode.VIDEO,num_faces=1,output_face_blendshapes=True)
    geometry = np.zeros((len(actors),count,len(POINTS),3),np.float32)
    values = np.zeros((len(actors),count,52),np.float32)
    valid=np.zeros((len(actors),count),bool)
    channels=CHANNELS
    with ExitStack() as stack:
        predict = [stack.enter_context(task.FaceLandmarker.create_from_options(options)) for _ in actors]
        scout = None
        latest = {}
        if any(actor not in boxes for actor in actors):
            still = task.FaceLandmarkerOptions(base_options=options.base_options,
                       running_mode=task.RunningMode.IMAGE,num_faces=1,output_face_blendshapes=True)
            scout=stack.enter_context(task.FaceLandmarker.create_from_options(still))
        reader=cv2.VideoCapture(str(media))
        stack.callback(reader.release)
        for frame in range(count):
            success,image=reader.read()
            if not success:
                raise ValueError(f'Could not decode face tracking frame {frame+1} of {count}.')
            available={k:track[frame] for k,(track,seen) in boxes.items()
                       if seen[frame] and np.isfinite(track[frame]).all()}
            candidates={}
            for slot,actor in enumerate(actors):
                if actor in boxes and actor not in available:
                    continue
                own=available.get(actor)
                timestamp=round(frame*1000/fps)
                if own is None:
                    result=find_unboxed(image,latest.get(actor),predict[slot],scout,timestamp,mp,cv2,np)
                else:
                    patch, placement=crop(image,own,cv2,np)
                    result=read_face(patch,placement,lambda rgb:predict[slot].detect_for_video(rgb,timestamp),
                                     mp,cv2,np)
                if result is None:
                    continue
                raw, blendshapes=result
                if own is None:
                    latest[actor]=(raw[1,:2],np.ptp(raw[:468,0]))
                if not belongs(raw[1],own,[v for k,v in available.items() if k!=actor],np):
                    continue
                scores={item.category_name:item.score for item in blendshapes}
                neutral=scores.pop('_neutral',None)
                scores['tongueOut']=0.
                if neutral is None or set(scores)!=set(channels):
                    raise ValueError('Face Landmarker returned a different blendshape schema.')
                geometry[slot,frame]=raw[POINTS]
                values[slot,frame]=np.clip([scores[name] for name in channels],0,1)
                candidates[slot]=raw
            rejected=duplicates(candidates,np)
            for slot in candidates.keys()-rejected:
                valid[slot,frame]=True
            if frame%15==0 or frame==count-1:
                emit('progress',current=frame+1,total=count,message='Tracking faces')
    offsets=np.zeros(geometry.shape[:-1]+(2,),dtype=np.float32)
    channel = {key:i for i,key in enumerate(channels)}
    for i,mask in enumerate(valid):
        if not mask.any():
            continue
        score=values[i,mask]@np.array([1.25 if c=='jawOpen' else .45 if c=='mouthPucker' else
            .25 if c.startswith('eyeBlink') else .18 if c.startswith('mouthSmile') else 0
            for c in channels])
        best=np.argmin(score)
        samples=geometry[i,mask]
        offsets[i,mask]=expression_offsets(samples,samples[best],canonical)
    for i, mask in enumerate(valid):
        emit('info',message=f'Face tracking Person {actors[i]}: {int(mask.sum())}/{count} frames detected.')
    media,info,tracks=capture_inputs(project)
    if fingerprint(project,media,info,tracks)!=identity:
        raise ValueError('The input or Person tracks changed during face capture.')
    write(project,identity=identity,ids=np.asarray(actors,dtype=np.int32),channels=np.asarray(channels),
          valid=valid,values=values,offsets=offsets,indices=indices,weights=weights,binding_type=binding)
    return len(wanted),False


def main(arguments, emit):
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--person',type=int,action='append',default=[])
    args=parser.parse_args(arguments)
    began=monotonic()
    try:
        count,cached=capture(args.project,set(args.person),emit)
    except Exception as error:
        import traceback
        traceback.print_exc()
        emit('failed',stage='face_track',message=str(error))
        return 1
    emit('face_ready',project=str(args.project),count=count)
    emit('complete',stage='face_track',output=str(args.project),elapsed=monotonic()-began,cached=cached)
    return 0
