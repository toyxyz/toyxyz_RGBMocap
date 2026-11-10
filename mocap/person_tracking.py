"""Disk-streamed multi-person discovery with BoT-SORT and small OSNet ReID.

All CUDA, detector and appearance work runs in a separate processing worker.
Only observations are saved: long absences become separate reconstruction takes.
"""
from __future__ import annotations
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sqlite3
from types import SimpleNamespace
import uuid

from .paths import ROOT, UPSTREAM
from .people_store import (VERSION, load_people, media_receipt, person_color, safe_child,
                           save_people, visible_runs)
from .storage import atomic_json, sha256


class AppearanceEncoder:
    def __init__(self, checkpoint):
        import torch
        spec = importlib.util.spec_from_file_location('rgb_person_osnet', UPSTREAM/'OSNet/osnet.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.torch = torch
        self.model = module.osnet_x0_25(num_classes=4101, pretrained=False).eval().cuda()
        self.model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        self.mean = torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]

    def inference(self, frame, boxes):
        import cv2
        import numpy as np
        torch = self.torch
        result = []
        def run(first, count):
            crops = []
            for box in boxes[first:first+count]:
                x1,y1,x2,y2 = np.asarray(box[:4]).astype(int)
                x1,y1 = max(0,x1),max(0,y1)
                x2,y2 = min(frame.shape[1],max(x1+1,x2)),min(frame.shape[0],max(y1+1,y2))
                crop = frame[y1:y2,x1:x2]
                if not crop.size:
                    crop = np.zeros((256,128,3),np.uint8)
                crops.append(cv2.resize(crop,(128,256))[:,:,::-1].copy())
            tensor = torch.from_numpy(np.stack(crops)).permute(0,3,1,2).cuda().float().div_(255)
            with torch.inference_mode():
                embedding = torch.nn.functional.normalize(self.model((tensor-self.mean)/self.std), dim=1)
            return embedding.cpu().numpy()
        # A small model; bounded batches and retry outside the OOM exception scope.
        from .memory_mode import effective_mode
        batch = 4 if effective_mode() == 'low_memory' else 16
        first = 0
        while first < len(boxes):
            failed = False
            try:
                values = run(first,min(batch,len(boxes)-first))
            except torch.cuda.OutOfMemoryError:
                if batch == 1:
                    raise RuntimeError('Not enough GPU memory for person identification.') from None
                failed = True
            if failed:
                batch = max(1,batch//2)
                gc.collect();torch.cuda.empty_cache()
                continue
            result.append(values)
            first += len(values)
        return np.concatenate(result) if result else np.empty((0,512),np.float32)


class IdentityRegistry:
    """Reconnect disjoint short tracker IDs conservatively, with bounded galleries."""
    def __init__(self, fps, max_people=256):
        self.fps = fps
        self.max_people = max_people
        self.aliases = {}
        self.people = {}
        self.decisions = []

    def identify(self, track_id, frame, box, feature, occupied=(), color=None):
        import numpy as np
        if track_id in self.aliases:
            return self.aliases[track_id]
        scores = []
        feature = np.asarray(feature)
        for ident, person in self.people.items():
            gap = frame-person['last']
            if ident in occupied or gap <= 0 or gap > self.fps*20:
                continue
            similarity = max(float(feature @ ref) for ref in person['gallery'])
            # A briefly occluded person may be detected as only a head/torso.
            # In that case full-body OSNet appearance alone is unreliable. Use
            # local color only with short-gap geometry AND a clear margin over
            # every other known identity, including currently occupied people.
            local_recovery=False
            own=competing=distance=height=None
            if color is not None and box is not None and person.get('box') is not None and gap<=self.fps*.5:
                old=np.asarray(person['box']);box=np.asarray(box)
                distance=float(np.linalg.norm((old[:2]+old[2:]-box[:2]-box[2:])*.5))
                height=float(max(1,old[3]-old[1],box[3]-box[1]))
                anchor=person.get('color_anchor')
                own=float(color@anchor) if anchor is not None else 0
                competing=max((float(color@other['color_anchor']) for other_id,other in self.people.items()
                               if other_id!=ident and other.get('color_anchor') is not None),default=1)
                local_recovery=distance<=height*.65 and own>=.45 and own-competing>=.12 and similarity>=.38
            self.decisions.append(dict(frame=frame,tracker=track_id,candidate=ident,similarity=similarity,
                gap=gap,color=own,competing_color=competing,distance=distance,height=height,recovered=local_recovery))
            self.decisions=self.decisions[-128:]
            if similarity >= .88 or local_recovery:
                scores.append((similarity,ident))
        scores.sort(reverse=True)
        if scores and (len(scores)==1 or scores[0][0]-scores[1][0] >= .08):
            ident = scores[0][1]
        else:
            if len(self.people) >= self.max_people:
                raise RuntimeError('Too many person tracks. Trim the video into shorter scenes and retry.')
            ident = len(self.people)+1
            self.people[ident] = dict(last=frame,gallery=[feature.copy()],count=0,
                colors=[color.copy()] if color is not None else [],box=box,
                color_anchor=color.copy() if color is not None else None,anchor_confirmed=False)
        self.aliases[track_id] = ident
        return ident

    def observe(self, ident, frame, feature, clear=True, color=None, box=None):
        import numpy as np
        person = self.people[ident]
        person['last'] = frame
        person['count'] += 1
        if box is not None:person['box']=np.asarray(box).copy()
        if clear and color is not None and not person['anchor_confirmed']:
            person['color_anchor']=color.copy();person['anchor_confirmed']=True
        if clear and frame-person.get('gallery_frame',-10000) >= max(1,round(self.fps/2)):
            person['gallery'].append(np.asarray(feature).copy())
            if len(person['gallery'])>8:person['gallery'] = person['gallery'][:1]+person['gallery'][-7:]
            person['gallery_frame'] = frame
            if color is not None:
                person['colors'].append(color.copy());person['colors']=person['colors'][-8:]


def color_signature(frame,box):
    """Bounded upper-body color cue; supplementary to learned appearance."""
    import cv2
    import numpy as np
    x1,y1,x2,y2=box[:4];width=x2-x1;height=y2-y1
    x1,y1,x2,y2=map(int,(x1+width*.2,y1+height*.05,x2-width*.2,y1+height*.5))
    crop=frame[max(0,y1):min(frame.shape[0],y2),max(0,x1):min(frame.shape[1],x2)]
    if not crop.size:return np.zeros(144,np.float32)
    hsv=cv2.cvtColor(cv2.resize(crop,(64,96)),cv2.COLOR_BGR2HSV)
    hist=cv2.calcHist([hsv],[0,1],None,[18,8],[0,180,0,256]).reshape(-1)
    return np.sqrt(hist/max(1,float(hist.sum())))


def _link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.samefile(source):
            return
        if target.stat().st_size==source.stat().st_size and sha256(target)==sha256(source):
            return  # A prior cross-volume or permission fallback made an ordinary copy.
        raise RuntimeError('Existing person media does not match the source.')
    try:
        os.link(source,target)
    except OSError:
        shutil.copy2(source,target)


def discover(project, emit):
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    import ultralytics
    import ultralytics.nn.tasks as tasks
    from ultralytics.trackers.bot_sort import BOTSORT
    from .models import BY_KEY
    project = Path(project)
    manifest = json.loads((project/'project.json').read_text(encoding='utf-8'))
    from .cache_compat import discovery_identity
    digest = hashlib.sha256(json.dumps([media_receipt(project),str(torch.__version__),
        torch.version.cuda,cv2.__version__,ultralytics.__version__,torch.cuda.get_device_name(),
        discovery_identity(Path(__file__)),sha256(UPSTREAM/'OSNet/osnet.py'),
        *[sha256(BY_KEY[key].path) for key in ('yolo','person_reid')]]).encode()).hexdigest()
    try:old = load_people(project) if (project/'people.json').exists() else None
    except (OSError,ValueError,KeyError):old = None
    if old and old.get('fingerprint') == digest:
        if all((safe_child(project,p['preview'])/'tracking.npz').is_file() for p in old['people']):
            emit('people_found',project=str(project),count=len(old['people']),cached=True)
            return project/'people.json'
    def load(weight):
        selected = Path(weight).resolve()
        if selected != BY_KEY['yolo'].path.resolve():
            raise ValueError('Unexpected person detector checkpoint')
        return torch.load(selected,map_location='cpu',weights_only=False),str(selected)
    tasks.torch_safe_load = load
    detector = YOLO(str(BY_KEY['yolo'].path))
    encoder = AppearanceEncoder(BY_KEY['person_reid'].path)
    args = SimpleNamespace(track_high_thresh=.5,track_low_thresh=.1,new_track_thresh=.6,
        track_buffer=60,match_thresh=.8,gmc_method='none',with_reid=True,
        proximity_thresh=.5,appearance_thresh=.25,fuse_score=True)
    tracker = BOTSORT(args,frame_rate=manifest['fps'])
    tracker.encoder = encoder
    registry = IdentityRegistry(manifest['fps'])
    generation = 'people/'+uuid.uuid4().hex
    folder = safe_child(project,generation)
    folder.mkdir(parents=True)
    capture = cv2.VideoCapture(str(project/'input.mp4'))
    db = sqlite3.connect(folder/'observations.sqlite')
    db.execute('CREATE TABLE observations(person INTEGER, frame INTEGER, x1 REAL,y1 REAL,x2 REAL,y2 REAL, score REAL, PRIMARY KEY(person,frame))')
    people = {}
    try:
        if not capture.isOpened():
            raise RuntimeError('Cannot open the video for person detection.')
        for index in range(manifest['frames']):
            ok,frame = capture.read()
            if not ok:
                raise RuntimeError(f'Cannot read video frame {index}.')
            prediction = detector.predict(frame,device=0,conf=.1,classes=0,verbose=False,max_det=128)[0]
            # Upstream calls this encoder only for high-confidence candidates.
            tracks = tracker.update(prediction.boxes.cpu().numpy(),frame)
            lookup = {int(t.track_id):t for t in tracker.tracked_stracks}
            occupied = set()
            # Existing identities must reserve their ID before new tracklets reconnect.
            occupied.update(registry.aliases[int(row[4])] for row in tracks if int(row[4]) in registry.aliases)
            for row in tracks:
                tid = int(row[4]); tracked = lookup.get(tid)
                if tracked is None or tracked.smooth_feat is None:
                    continue
                feature = tracked.smooth_feat
                color=color_signature(frame,row[:4])
                ident = registry.identify(tid,index,row[:4],feature,occupied,color)
                if ident in people and people[ident].get('last_frame') == index:
                    continue
                occupied.add(ident)
                x1,y1,x2,y2 = np.clip(row[:4],[0,0,0,0],[frame.shape[1],frame.shape[0]]*2)
                if x2-x1 < 2 or y2-y1 < 2:
                    continue
                others = tracks[tracks[:,4] != tid,:4]
                clear = True
                if len(others):
                    intersection = np.maximum(0,np.minimum(others[:,2:],[x2,y2])-np.maximum(others[:,:2],[x1,y1])).prod(1)
                    other_areas=(others[:,2]-others[:,0])*(others[:,3]-others[:,1])
                    clear = bool(np.max(intersection/np.maximum(1,np.minimum((x2-x1)*(y2-y1),other_areas))) < .15)
                registry.observe(ident,index,feature,clear,color,row[:4])
                db.execute('INSERT INTO observations VALUES(?,?,?,?,?,?,?)',(ident,index,float(x1),float(y1),float(x2),float(y2),float(row[5])))
                entry = people.setdefault(ident,dict(id=ident,name=f'Person {ident}',color=person_color(ident),checked=ident==1,
                    first=index,last=index+1,observations=0,preview=f'{generation}/person_{ident:04d}',takes=[]))
                entry['last'] = index+1;entry['last_frame'] = index;entry['observations'] += 1
                quality = float(row[5])*(y2-y1)*(1 if clear else .25)
                if quality > entry.get('_thumbnail_quality',0):
                    entry['_thumbnail_quality'] = quality
                    target = safe_child(project,entry['preview']);target.mkdir(exist_ok=True)
                    crop = frame[int(y1):int(y2),int(x1):int(x2)]
                    scale = min(96/max(1,crop.shape[1]),144/max(1,crop.shape[0]))
                    thumb = cv2.resize(crop,(max(1,round(crop.shape[1]*scale)),max(1,round(crop.shape[0]*scale))))
                    success,encoded = cv2.imencode('.png',thumb)
                    if success:encoded.tofile(str(target/'thumbnail.png'))
            tracker.removed_stracks = tracker.removed_stracks[-128:]
            # Expired short IDs do not need to remain resident for the entire recording.
            live = {int(t.track_id) for t in tracker.tracked_stracks+tracker.lost_stracks}
            registry.aliases = {key:value for key,value in registry.aliases.items() if key in live}
            if index%30 == 0 or index+1 == manifest['frames']:
                db.commit()
                emit('progress',stage='people',current=index+1,total=manifest['frames'],message=f'Finding people: {len(people)} tracks')
        db.commit()
        atomic_json(folder/'identity_decisions.json',registry.decisions)
        selected = publish_discovery(project, manifest, db, people, generation, digest,
                                     'BoT-SORT + OSNet x0.25 MSMT17')
    finally:
        capture.release();db.close()
        del detector,encoder,tracker
        gc.collect();torch.cuda.empty_cache()
    emit('people_found',project=str(project),count=len(selected),cached=False)
    return project/'people.json'


def publish_discovery(project, manifest, db, people, generation, digest, method):
    import numpy as np
    selected = []
    focal = float(np.hypot(manifest['width'],manifest['height']))
    K = np.array([[focal,0,manifest['width']/2],[0,focal,manifest['height']/2],[0,0,1]],np.float32)
    for ident,entry in people.items():
        if entry['observations'] < 3:
            continue
        target = safe_child(project,entry['preview'])
        boxes = np.lib.format.open_memmap(target/'boxes.partial.npy',mode='w+',dtype=np.float32,shape=(manifest['frames'],4))
        seen = np.lib.format.open_memmap(target/'seen.partial.npy',mode='w+',dtype=np.bool_,shape=(manifest['frames'],))
        boxes[:] = np.nan;seen[:] = False
        for row in db.execute('SELECT frame,x1,y1,x2,y2 FROM observations WHERE person=? ORDER BY frame',(ident,)):
            boxes[row[0]] = row[1:];seen[row[0]] = True
        np.savez_compressed(target/'tracking.npz',boxes=boxes,observed=seen,K=K)
        boxes._mmap.close();seen._mmap.close()
        (target/'boxes.partial.npy').unlink();(target/'seen.partial.npy').unlink()
        entry['observation_id']=ident
        entry['id']=len(selected)+1
        entry['name']=f"Person {entry['id']}"
        child = {**manifest,'stages':{},'people_root':os.path.relpath(project,target),'person_id':entry['id'],'person_preview':True}
        from .project_storage import inherit_storage
        inherit_storage(project,target,child)
        atomic_json(target/'project.json',child)
        for name in ('input.mp4','preview.mp4','preview.json'):
            if (project/name).is_file():_link(project/name,target/name)
        entry.pop('last_frame',None);entry.pop('_thumbnail_quality',None)
        selected.append(entry)
    if not selected:
        raise RuntimeError('No stable person was found. Use a clip with a visible person for at least a few frames.')
    # First displayed person is gray and checked even if short-lived ghosts were filtered.
    selected[0]['checked'] = True;selected[0]['color'] = '#b5b5b5'
    state = dict(version=VERSION,input=media_receipt(project),fingerprint=digest,people=selected,
        observations=f'{generation}/observations.sqlite',active=selected[0]['id'],
        method=method,max_gap_seconds=.5)
    save_people(project,state)
    return selected


def prepare_person(project, ident, first, end, result, options, emit):
    """Create independent, camera-preserving takes with shared source media."""
    import numpy as np
    import torch
    from .video import prepare_people_clip
    root = Path(project)
    state = load_people(root)
    person = next((p for p in state['people'] if p['id']==ident),None) if state else None
    if person is None:
        raise ValueError('Person no longer exists. Run Find person again.')
    observation_id=person.get('observation_id',ident)
    manifest = json.loads((root/'project.json').read_text(encoding='utf-8'))
    from .motion_tools import require_supported_project
    motion_tool=require_supported_project(root).key
    first,end = max(0,int(first)),min(manifest['frames'],int(end))
    db = sqlite3.connect(safe_child(root,state['observations']).as_uri()+'?mode=ro',uri=True)
    try:
        indices = (row[0] for row in db.execute('SELECT frame FROM observations WHERE person=? AND frame>=? AND frame<? ORDER BY frame',(observation_id,first,end)))
        runs = visible_runs(indices,manifest['fps'],first,end)
        if not runs:
            raise ValueError(f'{person["name"]} has no visible reconstruction interval in this trim range.')
        projects = []
        for begin,finish in runs:
            # Each output uses only a continuous visible interval. Short holes alone are interpolated.
            target = safe_child(root,person['preview'])/f'takes/{begin:09d}_{finish:09d}'
            if (target/'project.json').exists() and require_supported_project(target).key!=motion_tool:
                raise ValueError('This person take belongs to a different motion model; its files were preserved.')
            # Keep old resampled media and its inference intact. New partial
            # takes use the scene frame grid and cannot mix with those caches.
            if begin!=0 or finish!=manifest['frames']:
                target=target.with_name(target.name+'_scene_v1')
                if (target/'project.json').exists() and require_supported_project(target).key!=motion_tool:
                    raise ValueError('This person take belongs to a different motion model; its files were preserved.')
            shared = prepare_people_clip(root,begin,finish)
            shared_manifest = json.loads((shared/'project.json').read_text(encoding='utf-8'))
            if shared_manifest['frames']!=finish-begin:
                raise ValueError('Person take frame count does not match its scene interval')
            target.mkdir(parents=True,exist_ok=True)
            for name in ('input.mp4','preview.mp4','preview.json'):
                if (shared/name).is_file():_link(shared/name,target/name)
            rows = np.asarray(list(db.execute('SELECT frame,x1,y1,x2,y2 FROM observations WHERE person=? AND frame>=? AND frame<? ORDER BY frame',(observation_id,begin,finish))),np.float32)
            count = shared_manifest['frames']
            times = np.arange(count)*manifest['fps']/shared_manifest['fps']+begin
            boxes = np.column_stack([np.interp(times,rows[:,0],rows[:,j]) for j in range(1,5)]).astype(np.float32)
            observed = np.isin(np.rint(times).astype(int),rows[:,0].astype(int))
            with np.load(safe_child(root,person['preview'])/'tracking.npz') as data:K=data['K']
            tracking = target/'tracking.npz'
            valid=False
            if tracking.exists():
                try:
                    with np.load(tracking,allow_pickle=False) as saved:
                        valid=np.array_equal(saved['boxes'],boxes) and np.array_equal(saved['observed'],observed) and np.array_equal(saved['K'],K)
                except (OSError,ValueError,KeyError):pass
            if not valid:
                temporary=target/'tracking.partial.npz'
                np.savez_compressed(temporary,boxes=boxes,observed=observed,K=K)
                os.replace(temporary,tracking)
            camera = target/'camera.pt'
            if not camera.exists():torch.save({'K':torch.from_numpy(K)},camera)
            path = target/'project.json'
            child = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
            child = {**child, **shared_manifest, 'stages': child.get('stages',{})}
            from .project_storage import inherit_storage
            inherit_storage(shared,target,child)
            child.update(people_root=os.path.relpath(root,target),person_id=ident,person_preview=False,
                person_frames=[begin,finish],person_fingerprint=state['fingerprint'],
                person_media_version=1,pipeline={'model':motion_tool,'result':result},correction=options)
            if motion_tool=='exmr':child['exmr_options']=manifest.get('exmr_options',{})
            atomic_json(path,child)
            relative = target.relative_to(root).as_posix()
            take = dict(project=relative,first=begin,end=finish)
            person['takes'] = [t for t in person['takes'] if t['project']!=relative and
                               (t['first'],t['end'])!=(begin,finish)]+[take]
            projects.append(str(target))
        save_people(root,state)
    finally:
        db.close()
    emit('person_prepared',project=str(root),person_id=ident,projects=projects,result=result)
    return root/'people.json'
