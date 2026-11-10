"""Translate SAM 3.1 masks to the app's standard person observations."""

import argparse
import hashlib
import json
import math
import sqlite3
import uuid
from contextlib import closing
from itertools import chain
from pathlib import Path

from .people_store import load_people, media_receipt, person_color, safe_child
from .storage import sha256


def selected_ids(outputs, points):
    if not points:
        return None
    ids = outputs['out_obj_ids']
    masks = outputs['out_binary_masks']
    selected = set()
    for x, y in points:
        for ident, mask in zip(ids, masks):
            height, width = mask.shape[-2:]
            if bool(mask.reshape(height, width)[int(y * height), int(x * width)]):
                selected.add(int(ident))
                break
        else:
            raise ValueError('A marker did not hit a person matching the prompt. Mark the torso or change the prompt.')
    return selected


def discover(project, prompt, anchor_time, points, emit):
    import cv2
    import numpy as np
    import torch
    from . import sam31_engine
    from .compute_policy import configure_worker_cpu
    from .model_session import set_memory_budget
    from .models import BY_KEY, ready
    from .person_tracking import publish_discovery
    from .sam31_runtime import REVISION, SUPPORTED, activate, location

    manifest = json.loads((project / 'project.json').read_text(encoding='utf-8'))
    if manifest.get('pipeline', {}).get('model', 'gvhmr') not in SUPPORTED:
        raise ValueError('SAM 3.1 discovery is available for ExMR, FlowHMR and GVHMR.')
    count, width, height = (manifest.get(key) for key in ('frames', 'width', 'height'))
    if any(type(value) is not int or value < 1 for value in (count, width, height)):
        raise ValueError('SAM 3.1 needs the prepared working video for this motion model.')
    if not 1 <= count <= 1800:
        raise ValueError('Trim SAM 3.1 discovery to 1,800 working frames or fewer.')
    seed = 0
    if points:
        if not manifest['start'] <= anchor_time < manifest['end']:
            raise ValueError('The marked frame is outside the working clip.')
        # A valid source frame can round just beyond the end of the
        # resampled 30 fps clip. Use the nearest frame that actually exists.
        seed = min(count-1, round((anchor_time - manifest['start']) * manifest['fps']))
    weight = BY_KEY['sam31_people']
    if not ready(weight):
        raise ValueError('Prepare the SAM 3.1 checkpoint in Model Manager.')
    folder = location()
    identity = [media_receipt(project), REVISION, weight.expected_sha256, str(torch.__version__),
                sha256(Path(__file__)), sha256(Path(sam31_engine.__file__)),
                sha256(Path(__file__).with_name('person_tracking.py')), prompt, seed, points]
    digest = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    try:
        cached = load_people(project)
    except (OSError, ValueError, KeyError):
        cached = None
    if cached and cached['fingerprint'] == digest and all(
            (safe_child(project, p['preview']) / 'tracking.npz').is_file() for p in cached['people']):
        emit('people_found', project=str(project), count=len(cached['people']), cached=True)
        return project / 'people.json'

    torch.set_num_threads(configure_worker_cpu('sam31_people'))
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError('SAM 3.1 requires an NVIDIA CUDA GPU with native bfloat16 support.')
    set_memory_budget(*torch.cuda.mem_get_info())
    cv2.setNumThreads(2)
    activate(folder)
    emit('progress', current=0, total=count, message='Loading SAM 3.1')
    predictor = sam31_engine.predictor(weight.path, folder)
    generation = 'people/' + uuid.uuid4().hex
    output = safe_child(project, generation)
    output.mkdir(parents=True)
    people = {}
    processed = set()
    if count < 12 or (seed and min(seed, count-seed) < 12):
        predictor.model.batched_grounding_batch_size = 1
        predictor.model.postprocess_batch_size = 1
    with closing(sqlite3.connect(output / 'observations.sqlite')) as db:
        db.execute('CREATE TABLE observations(person INTEGER, frame INTEGER, x1 REAL,y1 REAL,x2 REAL,y2 REAL, score REAL, PRIMARY KEY(person,frame))')
        session = predictor.handle_request(dict(type='start_session', resource_path=str(project / 'input.mp4'),
                                                offload_video_to_cpu=True))['session_id']
        try:
            response = predictor.handle_request(dict(type='add_prompt', session_id=session,
                                                     frame_index=seed, text=prompt))
            chosen = selected_ids(response['outputs'], points)
            anchor = response if seed and seed == count-1 else None
            del response
            if seed == 0:
                directions = ('forward',)
            elif seed == count-1:
                directions = ('backward',)
            else:
                directions = ('forward', 'backward')
            for direction in directions:
                requests = predictor.handle_stream_request(dict(type='propagate_in_video', session_id=session,
                    start_frame_index=seed, max_frame_num_to_track=count, propagation_direction=direction,
                    evict_cached_frame_outputs=len(directions) == 1))
                if anchor is not None:
                    # A reverse pass starts one frame before the seed.
                    requests = chain((anchor,), requests)
                for record in requests:
                    frame = record['frame_index']
                    if frame in processed:
                        continue
                    processed.add(frame)
                    masks = record['outputs']
                    for ident, score, box in zip(masks['out_obj_ids'], masks['out_probs'], masks['out_boxes_xywh']):
                        ident = int(ident)
                        if chosen is not None and ident not in chosen:
                            continue
                        x, y, w, h = map(float, box)
                        score = float(score)
                        if not all(map(math.isfinite, (x, y, w, h, score))):
                            continue
                        x1, x2 = np.clip([x, x + w], 0, 1) * width
                        y1, y2 = np.clip([y, y + h], 0, 1) * height
                        if x2 - x1 < 2 or y2 - y1 < 2:
                            continue
                        db.execute('INSERT INTO observations VALUES(?,?,?,?,?,?,?)',
                                   (ident, frame, float(x1), float(y1), float(x2), float(y2), score))
                        entry = people.setdefault(ident, dict(id=ident, name='', color=person_color(ident + 1),
                            checked=chosen is not None, first=frame, last=frame+1, observations=0,
                            preview=f'{generation}/person_{ident + 1:04d}', takes=[]))
                        entry['first'] = min(entry['first'], frame)
                        entry['last'] = max(entry['last'], frame+1)
                        entry['observations'] += 1
                        quality = score * (x2-x1) * (y2-y1)
                        if '_sample' not in entry or quality > entry['_thumbnail_quality']:
                            entry['_thumbnail_quality'] = quality
                            entry['_sample'] = (frame, int(x1), int(y1), int(x2), int(y2))
                    if len(processed) % 15 == 0 or len(processed) == count:
                        emit('progress', current=len(processed), total=count,
                             message=f'SAM 3.1 tracking: {len(people)} people')
        finally:
            predictor.handle_request(dict(type='close_session', session_id=session))
        if len(processed) != count or min(processed, default=-1) != 0 or max(processed, default=-1) != count-1:
            raise RuntimeError('SAM 3.1 did not process the expected frame range.')
        capture = cv2.VideoCapture(str(project / 'input.mp4'))
        try:
            for entry in people.values():
                target = safe_child(project, entry['preview'])
                target.mkdir()
                frame, x1, y1, x2, y2 = entry.pop('_sample')
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = capture.read()
                if ok:
                    crop = image[y1:y2, x1:x2]
                    scale = min(96 / crop.shape[1], 144 / crop.shape[0])
                    thumb = cv2.resize(crop, (max(1, round(crop.shape[1]*scale)), max(1, round(crop.shape[0]*scale))))
                    success, encoded = cv2.imencode('.png', thumb)
                    if success:
                        (target / 'thumbnail.png').write_bytes(encoded.tobytes())
        finally:
            capture.release()
        db.commit()
        entries = dict(sorted(people.items(), key=lambda row: (-row[1]['observations'], row[0])))
        result = publish_discovery(project, manifest, db, entries, generation, digest, 'SAM 3.1')
    emit('people_found', project=str(project), count=len(result), cached=False)
    return project / 'people.json'


def main(arguments, emit):
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True, type=Path)
    parser.add_argument('--sam31-options', required=True)
    args = parser.parse_args(arguments)
    stage = 'sam31_people'
    try:
        from time import monotonic
        started = monotonic()
        request = json.loads(args.sam31_options)
        prompt = request['prompt'].strip() or 'person'
        points, anchor = request['points'], request['time']
        if len(prompt) > 256 or len(points) > 16 or not math.isfinite(anchor) or any(
                not 0 <= x < 1 or not 0 <= y < 1 for x, y in points):
            raise ValueError('Invalid SAM 3.1 prompt or markers.')
        output = discover(args.project, prompt, anchor, points, emit)
        emit('complete', stage=stage, output=str(output), elapsed=monotonic()-started)
        return 0
    except Exception as error:
        import traceback
        traceback.print_exc()
        message = str(error)
        if 'out of memory' in message.lower():
            message += ' Trim to a shorter clip or close other GPU applications and try again.'
        emit('failed', stage=stage, message=message)
        return 1
