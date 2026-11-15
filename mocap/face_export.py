"""Write timestamped ARKit-style TXT files beside exported motion."""
import csv
import json
import os
from pathlib import Path
from uuid import uuid4

from .face_data import FaceArchive, manifest, root_for
from .people_store import load_people, safe_child


def plan(request, stage):
    from .blend_support import read
    payload=read(request)
    if stage=='save_blend_scene':
        if not payload.get('export_face'):
            return []
        root=Path(payload['root']).resolve()
        state=load_people(root)
        rows=[dict(project=str(safe_child(root,t['project'])),target=payload['target'],
                   rig_name=actor['name'],export_face=True)
              for actor in state['people'] if actor['id'] in payload['people'] for t in actor['takes']]
        origin=float(manifest(root)['start'])
        leading=False
    elif isinstance(payload,dict):
        rows=payload['takes']
        if not any(row.get('export_face') for row in rows):
            return []
        origin=min(float(manifest(Path(row['project']))['start']) for row in rows)
        leading=bool(rows and rows[0].get('prepend_tpose'))
    else:
        rows=payload
        origin,leading=None,False
    return [(row,origin,leading) for row in rows if row.get('export_face')]


def text_file(destination, archive, project, state, offset, leading, name):
    fps=float(state['fps'])
    temp=destination.with_name('.'+uuid4().hex+'.partial')
    try:
        with temp.open('w',encoding='utf8',newline='') as stream:
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
            for frame in range(int(state['frames'])):
                slot,sample,source=archive.resolve(project,frame,state=state)
                label='missing' if sample<0 else 'tracked' if sample==source else 'held'
                coeff=['']*52 if sample<0 else [f'{v:.7f}' for v in archive.values[slot,sample]]
                writer.writerow([frame+offset,f'{(frame+offset)/fps:.9f}',source,
                    f'{state["start"]+frame/fps:.9f}',label,sample if sample>=0 else '',*coeff])
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
            written=text_file(target.with_name(target.stem+suffix+'.txt'),archive,project,state,
                              offset,leading,row.get('rig_name','Person '+str(person)))
            emit('info',message='Exported face TXT: '+str(written))
        except (OSError,ValueError,KeyError,IndexError) as error:
            emit('info',message=f'Face TXT for {row.get("rig_name","selected person")}: {error}')
