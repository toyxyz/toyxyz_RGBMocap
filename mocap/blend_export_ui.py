"""Small GUI request preparation; all baking/copying runs in a worker."""
import uuid
from pathlib import Path
from PySide6.QtWidgets import QFileDialog


def targets(files):
    used=set();result=[]
    for project,kind,source,target in files:
        target=Path(target);folder=target.with_suffix('');number=2
        while folder.exists() or str(folder.resolve()).casefold() in used:
            folder=target.parent/(target.stem+'_'+str(number));number+=1
        used.add(str(folder.resolve()).casefold());result.append((project,kind,source,folder/target.name))
    return result


def export_scene(w):
    from .storage import atomic_json
    from .people_store import safe_child
    from .motion_tools import selected_export
    from .blend_support import partial,read
    try:
        if w.people_root is None or read(w.people_root/'project.json').get('pipeline',{}).get('model')!='mamma':
            raise ValueError('Open a reconstructed MAMMA people scene first.')
        people=w.checked_people()
        if not people:raise ValueError('Check the people to include in the scene.')
        for person in people:
            if not person['takes']:raise ValueError(person['name']+': no reconstructed motion.')
            for take in person['takes']:
                if selected_export(safe_child(w.people_root,take['project']),w.active_result) is None:
                    raise ValueError(person['name']+': Apply Correction or Prepare Preview for the selected result first.')
        name,_=QFileDialog.getSaveFileName(w,'Export MAMMA Blender Scene','MAMMA scene.blend','Blender (*.blend)',options=QFileDialog.DontConfirmOverwrite)
        if not name:return
        target=Path(name)
        if target.suffix.lower()!='.blend':target=target.with_suffix('.blend')
        target=targets([(None,None,None,target)])[0][3]
        request=w.project/('.scene-export-'+uuid.uuid4().hex+'.json')
        atomic_json(request,dict(root=str(w.people_root),kind=w.active_result,people=[p['id'] for p in people],target=str(target)))
        w._motion_export_targets=[str(target)];w._fbx_export_cleanup=(request,[partial(target.parent,request)])
        w.start_jobs([('save_blend_scene',['--project',str(w.project),'--output',str(request)])])
    except (OSError,ValueError,KeyError) as error:w.status.setText('MAMMA scene export unavailable: '+str(error))
