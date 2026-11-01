"""Small GUI request builder; all mesh baking runs in a worker."""
from pathlib import Path
import json,uuid
from PySide6.QtWidgets import QFileDialog,QMessageBox


def export_scene(w):
    from .storage import atomic_json
    from .people_store import safe_child
    from .motion_tools import selected_export,result_is_current
    try:
        if w.people_root is None:raise ValueError('Open a reconstructed MAMMA people scene first.')
        state=json.loads((w.people_root/'project.json').read_text(encoding='utf8'))
        if state.get('pipeline',{}).get('model')!='mamma':raise ValueError('All-camera export is available for MAMMA multi-view projects.')
        people=w.checked_people()
        if not people:raise ValueError('Check the people to include in the scene.')
        for p in people:
            if not p['takes']:raise ValueError(p['name']+': no reconstructed motion.')
            for t in p['takes']:
                project=safe_child(w.people_root,t['project'])
                if not result_is_current(project,w.active_result) or selected_export(project,w.active_result) is None:
                    raise ValueError(p['name']+': Apply Correction or Prepare Preview for the selected result first.')
        name,_=QFileDialog.getSaveFileName(w,'Export MAMMA scene',str(Path(getattr(w,'_motion_export_folder',None) or w.workspace_root or w.people_root)/'MAMMA scene.blend'),'Blender (*.blend)',options=QFileDialog.DontUseNativeDialog)
        if not name:return
        target=Path(name)
        if target.suffix.lower()!='.blend':target=target.with_suffix('.blend')
        w._motion_export_folder=str(target.parent);folder=target.with_suffix('');number=2
        while folder.exists():folder=target.parent/(target.stem+'_'+str(number));number+=1
        target=folder/target.name
        request=w.project/('.scene-export-'+uuid.uuid4().hex+'.json')
        atomic_json(request,dict(root=str(w.people_root),kind=w.active_result,people=[p['id'] for p in people],target=str(target)))
        w._motion_export_targets=[str(target)]
        temporary=folder.with_name('.'+folder.name+'.rgbmocap-'+request.stem+'.partial')
        w._fbx_export_cleanup=(request,[temporary])
        w.start_jobs([('save_blend_scene',['--project',str(w.project),'--output',str(request)])])
    except (OSError,ValueError,KeyError) as error:
        QMessageBox.information(w,'MAMMA scene export unavailable',str(error))
