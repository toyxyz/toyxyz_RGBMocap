"""Person selection is separate from extraction checkboxes and model settings."""
from __future__ import annotations
from .tooltips import tooltip_text
from .ui_refresh import batch_readiness
import json
from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import Qt, QSize, QRect, Signal, QPersistentModelIndex, QEvent
from PySide6.QtGui import QColor, QIcon, QPixmap, QPainter, QPen
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,
    QListWidget,QListWidgetItem,QColorDialog,QComboBox,QFileDialog,
    QStyledItemDelegate,QStyleOptionViewItem,QToolTip,QStyle,QMenu,QInputDialog,QMessageBox)

from .people_store import load_people,save_people,root_for,safe_child,DEFAULT_COLOR


PEOPLE_HELP = (
    'Reconstruct 3D automatically tracks one person. For several people or a specific person, '
    'use Find person first.\n\n'
    'Check a person to reconstruct and show them. Select a row to preview their motion and '
    'edit their contact timings. Selecting a row does not change its checkbox.\n\n'
    'Right-click a row to rename it. Click its color square to change the display color. '
    'Names are used for FBX files and rigs; colors affect the preview only.\n\n'
    'If a person has separate visible sections, use the take list to switch between them.\n\n'
    'Automatic single-person tracking may switch people when they cross or overlap. '
    'Use Find person when keeping a specific identity matters.'
)


class PersonColorDelegate(QStyledItemDelegate):
    """Reserve a right-hand swatch without allocating a widget per person."""
    def paint(self,painter,option,index):
        if isinstance(index.data(Qt.UserRole),str):
            super().paint(painter,option,index);return
        content=QStyleOptionViewItem(option)
        content.rect.adjust(0,0,-42,0)
        super().paint(painter,content,index)
        rect=self.parent().color_rect(index)
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setOpacity(1 if option.state & QStyle.State_Enabled else .4)
        painter.setPen(QPen(QColor('#8d8d8d'),1))
        painter.setBrush(QColor(index.data(Qt.UserRole+1)))
        painter.drawRoundedRect(rect.adjusted(0,0,-1,-1),3,3)
        painter.restore()


class PeopleList(QListWidget):
    characterMenuRequested=Signal(str,object)
    colorRequested=Signal(int)
    renameRequested=Signal(int)

    def __init__(self):
        super().__init__()
        self.setObjectName('peopleList')
        self._color_press=QPersistentModelIndex()
        self.setItemDelegate(PersonColorDelegate(self))

    def color_rect(self,index):
        row=self.visualRect(index)
        return QRect(row.right()-33,row.center().y()-11,26,22)

    def mousePressEvent(self,event):
        index=self.indexAt(event.position().toPoint())
        if event.button()==Qt.RightButton:
            event.accept();return
        if event.button()==Qt.LeftButton and index.isValid() and not isinstance(index.data(Qt.UserRole),str) and self.color_rect(index).contains(event.position().toPoint()):
            self._color_press=QPersistentModelIndex(index)
            event.accept();return
        super().mousePressEvent(event)

    def contextMenuEvent(self,event):
        index=self.indexAt(event.pos())
        if not index.isValid():return
        ident=index.data(Qt.UserRole)
        if isinstance(ident,str) and ident.startswith('character:'):
            self.characterMenuRequested.emit(ident,event.globalPos());return
        menu=QMenu(self)
        action=menu.addAction('Rename...')
        try:
            if menu.exec(event.globalPos())==action:self.renameRequested.emit(ident)
        finally:menu.deleteLater()

    def mouseMoveEvent(self,event):
        if self._color_press.isValid():
            event.accept();return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self,event):
        if event.button()==Qt.LeftButton and self._color_press.isValid():
            pressed=self._color_press
            self._color_press=QPersistentModelIndex()
            index=self.indexAt(event.position().toPoint())
            if pressed==index and self.color_rect(index).contains(event.position().toPoint()):
                if not isinstance(index.data(Qt.UserRole),str):self.colorRequested.emit(index.data(Qt.UserRole))
            event.accept();return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self,event):
        index=self.indexAt(event.position().toPoint())
        if event.button()==Qt.LeftButton and index.isValid() and self.color_rect(index).contains(event.position().toPoint()):
            self.mousePressEvent(event);return
        super().mouseDoubleClickEvent(event)

    def viewportEvent(self,event):
        if event.type()==QEvent.ToolTip:
            index=self.indexAt(event.pos())
            if index.isValid() and not isinstance(index.data(Qt.UserRole),str) and self.color_rect(index).contains(event.pos()):
                QToolTip.showText(event.globalPos(),tooltip_text(f"Change color | {index.data()}"),self)
                return True
        return super().viewportEvent(event)


class PeopleWorkspace:
    def init_people(self):
        self.people_root = None
        self.people_state = None
        self.active_person_id = None
        self._single_actor = None
        self._single_actor_project = None
        self._people_syncing = False
        self._people_batch = False
        self._people_stages = []
        self._people_return_id = None
        self._people_return_project=None
        from .scene_data import SceneDataLoader
        self.scene_data=None;self._scene_key=None
        self.scene_cache=OrderedDict()
        self.scene_loader=SceneDataLoader(self)
        self.scene_loader.ready.connect(self.people_scene_ready)
        self.scene_loader.failed.connect(self.people_scene_failed)
        self.scene_loader.start()

    def clear_people_scene(self):
        self.scene_loader.cancel();self._scene_key=None
        self.motion_view.people_mode=False;self.motion_view.scene_fit_root=None
        self.motion_view.set_people_scene(None)
        for scene in self.scene_cache.values():scene.close()
        self.scene_cache.clear();self.scene_data=None

    def request_people_scene(self,force=False):
        if self.people_root is None:return
        from .motion_tools import base_result
        kind=self.active_result
        key=(str(self.people_root),kind)
        if key==self._scene_key and not force:return
        if self._scene_key and self.scene_data:
            self.scene_data.close_windows()
        self._scene_key=key
        self.motion_view.people_mode=True
        self.motion_view.scene_message='Loading shared scene...'
        self.motion_view.set_people_scene(None)
        self.scene_data=None
        if force:
            for scene in self.scene_cache.values():scene.close()
            self.scene_cache.clear()
        asset=self.preview_data.mesh_asset if self.preview_data else None
        self.scene_loader.request(self.people_root,kind,asset,self.scene_cache.get(key))

    def people_scene_ready(self,generation,scene):
        if generation!=self.scene_loader.generation:
            if not any(scene is cached for cached in self.scene_cache.values()):scene.close()
            return
        self.motion_view.set_people_scene(None)
        if self._scene_key in self.scene_cache:
            previous=self.scene_cache.pop(self._scene_key)
            if previous is not scene:previous.close()
        self.scene_data=scene;self.scene_cache[self._scene_key]=scene
        while len(self.scene_cache)>8:self.scene_cache.popitem(last=False)[1].close()
        self.motion_view.set_people_scene(scene)
        self.sync_shared_floor_controls()
        self.sync_scene_visibility()

        self.restore_workspace_camera(consume=True)

    def sync_shared_floor_controls(self):
        if self.people_root is None:
            self.auto_floor.setText('Auto Floor Height')
            self.floor_height_label.setText('Floor Height (World Y)')
            self.floor_description.setText('Enable Correction Floor below the viewport to preview this height.\nHigher values raise the correction floor.\nApply Correction updates the motion.')
            self.sync_correction_floor_preview()
            return
        from .scene_floor import ensure_config,current_resolution
        options=ensure_config(self.people_root);resolved=current_resolution(self.people_root)
        self.auto_floor.blockSignals(True);self.floor_height.blockSignals(True);self.correction_mode.blockSignals(True)
        try:
            self.auto_floor.setChecked(options['auto_floor'])
            self.correction_mode.setCurrentIndex(self.correction_mode.findData(options.get('mode','flat')))
            self.floor_height.setValue(resolved['floor'] if options['auto_floor'] and resolved else options['floor'])
        finally:
            self.auto_floor.blockSignals(False);self.floor_height.blockSignals(False);self.correction_mode.blockSignals(False)
        self.auto_floor.setText('Auto Scene Floor')
        self.floor_height_label.setText('Floor Height (Shared World Y)')
        self.floor_description.setText('Correction Floor below the viewport shows the shared plane for all people.\nHigher values raise the correction floor.\nApply Correction updates available actors.')
        if self.correction_mode.currentData()=='multi_level':self.update_correction_description()
        self.sync_correction_floor_preview()

    def apply_scene_correction(self,kind):
        """Rebuild all available actor/take corrections against one plane."""
        from .motion_tools import RESULTS,result_sources,base_result,result_is_current
        jobs=[]
        for person in self.people_state['people']:
            for take in person.get('takes',[]):
                project=safe_child(self.people_root,take['project'])
                bases=result_sources(base_result(kind))
                available=all((project/RESULTS[base][0]).is_file() and result_is_current(project,base) for base in bases)
                if not available:
                    if person.get('checked'):
                        self.status.setText(f"{person['name']}: selected base motion is unavailable. Reconstruct it first.");return
                    continue
                jobs.extend((stage,['--project',str(project),'--result',kind]) for stage in ('correct','export'))
        if not jobs:
            self.status.setText('Reconstruct a person before applying the shared scene floor.');return
        self._people_batch=True
        self._people_inference_jobs=[]
        self._people_post_jobs=[];self._people_return_id=self.active_person_id
        self._people_return_project=self.project
        self.start_jobs(jobs, refresh_characters=True)

    def people_scene_failed(self,generation,message):
        if generation==self.scene_loader.generation:
            self._scene_key=None
            self.motion_view.scene_message='Shared scene unavailable.\n'+message
            self.motion_view.update()
            self.status.setText('Shared scene unavailable: '+message)

    def sync_scene_visibility(self):
        self.motion_view.single_actor_visible=self._single_actor is None or self._single_actor.get('checked',True)
        self.motion_view.scene_visible={person['id'] for person in self.checked_people()}
        self.motion_view.scene_colors={p['id']:p['color'] for p in self.people_state['people']} if self.people_state else {}
        self.motion_view.scene_names={p['id']:p['name'] for p in self.people_state['people']} if self.people_state else {}
        self.motion_view.actor_name=self._single_actor['name'] if self._single_actor else 'Person 1'
        self.motion_view.scene_selected=self.active_person_id
        self.motion_view.scene_time=self.source_offset+self.motion_view.index/(self.info['fps'] if self.info else 30)
        if hasattr(self,'camera_tools'):self.camera_tools.canvas.mirror()
        self.motion_view.update()

    def build_people_editor(self, video):
        body = QWidget()
        layout = QHBoxLayout(body)
        layout.setContentsMargins(0,0,0,0);layout.setSpacing(2)
        self.people_panel = QWidget()
        self.people_panel.setObjectName('editor')
        self.people_panel.setMinimumWidth(158);self.people_panel.setMaximumWidth(240)
        column = QVBoxLayout(self.people_panel)
        column.setContentsMargins(6,6,6,6);column.setSpacing(5)
        header=QHBoxLayout()
        header.addWidget(QLabel('People'));header.addStretch()
        self.people_help_button=QPushButton('?')
        self.people_help_button.setFixedWidth(26)
        self.people_help_button.setAccessibleName('People Help')
        self.people_help_button.setToolTip('Show how to choose, rename and display people.')
        self.people_help_dialog=QMessageBox(self)
        self.people_help_dialog.setWindowTitle('People Help')
        self.people_help_dialog.setTextFormat(Qt.PlainText)
        self.people_help_dialog.setText(PEOPLE_HELP)
        self.people_help_dialog.setStandardButtons(QMessageBox.Close)
        self.people_help_button.clicked.connect(self.people_help_dialog.open)
        header.addWidget(self.people_help_button)
        column.addLayout(header)
        self.people_list = PeopleList()
        self.people_list.setToolTip(tooltip_text('Check to reconstruct and show; select to edit. Use ? for help.'))
        self.people_list.setMinimumWidth(140)
        self.people_list.setIconSize(QSize(42,56))
        self.people_list.setAccessibleName('People: check to extract and show, select to edit')
        self.people_list.currentItemChanged.connect(self.person_selected)
        self.people_list.itemChanged.connect(self.person_checked)
        self.people_list.colorRequested.connect(self.choose_person_color)
        self.people_list.renameRequested.connect(self.rename_person)
        column.addWidget(self.people_list,1)
        self.person_takes = QComboBox()
        self.person_takes.setAccessibleName('Visible motion interval')
        self.person_takes.setToolTip(tooltip_text('Choose a visible section of this person\'s motion. Separate takes avoid filling gaps while the person is out of view.'))
        self.person_takes.currentIndexChanged.connect(self.person_take_selected)
        self.person_takes.hide()
        column.addWidget(self.person_takes)
        self.people_note = QLabel('')
        self.people_note.setObjectName('muted');self.people_note.setWordWrap(True)
        self.people_note.hide()
        column.addWidget(self.people_note)
        layout.addWidget(self.people_panel)
        movie = QWidget()
        movie_layout = QVBoxLayout(movie)
        movie_layout.setContentsMargins(0,0,0,0);movie_layout.setSpacing(0)
        video.setMinimumWidth(260)
        movie_layout.addWidget(video,1)
        self.build_mamma_thumbnails(movie_layout)
        self.build_pear_thumbnails(movie_layout)
        footer = QWidget();footer.setObjectName('editorHeader')
        bar = QHBoxLayout(footer);bar.setContentsMargins(7,3,7,3)
        self.find_person_button = QPushButton('Find person')
        self.find_person_button.setToolTip(tooltip_text('Find people in the video so you can choose who to reconstruct. Useful for multiple people or a specific person.'))
        self.find_person_button.clicked.connect(self.find_people)
        bar.addWidget(self.find_person_button);bar.addStretch()
        movie_layout.addWidget(footer)
        layout.addWidget(movie,1)
        return body

    def update_people_readiness(self):
        enabled = not self.busy and not self.contact_saving
        self.find_person_button.setEnabled(enabled and self.video_readable)
        self.people_list.setEnabled(enabled)
        self.person_takes.setEnabled(enabled)

    def reset_people(self):
        self.clear_people_scene()
        self.people_root = self.people_state = self.active_person_id = None
        self._single_actor = self._single_actor_project = None
        self.motion_view.single_actor_visible=True
        self._people_syncing = True
        self.people_list.clear();self.person_takes.clear();self.person_takes.hide()
        self._people_syncing = False
        self.set_people_note('')
        self.sync_shared_floor_controls()
        self.apply_person_color()

        if hasattr(self,'append_character_rows'):self.append_character_rows()

    def sync_people(self, project, manifest):
        root = root_for(project,manifest)
        state = load_people(root)
        if state is None:
            self.reset_people()
            if (Path(project)/'tracking.npz').is_file():self.begin_single_tracking(project,loading=True)
            return
        self._single_actor = self._single_actor_project = None
        self.motion_view.single_actor_visible=True
        self.people_root,self.people_state = root,state
        self.sync_shared_floor_controls()
        if self._scene_key is not None and self._scene_key[0]!=str(root):self.clear_people_scene()
        ident = manifest.get('person_id',state.get('active',state['people'][0]['id']))
        self.active_person_id = ident
        self._people_syncing = True
        try:
            self.people_list.clear()
            for person in state['people']:
                item = QListWidgetItem(person['name'])
                item.setData(Qt.UserRole,person['id'])
                item.setData(Qt.UserRole+1,person['color'])
                item.setFlags(item.flags()|Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if person.get('checked') else Qt.Unchecked)
                item.setIcon(self.person_icon(person))
                item.setToolTip(tooltip_text(f"{person['name']} | {person['observations']} observed frames\nCheck to reconstruct and show; select to edit. Right-click to rename."))
                self.people_list.addItem(item)
                if person['id']==ident:self.people_list.setCurrentItem(item)
            self.populate_person_takes()
        finally:
            self._people_syncing = False
        count=len(state['people'])
        self.set_people_note(f"{count} {'person' if count==1 else 'people'} found")
        self.apply_person_color()
        self.request_people_scene()
        # Opening the group selects its current person; existing projects without
        # person metadata keep their original behavior and result selection.
        if Path(project).resolve()==root and not self.busy:
            self.activate_person(ident)

    def current_person(self):
        if self.people_state:
            return next((p for p in self.people_state['people'] if p['id']==self.active_person_id),None)
        return self._single_actor

    def set_people_note(self,text):
        """Reserve the footer for current state and actionable warnings."""
        self.people_note.setText(text)
        self.people_note.setVisible(bool(text))

    def begin_single_tracking(self,project,loading=False):
        if self.people_root is not None or project is None or self.project is None:return
        if Path(project).resolve()!=self.project.resolve():return
        if self._single_actor_project!=self.project:
            from .single_actor import preferences
            self._single_actor=preferences(self.project)
            self._single_actor.update(pending=True,observations=0,thumbnail=None)
            self._single_actor_project=self.project
        self.active_person_id=1
        self.update_single_actor_row('Loading tracked person...' if loading else 'Tracking one person...')

    def update_single_actor_row(self,message=None):
        person=self._single_actor
        if person is None:return
        self._people_syncing=True
        try:
            if self.people_list.count()<1 or self.people_list.item(0).data(Qt.UserRole)!=1:
                self.people_list.clear();self.people_list.addItem(QListWidgetItem(person['name']))
            item=self.people_list.item(0)
            item.setText(person['name'])
            item.setData(Qt.UserRole,1);item.setData(Qt.UserRole+1,person['color'])
            item.setFlags(item.flags()|Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if person['checked'] else Qt.Unchecked)
            thumbnail=person.get('thumbnail')
            item.setIcon(QIcon(QPixmap.fromImage(thumbnail)) if thumbnail is not None else QIcon())
            count=person.get('observations',0)
            item.setToolTip(tooltip_text(f"{person['name']} | Automatic single-person target | {count} observed frames\n"
                'Check to extract and show; select to edit. Use Find person for identity-aware selection.\n'
                'This tracker may switch people when they cross or overlap.'))
            self.people_list.setCurrentItem(item)
            self.person_takes.clear();self.person_takes.hide()
        finally:self._people_syncing=False
        self.set_people_note(message or ('Tracking one person...' if person.get('pending') else
            '1 person tracked automatically'))
        self.apply_person_color()

        if hasattr(self,'append_character_rows'):self.append_character_rows()

    def tracked_actor_ready(self,generation,project,key,snapshot):
        if generation!=self.data_loader.generation or self.people_root is not None or self.project!=Path(project):return
        from .single_actor import snapshot_key
        if key!=snapshot_key(project):return
        if snapshot is None:
            if self._single_actor is not None and self._single_actor.get('pending'):
                self.reset_people()
                self.set_people_note('No tracked person available.')
            return
        self.begin_single_tracking(project,loading=True)
        self._single_actor.update(snapshot)
        self.update_single_actor_row()

    def single_tracking_event(self,event):
        if self.people_root is not None:return
        job=getattr(self,'job_project',None)
        if job is None or self.project is None or Path(job).resolve()!=self.project.resolve():return
        if event['type']=='progress' and event.get('stage')=='track':
            self.begin_single_tracking(job)
            if self._single_actor:
                self.set_people_note(f"Tracking one person... {event.get('current',0)} / {event.get('total',0)} frames")
        elif event['type']=='complete' and event.get('stage')=='track':
            self.refresh_results()

    def finish_single_tracking(self):
        if self.people_root is None and self._single_actor and self._single_actor.get('pending') and (self.failed or self.cancelled):
            self.reset_people()
            self.set_people_note('Tracking failed.' if self.failed else 'Tracking cancelled.')

    def person_icon(self,person):
        pixmap=QPixmap(40,56);pixmap.fill(Qt.transparent)
        painter=QPainter(pixmap)
        thumbnail=safe_child(self.people_root,person['preview'])/'thumbnail.png'
        if thumbnail.is_file():
            image=QPixmap(str(thumbnail)).scaled(40,56,Qt.KeepAspectRatio,Qt.SmoothTransformation)
            painter.drawPixmap((40-image.width())//2,(56-image.height())//2,image)
        painter.end()
        return QIcon(pixmap)

    def checked_people(self):
        if self.people_state:return [p for p in self.people_state['people'] if p.get('checked')]
        return [self._single_actor] if self._single_actor and self._single_actor.get('checked') else []

    def populate_person_takes(self):
        person=self.current_person()
        self.person_takes.blockSignals(True)
        try:
            self.person_takes.clear()
            takes=person.get('takes',[]) if person else []
            for index,take in enumerate(takes):
                path=safe_child(self.people_root,take['project'])
                self.person_takes.addItem(f"Take {index+1} | frames {take['first']+1}-{take['end']}",take['project'])
                if path==self.project:self.person_takes.setCurrentIndex(index)
            self.person_takes.setVisible(len(takes)>1)
        finally:self.person_takes.blockSignals(False)

    def person_checked(self,item):
        if self._people_syncing:return
        if hasattr(self,'character_checked') and self.character_checked(item):return
        if self.people_state is None:
            if self._single_actor:
                from .single_actor import save_preferences
                self._single_actor['checked']=item.checkState()==Qt.Checked
                save_preferences(self.project,self._single_actor)
                self.sync_scene_visibility()
            return
        ident=item.data(Qt.UserRole)
        person=next(p for p in self.people_state['people'] if p['id']==ident)
        person['checked']=item.checkState()==Qt.Checked
        save_people(self.people_root,self.people_state)
        self.sync_scene_visibility()

    @batch_readiness
    def person_selected(self,item,previous=None):
        if self._people_syncing or item is None or self.busy or self.contact_saving:return
        if isinstance(item.data(Qt.UserRole),str):
            self.update_readiness();return
        self.activate_person(item.data(Qt.UserRole))
        self.update_readiness()

    @batch_readiness
    def activate_person(self,ident,path=None):
        if self.people_state is None:return
        self.active_person_id=ident
        person=self.current_person()
        if person is None:return
        self.people_state['active']=ident
        save_people(self.people_root,self.people_state)
        if path is None:
            takes=person.get('takes',[])
            relative=takes[-1]['project'] if takes else person['preview']
            path=safe_child(self.people_root,relative)
        if self.project!=path:
            source_time=self.source_offset+(self.slider.value()/self.info['fps'] if self.info else 0)
            self.load_project(path)
            self.motion_view.auto_fit=False
            if self.info:
                self.slider.setValue(max(0,min(self.slider.maximum(),round((source_time-self.source_offset)*self.info['fps']))))
        else:
            self.populate_person_takes();self.apply_person_color()
        self.update_readiness()

    def person_take_selected(self,index):
        if self._people_syncing or index<0 or self.busy or self.contact_saving:return
        relative=self.person_takes.itemData(index)
        if relative:self.activate_person(self.active_person_id,safe_child(self.people_root,relative))

    def apply_person_color(self):
        person=self.current_person()
        color=person['color'] if person else None
        self.motion_view.set_body_color(color or DEFAULT_COLOR)
        self.video_view.actor_color=color
        self.video_view.update()
        self.slider.person_color=color
        self.slider.person_name=person['name'] if person else None
        self.slider.update()
        self.sync_scene_visibility()

    def choose_person_color(self,ident):
        person=next((p for p in self.people_state['people'] if p['id']==ident),None) if self.people_state else self._single_actor
        if person is None:return
        color=QColorDialog.getColor(QColor(person['color']),self,f"Color | {person['name']}")
        if color.isValid():self.set_person_color(person['id'],color.name())

    def rename_person(self,ident):
        if self.busy:return
        person=next((p for p in self.people_state['people'] if p['id']==ident),None) if self.people_state else self._single_actor
        if person is None:return
        text,ok=QInputDialog.getText(self,'Rename Person','Name:',text=person['name'])
        if not ok:return
        try:self.set_person_name(ident,text)
        except (ValueError,OSError) as error:QMessageBox.warning(self,'Rename Person',str(error))

    def set_person_name(self,ident,name):
        from .person_names import validate_name
        from copy import deepcopy
        name=validate_name(name)
        if self.people_state:
            state=deepcopy(self.people_state)
            person=next(p for p in state['people'] if p['id']==ident)
            person['name']=name
            save_people(self.people_root,state)
            self.people_state=state
        else:
            if self._single_actor is None or ident!=1:return
            from .single_actor import save_preferences
            person={**self._single_actor,'name':name}
            save_preferences(self.project,person)
            self._single_actor=person
        self._people_syncing=True
        try:
            for row in range(self.people_list.count()):
                item=self.people_list.item(row)
                if item.data(Qt.UserRole)==ident:
                    item.setText(name)
                    item.setToolTip(tooltip_text(f"{name} | {person.get('observations',0)} observed frames\nCheck to extract and show; select to edit. Right-click to rename."))
        finally:self._people_syncing=False
        active=self.current_person()
        self.slider.person_name=active['name'] if active else None
        self.slider.update()
        self.sync_scene_visibility()

    def export_person_name(self,project):
        if self.people_state:
            for person in self.people_state['people']:
                paths=[t['project'] for t in person.get('takes',[])]+[person['preview']]
                if any(safe_child(self.people_root,path)==Path(project).resolve() for path in paths):return person['name']
        if self._single_actor and Path(project).resolve()==Path(self.project).resolve():return self._single_actor['name']
        return 'Person 1'

    def set_person_color(self,ident,color):
        chosen=QColor(color)
        if not chosen.isValid():raise ValueError('Invalid person color')
        if self.people_state is None:
            if self._single_actor is not None and ident==1:
                from .single_actor import save_preferences
                self._single_actor['color']=chosen.name();save_preferences(self.project,self._single_actor)
                self.update_single_actor_row()
            return
        person=next(p for p in self.people_state['people'] if p['id']==ident)
        person['color']=chosen.name()
        save_people(self.people_root,self.people_state)
        self._people_syncing=True
        try:
            for row in range(self.people_list.count()):
                item=self.people_list.item(row)
                if item.data(Qt.UserRole)==ident:
                    item.setData(Qt.UserRole+1,chosen.name())
        finally:self._people_syncing=False
        self.apply_person_color()

    def find_people(self):
        if self.busy:return
        from .models import missing
        unavailable=missing('people')
        if unavailable:
            self.status.setText('Prepare person detection models: '+', '.join(p.name for p in unavailable))
            self.open_model_manager();return
        if self.people_root is not None and self.info is not None and self.video_readable:
            manifest=json.loads((self.people_root/'project.json').read_text(encoding='utf-8'))
            first,end=self.processing_range()
            root_start=float(manifest['start'])
            root_end=min(float(manifest['end']),root_start+manifest['frames']/manifest['fps'])
            # Cached discovery belongs to the group's complete input, not to a
            # newly trimmed preview or a shorter selected person's take.
            if abs(first-root_start)<1e-6 and abs(end-root_end)<1e-6:
                self.start_jobs([('people',['--project',str(self.people_root)])])
                return
        self.input_jobs(['people'])

    def input_person_jobs(self,stages,result):
        if self.people_state is None:return False
        if self.busy:return True
        selected=self.checked_people()
        if not selected:
            self.status.setText('Check at least one person in the People list to extract motion.')
            return True
        from .models import missing
        unavailable=[model for stage in stages for model in ([] if self.model_choice.currentData()=='flowhmr' and stage=='flow_camera' and self.flow_sampling.currentData()==-1 else missing('flow_export' if self.model_choice.currentData()=='flowhmr' and stage in ('correct','export') else stage))]
        if unavailable:
            self.status.setText('Prepare the required motion models first.');self.open_model_manager();return True
        root_manifest=json.loads((self.people_root/'project.json').read_text(encoding='utf-8'))
        first,end=self.processing_range()
        fps=root_manifest['fps']
        begin=round((first-root_manifest['start'])*fps)
        finish=round((end-root_manifest['start'])*fps)
        self._people_stages=[s for s in stages if s!='track']
        self._people_batch=True
        self._people_inference_jobs=[];self._people_post_jobs=[]
        self._people_return_id=self.active_person_id if any(p['id']==self.active_person_id for p in selected) else selected[0]['id']
        self._people_return_project=None
        self._people_result=result or self.active_result
        options=self.selected_correction_options()
        jobs=[]
        for person in selected:
            args=['--project',str(self.people_root),'--person-id',str(person['id']),
                '--first-frame',str(begin),'--end-frame',str(finish),'--result',self._people_result,
                '--correction-preset',options['preset'],'--correction-mode',options['mode'],'--foot-lock',str(int(options['foot_lock'])),'--hand-lock',str(int(options['hand_lock'])),'--pelvis-offset',str(options['pelvis_offset'])]
            if not options['auto_floor']:args+=['--floor',str(options['floor'])]
            jobs.append(('person',args))
        self.start_jobs(jobs)
        return True

    def people_event(self,event):
        if event['type']=='person_prepared':
            # Explicit project arguments prevent a later UI selection from changing
            # which person's worker, contact edits, or exports are being processed.
            inference=[stage for stage in self._people_stages if stage not in ('correct','export','flow_export')]
            post=[stage for stage in self._people_stages if stage in ('correct','export','flow_export')]
            jobs=[(stage,['--project',path,'--result',self._people_result])
                  for path in event['projects'] for stage in inference]
            deferred=[(stage,['--project',path,'--result',self._people_result])
                      for path in event['projects'] for stage in post]
            # Prepare every actor's original camera/motion before resolving the
            # one automatic scene plane or committing any corrected exports.
            self._people_inference_jobs.extend(jobs)
            self._people_post_jobs.extend(deferred)
            if not any(stage=='person' for stage,_ in self.jobs):
                from .queue_groups import grouped_jobs
                self.jobs=grouped_jobs(self._people_inference_jobs,self.people_root)+self.jobs+self._people_post_jobs
                self._people_inference_jobs=[];self._people_post_jobs=[]
            return True
        if event['type']=='people_found':
            self._people_found_root=Path(event['project'])
            return True
        return False

    def finish_people_jobs(self):
        if self.people_root:self.request_people_scene(force=True)
        found=getattr(self,'_people_found_root',None)
        if found is not None:
            self._people_found_root=None
            self.load_project(found)
            self.status.setText(f'{self.people_list.count()} people found. Check people to extract; select a row to preview.')
        if not self._people_batch:return
        self._people_batch=False
        self._people_inference_jobs=[];self._people_post_jobs=[]
        root=self.people_root
        ident=self._people_return_id
        if root:
            self.people_state=load_people(root)
            path=self._people_return_project;self._people_return_project=None
            self.activate_person(ident,path)

    def export_checked_people(self):
        """Multiple selected people/takes export separately, without fabricated gaps."""
        if not self.people_state:return False
        people=self.checked_people()
        if len(people)<2 and not (people and len(people[0].get('takes',[]))>1):return False
        from .motion_tools import selected_export,result_is_current
        files=[]
        for person in people:
            takes=person.get('takes',[])
            if not takes:
                self.status.setText(f"No motion for {person['name']}. Run Reconstruct 3D first.");return True
            for number,take in enumerate(takes,1):
                project=safe_child(self.people_root,take['project'])
                source=selected_export(project,self.active_result) if result_is_current(project,self.active_result) else None
                if source is None:
                    self.status.setText(f"{person['name']}, take {number}: displayed result unavailable or correction pending.");return True
                from .person_names import export_stem
                stem=export_stem(person['name'])+(f'_take_{number:02d}' if len(takes)>1 else '')
                files.append((source,stem))
        folder=QFileDialog.getExistingDirectory(self,'Export Checked People')
        if folder:
            copies=[]
            from .person_names import export_path
            used={p.name.casefold() for p in Path(folder).iterdir()}
            for source,name in files:
                target=export_path(folder,name,used,suffix='.'+getattr(self,'_source_export_options',{}).get('format','fbx'))
                copies.append((source.parent,self.active_result,source,target))
            self.submit_fbx_exports(copies)
        return True
