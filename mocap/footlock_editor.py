"""Sparse contact tracks: bounded drawing, frame edits and command-based undo."""
from bisect import bisect_left,bisect_right
from PySide6.QtCore import Qt,QRectF,QPointF
from PySide6.QtGui import QColor,QPen
from PySide6.QtWidgets import QMenu
from .theme import is_light_theme, theme_color
from .foot_contacts import FOOT_COLORS
CONTACT_COLORS=(*FOOT_COLORS,"#79a9db","#bf8dcf")


class FootLockEditor:
    def init_contacts(self):
        self.contacts=[[],[]];self.automatic_contacts=[[],[]]
        self.contacts_available=False;self.contact_edit_enabled=False
        self.contact_revision=0;self._contact_identity=None;self._contact_token=None
        self._selected_contact=None;self._foot_drag=None;self._foot_draft=None
        self._undo_contacts=[];self._redo_contacts=[]
        self._contact_starts=[[],[]];self._contact_ends=[[],[]]
        self.contact_draw_queries=0
        self.contact_track_enabled=(True,True,True,True)

    def set_contacts(self,ranges=None,automatic=None,editable=False,identity=None,token=None):
        self.contact_edit_enabled=editable and ranges is not None
        if (identity,token)==(self._contact_identity,self._contact_token):self.update();return
        incoming=[list(map(tuple,track)) for track in ranges] if ranges is not None else [[],[]]
        if identity!=self._contact_identity or incoming!=self.contacts:
            self._undo_contacts.clear();self._redo_contacts.clear();self._selected_contact=None
        self._contact_identity=identity;self._contact_token=token
        self.contacts_available=ranges is not None
        self.contacts=incoming
        self.automatic_contacts=[list(map(tuple,track)) for track in automatic] if automatic is not None else [list(t) for t in incoming]
        self._foot_drag=None;self._foot_draft=None
        self._index_contacts()

    def _index_contacts(self):
        self._contact_starts=[[s for s,e in track] for track in self.contacts]
        self._contact_ends=[[e for s,e in track] for track in self.contacts]
        self.contact_revision+=1;self.update()
        self.contacts_preview_changed.emit()

    def active_contacts(self,frame):
        return self._active_tracks(frame,0)

    def active_hand_contacts(self,frame):
        return self._active_tracks(frame,2)

    def _active_tracks(self,frame,first):
        """Two logarithmic lookups, including the in-progress edit, without copies."""
        active=[False,False]
        if not self.contacts_available:return tuple(active)
        for side,foot in enumerate(range(first,min(first+2,len(self.contacts)))):
            if not self.contact_track_enabled[foot]:continue
            index=bisect_right(self._contact_starts[foot],frame)-1
            draft=self._foot_drag if self._foot_drag and self._foot_drag['foot']==foot else None
            replaced=draft is not None and draft['original'] is not None and draft['index']==index
            if index>=0 and not replaced:
                start,end=self.contacts[foot][index]
                active[side]=start<=frame<end
            if draft is not None and self._foot_draft is not None:
                start,end=self._foot_draft
                active[side]=active[side] or start<=frame<end
        return tuple(active)

    def _contact_row(self,event):
        position=event.position() if hasattr(event,'position') else event.pos()
        row=int((position.y()-self.TRACK_TOP)//self.TRACK_HEIGHT)-2
        return row if 0<=row<len(self.contacts) and self.contact_track_enabled[row] else None

    def _edge_position(self,event):
        position=event.position() if hasattr(event,'position') else event.pos()
        value=self.view_start+(position.x()-self.LEFT)*self.span()/max(1,self.width()-self.LEFT-self.RIGHT)
        return max(0,min(self.maximum()+1,round(value)))

    def _contact_hit(self,foot,event):
        position=event.position() if hasattr(event,'position') else event.pos()
        frame=self._edge_position(event);x=position.x()
        index=bisect_right(self._contact_starts[foot],frame)-1
        candidates=[]
        for i in (index-1,index,index+1):
            if not 0<=i<len(self.contacts[foot]):continue
            s,e=self.contacts[foot][i]
            for edge,mode in ((s,'left'),(e-1,'right')):
                distance=abs(x-self.frame_x(edge))
                if distance<=6:candidates.append((distance,0 if self._selected_contact==(foot,i) else 1,i,mode))
        if candidates:
            _,_,index,mode=min(candidates);return index,mode
        if index>=0 and self.contacts[foot][index][0]<=frame<self.contacts[foot][index][1]:return index,'move'
        return None,'new'

    def _foot_press(self,event):
        foot=self._contact_row(event)
        if event.button()!=Qt.LeftButton or foot is None or event.position().x()<self.LEFT or not self.contact_edit_enabled:return False
        self.contacts_edit_started.emit();self.setFocus(Qt.MouseFocusReason)
        index,mode=self._contact_hit(foot,event)
        frame=min(self.maximum(),self._edge_position(event))
        insertion=bisect_right(self._contact_starts[foot],frame) if index is None else index
        track=self.contacts[foot]
        low=track[insertion-1][1] if insertion else 0
        high=track[insertion+(0 if index is None else 1)][0] if insertion+(0 if index is None else 1)<len(track) else self.maximum()+1
        original=track[index] if index is not None else None
        self._foot_drag={'foot':foot,'index':insertion,'mode':mode,'original':original,
                         'frame':frame,'pixel':event.position().x(),'low':low,'high':high}
        self._foot_draft=original
        self._selected_contact=(foot,index) if index is not None else None
        self.update();self.contacts_preview_changed.emit();event.accept();return True

    def _foot_move(self,event):
        d=self._foot_drag
        if d is None:return False
        frame=self._edge_position(event);low,high=d['low'],d['high']
        if d['mode']=='new':
            s=max(low,min(d['frame'],frame));e=min(high,max(d['frame'],frame)+1)
        else:
            s,e=d['original']
            if d['mode']=='left':s=max(low,min(e-1,frame))
            elif d['mode']=='right':e=min(high,max(s+1,frame+1))
            else:
                delta=max(low-s,min(high-e,frame-d['frame']));s+=delta;e+=delta
        self._foot_draft=(s,e) if e>s else None
        self.setCursor(Qt.SizeHorCursor if d['mode'] in ('left','right') else Qt.ClosedHandCursor)
        self.update();self.contacts_preview_changed.emit();event.accept();return True

    def _foot_release(self,event):
        if event.button()!=Qt.LeftButton or self._foot_drag is None:return False
        self._foot_move(event)
        d=self._foot_drag;after=self._foot_draft
        self._foot_drag=None;self._foot_draft=None
        if d['mode']=='new' and abs(event.position().x()-d['pixel'])<3:
            after=None
        if after is not None and after!=d['original']:
            self._contact_command(d['foot'],d['index'],d['original'],after)
        self.setCursor(Qt.ArrowCursor);self.update();self.contacts_preview_changed.emit();event.accept();return True

    def _contact_command(self,foot,index,before,after,record=True):
        if before is None:self.contacts[foot].insert(index,after)
        elif after is None:self.contacts[foot].pop(index)
        else:self.contacts[foot][index]=after
        self._selected_contact=(foot,index) if after is not None else None
        if record:
            self._undo_contacts.append((foot,index,before,after));del self._undo_contacts[:-100]
            self._redo_contacts.clear()
        self._index_contacts();self._emit_contacts()

    def _emit_contacts(self):
        self.contacts_changed.emit([[tuple(pair) for pair in track] for track in self.contacts])

    def undo_contact(self):
        if not self.contact_edit_enabled or not self._undo_contacts:return
        foot,index,before,after=self._undo_contacts.pop()
        self._redo_contacts.append((foot,index,before,after))
        self._contact_command(foot,index,after,before,False)

    def redo_contact(self):
        if not self.contact_edit_enabled or not self._redo_contacts:return
        command=self._redo_contacts.pop();self._undo_contacts.append(command)
        self._contact_command(*command,False)

    def delete_contact(self):
        if not self.contact_edit_enabled or self._selected_contact is None:return
        foot,index=self._selected_contact
        self._contact_command(foot,index,self.contacts[foot][index],None)

    def add_contact(self,foot,frame):
        if not self.contact_edit_enabled:return
        index=bisect_right(self._contact_starts[foot],frame)
        if index and self.contacts[foot][index-1][1]>frame:return
        high=self.contacts[foot][index][0] if index<len(self.contacts[foot]) else self.maximum()+1
        end=min(high,frame+max(1,round(self.fps*.3)))
        if end>frame:self._contact_command(foot,index,None,(frame,end))

    def reset_contact_foot(self,foot):
        if not self.contact_edit_enabled:return
        self.contacts[foot]=list(self.automatic_contacts[foot])
        self._undo_contacts.clear();self._redo_contacts.clear();self._selected_contact=None
        self._index_contacts();self._emit_contacts()

    def mouseDoubleClickEvent(self,event):
        foot=self._contact_row(event)
        if event.button()==Qt.LeftButton and foot is not None and self.contact_edit_enabled and event.position().x()>=self.LEFT:
            self._foot_drag=None;self._foot_draft=None
            self.contacts_edit_started.emit();self.add_contact(foot,min(self.maximum(),self._edge_position(event)))
            event.accept();return
        # QSlider's default double-click handler can seek outside our frame handle.
        self.mousePressEvent(event)

    def keyPressEvent(self,event):
        if self.contact_edit_enabled:
            if event.key() in (Qt.Key_Delete,Qt.Key_Backspace) and self._selected_contact is not None:
                self.delete_contact();event.accept();return
            if event.modifiers() & Qt.ControlModifier:
                if event.key()==Qt.Key_Z:
                    self.redo_contact() if event.modifiers() & Qt.ShiftModifier else self.undo_contact()
                    event.accept();return
                if event.key()==Qt.Key_Y:self.redo_contact();event.accept();return
        if event.key()==Qt.Key_Escape:
            self._foot_drag=None;self._foot_draft=None;self.update();self.contacts_preview_changed.emit();event.accept();return
        super().keyPressEvent(event)

    def contextMenuEvent(self,event):
        foot=self._contact_row(event)
        if foot is None or not self.contact_edit_enabled:return
        index,_=self._contact_hit(foot,event)
        self._selected_contact=(foot,index) if index is not None else None
        self.update()
        menu=QMenu(self)
        menu.addAction('Add Foot Lock Here',lambda:self.add_contact(foot,min(self.maximum(),self._edge_position(event))))
        remove=menu.addAction('Delete Selected Lock',self.delete_contact);remove.setEnabled(self._selected_contact is not None)
        menu.addSeparator()
        undo=menu.addAction('Undo',self.undo_contact);undo.setEnabled(bool(self._undo_contacts))
        redo=menu.addAction('Redo',self.redo_contact);redo.setEnabled(bool(self._redo_contacts))
        menu.addSeparator();menu.addAction('Reset This Track to Automatic',lambda:self.reset_contact_foot(foot))
        menu.addAction('Reset All Locks to Automatic',self.contacts_reset.emit)
        menu.exec(event.globalPos());menu.deleteLater()

    def _contact_rect(self,foot,interval):
        s,e=interval;left=max(self.LEFT,self.frame_x(s));right=min(self.width()-self.RIGHT,self.frame_x(e-1))
        return QRectF(left,self.TRACK_TOP+(foot+2)*self.TRACK_HEIGHT+2,max(2,right-left),self.TRACK_HEIGHT-5)

    def paint_contact_tracks(self,p):
        self.contact_draw_queries=0
        if not getattr(self,'contacts_supported',True):return
        for foot,(label,color) in enumerate(zip(('L foot','R foot','L hand','R hand'),CONTACT_COLORS)):
            y=self.TRACK_TOP+(foot+2)*self.TRACK_HEIGHT
            p.setPen(QColor(theme_color('#aaaaaa')));p.drawText(QRectF(8,y,self.LEFT-11,self.TRACK_HEIGHT),Qt.AlignVCenter,label)
            if not self.contact_track_enabled[foot]:
                p.setPen(QColor(theme_color('#777777')));p.drawText(QRectF(self.LEFT+8,y,max(0,self.width()-self.LEFT-25),self.TRACK_HEIGHT),Qt.AlignVCenter,'Disabled');continue
            if not self.contacts_available or foot>=len(self.contacts):
                p.setPen(QColor(theme_color('#777777')));p.drawText(QRectF(self.LEFT+8,y,max(0,self.width()-self.LEFT-25),self.TRACK_HEIGHT),Qt.AlignVCenter,'Apply Correction to generate locks');continue
            p.save();p.setClipRect(QRectF(self.LEFT,y,max(1,self.width()-self.LEFT-self.RIGHT),self.TRACK_HEIGHT))
            tint=QColor(color);tint.setAlpha(90 if self.contact_edit_enabled else 40)
            lo=bisect_right(self._contact_ends[foot],self.view_start)
            hi=bisect_right(self._contact_starts[foot],self.view_start+self.span())
            if hi-lo>max(1,(self.width()-self.LEFT-self.RIGHT)//3):
                # At distant zoom, query coverage per pixel column rather than
                # scanning every contact in a multi-hour recording.
                for x in range(self.LEFT,self.width()-self.RIGHT,3):
                    self.contact_draw_queries+=1
                    a=self.view_start+(x-self.LEFT)*self.span()/max(1,self.width()-self.LEFT-self.RIGHT)
                    b=a+3*self.span()/max(1,self.width()-self.LEFT-self.RIGHT)
                    index=bisect_right(self._contact_ends[foot],a)
                    if index<len(self.contacts[foot]) and self.contacts[foot][index][0]<b:p.fillRect(QRectF(x,y+2,3,self.TRACK_HEIGHT-5),tint)
            else:
                for index in range(lo,hi):
                    self.contact_draw_queries+=1
                    if self._foot_drag and (foot,index)==(self._foot_drag['foot'],self._foot_drag['index']) and self._foot_drag['original'] is not None:continue
                    rect=self._contact_rect(foot,self.contacts[foot][index]);p.fillRect(rect,tint)
                    p.setPen(QPen(QColor(color),1));p.drawRect(rect)
                    if rect.width()>42:
                        s,e=self.contacts[foot][index]
                        if is_light_theme():p.setPen(QColor('#333e48'))
                        p.drawText(rect,Qt.AlignCenter,f'{s+1}–{e}' if self._selected_contact==(foot,index) else 'Lock')
            p.restore()

    def paint_contact_selection(self,p):
        target=None
        if self._foot_drag and self._foot_draft is not None:target=(self._foot_drag['foot'],self._foot_draft)
        elif self._selected_contact is not None:
            foot,index=self._selected_contact
            if index<len(self.contacts[foot]):target=(foot,self.contacts[foot][index])
        if target:
            foot,interval=target
            p.save();p.setClipRect(QRectF(self.LEFT,self.RULER_HEIGHT,max(1,self.width()-self.LEFT-self.RIGHT),self.height()-self.RULER_HEIGHT))
            rect=self._contact_rect(foot,interval)
            if self._foot_drag:p.fillRect(rect,QColor(126,165,208,80))
            p.setPen(QPen(QColor(theme_color('#d9e8fa')),1.5));p.setBrush(Qt.NoBrush);p.drawRect(rect)
            for x in (rect.left(),rect.right()):p.drawLine(QPointF(x,rect.top()+3),QPointF(x,rect.bottom()-3))
            if self._foot_drag and rect.width()>42:
                s,e=interval;p.drawText(rect,Qt.AlignCenter,f'{s+1}–{e}')
            p.restore()
