"""SMPL-X mesh viewport with an orbit camera and world-space grid."""
from __future__ import annotations

import math
import os
import time
import zipfile
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, QLineF, Signal, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF, QRadialGradient, QSurfaceFormat
from PySide6.QtWidgets import QWidget,QLabel
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from .preview_data import mapped_array, motion_summary, close_array
from .foot_contacts import FOOT_COLORS
from .footlock_editor import CONTACT_COLORS

OPENGL_VIEW = os.environ.get("QT_QPA_PLATFORM") != "offscreen"
ViewportBase = QOpenGLWidget if OPENGL_VIEW else QWidget


class MotionView(ViewportBase):
    view_changed = Signal(str)
    GRID_HALF_EXTENT = 10.0
    GRID_SPACING = 0.25

    def __init__(self):
        super().__init__()
        if OPENGL_VIEW:
            fmt = QSurfaceFormat()
            fmt.setVersion(4,3)
            fmt.setProfile(QSurfaceFormat.CoreProfile)
            fmt.setDepthBufferSize(24)
            fmt.setStencilBufferSize(8)
            fmt.setSamples(4)
            fmt.setSwapInterval(0)
            self.setFormat(fmt)
        self.setMinimumSize(350, 280)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self._pear_renderer=None
        self._pear_notice=QLabel(self);self._pear_notice.setWordWrap(True);self._pear_notice.setAlignment(Qt.AlignCenter);self._pear_notice.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._pear_notice.setStyleSheet("color:#dddddd; background:transparent; font-size:14px; padding:20px;");self._pear_notice.hide()
        self.live_active=False;self.live_data=None;self.live_asset=None;self.live_frame_first=True;self.live_message="";self.live_key=None
        self.joints = None
        self.rotations = None
        self.mesh_frames = self.mesh_asset = self.mesh_renderer = None
        self.mesh_binding=None
        self.mesh_error = None
        self.mesh_presented_index = None
        self.body_color = QColor('#b5b5b5')
        self.path = None
        self.signature = None
        self._owned_array = None
        self._owned_rotations = None
        self.scene_span = 1.7
        self.index = 0
        self.yaw, self.pitch = -2.76, 0.20
        self.distance = 3.2
        self.pan = np.zeros(3)
        self.follow = True
        self.floor = 0.0
        # The grid is a world reference, not the solved contact plane. Moving
        # it with correction concealed manual world-Y changes completely.
        self.grid_floor = 0.0
        self._camera_project = None
        self._camera_initialized = False
        self.body_height = 1.7
        self.scene_center = np.zeros(3)
        self.drag_point = None
        self._camera_paint_at = 0.
        self._camera_update_timer = QTimer(self)
        self._camera_update_timer.setSingleShot(True)
        self._camera_update_timer.setTimerType(Qt.PreciseTimer)
        self._camera_update_timer.timeout.connect(self.update)
        self.auto_fit = True
        self.view_name = "Perspective"
        self.show_grid = True
        self.show_correction_floor = False
        self.correction_floor = None
        self.correction_floor_auto = True
        self.correction_floor_pending = False
        self.correction_floor_enabled = True
        self.show_trajectory = True
        self.show_foot_locks = True
        self.show_names = True
        self.show_cameras=False;self.camera_records=[];self.camera_message='';self.camera_translation_scale=0.
        self.actor_name = 'Person 1'
        self.scene_names = {}
        self.name_labels_presented = []
        self._gizmo_hits = []
        self._paint_camera = None
        self.foot_lock_activity = None
        self.hand_lock_activity = None
        self.people_scene=None
        self.people_mode=False;self.scene_message='Loading shared scene...';self.scene_fit_root=None
        self.single_actor_visible=True
        self.scene_visible=set();self.scene_colors={};self.scene_selected=None
        self.scene_time=0.;self.scene_renderers={};self.scene_presented={}
        self.character_workspace=None;self.character_renderers={};self.character_error=None
        self.character_name_points=[]
        self.render_mode='raw';self.show_bones=True
        self.target_frames=None;self.target_asset=None;self.raw_asset=None;self.native_vertices=None
        self.target_renderers={};self.bone_renderers={};self.bone_error=None;self.native_required=False
        # A bounded world-space disk, reused for either foot on every frame.
        angles = np.arange(64) * (2 * math.pi / 64)
        self._contact_circle = np.column_stack((np.cos(angles), np.zeros(64), np.sin(angles))) * .16
        # Fixed world geometry, independent of camera zoom and actor following.
        minor, major = [], []
        steps = round(self.GRID_HALF_EXTENT / self.GRID_SPACING)
        for tick in range(-steps, steps + 1):
            value = tick * self.GRID_SPACING
            batch = major if tick % 4 == 0 else minor
            edge = self.GRID_HALF_EXTENT
            batch.extend([[[value, 0, -edge], [value, 0, edge]],
                          [[-edge, 0, value], [edge, 0, value]]])
        self.grid_minor = np.asarray(minor)
        self.grid_major = np.asarray(major)

    def set_joints(self, path: Path | None):
        signature = (str(path), path.stat().st_mtime_ns, path.stat().st_size) if path and path.exists() else None
        if signature == self.signature:
            return
        old = self._owned_array
        old_rotations = self._owned_rotations
        array = mapped_array(path, "joints") if signature is not None else None
        rotations = None
        if signature is not None:
            with zipfile.ZipFile(path) as archive:
                if "rotations.npy" in archive.namelist():
                    rotations = mapped_array(path, "rotations")
        self._owned_array = array
        self._owned_rotations = rotations
        self.set_motion(array, motion_summary(array) if array is not None else None, path, rotations)
        self.signature = signature
        close_array(old)
        close_array(old_rotations)

    def set_motion(self, joints, summary=None, path=None, rotations=None, mesh_frames=None, mesh_asset=None):
        if self.mesh_frames is not mesh_frames or self.mesh_asset is not mesh_asset:
            self.mesh_error=None
        self.mesh_binding=None
        if joints is None:
            self.target_frames=None;self.target_asset=None;self.raw_asset=None;self.native_vertices=None
            self.release_target_renderers()
        if self._owned_array is not None and self._owned_array is not joints:
            close_array(self._owned_array)
            self._owned_array = None
        if self._owned_rotations is not None and self._owned_rotations is not rotations:
            close_array(self._owned_rotations)
            self._owned_rotations = None
        self.signature = None
        if joints is None and path is None:
            self._camera_initialized=False;self._camera_project=None
        project=path.parent if path is not None else None
        fit=not self._camera_initialized or project!=self._camera_project
        self.path = path
        self.joints = joints
        self.rotations = rotations
        self.mesh_frames, self.mesh_asset = mesh_frames, mesh_asset
        self.mesh_presented_index = None
        if joints is None and not self.people_mode and self.mesh_renderer is not None and OPENGL_VIEW and self.isValid():
            self.makeCurrent()
            self.mesh_renderer.release_buffers()
            self.doneCurrent()
        if self.joints is not None:
            self.floor = self.people_scene.reference['floor'] if self.people_scene else summary["floor"]
            self.body_height = summary["height"] + .16  # Mesh extends above the head joint.
            self.scene_span = summary["span"]
            if fit:
                if not self.people_mode:
                    self.scene_center = summary["center"].copy()
                    self.scene_center[1] = self.floor + self.body_height * 0.50
                self._camera_project=project;self._camera_initialized=True
            if fit and self.auto_fit:
                self.fit_view()
        if self.people_scene and self.people_scene.summary:
            self.apply_scene_summary()
        self.update()

    def apply_scene_summary(self):
        summary=self.people_scene.summary
        self.floor=summary['floor'];self.body_height=summary['height']+.16
        self.scene_span=summary['span']
        if self.scene_fit_root!=self.people_scene.root:
            self.scene_center=np.asarray(summary['center']).copy()
            self.scene_center[1]=self.floor+self.body_height*.5

    def set_people_scene(self,scene):
        self.release_scene_renderers()
        self.people_scene=scene;self.scene_presented={}
        if scene and scene.summary:
            self.apply_scene_summary()
            if self.scene_fit_root!=scene.root:
                if self.auto_fit:self.fit_view()
                self.scene_fit_root=scene.root
        self.update()

    def release_scene_renderers(self):
        self.release_target_renderers()
        if self.scene_renderers and OPENGL_VIEW and self.isValid():
            self.makeCurrent()
            for renderer in self.scene_renderers.values():renderer.close()
            self.doneCurrent()
        self.scene_renderers={}

    def release_target_renderers(self):
        if (self.target_renderers or self.bone_renderers) and OPENGL_VIEW and self.isValid():
            self.makeCurrent()
            for renderer in [*reversed(list(self.target_renderers.values())),*self.bone_renderers.values()]:renderer.close()
            self.doneCurrent()
        self.target_renderers={};self.bone_renderers={};self.bone_error=None

    def draw_target(self,slot,asset,frame,key,color,transform=None,clear_depth=True):
        from .rig_gl import RigRenderer
        topology=asset.get('topology_key',asset['key'])
        if slot in self.target_renderers:
            previous=self.target_renderers[slot]
            if previous.shared and previous.shared.topology_key!=topology:self.target_renderers.pop(slot).close()
        if slot not in self.target_renderers:
            shared=next((r for r in self.target_renderers.values() if r.shared is None and r.topology_key==topology),None) if asset.get('mode')=='raw' else None
            self.target_renderers[slot]=RigRenderer(shared)
        renderer=self.target_renderers[slot];renderer.load(asset);renderer.deform(frame,key)
        renderer.draw(*self._paint_camera,self.width(),self.height(),color,transform=transform,clear_depth=clear_depth)

    def draw_bones(self,painter,actors=None):
        if not self.show_bones:return
        entries=[];missing=False
        if actors is not None:
            for take,frame in actors:
                if take.target_asset is None or 'bones' not in take.target_asset or 'rig' not in take.arrays:
                    missing=True;continue
                entries.append(((take.ident,str(take.project)),take.target_asset['bones'],take.arrays['rig'][frame],frame,take.transform))
        elif self.joints is not None:
            if self.target_asset is not None and 'bones' in self.target_asset and self.target_frames is not None:
                entries.append(('single',self.target_asset['bones'],self.target_frames[self.index],self.index,None))
            else:missing=True
        if OPENGL_VIEW and entries:
            painter.beginNativePainting()
            try:
                from .bone_gl import BoneRenderer
                used=set()
                # One fresh depth pass puts every bone in front of all Raw
                # meshes while retaining occlusion between bones and actors.
                for ordinal,(slot,asset,frame,index,transform) in enumerate(entries):
                    used.add(slot)
                    if slot not in self.bone_renderers:self.bone_renderers[slot]=BoneRenderer()
                    renderer=self.bone_renderers[slot];renderer.load(asset);renderer.deform(frame,(asset['key'],index))
                    renderer.draw(*self._paint_camera,self.width(),self.height(),transform=transform,clear_depth=ordinal==0)
                for slot in list(self.bone_renderers):
                    if slot not in used:self.bone_renderers.pop(slot).close()
                self.bone_error=None
            except Exception as error:self.bone_error='Cannot render export bones: '+str(error)
            finally:painter.endNativePainting()
        if missing or self.bone_error:
            painter.setPen(QColor('#e0b06c'))
            painter.drawText(self.rect().adjusted(18,105,-95,-80),Qt.AlignTop|Qt.TextWordWrap,
                self.bone_error or 'Export bone overlay unavailable for this result. Use Prepare Preview / Apply Correction.')

    def scene_actors(self):
        return self.people_scene.actors_at(self.scene_time,self.scene_visible,self.path.parent if self.path else None) if self.people_scene else []

    def draw_names(self,painter,actors=None):
        from .name_overlay import body_points,label_rect,separate_labels
        self.name_labels_presented=[]
        if not self.show_names:return
        entries=[]
        if actors is not None:
            for take,frame in actors:
                if self.scene_presented.get(take.ident)!=(str(take.project),frame):continue
                joints=take.arrays.get('raw_joints') if take.native_required else take.arrays.get('joints')
                if joints is None:joints=take.arrays['joints']
                rig=take.arrays.get('rig')
                points=body_points(joints[frame],rig[frame] if rig is not None else None,
                                   take.target_asset,take.transform)
                entries.append((self.scene_names.get(take.ident,f'Person {take.ident}'),points))
        elif self.single_actor_visible and self.joints is not None and not self.mesh_error and self.mesh_presented_index==self.index:
            points=body_points(self.joints[self.index],
                              self.target_frames[self.index] if self.target_frames is not None else None,self.target_asset)
            entries.append((self.actor_name,points))
        entries.extend(self.character_name_points)
        painter.save()
        font=QFont(painter.font());font.setPixelSize(14);font.setWeight(QFont.DemiBold)
        painter.setFont(font)
        metrics=painter.fontMetrics()
        for name,points in entries:
            text=metrics.elidedText(name,Qt.ElideRight,min(300,max(0,self.width()-36)))
            xy,depth,focal=self.project(points)
            rect=label_rect(xy,depth,focal,self.width(),self.height(),metrics.horizontalAdvance(text),metrics.height())
            if rect is None or not text:continue
            rect=separate_labels(rect,[r for _,r in self.name_labels_presented],self.width())
            painter.setPen(QPen(QColor(255,255,255,65),1))
            painter.setBrush(QColor(24,27,31,215))
            painter.drawRoundedRect(rect,5,5)
            painter.setPen(QColor('#f4f4f4'))
            painter.drawText(rect.adjusted(10,0,-10,0),Qt.AlignCenter,text)
            self.name_labels_presented.append((name,rect))
        painter.restore()

    def set_body_color(self,color):
        chosen=QColor(color)
        if chosen.isValid():
            self.body_color=chosen
            self.update()

    def target(self):
        if self.live_active:
            target=self.scene_center.copy()
            if self.follow and self.live_data is not None:target[[0,2]]=self.live_data["joints"][0,[0,2]]
            return target+self.pan
        target = self.scene_center.copy()
        if self.follow and self.people_scene is not None:
            actors=self.scene_actors()
            roots=[self.scene_joint_points(take,frame)[0] for take,frame in actors]
            if roots:target[[0,2]]=np.mean(roots,axis=0)[[0,2]]
        elif self.follow and self.joints is not None:
            target[[0, 2]] = self.joints[min(self.index, len(self.joints) - 1), 0, [0, 2]]
        return target + self.pan

    def scene_joint_points(self,take,frame):
        points=take.arrays.get('raw_joints') if getattr(take,'native_required',False) else take.arrays.get('joints')
        return points[frame]@take.transform[:3,:3].T+take.transform[:3,3] if points is not None else take.points(frame)

    def fit_view(self):
        self.pan[:] = 0
        span = max(self.body_height,self.scene_span) if self.people_scene else self.body_height
        if not self.follow and self.joints is not None:
            span = self.scene_span
        aspect = max(0.4, self.width() / max(1, self.height()))
        self.distance = max(1.0, span * 0.60 / math.tan(math.radians(20)) / min(1.0, aspect))
        self.auto_fit = True
        self.update()

    def frame_all(self):
        if self.live_active:
            if self.live_data is not None:
                v=self.live_data.get("bounds",self.live_data.get("vertices"));self.scene_center=(v.min(0)+v.max(0))*.5;self.body_height=float(max(1.,np.ptp(v,axis=0).max()));self.scene_span=self.body_height
            self.fit_view();return
        # Explicit user framing may recenter. Replacing a corrected preview or
        # resizing the widget must retain the existing world-space camera Y.
        if self.people_scene is not None and self.people_scene.summary:
            self.scene_center=np.asarray(self.people_scene.summary['center']).copy()
        self.scene_center[1]=self.floor+self.body_height*.5
        self.fit_view()
        if self.character_workspace:
            bounds=[]
            self.makeCurrent()
            try:
                for c,data,matrix,frame,transform in self.character_workspace.character_draws():
                    renderer=self.character_renderers.get(data['asset']['key'])
                    if renderer is None:continue
                    v=renderer.read_vertices()[:,:3]
                    if transform is not None:
                        T=np.asarray(transform);v=v@T[:3,:3].T+T[:3,3]
                    bounds.extend([v.min(0),v.max(0)])
            finally:self.doneCurrent()
            if bounds:
                points=np.array(bounds);center=(points.min(0)+points.max(0))*.5
                if self.joints is None and self.people_scene is None:self.scene_center=center
                span=2*float(np.max(np.linalg.norm(points-self.target(),axis=1)))
                aspect=max(.4,self.width()/max(1,self.height()))
                self.distance=max(self.distance,span*.60/math.tan(math.radians(20))/min(1.,aspect))
                self.update()

    def set_follow(self, enabled):
        self.follow = enabled
        self.frame_all()

    def set_view(self, name, fit=True):
        self.yaw, self.pitch = {"Perspective": (-2.76, .20), "Front": (math.pi, 0),
            "Back": (0, 0), "Right": (math.pi / 2, 0), "Left": (-math.pi / 2, 0),
            "Top": (0, math.pi / 2 - .001), "Bottom": (0, -math.pi / 2 + .001)}[name]
        self.view_name = name
        if fit:
            self.frame_all()
        else:
            self.auto_fit = False
            self.update()
        self.view_changed.emit(name)

    def camera(self):
        target = self.target()
        offset = self.distance * np.array([math.sin(self.yaw) * math.cos(self.pitch),
                                          math.sin(self.pitch), math.cos(self.yaw) * math.cos(self.pitch)])
        eye = target + offset
        forward = -offset / self.distance
        right = np.cross(forward, [0, 1, 0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        return eye, np.array([right, up, forward])

    def project(self, points):
        eye, basis = self._paint_camera if self._paint_camera is not None else self.camera()
        view = (np.asarray(points) - eye) @ basis.T
        focal = self.height() * 0.5 / math.tan(math.radians(20))
        depth = np.maximum(view[..., 2], 0.03)
        xy = np.stack([self.width() / 2 + view[..., 0] * focal / depth,
                       self.height() / 2 - view[..., 1] * focal / depth], axis=-1)
        return xy, view[..., 2], focal

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            for center, name in reversed(self._gizmo_hits):
                if (event.position() - center).manhattanLength() <= 14:
                    self.set_view(name, fit=False)
                    self.drag_point = None
                    event.accept()
                    return
        self.drag_point = event.position()

    def mouseReleaseEvent(self, event):
        self.drag_point = None
        if self._camera_update_timer.isActive():
            self._camera_update_timer.stop()
            self.update()

    def request_camera_update(self):
        # Keep every input delta, but submit only the latest camera per display
        # interval. A single timer bounds work without an idle render loop.
        screen = self.screen()
        rate = screen.refreshRate() if screen is not None else 60.
        rate = min(240., max(30., rate if math.isfinite(rate) and rate > 0 else 60.))
        remaining = 1. / rate - (time.monotonic() - self._camera_paint_at)
        if remaining <= 0:
            self._camera_update_timer.stop()
            self.update()
        elif not self._camera_update_timer.isActive():
            self._camera_update_timer.start(max(1, math.ceil(remaining * 1000)))

    def mouseMoveEvent(self, event):
        if self.drag_point is None:
            hovered = next((name for center, name in reversed(self._gizmo_hits)
                            if (event.position() - center).manhattanLength() <= 14), None)
            self.setCursor(Qt.PointingHandCursor if hovered else Qt.ArrowCursor)
            return
        delta = event.position() - self.drag_point
        if event.buttons() & Qt.LeftButton and not event.modifiers() & Qt.ShiftModifier:
            self.yaw -= delta.x() * 0.008
            self.pitch = np.clip(self.pitch + delta.y() * 0.006, -1.45, 1.55)
            if self.view_name != "Perspective":
                self.view_name = "Perspective"
                self.view_changed.emit("Perspective")
        elif event.buttons() & (Qt.RightButton | Qt.MiddleButton | Qt.LeftButton):
            _, basis = self.camera()
            scale = 2 * self.distance * math.tan(math.radians(20)) / max(1, self.height())
            self.pan += (-basis[0] * delta.x() + basis[1] * delta.y()) * scale
        self.auto_fit = False
        self.drag_point = event.position()
        self.request_camera_update()

    def wheelEvent(self, event):
        self.distance = np.clip(self.distance * math.exp(-event.angleDelta().y() / 120 * 0.12), 0.4, 80)
        self.auto_fit = False
        self.request_camera_update()
        event.accept()

    def mouseDoubleClickEvent(self, event):
        self.frame_all()

    def resizeEvent(self, event):
        if self.auto_fit:
            self.fit_view()
        super().resizeEvent(event)

    def closeEvent(self, event):
        self._camera_update_timer.stop()
        self.set_motion(None)
        self.cleanup_gl()
        super().closeEvent(event)

    def initializeGL(self):
        self.context().aboutToBeDestroyed.connect(self.cleanup_gl)

    def cleanup_gl(self):
        self.release_live_renderer()
        self.clear_characters()
        self.release_scene_renderers()
        if self.mesh_renderer is not None:
            self.makeCurrent()
            self.mesh_renderer.close()
            self.mesh_renderer = None
            self.doneCurrent()

    def clear_characters(self):
        self.character_name_points=[]
        if not self.character_renderers:return
        self.makeCurrent()
        for renderer in self.character_renderers.values():renderer.close()
        self.character_renderers.clear()
        self.doneCurrent()

    def draw_characters(self,painter):
        self.character_name_points=[]
        workspace=self.character_workspace
        if workspace is None or not OPENGL_VIEW:return False
        draws=list(workspace.character_draws())
        drawn=False
        painter.beginNativePainting()
        try:
            from .character_gl import CharacterRenderer
            active={data['asset']['key'] for _,data,*_ in draws}
            for key in list(self.character_renderers):
                if key not in active:self.character_renderers.pop(key).close()
            # Characters occupy their own depth layer above all source meshes.
            # Keep depth between characters so they still occlude each other.
            clear=True
            for character,data,matrix,frame,transform in draws:
                key=data['asset']['key']
                if key not in self.character_renderers:self.character_renderers[key]=CharacterRenderer()
                renderer=self.character_renderers[key]
                renderer.load(data['asset'],data['palette']);renderer.deform(matrix,(key,frame))
                renderer.draw(*self._paint_camera,self.width(),self.height(),transform=transform,clear_depth=clear)
                if self.show_names:
                    from .name_overlay import character_points
                    points=character_points(data['asset'].get('name_samples'),matrix,
                        np.asarray(transform) if transform is not None else None)
                    self.character_name_points.append((character['name'],points))
                drawn=True
                clear=False
            self.character_error=None
        except Exception as exc:self.character_error=str(exc)
        finally:painter.endNativePainting()
        if self.character_error:
            painter.setPen(QColor('#e0b06c'));painter.drawText(18,100,'Character: '+self.character_error[:160])
        return drawn

    def paintEvent(self,event):
        self._camera_paint_at = time.monotonic()
        if OPENGL_VIEW:
            super().paintEvent(event)
        else:
            self.paint_scene(event)

    def paintGL(self):
        if self.live_active:
            from .pear.live_view import LiveRenderer
            if self._pear_renderer is None:self._pear_renderer=LiveRenderer(self)
            self._paint_camera=self.camera()
            try:self._pear_renderer.paint(self)
            except Exception as e:self.live_message='PEAR preview: '+str(e)
            self._pear_notice.setGeometry(self.rect());self._pear_notice.setText(self.live_message)
            self._pear_notice.setVisible(bool(self.live_message));return
        self._pear_notice.hide()
        self.paint_scene(None)

    def foot_lock_discs(self,index,joints,activity=None,floor=None):
        """Return at most two fixed-size circles on the displayed floor plane."""
        if not self.show_foot_locks:return []
        active = activity if activity is not None else self.foot_lock_activity(index) if self.foot_lock_activity is not None else (False,False)
        discs = []
        if floor is None and getattr(self,'foot_lock_levels',None) is not None:
            floor=self.foot_lock_levels(index)
        for foot, locked in enumerate(active[:2]):
            if not locked:continue
            center = (joints[7+foot] + joints[10+foot]) * .5
            level=self.floor if floor is None else floor[foot] if isinstance(floor,(list,tuple,np.ndarray)) else floor
            center[1] = level + .008
            if np.all(np.isfinite(center)):
                discs.append((foot, self._contact_circle + center))
        return discs

    def set_correction_floor(self,height,automatic=True,pending=False,enabled=True):
        self.correction_floor=float(height) if height is not None and math.isfinite(height) else None
        self.correction_floor_auto=automatic
        self.correction_floor_pending=pending
        self.correction_floor_enabled=enabled
        self.update()

    def hand_lock_discs(self,index,frames,activity=None,transform=None):
        """Palm centres from the displayed rig, independent of floor and finger curl.

        The four non-thumb knuckle pivots and wrist bound the palm. They retain
        source positions; using their centre avoids following fingertips or
        confusing the wrist pivot with the palm. No inference/mesh work here.
        """
        if not self.show_foot_locks or frames is None:return []
        active=activity if activity is not None else self.hand_lock_activity(index) if self.hand_lock_activity else (False,False)
        if frames.shape!=(52,4,4):return []
        points=frames[:,:3,3]
        discs=[]
        for hand,locked in enumerate(active[:2]):
            if not locked:continue
            wrist=points[20+hand]
            knuckles=points[np.array([25,28,31,34])+15*hand].mean(axis=0)
            center=(wrist+knuckles)*.5
            if transform is not None:center=center@transform[:3,:3].T+transform[:3,3]
            # Horizontal support indicator at palm height: a table must not
            # be projected onto the foot floor. This does not infer a surface.
            if np.isfinite(center).all():discs.append((hand,center+self._contact_circle*(.10/.16)))
        return discs

    def draw_hand_locks(self,painter,index,frames,activity=None,transform=None):
        self.draw_contact_discs(painter,self.hand_lock_discs(index,frames,activity,transform),CONTACT_COLORS[2:])

    def draw_people_hand_locks(self,painter,actors):
        if not self.show_foot_locks:return
        for take,frame in actors:
            rig=take.arrays.get('rig')
            if rig is None:continue
            selected=take.ident==self.scene_selected and self.path is not None and take.project==self.path.parent
            activity=None if selected else take.active_contacts(frame)[2:4]
            self.draw_hand_locks(painter,frame,rig[frame],activity,take.transform)

    def correction_floor_geometry(self):
        if not getattr(self,'correction_floor_applicable',True):return None
        if not self.show_correction_floor or self.correction_floor is None:return None
        center=self.scene_center
        if self.people_mode:
            if self.people_scene is None:return None
            actors=self.scene_actors()
            if not actors:return None
            center=np.mean([self.scene_joint_points(take,frame)[0] for take,frame in actors],axis=0)
        elif self.joints is not None:
            center=self.joints[min(self.index,len(self.joints)-1),0]
        else:return None
        from .floor_overlay import geometry
        return geometry(center,self.correction_floor,self.grid_floor)

    def draw_correction_floor(self,painter,lines3d):
        geometry=self.correction_floor_geometry()
        if geometry is None:return False
        from .floor_overlay import COLOR,clip_polygon
        corners,lines,ruler=geometry
        eye,basis=self._paint_camera
        view=clip_polygon(corners,eye,basis)
        if len(view)>=3:
            focal=self.height()*.5/math.tan(math.radians(20))
            xy=view[:,:2]*focal/view[:,2:]
            xy[:,0]+=self.width()/2;xy[:,1]=self.height()/2-xy[:,1]
            tint=QColor(COLOR);tint.setAlpha(24)
            painter.setBrush(tint);painter.setPen(Qt.NoPen)
            painter.drawPolygon(QPolygonF([QPointF(*p) for p in xy]))
        lines3d(lines,'#9c8058',.75)
        lines3d(np.stack((corners,np.roll(corners,-1,axis=0)),axis=1),COLOR,1.8)
        lines3d(ruler,COLOR,1.7)
        if abs(self.correction_floor-self.grid_floor)>=.0005:
            xy,depth,_=self.project(ruler[0,0])
            if depth>.08 and 0<xy[0]<self.width()-65 and 20<xy[1]<self.height()-90:
                painter.save();font=painter.font();font.setPixelSize(11);painter.setFont(font)
                painter.setPen(QColor(COLOR))
                painter.drawText(QPointF(xy[0]+7,xy[1]-6),f'{(self.correction_floor-self.grid_floor)*100:+.1f} cm')
                painter.restore()
        return True

    def draw_foot_locks(self,painter,index,joints,activity=None,floor=None):
        self.draw_contact_discs(painter,self.foot_lock_discs(index,joints,activity,floor),FOOT_COLORS)

    def draw_contact_discs(self,painter,discs,colors):
        for side,points in discs:
            # Clip against the camera near plane before perspective division.
            eye,basis = self._paint_camera if self._paint_camera is not None else self.camera()
            view = (points-eye) @ basis.T
            inside = view[:,2] >= .08
            if not inside.any():continue
            if not inside.all():
                clipped=[]
                for a,b,visible_a,visible_b in zip(view,np.roll(view,-1,axis=0),inside,np.roll(inside,-1)):
                    if visible_a:clipped.append(a)
                    if visible_a != visible_b:
                        clipped.append(a+(b-a)*((.08-a[2])/(b[2]-a[2])))
                view=np.asarray(clipped)
            focal=self.height()*.5/math.tan(math.radians(20))
            xy=view[:,:2]*focal/view[:,2:]
            xy[:,0]+=self.width()/2
            xy[:,1]=self.height()/2-xy[:,1]
            color=QColor(colors[side]);color.setAlpha(125)
            painter.setBrush(color)
            painter.setPen(QPen(QColor(colors[side]),1.5))
            painter.drawPolygon(QPolygonF([QPointF(*point) for point in xy]))

    def draw_mesh(self,painter,index):
        if self.native_vertices is not None or self.native_required:
            if self.raw_asset is None or self.native_vertices is None:
                painter.setPen(QColor('#e0b06c'));painter.drawText(self.rect().adjusted(25,100,-25,-60),Qt.AlignCenter|Qt.TextWordWrap,
                    'Target preview unavailable. Use Prepare Preview to build it without AI inference.');return
            if OPENGL_VIEW:
                painter.beginNativePainting()
                try:
                    asset=self.raw_asset['raw_asset']
                    frame=self.native_vertices[index]
                    self.draw_target('single',asset,frame,(asset['key'],index),
                        (self.body_color.redF(),self.body_color.greenF(),self.body_color.blueF()))
                    self.mesh_presented_index=index;self.mesh_error=None
                except Exception as error:self.mesh_error='Cannot render target: '+str(error)
                finally:painter.endNativePainting()
                if self.mesh_error:
                    painter.setPen(QColor('#e0b06c'));painter.drawText(18,90,self.mesh_error)
            return
        if self.mesh_frames is None or self.mesh_asset is None:
            message = "Mesh data unavailable. Click Prepare Mesh."
        elif not OPENGL_VIEW:
            message = "SMPL-X Mesh requires a native OpenGL viewport."
        elif self.mesh_error:
            message = self.mesh_error
        else:
            painter.beginNativePainting()
            try:
                from .mesh_gl import MeshRenderer
                from .bound_mesh_gl import BoundMeshRenderer
                renderer_type=BoundMeshRenderer if self.mesh_binding is not None else MeshRenderer
                self.ensure_mesh_renderer(renderer_type)
                self.mesh_renderer.load(self.mesh_asset)
                if self.mesh_binding is not None:self.mesh_renderer.bind_skin(self.mesh_binding)
                self.mesh_renderer.deform(self.mesh_frames[index],(str(self.path),id(self.mesh_frames),index))
                self.mesh_renderer.draw(*self._paint_camera,self.width(),self.height(),
                    (self.body_color.redF(),self.body_color.greenF(),self.body_color.blueF()))
                self.mesh_presented_index = index
            except Exception as error:
                self.mesh_error = "Cannot render SMPL-X Mesh: " + str(error)
            finally:
                painter.endNativePainting()
            if not self.mesh_error:
                return
            message = self.mesh_error
        painter.setPen(QColor("#e0b06c"))
        painter.drawText(self.rect().adjusted(25,100,-25,-60),Qt.AlignCenter|Qt.TextWordWrap,message)

    def ensure_mesh_renderer(self,renderer_type):
        if self.mesh_renderer is not None and type(self.mesh_renderer)!=renderer_type:
            for renderer in self.scene_renderers.values():renderer.close()
            self.scene_renderers={};self.mesh_renderer.close();self.mesh_renderer=None
        if self.mesh_renderer is None:self.mesh_renderer=renderer_type()

    def draw_people_scene(self,painter,actors,lines3d):
        self.scene_presented={};self.mesh_presented_index=None
        for take,frame in actors:
            points=take.arrays.get('raw_joints') if getattr(take,'native_required',False) else take.arrays.get('joints')
            joints=self.scene_joint_points(take,frame)
            if self.show_trajectory:
                path_points=points if points is not None else take.arrays['joints']
                trail=path_points[max(0,frame-90):frame+1,0]@take.transform[:3,:3].T+take.transform[:3,3]
                trail[:,1]=self.floor+.006
                lines3d(np.stack([trail[:-1],trail[1:]],axis=1),self.scene_colors.get(take.ident,'#b5b5b5'),1.3)
            from .foot_contacts import contact_levels
            levels=contact_levels(getattr(take,'contact_track',None),frame)
            floor=self.people_scene.contact_floor if levels is None else levels
            if take.ident==self.scene_selected and self.path is not None and take.project==self.path.parent:
                self.draw_foot_locks(painter,frame,joints,floor=floor)
            else:self.draw_foot_locks(painter,frame,joints,take.active_contacts(frame),floor)
        # Both contact types use the same background pass; all actor meshes
        # and export bones subsequently occlude them.
        self.draw_people_hand_locks(painter,actors)
        if OPENGL_VIEW and actors:
            painter.beginNativePainting()
            try:
                if any('native' in take.arrays or take.native_required for take,_ in actors):
                    used=set();drawn=0
                    for take,frame in actors:
                        if take.raw_asset is None or 'native' not in take.arrays:
                            self.mesh_error='Target preview unavailable. Use Prepare Preview.';continue
                        slot=(take.ident,str(take.project));used.add(slot)
                        asset=take.raw_asset['raw_asset']
                        color=QColor(self.scene_colors.get(take.ident,'#b5b5b5'))
                        self.draw_target(slot,asset,take.arrays['native'][frame],(asset['key'],frame),
                            (color.redF(),color.greenF(),color.blueF()),take.transform,clear_depth=drawn==0)
                        drawn+=1;self.scene_presented[take.ident]=(str(take.project),frame)
                        if take.ident==self.scene_selected:self.mesh_presented_index=frame
                    for slot in list(self.target_renderers):
                        if slot not in used:
                            owner=self.target_renderers[slot]
                            if not any(self.target_renderers[key].shared is owner for key in used if key in self.target_renderers):
                                self.target_renderers.pop(slot).close()
                    if drawn==len(actors):self.mesh_error=None
                    return
                from .mesh_gl import MeshRenderer
                renderer_type=MeshRenderer
                self.ensure_mesh_renderer(renderer_type)
                if self.mesh_renderer.asset_key not in (None,actors[0][0].asset['key']):
                    for renderer in self.scene_renderers.values():renderer.close()
                    self.scene_renderers={}
                self.mesh_renderer.load(actors[0][0].asset)
                used=set()
                for slot,(take,frame) in enumerate(actors):
                    if slot==0:renderer=self.mesh_renderer
                    else:
                        used.add(take.ident)
                        if take.ident not in self.scene_renderers:self.scene_renderers[take.ident]=renderer_type(self.mesh_renderer)
                        renderer=self.scene_renderers[take.ident]
                        renderer.load(take.asset)
                    mesh=take.arrays['mesh']
                    renderer.deform(mesh[frame],(str(take.project),id(mesh),frame))
                    color=QColor(self.scene_colors.get(take.ident,'#b5b5b5'))
                    renderer.draw(*self._paint_camera,self.width(),self.height(),
                        (color.redF(),color.greenF(),color.blueF()),transform=take.transform,clear_depth=slot==0)
                    self.scene_presented[take.ident]=(str(take.project),frame)
                    if take.ident==self.scene_selected:self.mesh_presented_index=frame
                for ident in list(self.scene_renderers):
                    if ident not in used:self.scene_renderers.pop(ident).close()
                self.mesh_error=None
            except Exception as error:self.mesh_error='Cannot render people scene: '+str(error)
            finally:painter.endNativePainting()
        elif not actors:
            self._empty_people_notice='No checked person has motion at this time.\nCheck a person or reconstruct the selected result.'
        painter.setPen(QColor('#bebebe'))
        painter.drawText(18,53,f'{len(actors)} people visible | Shared camera / world')
        warnings=self.people_scene.warnings
        if self.mesh_error or warnings:
            painter.setPen(QColor('#e0b06c'))
            text=self.mesh_error or '; '.join(warnings[:2])
            painter.drawText(self.rect().adjusted(18,80,-95,-80),Qt.AlignTop|Qt.TextWordWrap,text)

    def release_live_renderer(self):
        self._pear_notice.hide()
        if self._pear_renderer is not None and OPENGL_VIEW and self.isValid():
            self.makeCurrent();self._pear_renderer.close();self.doneCurrent()
        self._pear_renderer=None

    def paint_scene(self, event):
        painter = QPainter(self)
        self._empty_people_notice=None
        self._paint_camera = self.camera()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#383838"))

        def line3d(a, b, color, width=1):
            lines3d([[a, b]], color, width)

        def lines3d(lines, color, width=1):
            if not len(lines):
                return
            eye, basis = self._paint_camera
            view = (np.asarray(lines, dtype=float) - eye) @ basis.T
            view = view[view[:, :, 2].max(axis=1) > 0.08]
            if not len(view):
                return
            for endpoint in (0, 1):
                mask = view[:, endpoint, 2] < 0.08
                behind, front = view[mask, endpoint], view[mask, 1 - endpoint]
                if len(behind):
                    factor = (0.08 - behind[:, 2]) / (front[:, 2] - behind[:, 2])
                    view[mask, endpoint] = behind + (front - behind) * factor[:, None]
            focal = self.height() * 0.5 / math.tan(math.radians(20))
            xy = view[:, :, :2] * focal / np.maximum(view[:, :, 2:], 0.08)
            xy[:, :, 0] += self.width() / 2
            xy[:, :, 1] = self.height() / 2 - xy[:, :, 1]
            painter.setPen(QPen(QColor(color), width, Qt.SolidLine, Qt.RoundCap))
            painter.drawLines([QLineF(*segment.reshape(-1)) for segment in xy])

        floor = self.floor
        offset = np.array([0, self.grid_floor, 0])
        # Thin grid lines use Qt's fast line rasterizer. Antialiased strokes
        # become expensive when the whole fixed floor is visible at a distance.
        # Keep every world-space grid line; only the stroke rasterization changes.
        painter.setRenderHint(QPainter.Antialiasing, False)
        edge = self.GRID_HALF_EXTENT
        if self.show_grid:
            lines3d(self.grid_minor + offset, "#454545", .65)
            lines3d(self.grid_major + offset, "#515151", 1.0)
            line3d([-edge, self.grid_floor + .001, 0], [edge, self.grid_floor + .001, 0], "#99515b", 1.1)
            line3d([0, self.grid_floor + .001, -edge], [0, self.grid_floor + .001, edge], "#4c759c", 1.1)
        painter.setRenderHint(QPainter.Antialiasing, True)
        if self.live_active:
            painter.setPen(QColor("#dddddd"));painter.drawText(self.rect(),Qt.AlignCenter|Qt.TextWordWrap,self.live_message);painter.end();return
        correction_plane_visible=self.draw_correction_floor(painter,lines3d)
        self.name_labels_presented=[]
        bone_actors=[]

        if self.people_mode:
            if self.people_scene is not None:
                actors=self.scene_actors()
                self.draw_people_scene(painter,actors,lines3d)
                bone_actors=actors
            else:
                painter.setPen(QColor('#bcbcbc'))
                painter.drawText(self.rect().adjusted(25,60,-25,-60),Qt.AlignCenter|Qt.TextWordWrap,self.scene_message)
            if self.mesh_error:
                painter.setPen(QColor('#e0b06c'));painter.drawText(self.rect().adjusted(18,80,-100,-80),Qt.AlignTop|Qt.TextWordWrap,self.mesh_error)
        elif not self.single_actor_visible:
            self._empty_people_notice=f'{self.actor_name} is hidden. Check it in the People list to show.'
        elif self.joints is None and not (self.character_workspace and self.character_workspace.characters):
            painter.setPen(QColor("#bcbcbc"))
            painter.drawText(self.rect().adjusted(25, 30, -25, -30), Qt.AlignCenter | Qt.TextWordWrap,
                             "No motion loaded\n\nReconstruct a motion or open a project\nto compare the initial and refined animation.")
        elif self.joints is not None:
            index = min(self.index, len(self.joints) - 1)
            joints = self.joints[index, :22]
            # Root trajectory preserves world translation when following the actor.
            trail = self.joints[max(0, index - 90):index + 1, 0].copy() if self.show_trajectory else []
            if len(trail):
                trail[:, 1] = floor + .006
            xy, depth, focal = self.project(joints)
            if len(trail) > 1:
                lines3d(np.stack([trail[:-1], trail[1:]], axis=1), "#92a4b8", 1.4)
            # A projected soft footprint makes the ground relationship readable.
            for foot in (joints[7], joints[8]):
                center2d, z, _ = self.project([foot[0], floor + 0.005, foot[2]])
                if z > 0.08:
                    size = focal * 0.13 / z
                    shadow = QRadialGradient(QPointF(*center2d), size)
                    shadow.setColorAt(0, QColor(0, 0, 0, 95))
                    shadow.setColorAt(1, QColor(0, 0, 0, 0))
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(shadow)
                    painter.drawEllipse(QRectF(center2d[0] - size, center2d[1] - size * 0.3, size * 2, size * 0.6))
            # Contact discs are behind the actual mesh, so feet occlude them.
            self.draw_foot_locks(painter,index,joints)
            if self.show_foot_locks and self.target_frames is not None:
                self.draw_hand_locks(painter,index,self.target_frames[index])
            # Floor arrow indicates the actor's forward direction, independent of camera.
            forward = self.rotations[index,0,:,2].copy() if self.rotations is not None else np.cross(joints[16]-joints[17],joints[12]-joints[0])
            forward[1] = 0
            forward /= max(np.linalg.norm(forward), 1e-7)
            root = np.array([joints[0, 0], floor + .012, joints[0, 2]])
            tip = root + forward * .40
            side = np.cross(forward, [0, 1, 0])
            line3d(root, tip - forward * .06, "#66baff", 2.5)
            arrow, arrow_depth, _ = self.project([tip, tip - forward * .11 + side * .055,
                                                 tip - forward * .11 - side * .055])
            if np.all(arrow_depth > .08):
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor("#66baff"))
                painter.drawPolygon(QPolygonF([QPointF(*point) for point in arrow]))
            self.draw_mesh(painter,index)
            bone_actors=None
            painter.setPen(QColor("#bebebe"))
            painter.drawText(QRectF(18, 39, self.width() - 36, 20), Qt.AlignLeft, f"Frame {index + 1} / {len(self.joints)}")
            painter.drawText(QRectF(18, 60, self.width() - 36, 20), Qt.AlignLeft,
                             f"X {joints[0, 0]:+.2f}   Z {joints[0, 2]:+.2f} m")
        if self.show_cameras:
            from .camera_tracks import at,geometry
            for camera,trail in self.camera_records:
                matrix=at(camera,self.scene_time,self.camera_translation_scale)
                if matrix is None:continue
                edges,origin=geometry(camera,matrix)
                lines3d(edges,'#ffae42',1.4)
                if len(trail)>1:lines3d(np.stack((trail[:-1],trail[1:]),axis=1),'#a48250',1.)
                xy,z,_=self.project(origin)
                if z>.08:
                    painter.setPen(QColor('#ffcc82'));painter.drawText(QPointF(xy[0]+5,xy[1]-5),camera['name'])
            if self.camera_message:
                painter.setPen(QColor('#e8c38d'));painter.drawText(self.rect().adjusted(18,125,-18,-18),Qt.AlignTop|Qt.TextWordWrap,self.camera_message)
        characters_drawn=self.draw_characters(painter)
        # Final independent depth layer: exported bones stay above characters,
        # with the same ordering whether Bone Overlay was on or off previously.
        self.draw_bones(painter,bone_actors)
        if self._empty_people_notice and not characters_drawn:
            painter.setPen(QColor('#bcbcbc'))
            painter.drawText(self.rect().adjusted(25,60,-25,-60),Qt.AlignCenter|Qt.TextWordWrap,
                self._empty_people_notice)
        painter.setPen(QColor("#dedede"))
        painter.drawText(18, 23, "User Perspective" if self.view_name == "Perspective" else self.view_name + " Perspective")
        if correction_plane_visible:
            from .floor_overlay import COLOR,label
            title,status=label(self.correction_floor,self.grid_floor,self.correction_floor_auto,
                               self.correction_floor_pending,self.correction_floor_enabled)
            painter.save()
            font=painter.font();font.setPixelSize(12);painter.setFont(font)
            painter.setPen(QColor(COLOR))
            for y,text in ((self.height()-84,title),(self.height()-65,status)):
                painter.drawText(15,y,painter.fontMetrics().elidedText(text,Qt.ElideRight,max(0,self.width()-30)))
            painter.restore()
        painter.save()
        hint_font = painter.font()
        hint_font.setPixelSize(11)
        painter.setFont(hint_font)
        painter.setPen(QColor("#999999"))
        hint = "LMB: Orbit  |  MMB / RMB: Pan  |  Wheel: Zoom  |  Double-click: Frame"
        hint_width = max(0, self.width() - 30)
        hint = painter.fontMetrics().elidedText(hint, Qt.ElideRight, hint_width)
        painter.drawText(QRectF(15, self.height() - 43, hint_width, 17), Qt.AlignLeft | Qt.AlignVCenter, hint)
        painter.restore()
        painter.setPen(QColor("#a8a8a8"))
        painter.drawText(QRectF(15, self.height() - 26, self.width() - 30, 20), Qt.AlignRight,
                         f"Grid Y {self.grid_floor:+.3f} m | 20 m | 0.25 / 1 m | Y up")
        # Six clickable axis endpoints, drawn back to front in camera space.
        _, basis = self.camera()
        origin = QPointF(self.width() - 56, 58)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(40, 40, 40, 160))
        painter.drawEllipse(origin, 45, 45)
        self._gizmo_hits = []
        endpoints = []
        for axis, color, letter, positive, negative in ((0, "#db6375", "X", "Right", "Left"),
            (1, "#91bb51", "Y", "Top", "Bottom"), (2, "#5795de", "Z", "Back", "Front")):
            for sign, name in ((1, positive), (-1, negative)):
                vector = basis[:2, axis] * 31 * sign
                end = origin + QPointF(float(vector[0]), float(-vector[1]))
                endpoints.append((float(basis[2, axis] * sign), end, color, letter, sign, name))
        for _, end, color, letter, sign, name in sorted(endpoints, key=lambda item: -item[0]):
            tint = QColor(color)
            painter.setPen(QPen(tint.darker(150) if sign < 0 else tint, 1.6))
            painter.drawLine(origin, end)
            painter.setBrush(tint.darker(180) if sign < 0 else tint)
            painter.drawEllipse(end, 8, 8)
            if sign > 0:
                painter.setPen(QColor("#171717"))
                painter.drawText(QRectF(end.x() - 8, end.y() - 8, 16, 16), Qt.AlignCenter, letter)
            self._gizmo_hits.append((end, name))
        # Keep names readable even when a badge approaches the viewport HUD.
        self.draw_names(painter,self.scene_actors() if self.people_scene is not None else [] if self.people_mode else None)
        self._paint_camera = None
        painter.end()


