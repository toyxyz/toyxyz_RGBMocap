"""One coalescing scene loader; bounded, source-timed actor windows."""
from __future__ import annotations
import bisect,heapq,itertools,json,subprocess,sys,time
from pathlib import Path
from threading import Condition
import numpy as np
from PySide6.QtCore import QThread,Signal
from .people_store import load_people,safe_child,media_receipt
from .scene_camera import read_camera,validate_transform,take_projects,VERSION
from .preview_data import FrameArray,mapped_array,close_array,motion_summary,LoadCancelled
from .motion_tools import result_is_current
from .result_artifacts import artifacts_current


class SceneTake:
    def __init__(self,ident,project,manifest,transform,arrays,summary,asset):
        self.ident,self.project=ident,project
        self.start,self.fps,self.frames=float(manifest['start']),float(manifest['fps']),int(manifest['frames'])
        self.end=self.start+self.frames/self.fps
        self.transform=validate_transform(transform)
        self.arrays,self.summary,self.asset=arrays,summary,asset
        self.target_asset=None;self.raw_asset=None
        self.native_required=False
        options=manifest.get('correction',{})
        self.contact_enabled=(bool(options.get('foot_lock',True)),)*2+(bool(options.get('hand_lock',False)),)*2
        self.raw_target_stamp=None
        self.target_stamp=None
        self.set_contacts(None)
        self.contact_revision=None
        self.contact_track=None
        from .scene_camera import camera_sources
        self.camera_sources=camera_sources(project)

    def frame_at(self,seconds):
        value=(seconds-self.start)*self.fps
        # Strict half-open bounds: never freeze an actor beyond its take.
        if value < -1e-6 or value>=self.frames-1e-6:return None
        return max(0,int(np.floor(value+1e-6)))

    def points(self,frame):
        return self.arrays['joints'][frame]@self.transform[:3,:3].T+self.transform[:3,3]

    def set_contacts(self,ranges):
        self.contacts=ranges or [[],[]]
        self.contact_starts=[[pair[0] for pair in track] for track in self.contacts]

    def active_contacts(self,frame):
        result=[]
        for foot,track in enumerate(self.contacts):
            index=bisect.bisect_right(self.contact_starts[foot],frame)-1
            result.append(self.contact_enabled[foot] and index>=0 and frame<track[index][1])
        return tuple(result)

    def close(self):
        for array in self.arrays.values():close_array(array)
        self.arrays.clear()


class PeopleScene:
    def __init__(self,root,reference):
        self.root=root;self.reference=reference;self.takes={};self.starts={};self.spans={};self.by_project={}
        self.warnings=[];self.active=[];self.asset=None;self.summary=None
        self.contact_floor=None

    def finish(self):
        centers=[];bounds=[];heights=[]
        for ident,takes in self.takes.items():
            # Later reconstruction entries win an overlap, without drawing the
            # same person twice. Build a logarithmic source-time lookup once.
            events=sorted([(take.start,1,index) for index,take in enumerate(takes)]+
                          [(take.end,0,index) for index,take in enumerate(takes)])
            active=set();heap=[];previous=None;spans=[]
            for instant,group in itertools.groupby(events,key=lambda event:event[0]):
                while heap and -heap[0] not in active:heapq.heappop(heap)
                if previous is not None and instant>previous and heap:
                    take=takes[-heap[0]]
                    if spans and spans[-1][2] is take and spans[-1][1]==previous:spans[-1]=(spans[-1][0],instant,take)
                    else:spans.append((previous,instant,take))
                for _,begin,index in group:
                    if begin:active.add(index);heapq.heappush(heap,-index)
                    else:active.discard(index)
                previous=instant
            self.spans[ident]=spans;self.starts[ident]=[span[0] for span in spans]
            for take in takes:
                self.by_project[take.project]=take
                center=np.asarray(take.summary['center'])@take.transform[:3,:3].T+take.transform[:3,3]
                centers.append(center);bounds.append(take.summary['span']);heights.append(take.summary['height'])
        if centers:
            center=np.mean(centers,axis=0);height=max(heights)
            self.summary=dict(floor=float(self.reference['floor']),height=height,
                center=center,span=max(bounds)+float(np.ptp(centers,axis=0).max()))

    def actors_at(self,seconds,visible,preferred=None):
        result=[]
        for ident in sorted(visible):
            selected=self.by_project.get(preferred) if preferred else None
            if selected is not None and selected.ident!=ident:selected=None
            frame=selected.frame_at(seconds) if selected else None
            if frame is not None:take=selected
            else:
                slot=bisect.bisect_right(self.starts.get(ident,()),seconds+1e-7)-1
                if slot<0:continue
                first,end,take=self.spans[ident][slot]
                if not first-1e-7<=seconds<end-1e-7:continue
                frame=take.frame_at(seconds)
            if frame is not None:result.append((take,frame))
        used={id(take) for take,_ in result}
        for take in self.active:
            if id(take) not in used:
                for array in take.arrays.values():array.release_windows()
        self.active=[take for take,_ in result]
        return result

    def close(self):
        for takes in self.takes.values():
            for take in takes:take.close()
        self.takes.clear();self.starts.clear();self.spans.clear();self.by_project.clear();self.active=[];self.asset=None

    def close_windows(self):
        for take in self.active:
            for array in take.arrays.values():array.release_windows()
        self.active=[]

    def __del__(self):
        self.close()


def prepare_camera_process(root,cancelled):
    from .paths import ROOT
    if getattr(sys,'frozen',False) or __import__('os').environ.get('RGB_MOCAP_MANAGED') == '1':command=[str(ROOT/'runtime/python.exe'),'-I','-X','utf8',str(ROOT/'scene_boot.py'),str(root)]
    else:command=[sys.executable,'-X','utf8','-m','mocap.scene_camera',str(root)]
    log=root/'.scene_camera.log'
    with log.open('wb') as output:
        child=subprocess.Popen(command,stdout=output,stderr=output,cwd=str(ROOT),
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        started=time.monotonic()
        try:
            while child.poll() is None:
                if cancelled():raise LoadCancelled()
                if time.monotonic()-started>90:raise RuntimeError('Shared scene camera preparation timed out; retry after freeing RAM.')
                time.sleep(.04)
            if child.returncode:
                with log.open('rb') as source:
                    source.seek(max(0,log.stat().st_size-4096));message=source.read().decode('utf-8',errors='replace')
                raise RuntimeError('Cannot prepare shared cameras: '+message)
        finally:
            if child.poll() is None:child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait()


class SceneDataLoader(QThread):
    ready=Signal(int,object)
    failed=Signal(int,str)

    def __init__(self,parent=None):
        super().__init__(parent);self.condition=Condition();self.generation=0
        self.pending=None;self.stopping=False;self.model_cache=None

    def request(self,root,kind,asset=None,cached=None):
        with self.condition:
            self.generation+=1;self.pending=(self.generation,Path(root),kind,asset,cached)
            self.condition.notify();return self.generation

    def cancel(self):
        with self.condition:self.generation+=1;self.pending=None

    def stop(self):
        with self.condition:self.stopping=True;self.generation+=1;self.pending=None;self.condition.notify()

    def load(self,root,kind,asset,cancelled):
        import psutil
        from .motion_tools import base_result, require_supported_project
        require_supported_project(root)
        if require_supported_project(root).key=='flowhmr':
            from .flow_scene import load_scene
            return load_scene(root,kind,cancelled)
        if psutil.virtual_memory().available<256*1024**2:
            raise MemoryError('Free at least 256 MiB RAM to open a shared people scene.')
        if (root/'mamma_pending.json').exists():
            raise ValueError('MAMMA publication is incomplete; finish reconstruction before opening this scene.')
        state=load_people(root)
        if state is None:raise ValueError('People group unavailable')
        projects=take_projects(root,state)
        if not projects:return PeopleScene(root,dict(floor=0.))
        from .scene_floor import read_config,current_resolution
        from .paths import MODELS
        needs_floor=read_config(root) is not None and current_resolution(root) is None and (MODELS/'body_models/smplx/SMPLX_NEUTRAL.npz').exists()
        path=root/'people_scene.json'
        if needs_floor or not path.exists() or any(read_camera(project) is None for _,project in projects):
            prepare_camera_process(root,cancelled)
        if cancelled():raise LoadCancelled()
        if not path.exists():return PeopleScene(root,dict(floor=0.))
        reference=json.loads(path.read_text(encoding='utf-8'))
        if reference.get('version')!=VERSION or reference.get('input')!=media_receipt(root):
            prepare_camera_process(root,cancelled);reference=json.loads(path.read_text(encoding='utf-8'))
        anchor=read_camera(safe_child(root,reference['anchor']))
        if anchor is None or anchor['sources']!=reference.get('sources'):
            prepare_camera_process(root,cancelled);reference=json.loads(path.read_text(encoding='utf-8'))
        camera_to_scene=np.linalg.inv(validate_transform(reference['world_to_camera']))
        scene=PeopleScene(root,reference)
        resolution=current_resolution(root)
        scene.contact_floor=resolution['floor'] if resolution else None
        contact_intervals=0
        try:
            if asset:self.model_cache=asset
            static_assets={}
            for ident,project in projects:
                if cancelled():raise LoadCancelled()
                manifest=json.loads((project/'project.json').read_text(encoding='utf-8'))
                raw_kind=kind
                if not result_is_current(project,raw_kind,ignore_contacts=True) or not artifacts_current(project,raw_kind):
                    scene.warnings.append(f'Person {ident}: {raw_kind} result unavailable');continue
                camera=read_camera(project)
                if not np.allclose(camera['K'],reference['K'],atol=1e-4):
                    scene.warnings.append(f'Person {ident}: camera intrinsics differ');continue
                if 'refined' in kind and not camera.get('htd_compatible'):
                    scene.warnings.append(f'Person {ident}: legacy HTD camera; reconstruct refinement');continue
                arrays={}
                try:
                    joints=mapped_array(project/f'{raw_kind}_joints.npz','joints',cancelled)
                    arrays['joints']=joints;joints.max_blocks=1
                    if joints.shape[1:]!=(22,3):raise ValueError('Invalid scene joints')
                    summary=motion_summary(joints)
                    info=json.loads((project/f'{raw_kind}_mesh.json').read_text(encoding='utf-8'))
                    from .mesh_data import MESH_VERSION,FRAME_FLOATS
                    if info.get('version')!=MESH_VERSION:raise ValueError('Mesh version differs')
                    from .project_storage import resolve_reference
                    asset_path=resolve_reference(project,info['asset'])
                    if not asset_path.exists():
                        from .paths import MODELS
                        asset_path=MODELS/'body_models/smplx'/asset_path.name
                    stat=asset_path.stat();key=(str(asset_path.resolve()),stat.st_size,stat.st_mtime_ns)
                    if manifest.get('pipeline',{}).get('model')=='mamma':
                        # The body filename includes the registered source,
                        # constant actor shape and adapter hash. Takes of one
                        # actor must not retain duplicate static skin arrays.
                        from .mamma_body import body_path
                        key=('mamma-body',body_path(project).name)
                    if key in static_assets:self.model_cache=static_assets[key]
                    if self.model_cache is None or self.model_cache['key']!=key:
                        with np.load(asset_path,allow_pickle=False) as data:
                            self.model_cache={name:data[name] for name in ('packed','faces','offsets','neighbors','vertex_count')}
                        self.model_cache['key']=key
                    static_assets[key]=self.model_cache
                    # Each take retains its actor's exact shape asset. Identical
                    # assets still share arrays and GPU buffers.
                    if scene.asset is None:scene.asset=self.model_cache
                    mesh=FrameArray(project/Path(info['frames_file']).name,max_blocks=1);arrays['mesh']=mesh
                    if mesh.shape!=(len(joints),FRAME_FLOATS) or mesh.dtype!=np.float32:raise ValueError('Invalid scene mesh')
                    manifest=json.loads((project/'project.json').read_text(encoding='utf-8'))
                    if len(joints)!=manifest['frames']:raise ValueError('Scene motion frame count differs')
                    if not artifacts_current(project,raw_kind) or not result_is_current(project,raw_kind,ignore_contacts=True):
                        raise ValueError('Motion changed while loading the scene; retry')
                    transform=camera_to_scene@validate_transform(camera['world_to_camera'])
                    take=SceneTake(ident,project,manifest,transform,arrays,summary,self.model_cache)
                    from .bone_overlay import open_preview
                    target=open_preview(project,kind,cancelled)
                    target_path=project/(kind+'_mixamo.json')
                    if target_path.exists():
                        from .result_artifacts import file_stat
                        take.target_stamp=file_stat(target_path)
                    if target:
                        more,target_asset,_=target;arrays.update(more);take.target_asset=target_asset
                    raw_path=project/(raw_kind+'_mixamo.json')
                    if raw_path.exists():
                        from .result_artifacts import file_stat
                        take.raw_target_stamp=file_stat(raw_path)
                    from .preview_data import load_contact_track
                    from .motion_tools import base_result
                    track=load_contact_track(project,base_result(kind),cancelled)
                    if track and track['context']['frames']==take.frames and track['context']['fps']==take.fps:
                        contact_intervals+=sum(len(ranges) for ranges in track['ranges'])
                        if contact_intervals>65536:raise ValueError('Scene foot-lock metadata exceeds 65536 intervals. Trim the video.')
                        take.set_contacts(track['ranges'])
                        take.contact_track=track
                        take.contact_revision=track['revision']
                    scene.takes.setdefault(ident,[]).append(take)
                    # Summary sampling must not leave one mapped window per take resident.
                    for array in arrays.values():array.release_windows()
                except LoadCancelled:
                    for array in arrays.values():close_array(array)
                    raise
                except Exception as error:
                    for array in arrays.values():close_array(array)
                    scene.warnings.append(f'Person {ident}: {error}')
            scene.finish()
            if cancelled():raise LoadCancelled()
            return scene
        except Exception:
            scene.close();raise

    def run(self):
        while True:
            with self.condition:
                while self.pending is None and not self.stopping:self.condition.wait()
                if self.stopping:self.model_cache=None;return
                generation,root,kind,asset,cached=self.pending;self.pending=None
            cancelled=lambda:self.stopping or generation!=self.generation
            try:
                if cached is not None and self.cache_current(cached,kind,cancelled):result=cached
                else:result=self.load(root,kind,asset,cancelled)
                if cancelled():
                    if result is not cached:result.close()
                else:self.ready.emit(generation,result)
            except LoadCancelled:pass
            except Exception as error:
                if not cancelled():self.failed.emit(generation,str(error))

    @staticmethod
    def cache_current(scene,kind,cancelled):
        from .scene_camera import camera_sources
        from .foot_contacts import contact_revision
        from .motion_tools import base_result
        try:
            if scene.reference.get('input')!=media_receipt(scene.root):return False
            anchor=read_camera(safe_child(scene.root,scene.reference['anchor']))
            if anchor is None or anchor['sources']!=scene.reference['sources']:return False
            for takes in scene.takes.values():
                for take in takes:
                    if cancelled():raise LoadCancelled()
                    if camera_sources(take.project)!=take.camera_sources or read_camera(take.project) is None:return False
                    display_kind=kind
                    if not artifacts_current(take.project,display_kind) or not result_is_current(take.project,display_kind,ignore_contacts=True):return False
                    from .result_artifacts import file_stat
                    from .mixamo_data import current as target_current
                    target_path=take.project/(kind+'_mixamo.json')
                    if (file_stat(target_path) if target_path.exists() else None)!=take.target_stamp:return False
                    if take.target_asset is not None and not target_current(take.project,kind):return False
                    raw_path=take.project/(display_kind+'_mixamo.json')
                    if (file_stat(raw_path) if raw_path.exists() else None)!=getattr(take,'raw_target_stamp',None):return False
                    if getattr(take,'raw_asset',None) is not None and not target_current(take.project,display_kind):return False
                    if take.contact_revision!=contact_revision(take.project,base_result(kind)):return False
            return not scene.warnings
        except (OSError,ValueError,KeyError):return False
