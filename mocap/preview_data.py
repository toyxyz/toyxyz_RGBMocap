"""Cancellable project loading with bounded RAM and read-only mapped arrays."""
from __future__ import annotations

import os
import json
from pathlib import Path
from threading import Condition, get_ident
import zipfile
from collections import OrderedDict

import numpy as np
from .motion_tools import RESULTS, result_is_current
from PySide6.QtCore import QThread, Signal


class LoadCancelled(Exception):
    pass


def close_array(array):
    if isinstance(array, FrameArray):
        array.close()
        return
    mapping = getattr(array, "_mmap", None)
    if mapping is not None and not mapping.closed:
        mapping.close()


class FrameArray:
    """Map small NPY windows, never the entire duration of a recording.

    Returned frames are copies, so evicting a mapped block cannot invalidate
    something the painter is using. Each array retains at most 8 x 64KiB.
    """
    def __init__(self, path, max_blocks=8):
        self.path = Path(path)
        self.source = self.path.open("rb")
        try:
            source = self.source
            version = np.lib.format.read_magic(source)
            reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            self.shape, fortran, self.dtype = reader(source)
            self.offset = source.tell()
            if fortran or self.dtype.hasobject or not self.shape:
                raise ValueError("Preview requires a C-order numeric frame array")
        except Exception:
            self.source.close()
            raise
        self.ndim = len(self.shape)
        self.stride = int(np.prod(self.shape[1:])) * self.dtype.itemsize
        self.block_frames = max(1, 65536 // self.stride)
        self.blocks = OrderedDict()
        self.max_blocks=max_blocks

    def __len__(self):
        return self.shape[0]

    def _block(self, block):
        if self.source.closed:
            raise ValueError("Preview array is closed")
        if block not in self.blocks:
            first = block * self.block_frames
            length = min(self.block_frames, len(self) - first)
            # Pin the immutable generation for the complete snapshot lifetime,
            # including arrays never painted yet and evicted windows. Windows
            # rejects worker garbage collection until this handle is closed;
            # POSIX can unlink it, but subsequent windows still use this fd.
            self.blocks[block] = np.memmap(self.source, mode="r", dtype=self.dtype,
                offset=self.offset + first * self.stride, shape=(length, *self.shape[1:]))
            while len(self.blocks) > self.max_blocks:
                _, old = self.blocks.popitem(last=False)
                old._mmap.close()
        self.blocks.move_to_end(block)
        return self.blocks[block]

    def _frame(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self._block(index // self.block_frames)[index % self.block_frames].copy()

    def __getitem__(self, item):
        row, *tail = item if isinstance(item, tuple) else (item,)
        if isinstance(row, (int, np.integer)):
            value = self._frame(int(row))
            return value[tuple(tail)] if tail else value
        indices = range(*row.indices(len(self))) if isinstance(row, slice) else row
        if len(indices) > 8192:
            raise ValueError("Use bounded frame windows for preview access")
        if isinstance(indices, range) and indices.step == 1 and len(indices):
            pieces = []
            first = indices.start
            while first < indices.stop:
                block = first // self.block_frames
                end = min(indices.stop, (block + 1) * self.block_frames)
                pieces.append(self._block(block)[first % self.block_frames:first % self.block_frames + end - first].copy())
                first = end
            value = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
        else:
            value = np.stack([self._frame(int(index)) for index in indices]) if len(indices) else np.empty((0, *self.shape[1:]), self.dtype)
        return value[(slice(None), *tail)] if tail else value

    @property
    def cached_bytes(self):
        return sum(block.nbytes for block in self.blocks.values())

    def release_windows(self):
        """Evict cached windows while retaining this immutable generation."""
        for block in getattr(self, 'blocks', {}).values():
            block._mmap.close()
        if hasattr(self, 'blocks'):self.blocks.clear()

    def close(self):
        self.release_windows()
        if hasattr(self, 'source'):self.source.close()

    def __del__(self):
        self.close()


def project_signature(project):
    """Small publication receipts, shared by the loader and its GUI consumer."""
    project=Path(project)
    names=dict.fromkeys(('tracking.npz','keypoints.npz','soma-result.json','soma-correction.json',
        'flowhmr_pending.json','flow_camera_sources.json',
        *(value[0] for value in RESULTS.values()),
        *(f'{kind}_{suffix}' for kind in RESULTS for suffix in
          ('joints.npz','mesh.json','sources.json','artifacts.json','correction.json',
           'correct_pending.json','export_pending.json','mixamo.json','mixamo_pending.json'))))
    stamps=[]
    for name in names:
        try:stat=(project/name).stat()
        except FileNotFoundError:continue
        stamps.append((name,stat.st_mtime_ns,stat.st_size))
    return str(project.resolve()),tuple(stamps)


def mapped_array(archive: Path, key: str, cancelled=lambda: False):
    stat = archive.stat()
    folder = archive.parent / ".preview_cache" / archive.stem
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{key}-{stat.st_size}-{stat.st_mtime_ns}.npy"
    if not target.exists():
        temporary = target.with_suffix(f".{os.getpid()}.{get_ident()}.partial")
        try:
            with zipfile.ZipFile(archive) as source, source.open(key + ".npy") as entry, temporary.open("wb") as output:
                while True:
                    if cancelled():
                        raise LoadCancelled()
                    chunk = entry.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            if not target.exists():
                try:
                    os.replace(temporary, target)
                except OSError:
                    if not target.exists():
                        raise
        finally:
            temporary.unlink(missing_ok=True)
    if cancelled():
        raise LoadCancelled()
    # Old revisions are no longer useful. Windows protects any mapping still in use.
    for old in folder.glob(f"{key}-*.npy"):
        if old != target:
            try:
                old.unlink()
            except OSError:
                pass
    return FrameArray(target,max_blocks=4)


def motion_summary(joints):
    # Never scan/copy an entire long sequence for camera fitting on the GUI thread.
    indices = np.linspace(0, len(joints) - 1, min(4096, len(joints)), dtype=np.int64)
    sample = np.asarray(joints[indices, :22])
    floor = float(np.percentile(sample[:, [7, 8, 10, 11], 1], 1))
    height = max(0.5, float(np.percentile(sample[:, :, 1], 99.5) - floor))
    center = sample[:, 0].mean(axis=0)
    center[1] = floor + height * 0.5
    return {"floor": floor, "height": height, "center": center,
            "span": max(height, float(np.ptp(sample[:, :, [0, 2]], axis=(0, 1)).max()))}


def load_contact_track(project,base,cancelled=lambda:False):
    """Sparse contact metadata, shared by single and multiple actor loaders."""
    from .foot_contacts import read_edit,normalize_ranges
    from .result_artifacts import pending_path
    if cancelled():raise LoadCancelled()
    stem='corrected_'+base
    if pending_path(project,stem,'correct').exists():return None
    report_path=project/(stem+'_correction.json');archive=project/(stem+'_contacts.npz')
    sources_path=project/(stem+'_sources.json')
    if not all(p.exists() for p in (report_path,archive,sources_path)):return None
    sources=json.loads(sources_path.read_text(encoding='utf-8'))
    if not all((project/name).is_file() and [(project/name).stat().st_size,(project/name).stat().st_mtime_ns]==stat for name,stat in sources.items()):return None
    report=json.loads(report_path.read_text(encoding='utf-8'))
    if 'contacts_stat' in report:
        from .result_artifacts import file_stat
        if file_stat(archive)!=report['contacts_stat']:return None
    context={'frames':int(report['frames']),'fps':float(report['fps']),'sources':sources}
    with zipfile.ZipFile(archive) as package:
        if sum(entry.file_size for entry in package.infolist())>8*1024**2:raise ValueError('Foot-lock timing metadata exceeds 8 MiB')
    with np.load(archive,allow_pickle=False) as data:
        rows=data['segments']
        if rows.ndim!=2 or rows.shape[1]!=6:raise ValueError('Invalid foot-lock segments')
        generated=normalize_ranges([[(int(s),int(e)) for f,s,e,*_ in rows if int(f)==foot] for foot in range(2)],context['frames'])
        automatic=normalize_ranges([[(int(s),int(e)) for f,s,e in data['automatic'] if int(f)==foot] for foot in range(2)],context['frames']) if 'automatic' in data else [list(t) for t in generated]
    hand=report.get('hand_lock',{})
    hand_generated=[[(int(s['start']),int(s['end'])) for s in hand.get('segments',[]) if s['foot']==h] for h in (0,1)]
    generated+=hand_generated;automatic+=hand.get('automatic',[[],[]])
    edited,revision,warning=read_edit(project,base,context)
    if edited is not None and len(edited)==2:edited+=hand_generated
    ranges=edited if edited is not None else (automatic if revision else generated)
    return {'ranges':ranges,'automatic':automatic,'context':context,'revision':revision,
        'mode':report.get('mode','flat'),
        'levels':[[[int(s),int(e),float(y)] for f,s,e,x,y,z in rows if int(f)==foot] for foot in range(2)],
        'level_starts':[[int(s) for f,s,e,x,y,z in rows if int(f)==foot] for foot in range(2)],
        'manual':edited is not None,'warning':warning,'applied_revision':report.get('contact_revision')}


class ProjectPreview:
    def __init__(self, project):
        self.project = project
        self.signature = project_signature(project)
        self.arrays = {}
        self.summaries = {}
        self.mesh_asset = None
        self.contact_tracks = {}
        self.quality_note = ''
        self.target_assets = {}
        self.target_info = {}

    def close(self):
        for array in self.arrays.values():
            close_array(array)
        self.arrays.clear()
        self.summaries.clear()
        self.mesh_asset = None
        self.contact_tracks.clear()
        self.target_assets.clear();self.target_info.clear()

    def __del__(self):
        self.close()


class PreviewDataLoader(QThread):
    ready = Signal(int, object)
    failed = Signal(int, str)
    tracked_actor_ready = Signal(int, object, object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.condition = Condition()
        self.generation = 0
        self.pending = None
        self.stopping = False
        self.model_cache = None
        self.single_actor_cache = None

    def load_tracked_actor(self,generation,project,arrays,cancelled,expected=None):
        from .single_actor import snapshot_key,snapshot
        key=snapshot_key(project)
        if expected is not None and project_signature(project)!=expected:raise LoadCancelled()
        if self.single_actor_cache is None or self.single_actor_cache[0]!=key:
            value=snapshot(project,arrays,cancelled) if key and len(arrays.get('boxes',()))==key[-1] else None
            if cancelled():raise LoadCancelled()
            if key and snapshot_key(project)!=key:raise LoadCancelled()
            self.single_actor_cache=(key,value)
        if expected is not None and project_signature(project)!=expected:raise LoadCancelled()
        if not cancelled():self.tracked_actor_ready.emit(generation,project,key,self.single_actor_cache[1])

    def request(self, project):
        with self.condition:
            self.generation += 1
            self.pending = (self.generation, Path(project))
            self.condition.notify()
            return self.generation

    def cancel(self):
        with self.condition:
            self.generation += 1
            self.pending = None

    def stop(self):
        with self.condition:
            self.stopping = True
            self.condition.notify()

    def run(self):
        while True:
            with self.condition:
                while self.pending is None and not self.stopping:
                    self.condition.wait()
                if self.stopping:
                    self.model_cache = None
                    self.single_actor_cache = None
                    return
                generation, project = self.pending
                self.pending = None
            result = ProjectPreview(project)
            cancelled = lambda: self.stopping or generation != self.generation
            try:
                from .gemx_result import is_project, open_preview
                if is_project(project):
                    open_preview(result, cancelled)
                    if cancelled():raise LoadCancelled()
                    self.ready.emit(generation, result)
                    continue
                for filename, fields in (("tracking.npz", ("boxes", "observed")),
                                         ("keypoints.npz", ("keypoints", "confidence")),
                                         *((f"{kind}_joints.npz", ("joints",)) for kind in RESULTS)):
                    archive = project / filename
                    if not archive.exists():
                        continue
                    if fields == ("joints",):
                        from .result_artifacts import artifacts_current
                        kind=filename.removesuffix('_joints.npz')
                        if not result_is_current(project,kind,ignore_contacts=True) or not artifacts_current(project,kind):continue
                    if fields == ("joints",):
                        with zipfile.ZipFile(archive) as package:
                            if "rotations.npy" in package.namelist():
                                fields = ("joints", "rotations")
                    for key in fields:
                        name = (filename.replace(".npz", "") if key == "joints" else
                                filename.replace("_joints.npz", "_rotations") if key == "rotations" else key)
                        result.arrays[name] = mapped_array(archive, key, cancelled)
                        expected = {"boxes": ((4,),), "observed": ((),), "keypoints": ((17, 2),),
                                    "confidence": ((17, 1), (17,)), "joints": ((22, 3),), "rotations": ((22, 3, 3),)}
                        if result.arrays[name].shape[1:] not in expected[key]:
                            raise ValueError(f"Invalid preview array shape: {filename}/{key}")
                        if key == "rotations" and len(result.arrays[name]) != len(result.arrays[filename.replace(".npz", "")]):
                            raise ValueError("Rotation and joint frame counts differ")
                        if key == "joints":
                            result.summaries[name] = motion_summary(result.arrays[name])
                            from .motion_tools import base_result,has_handflow
                            if has_handflow(filename.removesuffix('_joints.npz')):
                                hand_report=project/(base_result(filename.removesuffix('_joints.npz'))+'_handflow.json')
                                if hand_report.exists():
                                    result.summaries[name]['handflow_warning']=json.loads(hand_report.read_text(encoding='utf-8')).get('warning','')
                            correction=project/(filename.removesuffix('_joints.npz')+'_correction.json')
                            if correction.exists():
                                report=json.loads(correction.read_text(encoding='utf-8'))
                                result.summaries[name]['floor']=float(report['floor'])
                                result.summaries[name]['correction_mode']=report.get('mode','flat')
                                result.summaries[name]['correction_warning']=report.get('warning','')
                                if report.get('version',1)<5:
                                    result.summaries[name]['correction_warning']=' '.join(filter(None,(
                                        result.summaries[name]['correction_warning'],
                                        'Legacy knee correction. Apply Correction to update anatomical bend axes.')))
                                elif report.get('version',1)<6:
                                    result.summaries[name]['correction_warning']=' '.join(filter(None,(
                                        result.summaries[name]['correction_warning'],
                                        'Legacy sole correction. Apply Correction to update floor contacts.')))
                                result.summaries[name]['short_contact_segments']=report.get('short_contact_segments',0)
                                result.summaries[name]['pelvis_auto_drop']=report.get('max_reach_pelvis_drop_m')
                                result.summaries[name]['pelvis_auto_raise']=report.get('max_reach_pelvis_raise_m',0.)
                                result.summaries[name]['pelvis_applied_lift']=report.get('max_applied_pelvis_lift_m',0.)
                    if filename=='tracking.npz':
                        self.load_tracked_actor(generation,project,result.arrays,cancelled,result.signature)
                # Sparse contact metadata is loaded here, never in paint/seek.
                from .foot_contacts import BASES
                for base in BASES:
                    track=load_contact_track(project,base,cancelled)
                    if track is not None:result.contact_tracks[base]=track
                for kind in RESULTS:
                    if kind+"_joints" not in result.arrays:
                        continue
                    metadata = project / f"{kind}_mesh.json"
                    if not metadata.exists():
                        continue
                    info = json.loads(metadata.read_text(encoding="utf-8"))
                    if info.get('representation')=='smplh':continue  # Native surface is loaded with the same committed export rig below.
                    frames = project / Path(info.get("frames_file",f"{kind}_mesh.npy")).name
                    if not frames.exists():
                        continue
                    from .mesh_data import MESH_VERSION, FRAME_FLOATS
                    if info.get("version") != MESH_VERSION:
                        continue
                    original = project / RESULTS[kind][0]
                    if original.exists():
                        stat = original.stat()
                        if (info.get("source_size"),info.get("source_mtime_ns")) != (stat.st_size,stat.st_mtime_ns):
                            continue
                    from .project_storage import resolve_reference
                    asset = resolve_reference(project, info['asset'])
                    if not asset.exists():
                        from .paths import MODELS
                        asset = MODELS / "body_models/smplx" / asset.name
                    if not asset.exists():
                        continue  # Keep video/joints usable; the mesh view offers Prepare Mesh.
                    stat = asset.stat()
                    model = asset.parent / "SMPLX_NEUTRAL.npz"
                    if model.exists():
                        model_stat = model.stat()
                        if (info.get("model_size"),info.get("model_mtime_ns")) != (model_stat.st_size,model_stat.st_mtime_ns):
                            continue
                    signature = (str(asset.resolve()),stat.st_size,stat.st_mtime_ns)
                    if self.model_cache is None or self.model_cache["key"] != signature:
                        with np.load(asset,allow_pickle=False) as model:
                            self.model_cache = {key:model[key] for key in ("packed","faces","offsets","neighbors","vertex_count")}
                        self.model_cache["key"] = signature
                    result.mesh_asset = self.model_cache
                    array = FrameArray(frames,max_blocks=4)
                    result.arrays[kind+"_mesh"] = array
                    if array.shape != (len(result.arrays[kind+"_joints"]),FRAME_FLOATS) or array.dtype != np.float32:
                        raise ValueError("Invalid SMPL-X mesh frame array")
                from .bone_overlay import open_preview
                for kind in RESULTS:
                    target=open_preview(project,kind,cancelled)
                    if target:
                        arrays,asset,info=target
                        result.arrays.update({kind+'_'+key:value for key,value in arrays.items()})
                        result.target_assets[kind]=asset;result.target_info[kind]=info
                # Eight variants share the same total cache budget as four did.
                for array in result.arrays.values():
                    if isinstance(array,FrameArray):
                        array.max_blocks=4
                        while len(array.blocks)>4:
                            _,old=array.blocks.popitem(last=False);old._mmap.close()
                if cancelled():
                    raise LoadCancelled()
                self.ready.emit(generation, result)
            except LoadCancelled:
                result.close()
            except Exception as error:
                result.close()
                if not cancelled():
                    self.failed.emit(generation, str(error))
