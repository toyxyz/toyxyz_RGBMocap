import bpy,sys,json,time,numpy as np
from pathlib import Path
from mathutils import Matrix
from .file_paths import io_path, save_blend
A=np.array([[1.,0,0],[0,0,-1],[0,1,0]])
def curves(action):
    return action.fcurves if hasattr(action, 'fcurves') else next((b for l in action.layers for s in l.strips for b in s.channelbags)).fcurves

def key_values(obj, path, values):
    n = len(values)
    for index in range(values.shape[1]):
        f = next((c for c in curves(obj.animation_data.action) if c.data_path == path and c.array_index == index))
        f.keyframe_points.clear()
        f.keyframe_points.add(n)
        f.keyframe_points.foreach_set('co', np.column_stack((np.arange(1, n + 1), values[:, index])).reshape(-1))
        for k in f.keyframe_points:
            k.interpolation = 'LINEAR'
        f.update()

def rotation(M):
    (U, _, V) = np.linalg.svd(M)
    R = U @ V
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ V
    return R

def align(a, b):
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    v = np.cross(a, b)
    c = float(np.clip(a @ b, -1, 1))
    if c < -0.999999:
        axis = np.cross(a, np.eye(3)[np.argmin(abs(a))])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K / max(1 + c, 1e-10)

def bake(kind, label, method):
    start = time.perf_counter()
    timings = {}
    bpy.ops.wm.open_mainfile(filepath=io_path(OUT / (kind + '-rest.blend')))
    rig = json.loads(Path(io_path(OUT / (kind + '-rig.json'))).read_text(encoding='utf8'))
    arm = bpy.data.objects[rig['armature']]
    arm.animation_data_clear()
    data = np.load(io_path(OUT / (label + '-' + kind + '-' + method + '.npz')))
    n = len(data['root'])
    mapped = {name: i for (i, name) in enumerate(data['names'].tolist())}
    bones = list(arm.data.bones)
    indices = {b.name: i for (i, b) in enumerate(bones)}
    rest = np.array([arm.matrix_world @ b.head_local for b in bones])
    R0 = np.array([rotation(np.array(arm.matrix_world @ b.matrix_local)[:3, :3]) for b in bones])
    parents = np.array([indices[b.parent.name] if b.parent else -1 for b in bones])
    source_entry = next((e for e in json.loads(Path(io_path(OUT / 'sources.json')).read_text(encoding='utf-8')) if e['label'] == label))
    folder = Path(source_entry['folder'])
    info = source_entry['info']
    source_frames = np.load(io_path(folder / info['frames_file']), mmap_mode='r')
    source_asset = np.load(io_path(folder / info['asset_file']))
    source_names = info['names']
    source_rest = source_asset['rest']
    extra = {}
    base = np.load(io_path(OUT / (label + '-' + kind + '-rotation.npz')))
    for (j, name) in enumerate(source_names[22:], 22):
        side = 'left' if name.startswith('Left') else 'right'
        hand = 20 if side == 'left' else 21
        suffix = name.split('Hand')[1]
        finger = suffix[:-1]
        phalanx = int(suffix[-1])
        target = None
        if kind in {'mixamo', 'warrok', 'ch14'}:
            target = next((b.name for b in bones if b.name.split(':')[-1]==name), None)
        elif kind == 'vrm':
            semantic = side + ('Little' if finger == 'Pinky' else finger) + ['Proximal', 'Intermediate', 'Distal'][phalanx - 1]
            target = rig['semantic'].get(semantic)
        else:
            jp = {'Thumb': '親指', 'Index': '人指', 'Middle': '中指', 'Ring': '薬指', 'Pinky': '小指'}[finger]
            number = phalanx - 1 if finger == 'Thumb' else phalanx
            target = jp + '０１２３'[number] + ('.L' if side == 'left' else '.R')
        if target not in indices:
            continue
        target_bone = bones[indices[target]]
        vtarget = np.array(arm.matrix_world @ target_bone.tail_local) - rest[indices[target]]
        vsource = source_rest[j + 1] - source_rest[j] if phalanx < 3 else source_asset['ends'][j]
        C = align(vtarget @ A, vsource)
        Ch = source_frames[0, hand, :3, :3].T @ base['world'][0, hand]
        relative = source_frames[:, hand, :3, :3].swapaxes(-1, -2) @ source_frames[:, j, :3, :3]
        extra[target] = data['world'][:, hand] @ Ch.T @ relative @ C
    total_bones = len(bones)
    keyed = set(mapped) | set(extra) | ({'腰'} if kind == 'pmx' else set())
    needed = set(keyed)
    for name in keyed:
        b = arm.data.bones[name].parent
        while b is not None:
            needed.add(b.name)
            b = b.parent
    bones = [b for b in bones if b.name in needed]
    indices = {b.name: i for (i, b) in enumerate(bones)}
    rest = np.array([arm.matrix_world @ b.head_local for b in bones])
    R0 = np.array([rotation(np.array(arm.matrix_world @ b.matrix_local)[:3, :3]) for b in bones])
    parents = np.array([indices[b.parent.name] if b.parent else -1 for b in bones])
    p = np.zeros((n, len(bones), 3))
    rot = np.zeros((n, len(bones), 3, 3))
    channels = {}
    fk_start = time.perf_counter()
    original_length = np.array([b.length for b in bones])
    for pb in arm.pose.bones:
        pb.matrix_basis.identity()
        if pb.name in mapped or pb.name in extra or (kind == 'pmx' and pb.name.startswith('腰キャンセル')):
            for c in pb.constraints:
                c.mute = True
    for (j, b) in enumerate(bones):
        par = parents[j]
        parentR = np.tile(np.eye(3), (n, 1, 1)) if par < 0 else rot[:, par]
        before = R0[j] if par < 0 else R0[par].T @ R0[j]
        pre = parentR @ before
        if b.name in mapped:
            D = data['world'][:, mapped[b.name]]
            rot[:, j] = A @ D @ A.T @ R0[j]
        elif b.name in extra:
            rot[:, j] = A @ extra[b.name] @ A.T @ R0[j]
        elif kind == 'pmx' and b.name == '腰':
            rot[:, j] = A @ data['world'][:, 0] @ A.T @ R0[j]
        else:
            rot[:, j] = pre
        if par < 0:
            p[:, j] = rest[j]
        else:
            p[:, j] = p[:, par] + np.einsum('tij,j->ti', rot[:, par] @ R0[par].T, rest[j] - rest[par])
        if b.name in mapped or b.name in extra or (kind == 'pmx' and b.name == '腰'):
            basis = pre.swapaxes(-1, -2) @ rot[:, j]
            q = np.array([tuple(Matrix(m.tolist()).to_quaternion().normalized()) for m in basis])
            for t in range(1, n):
                if q[t] @ q[t - 1] < 0:
                    q[t] *= -1
            pb = arm.pose.bones[b.name]
            pb.rotation_mode = 'QUATERNION'
            pb.rotation_quaternion = q[0]
            pb.keyframe_insert('rotation_quaternion', frame=1)
            key_values(arm, pb.path_from_id('rotation_quaternion'), q)
    root_index = indices[data['names'][0]]
    shift = data['root'] @ A.T - p[:, root_index]
    from .mapping import body_trajectory_root
    root_name = body_trajectory_root(
        {b.name: b.parent.name if b.parent else None for b in arm.data.bones}, mapped)
    if rig.get('trajectory_root', root_name) != root_name:
        raise ValueError('Character hierarchy changed after import. Import the character again.')
    trajectory_root = arm.data.bones[root_name]
    rest_to_world = np.array(arm.matrix_world @ trajectory_root.matrix_local)[:3, :3]
    root_location = shift @ np.linalg.inv(rest_to_world).T
    pb = arm.pose.bones[trajectory_root.name]
    assert not any((not c.mute and c.influence > 0 for c in pb.constraints)), 'Constrained trajectory root requires explicit handling'
    pb.location = root_location[0]
    pb.keyframe_insert('location', frame=1)
    key_values(arm, pb.path_from_id('location'), root_location)
    keyed.add(trajectory_root.name)
    assert all((c.data_path.startswith('pose.bones[') for c in curves(arm.animation_data.action)))
    timings['fk_and_action_seconds'] = time.perf_counter() - fk_start
    scene=bpy.context.scene
    fps=float(np.load(io_path(OUT/'motion-source.npz'))['fps'])
    scene.frame_start=1;scene.frame_end=n;scene.render.fps=round(fps);scene.render.fps_base=round(fps)/fps
    scene.frame_set(1)
    save_blend(OUT/'animation.blend')
    return arm, sorted(keyed)
