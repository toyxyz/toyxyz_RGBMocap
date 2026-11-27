"""Registered neutral rest frames for FBX, without changing animated geometry."""
import numpy as np
from .bvh_bind import bind_pose
from .bone_overlay import reference_frame


def prepare(rig, source_rest, vertices, topology, fps):
    from .source_surface import layout, endpoints
    from .source_animation import fit
    ids, lookup = layout(topology)
    surface = endpoints(np.asarray(vertices)[None, ids], lookup)
    rest = np.asarray(source_rest)[:22]
    _, _, neutral = fit(rig, rest[None, 0], np.tile(np.eye(3), (1, 22, 1, 1)),
                        rest[None], fps=fps, surface_endpoints=surface)
    binding = bind_pose(rig)
    old = binding['rest']; parents = binding['parents']
    neutral_export = neutral[0] @ binding['basis']
    conversion = np.tile(np.eye(3), (52, 1, 1))
    limbs = [1, 2, 4, 5, 16, 17, 18, 19]
    conversion[limbs] = neutral_export[limbs].swapaxes(-1, -2)
    target = old.copy()
    display = []
    for j, p in enumerate(parents):
        if p >= 0:
            target[j] = target[p] + conversion[p].T @ (old[j] - old[p])
        children = np.flatnonzero(parents == j)
        delta = old[children].mean(0) - old[j] if len(children) else binding['ends'][j]
        if np.linalg.norm(delta) < .001:
            delta = delta + [0., .1, 0.]
        display.append(reference_frame(delta))
    display = np.asarray(display)
    # Stand the rest pose on the soles, as the motion does: the foot joints sit above the floor.
    sole = rest[[10,11],1].min() - np.asarray(vertices)[:,1].min()
    target -= [target[[7,8],0].mean(), target[[10,11],1].min() - sole, target[[7,8],2].mean()]
    neutral_frames = conversion.swapaxes(-1,-2) @ display
    # These translations must be constant in each parent's physical bone frame.
    for j, p in enumerate(parents):
        if p >= 0:
            np.testing.assert_allclose(neutral_frames[p].T @ (target[j]-target[p]),
                                       display[p].T @ (old[j]-old[p]), atol=1e-7)
    return dict(fbx_rest=target, fbx_rest_frames=neutral_frames,
                fbx_display_frames=display,
                fbx_tip_local=np.einsum('nji,nj->ni', display, binding['ends']))
