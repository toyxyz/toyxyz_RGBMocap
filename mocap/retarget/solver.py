"""Frozen, validated traditional contact solver (2026-10-26).
MIT RGB MoCap. No model inference, physics or target bone stretching.
Job-local paths are configured by pipeline.run in a dedicated worker process.
"""
from pathlib import Path
import os,json,time
import numpy as np
from scipy.ndimage import gaussian_filter1d,maximum_filter1d
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from mocap import contact_trajectory as trajectory
from mocap.contact_arm_plan import CONFIG as APP_CONFIG,align_many
from .foot_calibration import corrected_world
PARENTS=np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19])
CHAINS=[(1,4,7),(2,5,8),(16,18,20),(17,19,21)]
CFG=dict(APP_CONFIG,iterations=80,collar_limit_deg=3.,contact_sigma_m=.0002)
FPS=30.


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-12)

def align(a, b):
    a = unit(a)
    b = unit(b)
    v = np.cross(a, b)
    c = float(np.clip(a @ b, -1, 1))
    if c < -1 + 1e-08:
        x = unit(np.cross(a, np.eye(3)[np.argmin(abs(a))]))
        return 2 * np.outer(x, x) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / max(1 + c, 1e-10)

def fk(local, rest, root):
    world = local.copy()
    p = np.empty(local.shape[:-2] + (3,))
    p[:, 0] = root
    for j in range(1, len(rest)):
        parent = PARENTS[j]
        p[:, j] = p[:, parent] + world[:, parent] @ (rest[j] - rest[parent])
        world[:, j] = world[:, parent] @ local[:, j]
    return (p, world)

def local_from_world(world):
    q = world.copy()
    for j in range(1, 22):
        q[:, j] = world[:, PARENTS[j]].swapaxes(-1, -2) @ world[:, j]
    return q

def baseline(source_world, srest, trest):
    child = {0: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10, 8: 11, 9: 12, 12: 15, 13: 16, 14: 17, 16: 18, 17: 19, 18: 20, 19: 21}
    offsets = np.tile(np.eye(3), (22, 1, 1))
    for (j, k) in child.items():
        offsets[j] = align(trest[k] - trest[j], srest[k] - srest[j])
    for j in [10, 11, 15, 20, 21]:
        offsets[j] = offsets[PARENTS[j]]
    desired = source_world @ offsets
    return local_from_world(desired)

def skin(data, v, w):
    rest = data['rest']
    weighted = (v[None] - rest[:, None]) * w.T[:, :, None]
    return np.einsum('tjik,jvk->tvi', data['world'], weighted, optimize=True) + np.einsum('vj,tji->tvi', w, data['points'], optimize=True)

def geometry_samples(data, geo, palms):
    pins = []
    probes = []
    feet = []
    for side in [0, 1]:
        points = skin(data, geo[f'foot_v{side}'], geo[f'foot_w{side}'])
        feet.append(points)
        for ids in [geo[f'heel{side}'], geo[f'toe{side}']]:
            part = points[:, ids]
            sample = part.mean(1)
            sample[:, 1] = part[:, :, 1].min(1)
            probes.append(sample)
    for side in [0, 1]:
        points = skin(data, geo[f'palm_v{side}'][palms[side]], geo[f'palm_w{side}'][palms[side]])
        pins.append(points.mean(1))
    return (np.stack(probes, 1), feet, np.stack(pins, 1))

def setup_targets(base, source, active, geo):
    n = len(base['root'])
    palms = []
    for side in [0, 1]:
        points = skin(base, geo[f'palm_v{side}'], geo[f'palm_w{side}'])
        ids = np.flatnonzero(active[:, 2 + side])
        first = int(ids[0]) if len(ids) else 0
        palms.append(np.argsort(points[first, :, 1])[:max(4, min(12, len(points[first]) // 4))])
    (probes, feet, hand) = geometry_samples(base, geo, palms)
    weights = trajectory.support_weights(probes, active[:, :2], FPS)
    pin = trajectory.support_samples(probes, weights)
    for side in [0, 1]:
        pin[:, side, 1] = feet[side][:, :, 1].min(1)
    support = np.concatenate([pin, hand], axis=1)
    anchors = support.copy()
    segments = []
    for part in range(4):
        for (start, end) in trajectory.runs(active[:, part]):
            if part < 2:
                anchors[start:end, part] = trajectory.rolling_anchor(probes[start:end, 2 * part:2 * part + 2], weights[start:end, part], float(np.median(source['support_height'][start:end, part]) * base['ratio']) if 'support_height' in source else 0.0)
            else:
                a = np.median(support[start:min(end, start + 5), part], axis=0)
                a[1] = 0.0
                anchors[start:end, part] = a
            segments.append(dict(part=part, start=int(start), end=int(end), anchor=anchors[start, part].tolist(), anchor_end=anchors[end - 1, part].tolist()))
    offsets = np.zeros_like(support)
    transition = np.zeros_like(active)
    source_flight = ~active[:, :2].any(1) & (source['points'][:, [7, 8], 1].min(1) > 0.3)
    for parts in [[0, 1], [2, 3]]:
        (a, b) = parts
        seg = [dict(foot=x['part'] - a, start=x['start'], end=x['end']) for x in segments if x['part'] in parts]
        (off, tr) = trajectory.contact_offsets(probes, active[:, parts], anchors[:, parts], support[:, parts], np.zeros((n, 3)), seg, FPS, np.ones((n, 2)), source_flight if a == 0 else np.zeros(n, bool))
        offsets[:, parts] = off
        transition[:, parts] = tr
    goals = support + offsets
    for side in [0, 1]:
        free = ~active[:, side] & ~source_flight
        lift = np.where(free, np.minimum(np.maximum(-goals[:, side, 1] + 0.001, 0), 0.08), 0.0)
        lift = gaussian_filter1d(maximum_filter1d(lift, size=5, mode='nearest'), 1.0, mode='nearest') * free
        goals[:, side, 1] += lift
        transition[:, side] |= lift > 1e-05
    return dict(palms=palms, weights=weights, support=support, anchors=anchors, goals=goals, planned=active | transition, transition=transition, active=active, segments=segments, flight=source_flight)

def eval_pins(data, geo, plan):
    (probes, feet, hands) = geometry_samples(data, geo, plan['palms'])
    pin = trajectory.support_samples(probes, plan['weights'])
    for side in [0, 1]:
        pin[:, side, 1] = feet[side][:, :, 1].min(1)
    out = np.concatenate([pin, hands], 1)
    if 'skin_bias' in plan:
        out = out + plan['skin_bias']
    return (out, feet)

def source_safe_reach(rest, source, side):
    (a, b, c) = CHAINS[side]
    u = np.linalg.norm(rest[b] - rest[a])
    v = np.linalg.norm(rest[c] - rest[b])
    key = (side, float(u), float(v))
    cache = source.setdefault('_safe_reach', {})
    if key in cache:
        return cache[key]
    su = source['points'][:, b] - source['points'][:, a]
    sv = source['points'][:, c] - source['points'][:, b]
    flex = np.arccos(np.clip(np.einsum('ti,ti->t', su, sv) / (np.linalg.norm(su, axis=1) * np.linalg.norm(sv, axis=1)), -1.0, 1.0))
    minimum_flex = np.minimum(np.radians(30.0), np.maximum(flex - np.radians(6.0), 0.0))
    cache[key] = np.sqrt(np.maximum(u * u + v * v + 2 * u * v * np.cos(minimum_flex), 1e-12)) - 1e-06
    return cache[key]

def release_to_fk(data, base, active, fps=30.0):
    """Interpolate only added contact rotations; retain all source motion.

 No pole or endpoint IK is rebuilt for a released leg. Short gaps connect
 neighbouring contact deltas; longer swings return to FK within 0.20 sec.
 """
    n = len(active)
    radius = max(3, round(0.2 * fps))
    world = data['world'].copy()
    report = []

    def hermite(a, va, b, vb, count):
        return trajectory._hermite(a, va, b, vb, count)
    for (side, (upper, middle, end)) in enumerate(CHAINS[:2]):
        mask = active[:, side]
        correction = base['world'][:, [upper, middle]].swapaxes(-1, -2) @ world[:, [upper, middle]]
        vectors = R.from_matrix(correction.reshape(-1, 3, 3)).as_rotvec().reshape(n, 6)
        out = np.zeros_like(vectors)
        out[mask] = vectors[mask]
        for (first, stop) in trajectory.runs(~mask):
            left = first > 0 and mask[first - 1]
            right = stop < n and mask[stop]
            count = stop - first
            if not left and (not right):
                continue
            a = vectors[first - 1] if left else np.zeros(6)
            b = vectors[stop] if right else np.zeros(6)
            va = np.zeros(6)
            vb = np.zeros(6)
            if left and right and (count <= 2 * radius):
                out[first:stop] = hermite(a, va, b, vb, count)
            else:
                if left:
                    duration = min(radius, count) if stop < n else radius
                    take = min(duration, count)
                    out[first:first + take] = hermite(a, va, np.zeros(6), np.zeros(6), duration)[:take]
                if right:
                    duration = min(radius, count) if first > 0 else radius
                    take = min(duration, count)
                    out[stop - take:stop] = hermite(np.zeros(6), np.zeros(6), b, vb, duration)[-take:]
        world[:, [upper, middle]] = base['world'][:, [upper, middle]] @ R.from_rotvec(out.reshape(-1, 3)).as_matrix().reshape(n, 2, 3, 3)
        assert np.max(abs(world[mask][:, [upper, middle]] - data['world'][mask][:, [upper, middle]]), initial=0) < 1e-10
        report.append(dict(side=side, release_radius_frames=radius, extra_rotation_max_deg=float(np.degrees(np.linalg.norm(out.reshape(n, 2, 3), axis=-1)).max())))
    local = local_from_world(world)
    (p, w) = fk(local, data['rest'], data['root'])
    data.update(local=local, world=w, points=p)
    return report

def free_clearance(data, source, active, geo, plan):
    """Bounded floor clearance after FK release; never adds a contact or XZ pin.

 Raising a free ankle shortens its reach, so this does not reintroduce the
 maximum-extension target that caused release snapping.
 """
    maximum = 0.0
    frames = np.zeros_like(active)
    for _ in range(3):
        goals = data['points'][:, [7, 8, 20, 21]].copy()
        mask = np.zeros_like(active)
        amount = np.zeros((len(active), 2))
        for side in [0, 1]:
            vertices = skin(data, geo[f'foot_v{side}'], geo[f'foot_w{side}'])
            height = vertices[:, :, 1].min(1)
            free = ~active[:, side]
            raw = np.where(free, np.clip(0.001 - height, 0.0, 0.08), 0.0)
            for (first, end) in trajectory.runs(free):
                v = raw[first:end]
                smooth = gaussian_filter1d(maximum_filter1d(v, size=5, mode='nearest'), 1.0, mode='nearest')
                x = np.minimum(np.minimum((np.arange(len(v)) + 1) / 4, (len(v) - np.arange(len(v))) / 4), 1.0)
                taper = x * x * (3 - 2 * x)
                amount[first:end, side] = np.maximum(v, smooth * taper)
            amount[:, side] = np.minimum(amount[:, side], 0.08)
            mask[:, side] = free & (amount[:, side] > 0.0002)
        if not mask.any():
            break
        goals[:, :2, 1] += amount
        (raised, unreachable) = two_bone(data, data['root'], goals, mask, source)
        for key in ['local', 'world', 'points']:
            data[key] = raised[key]
        frames |= mask
        maximum = max(maximum, float(amount.max()))
    return dict(frames_per_foot=frames[:, :2].sum(0).tolist(), maximum_iteration_lift_mm=maximum * 1000)

def root_plan(base, goals, planned, active, source):
    planned = planned.copy()
    active = active.copy()
    planned[:, 2:] = False
    active[:, 2:] = False
    goals = goals.copy()
    goals[:, 2:] = base['points'][:, [20, 21]]
    n = len(base['root'])
    rest = base['rest']
    p = base['points']
    limit = 0.2 * float(rest[:, 1].max() - rest[:, 1].min())
    hi = np.full(n, limit)
    lo = np.full(n, -limit)
    moving = planned.any(1)
    hi[~moving] = 1e-10
    lo[~moving] = -1e-10
    lengths = []
    horizontal = []
    height = []
    reference = []
    legal = []
    for (part, (a, b, c)) in enumerate(CHAINS):
        u = np.linalg.norm(rest[b] - rest[a])
        v = np.linalg.norm(rest[c] - rest[b])
        lengths.append([u, v])
        d = p[:, a] - goals[:, part]
        h = (d[:, [0, 2]] ** 2).sum(1)
        horizontal.append(h)
        height.append(d[:, 1])
        old = np.linalg.norm(p[:, c] - p[:, a], axis=1)
        reference.append(np.arccos(np.clip((old ** 2 - u * u - v * v) / (2 * u * v), -1, 1)))
        maximum = source_safe_reach(rest, source, part) if part < 2 else u + v - 1e-05
        can = active[:, part] & (h < maximum ** 2) & (d[:, 1] >= 0)
        legal.append(can)
        minimum = abs(u - v) + 1e-05
        if os.environ.get('RETARGET_POLE', 'reach-comfort') == 'reach-comfort' and part < 2:
            can = active[:, part] & (h < maximum ** 2)
            su = source['points'][:, b] - source['points'][:, a]
            sv = source['points'][:, c] - source['points'][:, b]
            flex = np.arccos(np.clip(np.einsum('ti,ti->t', su, sv) / (np.linalg.norm(su, axis=1) * np.linalg.norm(sv, axis=1)), -1, 1))
            maximum_flex = np.minimum(flex + np.radians(5.0), np.radians(175.0))
            minimum = np.sqrt(np.maximum(u * u + v * v + 2 * u * v * np.cos(maximum_flex), 1e-10))
        upper = -d[:, 1] + np.sqrt(np.maximum(maximum ** 2 - h, 0))
        lower = -d[:, 1] + np.sqrt(np.maximum(minimum ** 2 - h, 0))
        if part < 2:
            hi = np.minimum(hi, np.where(can, upper, np.inf))
            lo = np.maximum(lo, np.where(can, lower, -np.inf))
        else:
            hnew = np.minimum(hi, np.where(can, upper, np.inf))
            lnew = np.maximum(lo, np.where(can, lower, -np.inf))
            ok = hnew > lnew + 1e-08
            hi = np.where(ok, hnew, hi)
            lo = np.where(ok, lnew, lo)
    bad = (hi <= lo + 1e-09) | (hi < -limit) | (lo > limit)
    hi = np.clip(hi, -limit, limit)
    lo = np.clip(lo, -limit, limit)
    bad |= hi <= lo + 1e-09
    fallback = np.clip(np.minimum(hi, 0), -limit, limit)
    lo[bad] = fallback[bad] - 1e-10
    hi[bad] = fallback[bad] + 1e-10
    horizontal = np.stack(horizontal, 1)
    height = np.stack(height, 1)
    reference = np.stack(reference, 1)
    L = np.array(lengths)
    mask = planned.astype(float)
    scale = max(0.4, float(np.mean(L[:2].sum(1))))
    sigma = 0.04 * scale

    def values(x):
        dy = height + x[:, None]
        c = (horizontal + dy ** 2 - (L[:, 0] ** 2 + L[:, 1] ** 2)[None]) / (2 * L[:, 0] * L[:, 1])[None]
        bounded = np.clip(c, -0.999999, 0.999999)
        a = np.arccos(bounded) - reference
        derivative = -dy / ((L[:, 0] * L[:, 1])[None] * np.sqrt(np.maximum(1 - bounded ** 2, 1e-10))) * (abs(c) < 0.999999)
        return (a * mask, derivative * mask)
    w0 = np.array([1.0, 1.0, 0.5, 0.5])[None] / 0.25
    w1 = 5.0
    w2 = 25.0
    wr = 350.0 / scale

    def fun(x):
        (a, _) = values(x)
        return np.r_[x / sigma, (a * w0).ravel(), (np.diff(a, axis=0) * w1).ravel(), (np.diff(a, n=2, axis=0) * w2).ravel(), np.diff(x, n=2) * wr]

    def jac(x):
        (_, d) = values(x)
        rs = []
        cs = []
        vs = []

        def add(r, c, v):
            rs.append(np.asarray(r).ravel())
            cs.append(np.asarray(c).ravel())
            vs.append(np.asarray(v).ravel())
        add(np.arange(n), np.arange(n), np.full(n, 1 / sigma))
        off = n
        add(off + np.arange(n * 4), np.repeat(np.arange(n), 4), d * w0)
        off += n * 4
        for (k, c) in [(0, -1), (1, 1)]:
            add(off + np.arange((n - 1) * 4), np.repeat(np.arange(n - 1) + k, 4), c * d[k:k + n - 1] * w1)
        off += (n - 1) * 4
        for (k, c) in [(0, 1), (1, -2), (2, 1)]:
            add(off + np.arange((n - 2) * 4), np.repeat(np.arange(n - 2) + k, 4), c * d[k:k + n - 2] * w2)
        off += (n - 2) * 4
        for (k, c) in [(0, 1), (1, -2), (2, 1)]:
            add(off + np.arange(n - 2), np.arange(n - 2) + k, np.full(n - 2, c * wr))
        return coo_matrix((np.concatenate(vs), (np.concatenate(rs), np.concatenate(cs))), shape=(off + n - 2, n)).tocsr()
    x0 = np.clip(np.zeros(n), lo + np.minimum(1e-09, (hi - lo) / 4), hi - np.minimum(1e-09, (hi - lo) / 4))
    opt = least_squares(fun, x0, jac=jac, bounds=(lo, hi), max_nfev=45, ftol=1e-05, xtol=1e-06, gtol=1e-05, tr_solver='lsmr')
    return (opt.x, dict(evaluations=int(opt.nfev), success=bool(opt.success), conflicting_bounds_frames=int(bad.sum()), max_root_shift_m=float(abs(opt.x).max())))

def two_bone(base, root, goals, planned, source):
    rest = base['rest']
    (p, w) = fk(base['local'], rest, root)
    unreachable = np.zeros(planned.shape, bool)
    for (part, (a, b, c)) in enumerate(CHAINS):
        u = np.linalg.norm(rest[b] - rest[a])
        v = np.linalg.norm(rest[c] - rest[b])
        ids = np.flatnonzero(planned[:, part])
        for i in ids:
            start = p[i, a]
            delta = goals[i, part] - start
            distance = np.linalg.norm(delta)
            axis = unit(delta)
            maximum = u + v - 1e-07
            if part < 2 and source.get('contact_active', planned)[i, part]:
                maximum = source_safe_reach(rest, source, part)[i]
            clipped = np.clip(distance, abs(u - v) + 1e-07, maximum)
            unreachable[i, part] = abs(clipped - distance) > 0.001
            goal = start + axis * clipped
            observed = p[i, b] - start - axis * ((p[i, b] - start) @ axis)
            confidence = np.clip((np.linalg.norm(observed) / max(u, 1e-09) - 0.02) / 0.08, 0, 1)
            prior = source['world'][i, a] @ np.array([0.0, 0.0, 1.0])
            prior -= axis * (prior @ axis)
            if np.linalg.norm(prior) < 1e-06:
                prior = np.cross(axis, source['world'][i, 0] @ np.array([1.0, 0.0, 0.0]))
            prior = unit(prior)
            observed = unit(observed) if np.linalg.norm(observed) > 1e-09 else prior
            angle = np.arctan2(axis @ np.cross(prior, observed), np.clip(prior @ observed, -1, 1))
            hint = R.from_rotvec(axis * angle * confidence).apply(prior)
            if os.environ.get('RETARGET_POLE', 'reach-comfort') == 'source-normal' and 'points' in source:
                su = source['points'][i, b] - source['points'][i, a]
                sv = source['points'][i, c] - source['points'][i, b]
                normal = np.cross(su, sv)
                normal -= axis * (normal @ axis)
                if np.linalg.norm(normal) > 1e-07:
                    normal_hint = unit(np.cross(axis, normal))
                    confidence_n = np.clip(np.linalg.norm(np.cross(su, sv)) / max(np.linalg.norm(su) * np.linalg.norm(sv), 1e-09) / 0.15, 0.0, 1.0)
                    angle_n = np.arctan2(axis @ np.cross(prior, normal_hint), np.clip(prior @ normal_hint, -1, 1))
                    hint = R.from_rotvec(axis * angle_n * confidence_n).apply(prior)
            along = (u * u - v * v + clipped * clipped) / (2 * clipped)
            height = np.sqrt(max(0.0, u * u - along * along))
            middle = start + axis * along + hint * height
            w[i, a] = align(p[i, b] - start, middle - start) @ w[i, a]
            w[i, b] = align(p[i, c] - p[i, b], goal - middle) @ w[i, b]
            p[i, b] = middle
            p[i, c] = goal
    local = local_from_world(w)
    (points, world) = fk(local, rest, root)
    return (dict(local=local, root=root, points=points, world=world, rest=rest, names=base['names']), unreachable)

def reachable_root_xz(base, root, goals, planned, source=None, mode=None):
    """Minimal bounded horizontal body adjustment for otherwise unreachable feet.

 A finite smoothing pass is followed by projection back into both reach disks.
 Anchors and timings never move; no limb stretching or residual relabeling.
 """
    planned = planned.copy()
    planned[:, 2:] = False
    rest = base['rest']
    delta = root - base['root']
    limit = 0.06 * float(rest[:, 1].max() - rest[:, 1].min())
    if os.environ.get('RETARGET_POLE', 'reach-comfort') == 'reach-comfort' and source is not None:
        span = float(rest[:, 1].max() - rest[:, 1].min())
        limit = 0.12 * span

        def project_shell(d):
            for _ in range(32):
                for (side, (a, b, c)) in enumerate(CHAINS[:2]):
                    u = np.linalg.norm(rest[b] - rest[a])
                    v = np.linalg.norm(rest[c] - rest[b])
                    su = source['points'][:, b] - source['points'][:, a]
                    sv = source['points'][:, c] - source['points'][:, b]
                    flex = np.arccos(np.clip(np.einsum('ti,ti->t', su, sv) / (np.linalg.norm(su, axis=1) * np.linalg.norm(sv, axis=1)), -1, 1))
                    maximum_flex = np.minimum(flex + np.radians(5.0), np.radians(175.0))
                    minimum = np.sqrt(np.maximum(u * u + v * v + 2 * u * v * np.cos(maximum_flex), 1e-10))
                    maximum = np.where(source['contact_active'][:, side], source_safe_reach(rest, source, side), u + v - 1e-05)
                    vec = goals[:, side] - (base['points'][:, a] + d)
                    dist = np.linalg.norm(vec, axis=1)
                    correction = (dist - np.clip(dist, minimum, maximum)) * planned[:, side]
                    d += vec / np.maximum(dist[:, None], 1e-09) * correction[:, None]
                    norm = np.linalg.norm(d[:, [0, 2]], axis=1)
                    d[:, [0, 2]] *= np.minimum(1.0, limit / np.maximum(norm, 1e-09))[:, None]
                    d[:, 1] = np.clip(d[:, 1], -0.2 * span, 0.2 * span)
            return d
        if (mode or os.environ.get('ROOT_MODE', 'pose-aware')) == 'smooth-stance':
            reference = delta.copy()
            delta = project_shell(delta)
            support = maximum_filter1d(planned[:, :2].any(1).astype(float), size=31, mode='nearest') > 0
            for _ in range(8):
                proposed = 0.1 * reference + 0.9 * gaussian_filter1d(delta, 1.4, axis=0, truncate=3.0, mode='nearest')
                proposed[~support] = reference[~support]
                delta = project_shell(proposed)
            return base['root'] + delta
        delta = project_shell(delta)
        delta = gaussian_filter1d(delta, 1.5, axis=0, truncate=3.0, mode='nearest')
        delta = project_shell(delta)
        delta[~planned[:, :2].any(1), 0] = 0
        delta[~planned[:, :2].any(1), 2] = 0
        return base['root'] + delta

    def project(d):
        for _ in range(12):
            for (side, (a, b, c)) in enumerate(CHAINS[:2]):
                upper = np.linalg.norm(rest[b] - rest[a])
                lower = np.linalg.norm(rest[c] - rest[b])
                L = np.where(source['contact_active'][:, side], source_safe_reach(rest, source, side), upper + lower - 1e-05)
                minimum = abs(upper - lower) + 1e-05
                hip = base['points'][:, a] + d
                v = goals[:, side] - hip
                radius = np.sqrt(np.maximum(L * L - v[:, 1] ** 2, 0))
                horizontal = np.linalg.norm(v[:, [0, 2]], axis=1)
                excess = np.maximum(horizontal - radius, 0) * planned[:, side]
                direction = v[:, [0, 2]] / np.maximum(horizontal[:, None], 1e-09)
                d[:, [0, 2]] += direction * excess[:, None]
                inner = np.sqrt(np.maximum(minimum * minimum - v[:, 1] ** 2, 0))
                inside = np.maximum(inner - horizontal, 0) * planned[:, side]
                d[:, [0, 2]] -= direction * inside[:, None]
                norm = np.linalg.norm(d[:, [0, 2]], axis=1)
                d[:, [0, 2]] *= np.minimum(1.0, limit / np.maximum(norm, 1e-09))[:, None]
        return d
    delta = project(delta)
    delta[:, [0, 2]] = gaussian_filter1d(delta[:, [0, 2]], 1.5, axis=0, truncate=3.0, mode='nearest')
    delta = project(delta)
    delta[~planned[:, :2].any(1), 0] = 0
    delta[~planned[:, :2].any(1), 2] = 0
    return base['root'] + delta

def polish_feet(data, source, geo, plan):
    count = 0
    largest = 0.0
    for (side, (a, b, c)) in enumerate(CHAINS[:2]):
        for i in np.flatnonzero(plan['active'][:, side]):

            def point(d):
                points = skin(d, geo[f'foot_v{side}'], geo[f'foot_w{side}'])[0]
                weight = plan['weights'][i, side]
                pin = points[geo[f'heel{side}']].mean(0) * (1 - weight) + points[geo[f'toe{side}']].mean(0) * weight
                pin[1] = points[:, 1].min()
                return pin + (plan['skin_bias'][i, side] if 'skin_bias' in plan else 0)
            frozen = {k: data[k][i:i + 1].copy() for k in ['local', 'root', 'points', 'world']}
            frozen.update(rest=data['rest'], names=data['names'])
            one_source = {'world': source['world'][i:i + 1], 'points': source['points'][i:i + 1], 'contact_active': plan['active'][i:i + 1]}
            goal = frozen['points'][:, [7, 8, 20, 21]].copy()
            planned = np.zeros((1, 4), bool)
            planned[0, side] = True
            target = plan['goals'][i, side]
            before = point(frozen)
            error = target - before
            if np.linalg.norm(error) < 0.0003:
                continue
            count += 1
            initial = frozen['world'][0, [a, b]].copy()

            def predict(x):
                g = goal.copy()
                g[0, side] = x
                (d, u) = two_bone(frozen, frozen['root'], g, planned, one_source)
                return (point(d), d)
            x = goal[0, side].copy()
            best = frozen
            best_error = np.linalg.norm(error)
            for _ in range(5):
                (value, d) = predict(x)
                r = target - value
                if np.linalg.norm(r) < best_error:
                    best = d
                    best_error = np.linalg.norm(r)
                if best_error < 0.00015:
                    break
                J = np.empty((3, 3))
                eps = 1e-05
                for k in range(3):
                    e = np.eye(3)[k] * eps
                    J[:, k] = (predict(x + e)[0] - predict(x - e)[0]) / (2 * eps)
                step = np.linalg.lstsq(J, r, rcond=0.0001)[0]
                step *= min(1.0, 0.015 / max(np.linalg.norm(step), 1e-09))
                x += step
            for k in ['local', 'points', 'world']:
                data[k][i] = best[k][0]
            largest = max(largest, float(np.max(R.from_matrix(initial.swapaxes(-1, -2) @ best['world'][0, [a, b]]).magnitude())))
    return dict(frames=count, max_extra_rotation_deg=float(np.degrees(largest)))

def refine(base, source, active, geo, skin_bias=None):
    source['contact_active'] = active
    plan = setup_targets(base, source, active, geo)
    if skin_bias is not None:
        plan['skin_bias'] = skin_bias
    current = base
    reports = []
    stop_enabled = os.environ.get('STOP_STAGNATION', '0') == '1'
    best = None
    best_error = float('inf')
    stale = 0
    trace = []
    stop_reason = 'iteration_limit'
    for iteration in range(20):
        (pin, _) = eval_pins(current, geo, plan)
        gain = 1.0 if iteration == 0 else 0.6
        goals = current['points'][:, [7, 8, 20, 21]] + gain * (plan['goals'] - pin)
        planned = plan['planned'].copy()
        (root_shift, report) = root_plan(base, goals, planned, active, source)
        root = base['root'].copy()
        root[:, 1] += root_shift
        mode = os.environ.get('ROOT_MODE', 'pose-aware')
        if mode in {'air-transition', 'pose-aware'}:
            legacy = reachable_root_xz(base, root, goals, planned, source, mode='legacy')
            smooth = reachable_root_xz(base, root, goals, active, source, mode='smooth-stance')
            L = np.mean(np.linalg.norm(base['rest'][[4, 5]] - base['rest'][[1, 2]], axis=1) + np.linalg.norm(base['rest'][[7, 8]] - base['rest'][[4, 5]], axis=1)) / float(base.get('ratio', 1.0))
            if mode == 'pose-aware':
                x = np.clip((source['points'][:, 0, 1] / L - 0.6) / 0.25, 0.0, 1.0)
                blend = gaussian_filter1d(x * x * (3 - 2 * x), 1.0, truncate=3.0, mode='nearest')
            else:
                departure = ~active[:, :2].any(1) & (source['points'][:, 0, 1] > 0.7 * L)
                blend = gaussian_filter1d(maximum_filter1d(departure.astype(float), size=9, mode='nearest'), 1.5, truncate=3.0, mode='nearest')
            root = legacy * (1 - blend[:, None]) + smooth * blend[:, None]
        else:
            hard_contacts = active if mode in {'stance', 'smooth-stance'} else planned
            root = reachable_root_xz(base, root, goals, hard_contacts, source, mode=mode)
        solve_mask = planned.copy()
        solve_mask[:, :2] = active[:, :2]
        (current, unreachable) = two_bone(base, root, goals, solve_mask, source)
        release_report = release_to_fk(current, base, active,FPS)
        clearance_report = free_clearance(current, source, active, geo, plan)
        reports.append(report)
        (pin, _) = eval_pins(current, geo, plan)
        error = max(np.linalg.norm((pin - plan['goals'])[active], axis=-1), default=0)
        if stop_enabled:
            trace.append(float(error))
            significant = error < best_error - max(1e-05, best_error * 0.01) if np.isfinite(best_error) else True
            if error < best_error:
                best_error = error
                best = ({k: v.copy() if isinstance(v, np.ndarray) else v for (k, v) in current.items()}, unreachable.copy(), release_report, clearance_report, iteration + 1)
            stale = 0 if significant else stale + 1
            if error < 0.0005:
                stop_reason = 'contact_tolerance'
                break
            if stale >= 3 and iteration >= 3:
                stop_reason = 'no_material_improvement'
                break
        elif error < 0.0005:
            break
    if stop_enabled and best is not None:
        (current, unreachable, release_report, clearance_report, best_iteration) = best
    polished = polish_feet(current, source, geo, plan)
    current.update(weights=active.astype(float), anchors=plan['anchors'], root_shift=current['root'] - base['root'], planned=plan['planned'], support_weights=plan['weights'], palm_ids_left=plan['palms'][0], palm_ids_right=plan['palms'][1])
    return (current, plan, dict(root_iterations=reports, unreachable_frames_per_part=unreachable.sum(0).tolist(), geometry_iterations=len(reports), skin_polish=polished, release=release_report, free_clearance=clearance_report, actual_skin_feedback=skin_bias is not None, early_stop=dict(enabled=stop_enabled, reason=stop_reason, best_iteration=best_iteration if best is not None else None, contact_max_trace_m=trace)))

def score(data, source, active, geo, plan):
    (pin, feet) = eval_pins(data, geo, plan)
    error = pin - plan['goals']
    rows = []
    for part in range(4):
        mask = active[:, part]
        e = error[mask, part]
        if not mask.any():
            rows.append(None)
            continue
        rows.append(dict(frames=int(mask.sum()), xz_max_mm=float(np.linalg.norm(e[:, [0, 2]], axis=-1).max() * 1000), height_abs_max_mm=float(abs(e[:, 1]).max() * 1000), position_mean_mm=float(np.linalg.norm(e, axis=-1).mean() * 1000), position_p95_mm=float(np.percentile(np.linalg.norm(e, axis=-1), 95) * 1000)))
    p = data['points']
    angles = []
    for (a, b) in [(1, 4), (4, 7), (2, 5), (5, 8), (16, 18), (18, 20), (17, 19), (19, 21)]:
        v = p[:, b] - p[:, a]
        s = source['points'][:, b] - source['points'][:, a]
        v /= np.linalg.norm(v, axis=-1, keepdims=True)
        s /= np.linalg.norm(s, axis=-1, keepdims=True)
        angles.extend(np.degrees(np.arccos(np.clip((v * s).sum(-1), -1, 1))).tolist())
    steps = np.degrees(R.from_matrix((data['local'][:-1].swapaxes(-1, -2) @ data['local'][1:]).reshape(-1, 3, 3)).magnitude())
    return dict(contacts=rows, limb_direction_mean_deg=float(np.mean(angles)), limb_direction_p95_deg=float(np.percentile(angles, 95)), local_step_max_deg=float(steps.max()), foot_skin_min_y_mm=[float(f[:, :, 1].min() * 1000) for f in feet])

def arm_reference(seed):
    """One actor-local normal per elbow, calibrated from well-bent frames.

 The normal follows the original proximal rotation at every frame. It is
 never a world-space fixed pole and is not switched at extension thresholds.
 """
    refs = []
    for (a, b, c) in CHAINS[2:]:
        upper = seed['points'][:, b] - seed['points'][:, a]
        lower = seed['points'][:, c] - seed['points'][:, b]
        reach = upper + lower
        reach /= np.linalg.norm(reach, axis=1)[:, None]
        normal = np.cross(upper, lower)
        normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-12)
        local_normal = np.einsum('tji,tj->ti', seed['world'][:, a], normal)
        flex = np.degrees(np.arccos(np.clip(np.sum(upper * lower, axis=1) / np.linalg.norm(upper, axis=1) / np.linalg.norm(lower, axis=1), -1, 1)))
        reliable = flex >= 60.0
        if reliable.sum() < 5:
            raise ValueError('This experiment requires at least five well-bent elbow calibration frames')
        calibrated_normal = unit(np.median(local_normal[reliable], axis=0))
        prior_normal = seed['world'][:, a] @ calibrated_normal
        prior_normal -= reach * np.sum(prior_normal * reach, axis=1)[:, None]
        pole = np.cross(reach, prior_normal)
        pole /= np.maximum(np.linalg.norm(pole, axis=1)[:, None], 1e-12)
        refs.append(dict(axis=reach, pole=pole, calibrated_local_normal=calibrated_normal, calibration_mask=reliable, source_flex_deg=flex))
    return refs

def arm_solve(seed, goals, planned, refs):
    p = seed['points'].copy()
    w = seed['world'].copy()
    unreachable = np.zeros_like(planned)
    for (side, (a, b, c)) in enumerate(CHAINS[2:]):
        u = np.linalg.norm(seed['rest'][b] - seed['rest'][a])
        v = np.linalg.norm(seed['rest'][c] - seed['rest'][b])
        for i in np.flatnonzero(planned[:, 2 + side]):
            start = p[i, a]
            delta = goals[i, 2 + side] - start
            distance = np.linalg.norm(delta)
            axis = unit(delta)
            clipped = np.clip(distance, abs(u - v) + 1e-07, u + v - 1e-07)
            unreachable[i, side + 2] = abs(distance - clipped) > 0.001
            ref = refs[side]
            hint = ref['pole'][i]
            if ref['axis'][i] @ axis > -0.999999:
                hint = align(ref['axis'][i], axis) @ hint
            hint -= axis * (hint @ axis)
            hint = unit(hint)
            along = (u * u - v * v + clipped * clipped) / (2 * clipped)
            height = np.sqrt(max(0.0, u * u - along * along))
            middle = start + axis * along + hint * height
            end = start + axis * clipped
            w[i, a] = align(p[i, b] - start, middle - start) @ w[i, a]
            w[i, b] = align(p[i, c] - p[i, b], end - middle) @ w[i, b]
            p[i, b] = middle
            p[i, c] = end
    local = local_from_world(w)
    (points, world) = fk(local, seed['rest'], seed['root'])
    return (dict(seed, local=local, points=points, world=world), unreachable)

def contact_reference_envelopes(refs, seed, active):
    """Finite source-relative handoff for real contact starts/releases.

 The earlier continuous push-up had no boundaries. On this full clip the
 preferred proximal reference must approach the original FK pole smoothly.
 """
    for (side, (a, b, c)) in enumerate(CHAINS[2:]):
        envelope = active[:, 2 + side].astype(float)
        radius = 6
        for (start, end) in trajectory.runs(active[:, 2 + side]):
            for (edge, direction) in [(start, -1), (end - 1, 1)]:
                for d in range(1, radius + 1):
                    i = edge + direction * d
                    if 0 <= i < len(envelope):
                        envelope[i] = max(envelope[i], 0.5 * (1 + np.cos(np.pi * d / (radius + 1))))
        ref = refs[side]
        axis = ref['axis']
        observed = seed['points'][:, b] - seed['points'][:, a]
        observed -= axis * np.sum(observed * axis, axis=1)[:, None]
        norm = np.linalg.norm(observed, axis=1)
        observed /= np.maximum(norm[:, None], 1e-12)
        observed[norm < 1e-08] = ref['pole'][norm < 1e-08]
        angle = np.arctan2(np.einsum('ti,ti->t', axis, np.cross(observed, ref['pole'])), np.clip(np.einsum('ti,ti->t', observed, ref['pole']), -1, 1))
        ref['pole'] = R.from_rotvec(axis * (angle * envelope)[:, None]).apply(observed)
        ref['contact_envelope'] = envelope
    return refs

def optimize_hand(original, seed, ref, geo, plan, side):
    n = len(original['root'])
    (collar, upper, elbow, wrist) = (13 + side, 16 + side, 18 + side, 20 + side)
    points = original['points']
    world = original['world']
    rest = original['rest']
    u = np.linalg.norm(rest[elbow] - rest[upper])
    v = np.linalg.norm(rest[wrist] - rest[elbow])
    su = points[:, elbow] - points[:, upper]
    sv = points[:, wrist] - points[:, elbow]
    flex0 = np.arccos(np.clip(np.einsum('ti,ti->t', su, sv) / (u * v), -1, 1))
    x = np.clip((np.degrees(flex0) - 15.0) / 20.0, 0, 1)
    focus = (1.0 - x * x * (3.0 - 2 * x)) * ref.get('contact_envelope', plan['active'][:, 2 + side])
    planned = focus > 1e-08
    if not planned.any():
        return (original, dict(bypassed=True, reason='No near-extension hand contact'))
    offset = points[:, upper] - points[:, collar]
    axis = np.cross(offset, [0.0, -1.0, 0.0])
    axis /= np.maximum(np.linalg.norm(axis, axis=1)[:, None], 1e-12)
    srcu = seed['points'][:, elbow] - seed['points'][:, upper]
    srcv = seed['points'][:, wrist] - seed['points'][:, elbow]
    source_flex = np.arccos(np.clip(np.einsum('ti,ti->t', srcu, srcv) / (u * v), -1, 1))
    (pins, _) = eval_pins(original, geo, plan)
    nominal = points[:, wrist] + plan['goals'][:, 2 + side] - pins[:, 2 + side]
    desired = nominal - points[:, collar]
    parallel = axis * np.einsum('ti,ti->t', axis, offset)[:, None]
    perp = offset - parallel
    ca = np.einsum('ti,ti->t', desired, perp)
    cb = np.einsum('ti,ti->t', desired, np.cross(axis, offset))
    phase = np.arctan2(cb, ca)
    bound = np.radians(CFG['collar_limit_deg']) * focus
    candidates = np.stack([-bound, bound, np.clip(phase, -bound, bound), np.clip(phase + np.pi, -bound, bound), np.clip(phase - np.pi, -bound, bound)], 1)
    squared = np.sum((desired - parallel) ** 2, axis=1) + np.sum(perp * perp, axis=1)
    distances = np.sqrt(np.maximum(squared[:, None] - 2 * (ca[:, None] * np.cos(candidates) + cb[:, None] * np.sin(candidates)), 0))
    max_reach = np.sqrt(u * u + v * v + 2 * u * v * np.cos(np.maximum(flex0 - np.radians(15.0) * focus, 0)))
    min_reach = np.sqrt(u * u + v * v + 2 * u * v * np.cos(np.minimum(flex0 + np.radians(15.0) * focus, np.pi)))
    reach_gap = np.maximum(np.maximum(distances.min(1) - max_reach, min_reach - distances.max(1)), 0)
    reach_gain = gaussian_filter1d(1.0 / (1.0 + (reach_gap / 0.03) ** 4), 2.4, truncate=3.0, mode='nearest')
    focus *= reach_gain
    planned = focus > 1e-08
    palm_ids = plan['palms'][side]
    vertices = geo[f'palm_v{side}'][palm_ids]
    weights = geo[f'palm_w{side}'][palm_ids]
    weighted = (vertices[None] - rest[:, None]) * weights.T[:, :, None]
    bias = plan.get('skin_bias', np.zeros((n, 4, 3)))[:, 2 + side]
    speed = np.r_[0.0, np.degrees(R.from_matrix(seed['local'][:-1, elbow].swapaxes(-1, -2) @ seed['local'][1:, elbow]).magnitude()) * FPS]
    motion = 0.25 + 0.75 / (1.0 + (speed / 30.0) ** 2)
    tw = CFG['temporal_weight']
    one = planned[1:] | planned[:-1]
    two = planned[2:] | planned[1:-1] | planned[:-2]
    w1 = one * np.sqrt(np.minimum(motion[1:], motion[:-1])) * tw / CFG['velocity_sigma']
    w2 = two * np.sqrt(np.minimum(np.minimum(motion[2:], motion[1:-1]), motion[:-2])) * tw / CFG['acceleration_sigma']
    wf = two * np.sqrt(np.minimum(np.minimum(motion[2:], motion[1:-1]), motion[:-2])) * tw / CFG['actual_flex_acceleration_sigma']
    lo = np.c_[-np.radians(CFG['collar_limit_deg']) * focus, -np.minimum(flex0, np.radians(15.0) * focus)]
    hi = np.c_[np.radians(CFG['collar_limit_deg']) * focus, np.minimum(np.pi - flex0, np.radians(15.0) * focus)]
    lo[~planned] = -1e-12
    hi[~planned] = 1e-12

    def predict(z, save=False):
        q = z.reshape(n, 2)
        theta = flex0 + q[:, 1]
        C = R.from_rotvec(axis * q[:, 0, None]).as_matrix()
        S = points[:, collar] + np.einsum('tij,tj->ti', C, offset)
        target = nominal - S
        direction = target / np.maximum(np.linalg.norm(target, axis=1)[:, None], 1e-12)
        ref_axis = np.einsum('tij,tj->ti', C, ref['axis'])
        pole = np.einsum('tij,tj->ti', C, ref['pole'])
        pole = np.einsum('tij,tj->ti', align_many(ref_axis, direction), pole)
        pole -= direction * np.einsum('ti,ti->t', direction, pole)[:, None]
        pole /= np.maximum(np.linalg.norm(pole, axis=1)[:, None], 1e-12)
        distance = np.sqrt(np.maximum(u * u + v * v + 2 * u * v * np.cos(theta), 1e-12))
        along = (u * u - v * v + distance * distance) / (2 * distance)
        height = u * v * np.sin(theta) / distance
        E = S + direction * along[:, None] + pole * height[:, None]
        W = S + direction * distance[:, None]
        U = align_many(np.einsum('tij,tj->ti', C, srcu), E - S) @ (C @ seed['world'][:, upper])
        L = align_many(np.einsum('tij,tj->ti', C, srcv), W - E) @ (C @ seed['world'][:, elbow])
        K = C @ world[:, collar]
        positions = points.copy()
        matrices = world.copy()
        for (j, val) in [(upper, S), (elbow, E), (wrist, W)]:
            positions[planned, j] = val[planned]
        for (j, val) in [(collar, K), (upper, U), (elbow, L)]:
            matrices[planned, j] = val[planned]
        sample = np.einsum('tjik,jvk->tvi', matrices, weighted, optimize=True) + np.einsum('vj,tji->tvi', weights, positions, optimize=True)
        sample = sample.mean(1) + bias
        relative = R.from_matrix(seed['world'][:, upper].swapaxes(-1, -2) @ matrices[:, upper]).as_rotvec()
        state = np.c_[theta - source_flex, relative, q[:, 0]]
        if save:
            local = local_from_world(matrices)
            (p, w) = fk(local, rest, original['root'])
            return (dict(original, local=local, world=w, points=p), sample, theta, S)
        return (sample, positions[:, elbow], state, theta)

    def residual(z):
        (sample, E, state, theta) = predict(z)
        q = z.reshape(n, 2)
        contact = (sample - plan['goals'][:, 2 + side]) * planned[:, None]
        contact *= np.sqrt(ref.get('contact_envelope', plan['active'][:, 2 + side]))[:, None]
        return np.r_[(contact / CFG['contact_sigma_m']).ravel(), (q / np.array([0.03, 0.175])).ravel(), ((E - points[:, elbow]) * planned[:, None] / 0.03).ravel(), (np.diff(state, axis=0) * w1[:, None] * np.array([1.0, 0.4, 0.4, 0.4, 1.0])).ravel(), (np.diff(state, n=2, axis=0) * w2[:, None] * np.array([1.0, 0.4, 0.4, 0.4, 1.0])).ravel(), np.diff(theta, n=2) * wf]
    rows = []
    cols = []
    cursor = 0

    def block(frames, count):
        nonlocal cursor
        f = np.asarray(frames)
        f = f[:, None] if f.ndim == 1 else f
        rr = cursor + np.arange(len(f))[:, None] * count + np.arange(count)
        variables = (f[:, :, None] * 2 + np.arange(2)).reshape(len(f), -1)
        for k in range(variables.shape[1]):
            rows.append(rr.ravel())
            cols.append(np.repeat(variables[:, k], count))
        cursor += len(f) * count
    block(np.arange(n), 3)
    block(np.arange(n), 2)
    block(np.arange(n), 3)
    block(np.stack([np.arange(n - 1), np.arange(1, n)], 1), 5)
    block(np.stack([np.arange(n - 2), np.arange(1, n - 1), np.arange(2, n)], 1), 5)
    block(np.stack([np.arange(n - 2), np.arange(1, n - 1), np.arange(2, n)], 1), 1)
    sparsity = coo_matrix((np.ones(sum(map(len, rows))), (np.concatenate(rows), np.concatenate(cols))), shape=(cursor, n * 2)).tocsr()
    opt = least_squares(residual, np.zeros(n * 2), jac_sparsity=sparsity, bounds=(lo.ravel(), hi.ravel()), max_nfev=CFG['iterations'], ftol=1e-05, xtol=2e-06, gtol=1e-05, tr_options={'maxiter': 150})
    (data, sample, theta, S) = predict(opt.x, save=True)
    report = dict(success=bool(opt.success), evaluations=int(opt.nfev), message=opt.message, planned_frames=int(planned.sum()), collar_rotation_max_deg=float(np.degrees(abs(opt.x.reshape(n, 2)[:, 0])).max()), shoulder_shift_max_mm=float(np.linalg.norm(S - points[:, upper], axis=1).max() * 1000), flex_change_max_deg=float(np.degrees(abs(theta - flex0)).max()), palm_contact_max_mm=float(np.linalg.norm((sample - plan['goals'][:, 2 + side])[plan['active'][:, 2 + side]], axis=-1).max(initial=0) * 1000), reach_gap_max_mm=float(reach_gap.max() * 1000), reach_gain_min=float(reach_gain.min()), seconds=None)
    return (data, report)

def optimize_hand_windows(original, seed, ref, geo, plan, side):
    """Only solve finite near-extension contact episodes with six-frame joins."""
    u = original['points'][:, 18 + side] - original['points'][:, 16 + side]
    v = original['points'][:, 20 + side] - original['points'][:, 18 + side]
    flex = np.degrees(np.arccos(np.clip(np.sum(u * v, axis=1) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1), -1, 1)))
    mask = (flex < 35.0) & (ref['contact_envelope'] > 0)
    windows = []
    n = len(mask)
    for (a, b) in trajectory.runs(mask):
        a = int(max(0, a - 6))
        b = int(min(n, b + 6))
        if windows and a <= windows[-1][1]:
            windows[-1][1] = max(windows[-1][1], b)
        else:
            windows.append([a, b])
    current = {k: v.copy() for (k, v) in original.items()}
    reports = []

    def slice_data(data, a, b):
        d = dict(data)
        for k in ['local', 'root', 'points', 'world']:
            d[k] = data[k][a:b].copy()
        return d
    for (a, b) in windows:
        rr = dict(ref)
        for k in ['axis', 'pole', 'source_flex_deg', 'calibration_mask', 'contact_envelope']:
            if k in rr:
                rr[k] = rr[k][a:b]
        pp = dict(plan)
        for k in ['weights', 'goals', 'active', 'skin_bias']:
            if k in pp:
                pp[k] = pp[k][a:b]
        (data, r) = optimize_hand(slice_data(original, a, b), slice_data(seed, a, b), rr, geo, pp, side)
        for k in ['local', 'world', 'points']:
            current[k][a:b] = data[k]
        r['window'] = [a, b]
        reports.append(r)
    return (current, dict(windows=reports, window_count=len(windows), bypassed=not windows))

def prepare(seed, geo, saved, bias=None):
    active = saved['active']
    plan = dict(palms=[saved['palm_ids_left'], saved['palm_ids_right']], weights=saved['support_weights'], active=active, goals=saved['goals'].copy())
    if bias is not None:
        plan['skin_bias'] = bias
    (support, _) = eval_pins(seed, geo, plan)
    (probes, _, _) = geometry_samples(seed, geo, plan['palms'])
    segments = [dict(foot=s, start=a, end=b) for s in [0, 1] for (a, b) in trajectory.runs(active[:, 2 + s])]
    (off, transition) = trajectory.contact_offsets(probes, active[:, 2:], saved['anchors'][:, 2:], support[:, 2:], np.zeros_like(seed['root']), segments, FPS, np.ones((len(active), 2)), np.zeros(len(active), bool))
    plan['goals'][:, 2:] = support[:, 2:] + off
    planned = active[:, 2:] | transition
    envelope = active[:, 2:].astype(float)
    for side in [0, 1]:
        for (a, b) in trajectory.runs(transition[:, side] & ~active[:, 2 + side]):
            left = a > 0 and active[a - 1, 2 + side]
            right = b < len(active) and active[b, 2 + side]
            if left and right:
                envelope[a:b, side] = 1.0
                continue
            t = np.arange(b - a, dtype=float) / max(1, b - a)
            if left:
                t = 1.0 - (np.arange(b - a) + 1.0) / max(1, b - a)
            envelope[a:b, side] = t * t * t * (10 + t * (-15 + 6 * t))
    return (plan, planned, envelope)

def normalise(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)

def angle(p, a, b, c):
    return np.arccos(np.clip(np.sum(normalise(p[:, b] - p[:, a]) * normalise(p[:, c] - p[:, b]), axis=-1), -1, 1))

def characterize(old, rotation, source, native):
    w = old['world'].copy()
    body = [0, 3, 6, 9, 12, 15]
    w[:, body] = R.from_matrix(native[:, body].reshape(-1, 3, 3)).as_matrix().reshape(len(w), 6, 3, 3)
    offset = old['rest'][[1, 2]].mean(0) - old['rest'][0]
    root = old['root'] + old['world'][:, 0] @ offset - w[:, 0] @ offset
    local = local_from_world(w)
    (p, w) = fk(local, old['rest'], root)
    result = dict(old, root=root, local=local, world=w, points=p)
    rw = rotation['world'].copy()
    rw[:, body] = w[:, body]
    rl = local_from_world(rw)
    (rp, rw) = fk(rl, old['rest'], root)
    seed = dict(rotation, root=root.copy(), local=rl, world=rw, points=rp)
    return (result, seed)

def leg_reference(seed, side):
    (a, b, c) = CHAINS[side]
    p = seed['points']
    u = p[:, b] - p[:, a]
    v = p[:, c] - p[:, b]
    axis = normalise(u + v)
    normals = normalise(np.cross(u, v))
    flex = angle(p, a, b, c)
    reliable = flex > np.radians(40)
    if reliable.sum() < 5:
        reliable = flex >= np.percentile(flex, 85)
    normal = unit(np.median(np.einsum('tji,tj->ti', seed['world'][:, a], normals)[reliable], axis=0))
    prior = normalise(np.cross(axis, seed['world'][:, a] @ normal))
    obs = normalise(u - axis * np.sum(u * axis, axis=1)[:, None])
    ph = np.arctan2(np.sum(axis * np.cross(prior, obs), 1), np.clip(np.sum(prior * obs, 1), -1, 1))
    x = np.clip((np.degrees(flex) - 5) / 20, 0, 1)
    weight = x * x * (3 - 2 * x)
    pole = R.from_rotvec(axis * (ph * weight)[:, None]).apply(prior)
    return (axis, pole, flex)

def legs(old, body, seed, geo, saved):
    start = time.perf_counter()
    n = len(old['root'])
    active = saved['active'][:, :2]
    p = body['points']
    rest = body['rest']
    original = old['points']
    root0 = body['root']
    scale = float(seed['ratio'])
    refs = [leg_reference(seed, s) for s in [0, 1]]
    goals = original[:, [7, 8]].copy()
    if '_contact_ankle_goals' in old:
        goals[active] = old['_contact_ankle_goals'][active]
    theta0 = np.stack([angle(original, *ch) for ch in CHAINS[:2]], 1)
    extent = maximum_filter1d(active.astype(float), size=13, axis=0, mode='nearest')
    envelope = gaussian_filter1d(extent, 1.5, axis=0, truncate=3, mode='nearest')
    envelope[active] = 1.0
    envelope[envelope < 1e-06] = 0
    enabled = envelope > 0
    hipoffset = p[:, [1, 2]] - root0[:, None]
    lengths = [(np.linalg.norm(rest[b] - rest[a]), np.linalg.norm(rest[c] - rest[b])) for (a, b, c) in CHAINS[:2]]
    poles = []
    phis = []
    for (s, (a, b, c)) in enumerate(CHAINS[:2]):
        (axis0, pole0, _) = refs[s]
        direction = normalise(goals[:, s] - p[:, a])
        pole = normalise(np.einsum('tij,tj->ti', align_many(axis0, direction), pole0))
        obs = original[:, b] - original[:, a]
        obs = normalise(obs - direction * np.sum(obs * direction, 1)[:, None])
        phi = np.unwrap(np.arctan2(np.sum(direction * np.cross(pole, obs), 1), np.clip(np.sum(pole * obs, 1), -1, 1)))
        phi -= round(np.median(phi) / (2 * np.pi)) * 2 * np.pi
        phis.append(phi)
        poles.append(pole)
    phi0 = np.stack(phis, 1)
    speed = np.zeros((n, 2))
    for (s, (a, b, c)) in enumerate(CHAINS[:2]):
        inc = np.degrees(R.from_matrix(seed['local'][:-1, b].swapaxes(-1, -2) @ seed['local'][1:, b]).magnitude()) * FPS
        speed[1:, s] = inc
    motion = 0.3 + 0.7 / (1 + (speed / 50) ** 2)
    w1 = np.sqrt(np.minimum(motion[1:], motion[:-1])) * (enabled[1:] | enabled[:-1])
    w2 = np.sqrt(np.minimum(np.minimum(motion[2:], motion[1:-1]), motion[:-2])) * (enabled[2:] | enabled[1:-1] | enabled[:-2])
    sourceflex = np.stack([x[2] for x in refs], 1)
    rootfocus = envelope.max(1)
    lo = np.c_[-np.tile(0.025 * max(scale, 0.5) * rootfocus[:, None], (1, 3)), np.maximum(-theta0, -np.radians(25) * envelope), -np.radians(80) * envelope]
    hi = np.c_[np.tile(0.025 * max(scale, 0.5) * rootfocus[:, None], (1, 3)), np.minimum(np.pi - 0.001 - theta0, np.radians(25) * envelope), np.radians(80) * envelope]
    lo = np.minimum(lo, -1e-12)
    hi = np.maximum(hi, 1e-12)

    def predict(z, save=False):
        q = z.reshape(n, 7)
        theta = theta0 + q[:, 3:5]
        phi = phi0 + q[:, 5:7]
        root = root0 + q[:, :3]
        if os.environ.get('PERFORMANCE_FAST') == '1' and (not save):
            ends = []
            state = []
            for (s, (a, b, c)) in enumerate(CHAINS[:2]):
                (u, v) = lengths[s]
                hip = root + hipoffset[:, s]
                goal = goals[:, s] + q[:, :3] * (1 - envelope[:, s, None])
                direction = normalise(goal - hip)
                distance = np.sqrt(np.maximum(u * u + v * v + 2 * u * v * np.cos(theta[:, s]), 1e-12))
                end = hip + direction * distance[:, None]
                ends.append(end - goal)
                state.append(theta[:, s] - sourceflex[:, s])
            return (np.stack(ends, 1), None, np.stack(state, 1), theta, phi, q)
        world = body['world'].copy()
        ends = []
        knees = []
        state = []
        for (s, (a, b, c)) in enumerate(CHAINS[:2]):
            (u, v) = lengths[s]
            hip = root + hipoffset[:, s]
            goal = goals[:, s] + q[:, :3] * (1 - envelope[:, s, None])
            direction = normalise(goal - hip)
            (axis0, pole0, flex) = refs[s]
            pole = np.einsum('tij,tj->ti', align_many(axis0, direction), pole0)
            pole = R.from_rotvec(direction * phi[:, s, None]).apply(pole)
            distance = np.sqrt(np.maximum(u * u + v * v + 2 * u * v * np.cos(theta[:, s]), 1e-12))
            along = (u * u - v * v + distance * distance) / (2 * distance)
            height = u * v * np.sin(theta[:, s]) / distance
            knee = hip + direction * along[:, None] + pole * height[:, None]
            end = hip + direction * distance[:, None]
            world[:, a] = align_many(seed['points'][:, b] - seed['points'][:, a], knee - hip) @ seed['world'][:, a]
            world[:, b] = align_many(seed['points'][:, c] - seed['points'][:, b], end - knee) @ seed['world'][:, b]
            world[~enabled[:, s], a] = old['world'][~enabled[:, s], a]
            world[~enabled[:, s], b] = old['world'][~enabled[:, s], b]
            ends.append(end - goal)
            knees.append(knee)
            state.append(theta[:, s] - sourceflex[:, s])
        if save:
            local = local_from_world(world)
            (pp, ww) = fk(local, rest, root)
            return dict(body, root=root, local=local, world=ww, points=pp)
        return (np.stack(ends, 1), np.stack(knees, 1), np.stack(state, 1), theta, phi, q)

    def residual(z):
        (error, knee, state, theta, phi, q) = predict(z)
        return np.r_[(error * envelope[:, :, None] / 0.00025).ravel(), (q[:, :3] / (0.012 * max(scale, 0.5))).ravel(), (q[:, 3:5] / 0.25).ravel(), (q[:, 5:] / 0.5).ravel(), (np.diff(state, axis=0) * w1 / 0.07).ravel(), (np.diff(state, n=2, axis=0) * w2 / 0.006).ravel(), (np.diff(phi, axis=0) * w1 / 0.07).ravel(), (np.diff(phi, n=2, axis=0) * w2 / 0.012).ravel(), (np.diff(q[:, :3], n=2, axis=0) / (0.0005 * max(scale, 0.5))).ravel()]
    rows = []
    cols = []
    cursor = 0

    def block(frames, count):
        nonlocal cursor
        f = np.asarray(frames)
        f = f[:, None] if f.ndim == 1 else f
        rr = cursor + np.arange(len(f))[:, None] * count + np.arange(count)
        variables = (f[:, :, None] * 7 + np.arange(7)).reshape(len(f), -1)
        for k in range(variables.shape[1]):
            rows.append(rr.ravel())
            cols.append(np.repeat(variables[:, k], count))
        cursor += len(f) * count
    f = np.arange(n)
    one = np.stack([f[:-1], f[1:]], 1)
    two = np.stack([f[:-2], f[1:-1], f[2:]], 1)
    for count in [6, 3, 2, 2]:
        block(f, count)
    for (frames, count) in [(one, 2), (two, 2), (one, 2), (two, 2), (two, 3)]:
        block(frames, count)
    sparsity = coo_matrix((np.ones(sum(map(len, rows))), (np.concatenate(rows), np.concatenate(cols))), shape=(cursor, n * 7)).tocsr()
    initial = residual(np.zeros(n * 7))
    result = least_squares(residual, np.zeros(n * 7), bounds=(lo.ravel(), hi.ravel()), jac_sparsity=sparsity, max_nfev=120, ftol=2e-05, xtol=2e-06, gtol=1e-05, tr_options={'maxiter': 150})
    output = predict(result.x, True)
    return (output, dict(seconds=time.perf_counter() - start, success=bool(result.success), evaluations=result.nfev, cost_before=float(initial @ initial / 2), cost_after=float(result.cost), root_extra_max_mm=float(np.linalg.norm(output['root'] - root0, axis=1).max() * 1000), leg_ankle_target_max_mm=float(np.linalg.norm(output['points'][:, [7, 8]] - goals, axis=-1)[active].max(initial=0) * 1000)))

def feet_only_body(kind, old, source, native, saved, geo):
    start = time.perf_counter()
    raw = dict(np.load(OUT / f'{LABEL}-{kind}-rotation.npz'))
    (baseline, seed) = characterize(raw, raw, source, native)
    source['world'] = R.from_matrix(source['world'].reshape(-1, 3, 3)).as_matrix().reshape(-1, 22, 3, 3)
    active = saved['active'].copy()
    active[:, 2:] = False
    source['contact_active'] = active
    goals = baseline['points'][:, [7, 8, 20, 21]].copy()
    goals[:, :2] = old['points'][:, [7, 8]]
    (shift, diagnostic) = root_plan(baseline, goals, active, active, source)
    root = baseline['root'].copy()
    root[:, 1] += shift
    root = reachable_root_xz(baseline, root, goals, active, source, mode='smooth-stance')
    root = baseline['root'] + gaussian_filter1d(root - baseline['root'], 2.0, axis=0, truncate=3.0, mode='nearest')
    (candidate, _) = two_bone(baseline, root, goals, active, source)
    release_to_fk(candidate, baseline, active,FPS)
    clear_plan = dict(palms=[saved['palm_ids_left'], saved['palm_ids_right']], weights=saved['support_weights'])
    diagnostic['free_clearance'] = free_clearance(candidate, source, active, geo, clear_plan)
    candidate = dict(old, **candidate, ratio=raw['ratio'], _contact_ankle_goals=goals[:, :2].copy())
    np.savez(OUT / f'{LABEL}-{kind}-root-base.npz', **baseline)
    np.savez(OUT / f'{LABEL}-{kind}-feet-replanned.npz', **candidate)
    diagnostic.update(seconds=time.perf_counter() - start, hands_used=False)
    return (candidate, seed, diagnostic)

def run(kind):
    started = time.perf_counter()
    label = LABEL
    old = dict(np.load(PREVIOUS / f'{label}-{kind}-stable-contact.npz'))
    rotation = dict(np.load(PREVIOUS / f'{label}-{kind}-rotation.npz'))
    source = dict(np.load(OUT / f'{label}-source.npz'))
    native = np.load(OUT / f'{label}-source-neutral.npz')['native_world']
    geo = np.load(OUT / f'{kind}-contact-geometry.npz')
    saved = dict(np.load(PREVIOUS / f'{label}-{kind}-contact-plan.npz'))
    (old, rotation, body_report) = feet_only_body(kind, old, source, native, saved, geo)
    (body, seed) = characterize(old, rotation, source, native)
    np.savez(OUT / f'{label}-{kind}-body-only.npz', **body)
    (fixed, report) = legs(old, body, seed, geo, saved)
    np.savez(OUT / f'{label}-{kind}-legs-only.npz', **fixed)
    world = fixed['world'].copy()
    world[:, [13, 14, 16, 17, 18, 19]] = seed['world'][:, [13, 14, 16, 17, 18, 19]]
    local = local_from_world(world)
    (p, w) = fk(local, fixed['rest'], fixed['root'])
    handseed = dict(fixed, world=w, points=p, local=local)
    biaspath = OUT / f'{label}-{kind}-skin-bias.npy'
    bias = np.load(biaspath) if biaspath.exists() else None
    (plan, planned, envelope) = prepare(handseed, geo, saved, bias)
    refs = arm_reference(handseed) if planned.any() else []
    for (s, ref) in enumerate(refs):
        (a, b, c) = CHAINS[2 + s]
        axis = ref['axis']
        observed = handseed['points'][:, b] - handseed['points'][:, a]
        observed -= axis * np.sum(observed * axis, axis=1)[:, None]
        norm = np.linalg.norm(observed, axis=1)
        observed /= np.maximum(norm[:, None], 1e-12)
        observed[norm < 1e-08] = ref['pole'][norm < 1e-08]
        blend = np.clip((35 - ref['source_flex_deg']) / 20, 0, 1)
        blend = blend * blend * (3 - 2 * blend) * envelope[:, s]
        ph = np.arctan2(np.sum(axis * np.cross(observed, ref['pole']), 1), np.clip(np.sum(observed * ref['pole'], 1), -1, 1))
        ref['pole'] = R.from_rotvec(axis * (ph * blend)[:, None]).apply(observed)
        ref['contact_envelope'] = envelope[:, s]
    mask = np.zeros_like(saved['active'])
    mask[:, 2:] = planned
    current = handseed
    if os.environ.get('PERFORMANCE_FAST') == '1' and (not planned.any()):
        (current, _) = arm_solve(handseed, np.zeros((len(mask), 4, 3)), mask, refs)
    else:
        for it in range(10):
            (pins, _) = eval_pins(current, geo, plan)
            goals = current['points'][:, [7, 8, 20, 21]] + (plan['goals'] - pins) * (0.7 if it else 1.0)
            (current, _) = arm_solve(handseed, goals, mask, refs)
    current['world'][~planned[:, 0], 16] = handseed['world'][~planned[:, 0], 16]
    local = local_from_world(current['world'])
    (p, w) = fk(local, current['rest'], current['root'])
    current.update(local=local, points=p, world=w)
    arms = []
    for side in range(len(refs)):
        (current, r) = optimize_hand_windows(current, handseed, refs[side], geo, plan, side)
        arms.append(r)
    current['root_shift'] = current['root'] - np.load(OUT / f'{label}-{kind}-root-base.npz')['root']
    np.savez(OUT / f'{label}-{kind}-stable-contact.npz', **current)
    np.savez(OUT / f'{label}-{kind}-rotation.npz', **seed)
    np.savez(OUT / f'{label}-{kind}-fk.npz', **seed)
    updated = dict(saved, goals=plan['goals'], handoff_envelope=envelope)
    np.savez(OUT / f'{label}-{kind}-contact-plan.npz', **updated)
    length = np.linalg.norm(p[:, 1:] - p[:, PARENTS[1:]], axis=-1)
    expected = np.linalg.norm(current['rest'][1:] - current['rest'][PARENTS[1:]], axis=-1)
    assert abs(length - expected).max() < 1e-08
    record = dict(target=kind, clip=label, frames=len(p), seconds=time.perf_counter() - started, body=body_report, legs=report, arms=arms, stable=score(current, source, saved['active'], geo, plan), fixed_length_max_error_m=float(abs(length - expected).max()), source_timing_changed=False)
    print('READY', kind, json.dumps(record), flush=True)
    return record
