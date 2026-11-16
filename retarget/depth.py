"""Tiefe aus der perspektivischen Verkürzung; das Bild selbst bleibt unangetastet."""

import math

import numpy as np

from .config import (ARM_SEGMENTS, DEPTH_2D_BASE, DEPTH_2D_FORWARD,
                     DEPTH_MAG_SMOOTH_S, DEPTH_SEGMENTS, DEPTH_SIGN_SCALE,
                     DEPTH_SIGN_SMOOTH_S, DEPTH_WEIGHT, FRAME_MARGIN,
                     LEN_MIN_SAMPLES, LEN_PCT, MAX_OOP_DEG, WEAK_CONF)
from .filters import odd, savgol, series_fill
from .sequence import P, in_frame


def proj_px(fd, ai, bi, W, H):
    p2 = fd.get("pose")
    if not p2:
        return None
    a, b = P(p2, ai), P(p2, bi)
    if a is None or b is None:
        return None
    return math.hypot((b[0] - a[0]) * W, (b[1] - a[1]) * H)


def _pct(values, pct):
    values = sorted(values)
    return values[min(len(values) - 1, int(pct / 100.0 * len(values)))]


def estimate_px_lengths(frames, W, H):
    ref = []
    ai, bi = DEPTH_SEGMENTS["shoulders"][0], DEPTH_SEGMENTS["shoulders"][1]
    for fd in frames:
        v = proj_px(fd, ai, bi, W, H)
        if v:
            ref.append(v)
    shoulder = _pct(ref, 90) if len(ref) >= LEN_MIN_SAMPLES else 0.0

    out = {}
    for name, (ai, bi, lo, hi) in DEPTH_SEGMENTS.items():
        lens = [v for v in (proj_px(fd, ai, bi, W, H) for fd in frames) if v]
        if not lens:
            continue
        L = _pct(lens, LEN_PCT) if len(lens) >= LEN_MIN_SAMPLES else max(lens)
        if shoulder > 0.0:
            L = min(max(L, lo * shoulder), hi * shoulder)
        out[name] = L
    return out


def world_seg_len(frames, ai, bi):
    ds = []
    for fd in frames:
        pw = fd.get("pose_world")
        if not pw:
            continue
        a, b = P(pw, ai), P(pw, bi)
        if a is None or b is None:
            continue
        ds.append(math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
                            + (b[2] - a[2]) ** 2))
    if not ds:
        return None
    ds.sort()
    return ds[len(ds) // 2]


def depth_series(frames, W, H, lengths, fps, strength=1.0):
    n = len(frames)
    max_ratio = math.tan(math.radians(MAX_OOP_DEG))
    sign_win = odd(max(5.0, DEPTH_SIGN_SMOOTH_S * fps))
    mag_win = odd(max(5.0, DEPTH_MAG_SMOOTH_S * fps))
    out = {}
    for name, (ai, bi, _lo, _hi) in DEPTH_SEGMENTS.items():
        Lpx = lengths.get(name)
        if not Lpx:
            continue
        proj = np.full(n, np.nan)
        wzd = np.full(n, np.nan)
        for i, fd in enumerate(frames):
            v = proj_px(fd, ai, bi, W, H)
            if v is None:
                continue
            proj[i] = v
            pw = fd.get("pose_world") or fd.get("pose")
            aw, bw = P(pw, ai), P(pw, bi)
            if aw is not None and bw is not None:
                wzd[i] = bw[2] - aw[2]
        valid = ~np.isnan(proj)
        if not valid.any():
            continue
        pj = proj.copy()
        series_fill(pj, valid, fps)

        # Echte 3D-Daten erkennbar an einer Tiefe ungleich 0. Der Adapter
        # synthetisiert fuer 2D-Quellen z = 0, dann fehlt das Vorzeichen.
        vz = ~np.isnan(wzd)
        has_z = bool(vz.any() and np.nanmax(np.abs(wzd)) > 1e-9)

        mag = np.sqrt(np.maximum(0.0, Lpx * Lpx - pj * pj))
        if not has_z and name in DEPTH_2D_BASE:
            # Grundtiefe: Arme liegen vor dem Rumpf, auch wenn sie im Bild
            # nicht verkuerzt sind. Ohne diesen Anteil verschwinden die
            # Haende bei bildparallelen Unterarmen im Brustkorb.
            base = DEPTH_2D_BASE[name] * Lpx
            mag = np.sqrt(mag * mag + base * base)
        mag = np.minimum(mag, max_ratio * pj)
        mag = savgol(mag, mag_win)

        Lw = world_seg_len(frames, ai, bi)
        scale = DEPTH_SIGN_SCALE * Lw if Lw else None
        if has_z and scale and scale > 1e-6:
            wz = wzd.copy()
            series_fill(wz, vz, fps)
            wz = savgol(wz, sign_win)
            direction = np.tanh(wz / scale)
        elif name in ARM_SEGMENTS:
            direction = np.full(n, DEPTH_2D_FORWARD)
        else:
            direction = np.zeros(n)
        Y = savgol(strength * DEPTH_WEIGHT.get(name, 1.0) * direction * mag,
                   sign_win)
        cap = max_ratio * pj
        Y = np.clip(Y, -cap, cap)
        Y[~valid] = np.nan
        out[name] = Y
    return out


def visible(p2, idx, min_conf, weak=False):
    lm = P(p2, idx)
    if lm is None:
        return False
    thr = WEAK_CONF if weak else min_conf
    if len(lm) > 3 and lm[3] < thr:
        return False
    return in_frame(lm, FRAME_MARGIN + (0.1 if weak else 0.0))


def segment_directions(frames, W, H, depth, name, min_conf=0.5, weak=False):
    ai, bi = DEPTH_SEGMENTS[name][0], DEPTH_SEGMENTS[name][1]
    n = len(frames)
    vec = np.zeros((n, 3))
    valid = np.zeros(n, bool)
    ser = depth.get(name)
    for i, fd in enumerate(frames):
        p2 = fd.get("pose")
        if not p2:
            continue
        if not (visible(p2, ai, min_conf, weak) and visible(p2, bi, min_conf, weak)):
            continue
        a, b = P(p2, ai), P(p2, bi)
        y = 0.0
        if ser is not None and not np.isnan(ser[i]):
            y = float(ser[i])
        vec[i] = ((b[0] - a[0]) * W, y, -(b[1] - a[1]) * H)
        valid[i] = np.linalg.norm(vec[i]) > 1e-6
    return vec, valid


def reconstruct(seq, strength=1.0):
    W, H = seq.size
    lengths = estimate_px_lengths(seq.frames, W, H)
    return depth_series(seq.frames, W, H, lengths, seq.fps, strength)