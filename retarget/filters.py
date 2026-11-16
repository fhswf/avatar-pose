"""Signalfilter. Alle Fenster liegen mittig, verschieben also zeitlich nichts."""

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .config import (GAP_BLEND_S, GAP_FULL_S, HAMPEL_WIN, JUMP_SEG_S,
                     QUAT_SPIKE_DEG, VMAX_DEG_S)


def odd(x):
    n = max(3, int(round(x)))
    return n if n % 2 else n + 1


def savgol_kernel(window, order=2):
    half = window // 2
    x = np.arange(-half, half + 1, dtype=float)
    A = np.vander(x, min(order, window - 1) + 1, increasing=True)
    return np.linalg.pinv(A)[0]


def savgol(arr, window, order=2):
    n = len(arr)
    if window < 5 or n < window:
        return arr
    k = savgol_kernel(window, order)
    padded = np.pad(arr, window // 2, mode="reflect")
    return np.convolve(padded, k[::-1], mode="valid")


def hampel(arr, window=HAMPEL_WIN, nsig=3.0, floor=0.004):
    n = len(arr)
    if n < window:
        return arr
    half = window // 2
    padded = np.pad(arr, half, mode="edge")
    win = sliding_window_view(padded, window)
    med = np.median(win, axis=1)
    mad = np.median(np.abs(win - med[:, None]), axis=1)
    thr = np.maximum(nsig * 1.4826 * mad, floor)
    out = arr.copy()
    bad = np.abs(arr - med) > thr
    out[bad] = med[bad]
    return out


def smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def median_filter(arr, window):
    """Median ueber ein gleitendes Fenster (Rand mit dem Randwert aufgefuellt)."""
    n = len(arr)
    if window < 3 or n < window:
        return arr
    half = window // 2
    padded = np.pad(arr, half, mode="edge")
    return np.median(sliding_window_view(padded, window), axis=1)


def rolling_max(arr, window):
    n = len(arr)
    if window < 3 or n < window:
        return arr
    half = window // 2
    padded = np.pad(arr, half, mode="edge")
    return sliding_window_view(padded, window).max(axis=1)


def rolling_mean(arr, window):
    n = len(arr)
    if window < 3 or n < window:
        return arr
    half = window // 2
    padded = np.pad(arr, half, mode="edge")
    return sliding_window_view(padded, window).mean(axis=1)


def runs(mask):
    out, s = [], None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        elif not v and s is not None:
            out.append((s, i))
            s = None
    if s is not None:
        out.append((s, len(mask)))
    return out


def series_fill(v, valid, fps, full_s=GAP_FULL_S, blend_s=GAP_BLEND_S):
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return False
    v[:idx[0]] = v[idx[0]]
    v[idx[-1] + 1:] = v[idx[-1]]
    full_n = max(1, int(round(full_s * fps)))
    blend_n = max(2, int(round(blend_s * fps)))
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if gap <= 0:
            continue
        if gap <= full_n:
            for j in range(a + 1, b):
                t = smoothstep((j - a) / (b - a))
                v[j] = v[a] + (v[b] - v[a]) * t
        else:
            hold_end = max(a + 1, b - blend_n)
            v[a + 1:hold_end] = v[a]
            for j in range(hold_end, b):
                t = smoothstep((j - hold_end + 1) / (b - hold_end))
                v[j] = v[a] + (v[b] - v[a]) * t
    return True


def qnorm(q):
    return q / (np.linalg.norm(q) + 1e-12)


def qangle(a, b):
    d = abs(float(np.dot(qnorm(a), qnorm(b))))
    return 2.0 * math.acos(min(1.0, d))


def qslerp(a, b, t):
    d = float(np.dot(a, b))
    if d < 0.0:
        b, d = -b, -d
    if d > 0.9995:
        r = a + t * (b - a)
    else:
        th = math.acos(min(1.0, d))
        r = (math.sin((1.0 - t) * th) * a + math.sin(t * th) * b) / math.sin(th)
    return qnorm(r)


def quat_fix_sign(qs, valid):
    idx = np.where(valid)[0]
    for i, j in zip(idx[:-1], idx[1:]):
        if float(np.dot(qs[j], qs[i])) < 0.0:
            qs[j] = -qs[j]


def quat_despike(qs, valid, thr_deg=QUAT_SPIKE_DEG, passes=2):
    thr = math.radians(thr_deg)
    idx = np.where(valid)[0]
    fixed = 0
    for _ in range(passes):
        changed = False
        for k in range(1, len(idx) - 1):
            i0, i1, i2 = idx[k - 1], idx[k], idx[k + 1]
            if i1 - i0 > 2 or i2 - i1 > 2:
                continue
            d1, d2 = qangle(qs[i1], qs[i0]), qangle(qs[i2], qs[i1])
            if d1 > thr and d2 > thr and qangle(qs[i2], qs[i0]) < 0.5 * (d1 + d2):
                qs[i1] = qslerp(qs[i0], qs[i2], 0.5)
                fixed += 1
                changed = True
        if not changed:
            break
    return fixed


def reject_jumps(qs, valid, fps, vmax_deg_s=VMAX_DEG_S, max_seg_s=JUMP_SEG_S):
    idx = np.where(valid)[0]
    if len(idx) < 3:
        return 0
    vmax = math.radians(vmax_deg_s) / fps
    cut = [0]
    for k in range(1, len(idx)):
        if qangle(qs[idx[k]], qs[idx[k - 1]]) / (idx[k] - idx[k - 1]) > vmax:
            cut.append(k)
    cut.append(len(idx))
    if len(cut) == 2:
        return 0
    segs = [(cut[j], cut[j + 1]) for j in range(len(cut) - 1)]
    anchor = max(range(len(segs)), key=lambda j: segs[j][1] - segs[j][0])
    max_seg = max(2, int(round(max_seg_s * fps)))
    removed = 0
    for j, (a, b) in enumerate(segs):
        if j == anchor:
            continue
        if idx[b - 1] - idx[a] + 1 <= max_seg:
            valid[idx[a:b]] = False
            removed += b - a
    return removed


def quat_fill_gaps(qs, valid, fps, full_s=GAP_FULL_S, blend_s=GAP_BLEND_S):
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return
    qs[:idx[0]] = qs[idx[0]]
    qs[idx[-1] + 1:] = qs[idx[-1]]
    full_n = max(1, int(round(full_s * fps)))
    blend_n = max(2, int(round(blend_s * fps)))
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if gap <= 0:
            continue
        if gap <= full_n:
            for j in range(a + 1, b):
                qs[j] = qslerp(qs[a], qs[b], smoothstep((j - a) / (b - a)))
        else:
            hold_end = max(a + 1, b - blend_n)
            qs[a + 1:hold_end] = qs[a]
            for j in range(hold_end, b):
                t = smoothstep((j - hold_end + 1) / (b - hold_end))
                qs[j] = qslerp(qs[a], qs[b], t)


def soften_jumps(qs, fps, vmax_deg_s=VMAX_DEG_S, passes=6):
    n = len(qs)
    if n < 3:
        return 0
    vmax = math.radians(vmax_deg_s) / fps
    max_half = max(3, int(round(0.25 * fps)))
    softened = 0
    for _ in range(passes):
        bad = [i for i in range(1, n) if qangle(qs[i], qs[i - 1]) > vmax]
        if not bad:
            break
        for i in bad:
            ang = qangle(qs[i], qs[i - 1])
            half = min(max_half, int(math.ceil(ang / vmax)) + 2)
            a, b = max(0, i - half), min(n - 1, i - 1 + half)
            if b - a < 2:
                continue
            qa, qb = qs[a].copy(), qs[b].copy()
            for j in range(a + 1, b):
                qs[j] = qslerp(qa, qb, smoothstep((j - a) / (b - a)))
            softened += 1
    return softened


def quat_smooth(qs, window):
    if window < 5 or len(qs) < window:
        return qs
    out = np.column_stack([savgol(qs[:, c], window) for c in range(4)])
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def filter_quat_series(qs, valid, fps, window, clean=True, vmax_deg_s=VMAX_DEG_S):
    if not valid.any():
        return qs
    quat_fix_sign(qs, valid)
    if clean:
        reject_jumps(qs, valid, fps, vmax_deg_s)
        quat_despike(qs, valid)
    quat_fill_gaps(qs, valid, fps)
    if clean:
        soften_jumps(qs, fps, vmax_deg_s)
    return quat_smooth(qs, window)


def filter_angle_series(v, valid, fps, window, clean=True, hampel_floor_deg=2.0):
    if not valid.any():
        return v, False
    if clean:
        for a, b in runs(valid):
            if b - a >= HAMPEL_WIN:
                v[a:b] = hampel(v[a:b], floor=math.radians(hampel_floor_deg))
    series_fill(v, valid, fps)
    return (savgol(v, window) if window else v), True


def hampel_angle(th, valid, window=HAMPEL_WIN, nsig=3.0):
    out = th.copy()
    c, s = np.cos(np.nan_to_num(th)), np.sin(np.nan_to_num(th))
    for comp in (c, s):
        for a, b in runs(valid):
            if b - a >= window:
                comp[a:b] = hampel(comp[a:b], window, nsig, floor=0.05)
    idx = np.where(valid)[0]
    out[idx] = np.arctan2(s[idx], c[idx])
    return out


def angles_to_axis_quats(v, valid, axis_index=2):
    qs = np.zeros((len(v), 4))
    qs[:, 0] = 1.0
    half = 0.5 * np.nan_to_num(v)
    qs[valid, 0] = np.cos(half[valid])
    qs[valid, axis_index] = np.sin(half[valid])
    return qs


def quat_median(qs):
    if len(qs) == 0:
        return None
    ref = qs[len(qs) // 2]
    aligned = np.where((qs @ ref)[:, None] < 0.0, -qs, qs)
    return qnorm(np.median(aligned, axis=0))