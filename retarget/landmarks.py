"""Landmarks bereinigen: Handseiten, Mikro-Lücken, Ausreißer, Pose/Hand-Naht."""

import math

import numpy as np

from .config import HAMPEL_WIN, POSE, TINY_GAP_S, WEAK_CONF
from .filters import hampel, odd, runs, savgol, series_fill
from .sequence import P, n_landmarks


def reassign_hands(frames, aspect=1.0):
    swapped = dropped = 0
    for fd in frames:
        hands = []
        for key in ("left_hand", "right_hand"):
            if fd.get(key):
                hands.append((fd[key], fd.get(key + "_world")))
        if not hands:
            continue
        pose = fd.get("pose")
        lw = P(pose, POSE["left_wrist"]) if pose else None
        rw = P(pose, POSE["right_wrist"]) if pose else None
        if lw is None or rw is None:
            continue

        def dist(a, b):
            return math.hypot((a[0] - b[0]) * aspect, a[1] - b[1])

        Lsh, Rsh = P(pose, POSE["left_shoulder"]), P(pose, POSE["right_shoulder"])
        shoulder = dist(Lsh, Rsh) if (Lsh and Rsh) else 0.2
        max_d = max(0.05, 0.6 * shoulder)

        new = {"left_hand": None, "left_hand_world": None,
               "right_hand": None, "right_hand_world": None}
        if len(hands) == 2:
            if (dist(hands[0][0][0], lw) + dist(hands[1][0][0], rw) >
                    dist(hands[0][0][0], rw) + dist(hands[1][0][0], lw)):
                hands.reverse()
                swapped += 1
            for (h, w), wrist, side in ((hands[0], lw, "left"), (hands[1], rw, "right")):
                if dist(h[0], wrist) <= max_d:
                    new[side + "_hand"] = h
                    new[side + "_hand_world"] = w
                else:
                    dropped += 1
        else:
            h, w = hands[0]
            dl, dr = dist(h[0], lw), dist(h[0], rw)
            if min(dl, dr) > max_d:
                dropped += 1
            else:
                side = "left" if dl <= dr else "right"
                if (side == "left") != (fd.get("left_hand") is h):
                    swapped += 1
                new[side + "_hand"] = h
                new[side + "_hand_world"] = w
        for key, value in new.items():
            fd[key] = value
    return swapped, dropped


def _fill_hand_micro_gaps(frames, base, tiny):
    keyw = base + "_world"
    idx = [i for i, fd in enumerate(frames) if fd.get(base)]
    filled = 0
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if not (0 < gap <= tiny):
            continue
        for j in range(a + 1, b):
            t = (j - a) / (b - a)
            for key in (base, keyw):
                A, B = frames[a].get(key), frames[b].get(key)
                if not A or not B or len(A) != len(B):
                    continue
                frames[j][key] = [[A[k][0] * (1 - t) + B[k][0] * t,
                                   A[k][1] * (1 - t) + B[k][1] * t,
                                   A[k][2] * (1 - t) + B[k][2] * t,
                                   min(A[k][3], B[k][3])] for k in range(len(A))]
            filled += 1
    return filled


def despike_landmarks(frames, fps, clean=True):
    n = len(frames)
    if n < 3:
        return
    tiny = max(1, int(round(TINY_GAP_S * fps)))
    for base in ("left_hand", "right_hand"):
        _fill_hand_micro_gaps(frames, base, tiny)

    specs = (("pose", 0.004, True), ("pose_world", 0.008, False),
             ("left_hand", 0.004, False), ("right_hand", 0.004, False),
             ("left_hand_world", 0.008, False), ("right_hand_world", 0.008, False))
    for key, floor, conf_gated in specs:
        n_lm = n_landmarks(frames, key)
        if n_lm is None:
            continue
        for kk in range(n_lm):
            vals = np.zeros((n, 3))
            conf = np.zeros(n)
            has = np.zeros(n, bool)
            for i, fd in enumerate(frames):
                lst = fd.get(key)
                if lst and lst[kk] is not None:
                    vals[i] = lst[kk][:3]
                    conf[i] = lst[kk][3] if len(lst[kk]) > 3 else 1.0
                    has[i] = True
            if not has.any():
                continue
            valid = (has & (conf >= WEAK_CONF)) if conf_gated else has.copy()
            if conf_gated:
                idx = np.where(valid)[0]
                for a, b in zip(idx[:-1], idx[1:]):
                    gap = b - a - 1
                    if 0 < gap <= tiny and has[a + 1:b].all():
                        for j in range(a + 1, b):
                            t = (j - a) / (b - a)
                            vals[j] = vals[a] * (1.0 - t) + vals[b] * t
                            conf[j] = min(conf[a], conf[b])
                        valid[a + 1:b] = True
            if clean:
                for a, b in runs(valid):
                    if b - a < 5:
                        continue
                    for c in range(3):
                        vals[a:b, c] = hampel(vals[a:b, c], floor=floor)
            for i, fd in enumerate(frames):
                if valid[i]:
                    lm = fd[key][kk]
                    lm[0], lm[1], lm[2] = (float(vals[i, 0]), float(vals[i, 1]),
                                           float(vals[i, 2]))
                    if len(lm) > 3:
                        lm[3] = float(conf[i])


def snap_wrist_to_hand(frames, fps):
    n = len(frames)
    win = odd(max(5.0, 0.5 * fps))
    total = 0
    for key, idx in (("left_hand", POSE["left_wrist"]),
                     ("right_hand", POSE["right_wrist"])):
        off = np.zeros((n, 2))
        valid = np.zeros(n, bool)
        for i, fd in enumerate(frames):
            p, h = fd.get("pose"), fd.get(key)
            lm = P(p, idx) if p else None
            if not h or lm is None:
                continue
            off[i] = (h[0][0] - lm[0], h[0][1] - lm[1])
            valid[i] = True
        if not valid.any():
            continue
        for c in range(2):
            col = off[:, c]
            for a, b in runs(valid):
                if b - a >= HAMPEL_WIN:
                    col[a:b] = hampel(col[a:b], floor=0.002)
            series_fill(col, valid, fps)
            off[:, c] = savgol(col, win)
        for i, fd in enumerate(frames):
            lm = P(fd.get("pose"), idx) if fd.get("pose") else None
            if lm is None:
                continue
            lm[0] += float(off[i, 0])
            lm[1] += float(off[i, 1])
            total += 1
    return total


def process(seq, clean=True, snap_wrist=True):
    frames = seq.frames
    if clean and seq.get("hands") != "wilor":
        reassign_hands(frames, seq.aspect)
    if snap_wrist:
        snap_wrist_to_hand(frames, seq.fps)
    despike_landmarks(frames, seq.fps, clean)
    return seq