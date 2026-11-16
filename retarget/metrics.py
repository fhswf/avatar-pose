"""Masse fuer die Bewertung: Guete des Eingangssignals und Unruhe der Bewegung."""

import math

import numpy as np

from .config import POSE
from .filters import qangle
from .sequence import P


def _finite(a):
    return a[np.isfinite(a)]


def step_series(quats):
    q = np.asarray(quats, dtype=float)
    return np.array([math.degrees(qangle(q[i - 1], q[i]))
                     for i in range(1, len(q))])


def jerk(quats, jump_deg=30.0):
    s = step_series(quats)
    if len(s) < 2:
        return {"zappel": 0.0, "max_step": 0.0, "median_step": 0.0, "jumps": 0,
                "n": len(s)}
    d = np.diff(s)
    return {"zappel": float(np.sqrt(np.mean(d * d))),
            "max_step": float(np.max(s)),
            "median_step": float(np.median(s)),
            "jumps": int(np.sum(s > jump_deg)),
            "n": int(len(s))}


def image_direction_change(frames, a_idx, b_idx, aspect=1.0):
    ang = []
    for fd in frames:
        p = fd.get("pose")
        a, b = (P(p, a_idx), P(p, b_idx)) if p else (None, None)
        if a is None or b is None:
            ang.append(np.nan)
            continue
        ang.append(math.atan2(b[1] - a[1], (b[0] - a[0]) * aspect))
    ang = np.array(ang)
    d = np.abs(np.diff(ang))
    d = np.minimum(d, 2 * math.pi - d)
    return np.degrees(_finite(d))


def world_depth_jumps(frames, idx):
    z = []
    for fd in frames:
        pw = fd.get("pose_world")
        lm = P(pw, idx) if pw else None
        z.append(np.nan if lm is None else lm[2])
    return np.abs(np.diff(_finite(np.array(z)))) * 100.0


def confidence(frames, idx):
    c = []
    for fd in frames:
        p = fd.get("pose")
        lm = P(p, idx) if p else None
        if lm is not None and len(lm) > 3:
            c.append(lm[3])
    return np.array(c)


def signal_quality(seq, conf_thr=0.5):
    frames = seq.frames
    out = {}
    for side in ("left", "right"):
        wrist = POSE[side + "_wrist"]
        conf = confidence(frames, wrist)
        d2 = image_direction_change(frames, POSE[side + "_elbow"], wrist, seq.aspect)
        dz = world_depth_jumps(frames, wrist)
        out[side] = {
            "conf_median": float(np.median(conf)) if len(conf) else float("nan"),
            "conf_below": int(np.sum(conf < conf_thr)),
            "n_frames": len(frames),
            "dir2d_median_deg": float(np.median(d2)) if len(d2) else float("nan"),
            "dir2d_p95_deg": float(np.percentile(d2, 95)) if len(d2) else float("nan"),
            "depth_median_cm": float(np.median(dz)) if len(dz) else float("nan"),
            "depth_max_cm": float(np.max(dz)) if len(dz) else float("nan"),
        }
    return out


def hand_coverage(seq):
    frames = seq.frames
    return {
        "n_frames": len(frames),
        "pose": sum(1 for f in frames if f.get("pose")),
        "left_hand": sum(1 for f in frames if f.get("left_hand")),
        "right_hand": sum(1 for f in frames if f.get("right_hand")),
        "both_hands": sum(1 for f in frames
                          if f.get("left_hand") and f.get("right_hand")),
    }
