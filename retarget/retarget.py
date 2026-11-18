"""Von den bereinigten Landmarks zu Knochendrehungen.

Portiert aus VideoPose/signpose/retarget.py. Heraus kommt je Knochen eine
Reihe Quaternionen (w, x, y, z) über alle ausgewerteten Bilder.
"""

import json
import math

import numpy as np

from . import rotations as rot
from .config import (ARM_MODE, ARM_SEGMENTS, FINGER_BEND, FINGER_CURL_MAX,
                     FINGER_MAP, FINGER_PRESETS, FINGER_SPREAD_MAX,
                     HEAD_MAX_DEG, HIP_MIN_FRAC, LEG_SEGMENTS, NECK_SHARE, POSE,
                     RELAXED_ARM_LOW, RELAXED_ARM_UP, ROLL_MIN_SIN,
                     ROLL_SMOOTH_FACTOR, ROLL_VMAX_DEG_S, SMOOTH_SEC,
                     WRIST_MAX_DEG, WRIST_VMAX_DEG_S)
from .depth import depth_series, estimate_px_lengths, segment_directions, visible
from .filters import (angles_to_axis_quats, filter_quat_series, hampel_angle, odd,
                      qslerp, quat_fill_gaps, quat_fix_sign, quat_median)
from .hands import finger_series
from .sequence import P, mid

Y_AXIS = np.array([0.0, 1.0, 0.0])


def mid_world_dir(frames, pairs, damp, min_conf, gate):
    (a0, a1), (b0, b1) = pairs
    n = len(frames)
    vec = np.zeros((n, 3))
    valid = np.zeros(n, bool)
    for i, fd in enumerate(frames):
        p2, pw = fd.get("pose"), fd.get("pose_world")
        if not p2 or not pw:
            continue
        if not all(visible(p2, g, min_conf) for g in gate):
            continue
        a = mid(P(pw, a0), P(pw, a1))
        b = mid(P(pw, b0), P(pw, b1))
        if a is None or b is None:
            continue
        v = np.array((b[0] - a[0], (b[2] - a[2]) * damp, -(b[1] - a[1])))
        if np.linalg.norm(v) > 1e-9:
            vec[i] = v
            valid[i] = True
    return vec, valid


def hand_frame_series(frames, key):
    n = len(frames)
    mats = np.zeros((n, 3, 3))
    valid = np.zeros(n, bool)
    for i, fd in enumerate(frames):
        h = fd.get(key + "_world")
        if not h or len(h) < 18:
            continue

        def bl(k):
            return np.array((h[k][0], h[k][2], -h[k][1]))

        forward = bl(9) - bl(0)
        nor = np.cross(forward, bl(5) - bl(17))
        if np.linalg.norm(forward) < 1e-9 or np.linalg.norm(nor) < 1e-9:
            continue
        F = rot.ortho_frame(forward, nor)
        if F is None:
            continue
        mats[i] = F
        valid[i] = True
    return mats, valid


def palm_normal_series(frames, key):
    n = len(frames)
    vec = np.zeros((n, 3))
    valid = np.zeros(n, bool)
    for i, fd in enumerate(frames):
        h = fd.get(key + "_world")
        if not h or len(h) < 18:
            continue
        w = np.asarray(h[0][:3], dtype=float)
        forward = np.asarray(h[9][:3], dtype=float) - w
        across = np.asarray(h[5][:3], dtype=float) - np.asarray(h[17][:3], dtype=float)
        nor = np.cross(forward, across)
        if np.linalg.norm(nor) < 1e-9:
            continue
        vec[i] = (nor[0], nor[2], -nor[1])
        valid[i] = True
    return vec, valid


def head_rotation_series(frames, W, H, depth, fps, spine_damp, min_conf, window, clean):
    n = len(frames)
    L = POSE
    up, up_ok = mid_world_dir(
        frames, ((L["left_shoulder"], L["right_shoulder"]), (L["left_ear"], L["right_ear"])),
        spine_damp, min_conf,
        (L["left_shoulder"], L["right_shoulder"], L["left_ear"], L["right_ear"]))
    across, across_ok = segment_directions(frames, W, H, depth, "ears", min_conf, False)

    qs = np.zeros((n, 4))
    qs[:, 0] = 1.0
    valid = np.zeros(n, bool)
    for i in range(n):
        if not (up_ok[i] and across_ok[i]):
            continue
        F = rot.ortho_frame(up[i], across[i])
        if F is None:
            continue
        qs[i] = rot.mat_to_quat(F)
        valid[i] = True
    if not valid.any():
        return None, valid

    inv = rot.quat_conj(quat_median(qs[valid]))
    lim = math.radians(HEAD_MAX_DEG)
    for i in np.where(valid)[0]:
        qs[i] = rot.cap(rot.quat_mul(qs[i], inv), lim)
    return filter_quat_series(qs, valid, fps, window, clean), valid


def arm_mode_for(seq, arm_mode=None):
    """Loest "auto" auf: seit den neu erzeugten ``.pev``-Dateien folgen beide
    Arme den Landmarks. ``sign`` bleibt als explizite Option erhalten."""
    mode = (arm_mode or ARM_MODE or "auto").strip().lower()
    if mode not in ("both", "sign", "auto"):
        mode = "auto"
    if mode == "auto":
        return "both"
    return mode


def build_targets(rig, seq, opts):
    frames = seq.frames
    n = len(frames)
    W, H = seq.size
    fps = seq.fps
    min_conf = opts["min_conf"]
    window = odd(max(5.0, opts["smooth_sec"] * fps)) if opts["smooth_sec"] > 0 else 0

    lengths = estimate_px_lengths(frames, W, H)
    depth = depth_series(frames, W, H, lengths, fps, opts["arm_depth"])

    T = {}
    L = POSE
    mode = opts.get("arm_mode", "both")
    sign_side = opts.get("sign_side") or "r"
    other_side = "l" if sign_side == "r" else "r"
    segs = ARM_SEGMENTS if opts["upper_only"] else ARM_SEGMENTS + LEG_SEGMENTS
    for canon in segs:
        name = rig.find(canon)
        if name is None:
            continue
        # Im "sign"-Modus haengt der zweite Arm entspannt herab; seine
        # Landmarks sind bei .pev zu unzuverlaessig (verdeckt/verrauscht).
        # Die Richtung wird absolut vorgegeben, damit sie nicht mit einem
        # schief geschaetzten Rumpf mitkippt.
        if mode == "sign" and canon in ARM_SEGMENTS and not canon.endswith("_" + sign_side):
            relaxed = RELAXED_ARM_UP if canon.startswith("upperarm") else RELAXED_ARM_LOW
            vec = np.tile(np.asarray(relaxed[other_side], float), (n, 1))
            T[name] = {"mode": "aim", "along": vec, "valid": np.ones(n, bool)}
            continue
        vec, valid = segment_directions(frames, W, H, depth, canon, min_conf, weak=True)
        T[name] = {"mode": "aim", "along": vec, "valid": valid}

    name = rig.find("spine")
    rest = rig.rest_frames.get("spine")
    # Bei .pev sind Hueft- und Schulterpunkte zu unzuverlaessig fuer eine
    # Rumpfdrehung (die Hueftpunkte liegen oft weit ausserhalb). Im
    # "sign"-Modus bleibt der Rumpf deshalb ruhig; der Gebaerdearm wird
    # ohnehin absolut ausgerichtet.
    if mode != "sign" and name is not None and rest is not None:
        hips_ok = sum(1 for fd in frames
                      if fd.get("pose") and visible(fd["pose"], L["left_hip"], min_conf)
                      and visible(fd["pose"], L["right_hip"], min_conf))
        if hips_ok >= HIP_MIN_FRAC * n:
            up, up_ok = mid_world_dir(
                frames, ((L["left_hip"], L["right_hip"]),
                         (L["left_shoulder"], L["right_shoulder"])),
                opts["spine_depth"], min_conf,
                (L["left_hip"], L["right_hip"], L["left_shoulder"], L["right_shoulder"]))
        else:
            up = np.tile((0.0, 0.0, 1.0), (n, 1))
            up_ok = np.ones(n, bool)
        across, across_ok = segment_directions(frames, W, H, depth, "shoulders",
                                               min_conf, False)
        valid = up_ok & across_ok
        if valid.any():
            T[name] = {"mode": "frame", "along": up, "ref": across,
                       "rest": rest, "valid": valid}

    hq = None
    if mode != "sign":
        hq, _ok = head_rotation_series(frames, W, H, depth, fps, opts["spine_depth"],
                                       min_conf, window, opts["clean"])
    if hq is not None:
        head, neck = rig.find("head"), rig.find("neck")
        full = np.ones(n, bool)
        if head is not None:
            T[head] = {"mode": "quat", "quat": hq, "valid": full, "filtered": True}
        if neck is not None:
            nq = np.zeros((n, 4))
            for i in range(n):
                nq[i] = qslerp(rot.IDENTITY, rot.positive(hq[i]), NECK_SHARE)
            T[neck] = {"mode": "quat", "quat": nq, "valid": full, "filtered": True}

    normals = {}
    hands = {}
    for side, key in (("l", "left_hand"), ("r", "right_hand")):
        # Der entspannte Arm bekommt keine Hand-/Handgelenksdaten, sonst
        # wuerde das verrauschte Handgelenk die feste Haltung wieder kippen.
        if mode == "sign" and side != sign_side:
            continue
        normals[side] = palm_normal_series(frames, key)
        rest = rig.rest_frames.get("hand_" + side)
        mats, ok = hand_frame_series(frames, key)
        if rest is None or not ok.any():
            continue
        qs = np.zeros((n, 4))
        qs[:, 0] = 1.0
        for i in np.where(ok)[0]:
            qs[i] = rot.positive(rot.mat_to_quat(mats[i] @ rest.T))
        hands[side] = (qs, ok)
    return T, normals, hands


def roll_series(rig, M, normals, n):
    out = {}
    for side in ("l", "r"):
        bname = rig.find("lowerarm_" + side)
        vec_ok = normals.get(side)
        if (bname is None or side not in rig.rest_palms or bname not in M[0]
                or vec_ok is None or not vec_ok[1].any()):
            continue
        vec, ok = vec_ok
        u = rig.rest_palms[side]
        up = rot.unit(u - float(np.dot(u, Y_AXIS)) * Y_AXIS)
        if up is None:
            continue
        th = np.full(n, np.nan)
        for i in range(n):
            if not ok[i]:
                continue
            target = rot.unit(vec[i])
            if target is None:
                continue
            v = M[i][bname].T @ target
            vp = v - float(np.dot(v, Y_AXIS)) * Y_AXIS
            if float(np.linalg.norm(vp)) < ROLL_MIN_SIN:
                continue
            vp = vp / float(np.linalg.norm(vp))
            th[i] = math.atan2(float(np.dot(np.cross(up, vp), Y_AXIS)),
                               float(np.dot(up, vp)))
        out[side] = th
    return out


def apply_wrist(rig, rec, A_of, hands, n, fps, window, clean):
    lim = math.radians(WRIST_MAX_DEG)
    for side, (R, ok) in hands.items():
        low = rig.find("lowerarm_" + side)
        bones = rig.wrist_bones(side)
        if low is None or not bones or low not in A_of or not ok.any():
            continue
        lowrest = rig.rest[low]
        ks = np.zeros((n, 4))
        ks[:, 0] = 1.0
        for i in np.where(ok)[0]:
            qb = rec[low][0][i] if low in rec else rot.IDENTITY
            Mlow = A_of[low][i] @ rot.quat_to_mat(qb)
            ks[i] = rot.cap(rot.mat_to_quat(lowrest @ Mlow.T @ rot.quat_to_mat(R[i])), lim)
        ks = filter_quat_series(ks, ok.copy(), fps, window, clean, WRIST_VMAX_DEG_S)
        for b in bones:
            r = rig.rest[b]
            qs = np.zeros((n, 4))
            for i in range(n):
                qs[i] = rot.mat_to_quat(r.T @ rot.quat_to_mat(ks[i]) @ r)
            quat_fix_sign(qs, np.ones(n, bool))
            rec[b] = [qs, ok.copy()]


def solve_body(rig, targets, normals, hands, n, fps, smooth_sec=SMOOTH_SEC, clean=True):
    window = odd(max(5.0, smooth_sec * fps)) if smooth_sec > 0 else 0
    ident = np.tile(rot.IDENTITY, (n, 1))
    M = [dict() for _ in range(n)]
    A_of = {}
    rec = {}
    for name in rig.chain(targets):
        p = rig.parent[name]
        A = [(M[i][p] @ rig.rel[name]) if p else rig.rel[name] for i in range(n)]
        A_of[name] = A
        t = targets.get(name)
        qs, valid = ident.copy(), np.zeros(n, bool)
        if t is not None:
            qs = np.zeros((n, 4))
            for i in range(n):
                if not t["valid"][i]:
                    continue
                b = rig.basis(name, A[i], t, i)
                if b is None:
                    continue
                qs[i] = b
                valid[i] = True
            if valid.any():
                if t.get("filtered"):
                    quat_fix_sign(qs, valid)
                    quat_fill_gaps(qs, valid, fps)
                else:
                    qs = filter_quat_series(qs, valid, fps, window, clean)
                rec[name] = [qs, valid]
            else:
                qs = ident.copy()
        for i in range(n):
            M[i][name] = A[i] @ rot.quat_to_mat(qs[i])

    roll_win = odd(max(9.0, smooth_sec * ROLL_SMOOTH_FACTOR * fps)) if smooth_sec > 0 else 0
    for side, th in roll_series(rig, M, normals, n).items():
        bname = rig.find("lowerarm_" + side)
        valid = ~np.isnan(th)
        if bname is None or bname not in rec or not valid.any():
            continue
        if clean:
            th = hampel_angle(th, valid)
        roll = filter_quat_series(angles_to_axis_quats(th, valid), valid, fps,
                                  roll_win, clean, ROLL_VMAX_DEG_S)
        qs = rec[bname][0]
        for i in range(n):
            qs[i] = rot.quat_mul(qs[i], roll[i])
        quat_fix_sign(qs, np.ones(n, bool))

    apply_wrist(rig, rec, A_of, hands, n, fps, roll_win, clean)
    return rec


def finger_quaternions(fingers, side, n, preset=None, intensity=0.7):
    out = {}
    if preset is not None:
        for finger, curls in FINGER_PRESETS[preset].items():
            for j, (c, (lo, hi)) in enumerate(zip(curls, FINGER_BEND[finger]), start=1):
                ax = math.radians((lo + (hi - lo) * c) * intensity)
                q = rot.axis_angle((1.0, 0.0, 0.0), ax)
                out["%s_%02d_%s" % (finger, j, side)] = np.tile(q, (n, 1))
        return out

    f = fingers[side]
    fist = f.get("fist")
    for jname, _, _ in FINGER_MAP:
        cmax = FINGER_CURL_MAX.get(jname, (95, 110, 80))
        smax = math.radians(FINGER_SPREAD_MAX.get(jname, 20))
        curls, spread = f["curl"][jname], f["spread"][jname]
        straight = f.get("straight", {}).get(jname)
        for k in range(1, 4):
            qs = np.tile(rot.IDENTITY, (n, 1))
            for i in range(n):
                c = 0.0 if math.isnan(curls[i, k - 1]) else float(curls[i, k - 1])
                c = max(0.0, min(c, math.radians(cmax[k - 1])))
                cl = 0.0
                if fist is not None:
                    # Faustanteil aus der 2D-Handgeometrie. Die rohen
                    # Fingergliedwinkel sind bei 2D oft 0 (Finger zeigen in
                    # die Kamera) - der Reach-Wert erkennt die Faust trotzdem.
                    cl = max(0.0, min(1.0, float(fist[i])))
                    # Ein nachweislich gestreckter Finger bleibt gestreckt
                    # (z. B. der Zeigefinger bei "Ablauf"), auch wenn die
                    # uebrigen Finger zur Faust gehoeren. "straight" ist
                    # bereits 0..1 normiert (siehe hands.finger_series).
                    if straight is not None and math.isfinite(float(straight[i])):
                        cl *= 1.0 - max(0.0, min(1.0, float(straight[i])))
                    c = (1.0 - cl) * c + cl * math.radians(cmax[k - 1])
                q = rot.axis_angle((1.0, 0.0, 0.0), c)
                if k == 1 and not math.isnan(spread[i]):
                    sv = float(spread[i]) if side == "l" else -float(spread[i])
                    sv *= 1.0 - cl
                    q = rot.quat_mul(rot.axis_angle((0.0, 0.0, 1.0),
                                                    max(-smax, min(smax, sv))), q)
                qs[i] = q
            out["%s_%02d_%s" % (jname, k, side)] = qs
    return out


def solve(rig, seq, upper_only=True, arm_depth=1.0, spine_depth=0.2, min_conf=0.5,
          finger_fallback="relaxed", hand_min=0.7, smooth_sec=SMOOTH_SEC, clean=True,
          arm_mode=None):
    n = len(seq.frames)
    fps = seq.fps
    mode = arm_mode_for(seq, arm_mode)
    sign_side = (seq.get("sign_arm") or "r").strip().lower()
    if sign_side not in ("l", "r"):
        sign_side = "r"
    opts = {"upper_only": upper_only, "arm_depth": arm_depth, "spine_depth": spine_depth,
            "min_conf": min_conf, "smooth_sec": smooth_sec, "clean": clean,
            "arm_mode": mode, "sign_side": sign_side}

    targets, normals, hands = build_targets(rig, seq, opts)
    body = solve_body(rig, targets, normals, hands, n, fps, smooth_sec, clean)
    fingers = finger_series(seq, hand_min, smooth_sec, clean)

    curves = {}
    for name, (qs, valid) in body.items():
        if valid.any():
            curves[name] = qs
    for side in ("l", "r"):
        # Im "sign"-Modus keine rohen Fingerpunkte: die sind bei .pev
        # verrauscht und erzeugen Klauen. Stattdessen die entspannte Hand.
        if fingers[side]["has_data"] and mode != "sign":
            fq = finger_quaternions(fingers, side, n)
        elif finger_fallback != "none":
            fq = finger_quaternions(fingers, side, n, preset=finger_fallback)
        else:
            continue
        for _jname, mcname, _ in FINGER_MAP:
            mc = rig.find("%s_%s" % (mcname, side))
            if mc and mc not in curves:
                curves[mc] = np.tile(rot.IDENTITY, (n, 1))
        for bname, qs in fq.items():
            name = rig.find(bname)
            if name:
                curves[name] = qs
    return curves


def frame_numbers(seq, start=1):
    step = max(1, int(seq.get("frame_step", 1)))
    return [start + int(fd.get("frame_video", i * step))
            for i, fd in enumerate(seq.frames)]


def save_curves(curves, frames, path, fps=30.0):
    data = {"fps": float(fps), "frames": [int(f) for f in frames],
            "bones": {k: np.asarray(v, float).tolist() for k, v in sorted(curves.items())}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path