"""Handfläche und Finger: ein Beugewinkel je Glied, dazu die Spreizung."""

import math

import numpy as np

from .config import (FINGER_MAP, FINGER_STRAIGHT_FRAC, FINGER_STRAIGHT_FRAC_SPAN,
                     FINGER_STRAIGHT_HI, FINGER_STRAIGHT_LO, FINGER_STRAIGHT_SURE,
                     FIST_REACH_CLOSED, FIST_REACH_OPEN)
from .filters import (filter_angle_series, median_filter, odd, rolling_max,
                      rolling_mean, savgol, series_fill, smoothstep)
from .sequence import P


def _v(p):
    return np.asarray(p[:3], dtype=float)


def _unit(v):
    n = float(np.linalg.norm(v))
    return None if n < 1e-9 else v / n


def _sat(x, lo, hi):
    return max(0.0, min(1.0, (float(x) - lo) / (hi - lo)))


def palm_frame(world_hand, side="l"):
    w, im, mm, pm = P(world_hand, 0), P(world_hand, 5), P(world_hand, 9), P(world_hand, 17)
    if not (w and im and mm and pm):
        return None
    forward = _unit(_v(mm) - _v(w))
    across = _v(im) - _v(pm)
    if forward is None or np.linalg.norm(across) < 1e-6:
        return None
    normal = _unit(np.cross(forward, across))
    if normal is None:
        return None
    across = _unit(np.cross(normal, forward))
    if across is None:
        return None
    if side == "r":
        normal = -normal
    return forward, across, normal


def palm_normal(world_hand):
    if not world_hand or len(world_hand) < 18:
        return None
    w = _v(world_hand[0])
    forward = _v(world_hand[9]) - w
    across = _v(world_hand[5]) - _v(world_hand[17])
    nor = np.cross(forward, across)
    if np.linalg.norm(nor) < 1e-9:
        return None
    return np.array((nor[0], nor[2], -nor[1]))


def finger_joint_angles(world_hand, side="l"):
    frame = palm_frame(world_hand, side) if world_hand else None
    out = {}
    for jname, _, idxs in FINGER_MAP:
        curl = [None, None, None]
        spread = None
        wpts = [P(world_hand, i) for i in idxs] if world_hand else [None] * 5
        segs = []
        for k in range(4):
            pa, pb = wpts[k], wpts[k + 1]
            if pa is None or pb is None:
                segs = None
                break
            v = _unit(_v(pb) - _v(pa))
            if v is None:
                segs = None
                break
            segs.append(v)
        if segs is None:
            out[jname] = {"curl": curl, "spread": None}
            continue

        is_long = (jname != "thumb") and (frame is not None)
        hinge = None
        if is_long:
            f, a, nrm = frame
            hinge = _unit(np.cross(f, nrm))

        for k in range(1, 4):
            if k == 1 and is_long:
                f, a, nrm = frame
                d = segs[1] - float(np.dot(segs[1], a)) * a
                curl[0] = (math.atan2(float(np.dot(d, nrm)), float(np.dot(d, f)))
                           if np.linalg.norm(d) > 1e-6 else 0.0)
                spread = (math.atan2(float(np.dot(segs[1], a)), float(np.dot(segs[1], f)))
                          - math.atan2(float(np.dot(segs[0], a)), float(np.dot(segs[0], f))))
                continue
            u, v = segs[k - 1], segs[k]
            if hinge is not None:
                curl[k - 1] = math.atan2(float(np.dot(np.cross(u, v), hinge)),
                                         float(np.dot(u, v)))
            else:
                curl[k - 1] = math.acos(max(-1.0, min(1.0, float(np.dot(u, v)))))
        out[jname] = {"curl": curl, "spread": spread}
    return out


def finger_straight(world_hand):
    """Gestrecktheit je Finger: Sehne geteilt durch die Laenge des Fingerpolygons.

    Ein gestreckter Finger hat fast den Wert 1, ein eingerollter deutlich
    weniger. Bei einer 2D-Hand (z = 0) ist das robuster als die einzelnen
    Fingergliedwinkel, weil sich die Glieder beim Drehen zur Kamera kaum
    verkuerzen. Wichtig fuer "Ablauf": dort zeigt der Zeigefinger, waehrend
    Mittel-, Ring- und kleiner Finger eingerollt sind - eine globale Faust
    wuerde den Zeigefinger faelschlich mit einrollen.
    """
    if not world_hand or len(world_hand) < 21:
        return {}
    out = {}
    for jname, _, idxs in FINGER_MAP:
        pts = [_v(world_hand[i])[:2] for i in idxs[1:]]
        path = sum(float(np.linalg.norm(pts[k + 1] - pts[k]))
                   for k in range(len(pts) - 1))
        chord = float(np.linalg.norm(pts[-1] - pts[0]))
        out[jname] = chord / path if path > 1e-9 else None
    return out


def fist_reach(world_hand):
    """Mittlerer Abstand Handgelenk -> Fingerspitze, geteilt durch die
    Handflaechenlaenge (Handgelenk -> Mittelhandknochen des Mittelfingers).

    Kleine Werte heissen geschlossene Finger. Der Wert ist robuster als die
    einzelnen Fingergliedwinkel: bei einer Faust *zum Betrachter* verkuerzen
    sich die Glieder im Bild kaum, der Abstand zur Spitze dagegen deutlich.
    """
    if not world_hand or len(world_hand) < 21:
        return None
    wrist = _v(world_hand[0])
    palm = float(np.linalg.norm(_v(world_hand[9]) - wrist))
    if palm < 1e-9:
        return None
    tips = [_v(world_hand[i]) for i in (4, 8, 12, 16, 20)]
    reach = float(np.mean([np.linalg.norm(t - wrist) for t in tips]))
    return reach / palm


def fist_closure(reach):
    """Reach-Verhaeltnis -> 0 (offen) bis 1 (Faust)."""
    if reach is None or math.isnan(reach):
        return None
    t = (FIST_REACH_OPEN - float(reach)) / (FIST_REACH_OPEN - FIST_REACH_CLOSED)
    return smoothstep(max(0.0, min(1.0, t)))


def fist_envelope(raw, fps, attack_s=0.06, release_s=0.6):
    """Faust schliesst schnell, oeffnet langsam.

    Die 2D-Handpunkte springen bei RTMLib oft zwischen Faust und offener Hand
    (die Hand dreht sich zur Kamera, die Spitze wird verdeckt). Ein reines
    Glaetten erzeugt dann ein Zucken der Finger. Die Huellkurve haelt die
    Faust deshalb kurz fest und gibt sie erst nach etwa einer halben
    Sekunde frei - so bleibt z. B. bei "Auto" die Faust stehen.
    """
    if len(raw) == 0:
        return raw
    dt = 1.0 / max(1.0, float(fps))
    ka = 1.0 - math.exp(-dt / max(1e-3, attack_s))
    kr = 1.0 - math.exp(-dt / max(1e-3, release_s))
    out = np.zeros_like(raw)
    v = float(raw[0])
    for i, target in enumerate(raw):
        k = ka if target > v else kr
        v += (float(target) - v) * k
        out[i] = v
    return out


def hand_reliable(hand, min_inframe=0.7, min_conf=0.2):
    if hand is None or not hand:
        return False
    conf = hand[0][3] if len(hand[0]) > 3 else 1.0
    if conf < min_conf:
        return False
    inside = sum(1 for lm in hand if 0.0 <= lm[0] <= 1.0 and 0.0 <= lm[1] <= 1.0)
    return (inside / len(hand)) >= min_inframe


def finger_series(seq, hand_min=0.7, smooth_sec=0.20, clean=True):
    frames = seq.frames
    fps = seq.fps
    n = len(frames)
    window = odd(max(9.0, smooth_sec * 1.5 * fps)) if smooth_sec > 0 else 0
    out = {}
    for side, key in (("l", "left_hand"), ("r", "right_hand")):
        curls = {j: np.full((n, 3), np.nan) for j, _, _ in FINGER_MAP}
        spreads = {j: np.full(n, np.nan) for j, _, _ in FINGER_MAP}
        straights = {j: np.full(n, np.nan) for j, _, _ in FINGER_MAP}
        reaches = np.full(n, np.nan)
        has_data = False
        for i, fd in enumerate(frames):
            if not hand_reliable(fd.get(key), hand_min):
                continue
            world = fd.get(key + "_world")
            if not world:
                continue
            r = fist_reach(world)
            if r is not None and math.isfinite(r):
                reaches[i] = r
            straight = finger_straight(world)
            for jname, _, _ in FINGER_MAP:
                s = straight.get(jname)
                if s is not None and math.isfinite(s):
                    straights[jname][i] = s
            ang = finger_joint_angles(world, side)
            for jname, _, _ in FINGER_MAP:
                t = ang.get(jname)
                if not t:
                    continue
                for k in range(3):
                    if t["curl"][k] is not None:
                        curls[jname][i, k] = t["curl"][k]
                        has_data = True
                if t.get("spread") is not None:
                    spreads[jname][i] = t["spread"]
        if has_data and window:
            for jname, _, _ in FINGER_MAP:
                for k in range(3):
                    col = curls[jname][:, k]
                    curls[jname][:, k] = filter_angle_series(
                        col, ~np.isnan(col), fps, window, clean)[0]
                sp = spreads[jname]
                spreads[jname] = filter_angle_series(
                    sp, ~np.isnan(sp), fps, window, clean)[0]
        if has_data:
            med_win = odd(max(5.0, 0.10 * fps))
            hold_win = odd(max(5.0, 0.35 * fps))
            frac_win = odd(max(5.0, 0.50 * fps))
            for jname, _, _ in FINGER_MAP:
                st = straights[jname]
                sv = ~np.isnan(st)
                if not sv.any():
                    continue
                # Gestrecktheit ist kein Winkel. Fehlende Frames zaehlen
                # nicht als "sicher gestreckt".
                sure = np.where(sv & (st > FINGER_STRAIGHT_SURE), 1.0, 0.0)
                filled = np.where(sv, st, 0.0)
                # Median gegen frame-weises Springen (die 2D-Punkte kippen oft
                # zwischen gestreckt und eingerollt), danach kurz halten,
                # damit ein einzelner Ausreisser den Finger nicht einrollt.
                shape = rolling_max(median_filter(filled, med_win), hold_win)
                shape = np.array([smoothstep(_sat(v, FINGER_STRAIGHT_LO,
                                                 FINGER_STRAIGHT_HI))
                                  for v in shape])
                # Nur wenn der Finger auch ueber die Zeit gestreckt bleibt,
                # wird er als gestreckt behandelt (siehe config).
                frac = rolling_mean(sure, frac_win)
                gain = np.array([smoothstep(_sat(
                    v, FINGER_STRAIGHT_FRAC,
                    FINGER_STRAIGHT_FRAC + FINGER_STRAIGHT_FRAC_SPAN))
                    for v in frac])
                sf = shape * gain
                straights[jname] = savgol(sf, window) if window else sf
        # Faustanteil aus dem Reach-Verhaeltnis. Das Verhaeltnis ist kein
        # Winkel, deshalb fuellen/glaetten wir es direkt statt ueber
        # filter_angle_series.
        fist = np.zeros(n)
        rv = ~np.isnan(reaches)
        if rv.any():
            series_fill(reaches, rv, fps)
            if window:
                reaches = savgol(reaches, window)
            raw = np.zeros(n)
            for i in range(n):
                c = fist_closure(reaches[i])
                raw[i] = 0.0 if c is None else c
            fist = fist_envelope(raw, fps)
        out[side] = {"curl": curls, "spread": spreads, "fist": fist,
                     "straight": straights, "reach": reaches,
                     "has_data": has_data}
    return out