"""RTMLib/COCO-WholeBody -> Landmark-Sequenz der Retarget-Kette.

RTMLib liefert je Bild ``keypoints`` der Form ``(N, 133, 2)`` und ``scores``
der Form ``(N, 133)`` - Pixelkoordinaten im COCO-WholeBody-Layout. Die
Retarget-Kette erwartet dagegen das MediaPipe-Holistic-Layout mit
normalisierten Bildkoordinaten, getrennten Haenden und einem ``pose_world``.

Dieses Modul ist die Bruecke:

* Koordinaten werden auf ``[0, 1]`` normiert und die Handpunkte an den
  COCO-Offsets 91 bzw. 112 herausgeschnitten.
* RTMLib ist 2D. ``pose_world`` und die Hand-Weltpunkte werden deshalb aus
  den normierten Bildkoordinaten synthetisiert (x mit Seitenverhaeltnis,
  y nach unten, z = 0) - dieselbe Achsenlage wie bei MediaPipe.
* Liefert eine Quelle echte 3D-Punkte (``keypoints_3d``), werden diese
  bevorzugt und unveraendert als Weltpunkte uebernommen.

Unterstuetzt werden unter anderem:

* ``PoseResult``/``VideoResult`` aus ``pose_estimator_2d.py`` direkt,
* das JSON von ``export_to_json`` (``{"frames": [...]}``),
* die Liste aus ``process_side_by_side_video`` (``left``/``right``),
* eine nackte Liste von Frame-Dicts,
* ein Pfad zu einer dieser JSON-Dateien.

Mehrere Personen: standardmaessig wird die Person mit der hoechsten mittleren
Koerperkonfidenz genommen (``person="best"``); alternativ ein fester Index.

Zusaetzlich gibt es :func:`sequence_from_pev` fuer die ``.pev``-Dateien aus
``plugins/gsg/extracted_videos``. Dieses Format enthaelt bereits 3D-Punkte,
aber in einer anderen Skala: x/y sind grosse Pixelwerte, y zeigt nach oben und
die Videos sind gespiegelt. Der Adapter normiert deshalb robust auf ein
quadratisches Bild, dreht y um und tauscht - wie das alte ``bone_mapping.json``
- linke und rechte Koerperseite.
"""

import json
import os

import numpy as np

from .config import HAND_N, LEFT_HAND_BASE, RIGHT_HAND_BASE
from .sequence import Sequence

BODY_N = 17
EMPTY_FRAME = {"pose": None, "pose_world": None,
               "left_hand": None, "left_hand_world": None,
               "right_hand": None, "right_hand_world": None}

WORLD_3D_KEYS = ("keypoints_3d", "keypoints_world", "landmarks_3d", "world")


class AdapterError(ValueError):
    pass


def _as_array(value):
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    return None if arr.size == 0 else arr


def _meta_value(data, *names):
    if isinstance(data, dict):
        for name in names:
            if data.get(name) is not None:
                try:
                    return float(data[name])
                except (TypeError, ValueError):
                    pass
    return None


def _records_and_meta(data):
    """Gibt (Frame-Liste, Metadaten-Dict) fuer alle unterstuetzten Quellen."""
    meta = {"width": None, "height": None, "fps": None, "frame_step": None}
    if isinstance(data, (str, os.PathLike)):
        with open(data, "r", encoding="utf-8") as f:
            data = json.load(f)

    # PoseEstimator2D.VideoResult / PoseResult
    if hasattr(data, "frame_results"):
        meta["fps"] = getattr(data, "fps", None)
        return list(data.frame_results), meta
    if hasattr(data, "keypoints"):
        return [data], meta

    if isinstance(data, dict):
        meta["width"] = _meta_value(data, "width", "image_width", "W")
        meta["height"] = _meta_value(data, "height", "image_height", "H")
        meta["fps"] = _meta_value(data, "fps", "video_fps")
        step = data.get("frame_step")
        meta["frame_step"] = int(step) if step is not None else None

        for key in ("frame_results", "frames", "results", "predictions"):
            if isinstance(data.get(key), list):
                return data[key], meta
        if any(k in data for k in ("keypoints", "left", "right", "people")):
            return [data], meta
        raise AdapterError("kein Frame-Array in der JSON-Struktur gefunden")

    if isinstance(data, list):
        return data, meta
    raise AdapterError("unbekanntes Eingabeformat: %r" % type(data).__name__)


def _record_keypoints(rec, side):
    """Holt ``(keypoints, scores)`` aus einem Frame - auch side-by-side."""
    if rec is None:
        return None, None
    if hasattr(rec, "keypoints"):
        return _as_array(rec.keypoints), _as_array(getattr(rec, "scores", None))
    if not isinstance(rec, dict):
        return None, None

    if rec.get("keypoints") is not None:
        return _as_array(rec["keypoints"]), _as_array(rec.get("scores"))

    # process_side_by_side_video: {"left": {...}, "right": {...}}
    if side in ("left", "right") and isinstance(rec.get(side), (dict,)):
        return _record_keypoints(rec[side], None)
    for key in ("left", "right"):
        if isinstance(rec.get(key), dict):
            return _record_keypoints(rec[key], None)
    if isinstance(rec.get("people"), list) and rec["people"]:
        return _record_keypoints(rec["people"][0], None)
    return None, None


def _record_3d(rec, side):
    """Optionale echte 3D-Punkte (z. B. aus einem spaeteren Sapiens-Modell)."""
    if rec is None or not isinstance(rec, dict):
        return None
    for key in WORLD_3D_KEYS:
        if rec.get(key) is not None:
            return _as_array(rec[key])
    if side in ("left", "right") and isinstance(rec.get(side), dict):
        return _record_3d(rec[side], None)
    for key in ("left", "right"):
        if isinstance(rec.get(key), dict):
            found = _record_3d(rec[key], None)
            if found is not None:
                return found
    return None


def _infer_scores(kp):
    present = ~np.all(np.isclose(kp[..., :2], 0.0), axis=-1)
    return present.astype(float)


def _split_persons(kp, sc):
    """Bringt ``(133,2)``/``(N,133,2)`` auf ``(N,133,2)``/``(N,133)``."""
    if kp.ndim == 2:
        kp = kp[np.newaxis, ...]
    if kp.ndim != 3 or kp.shape[-1] < 2:
        raise AdapterError("keypoints muessen (N, 133, 2) sein, nicht %s" % (kp.shape,))
    if sc is None or sc.ndim == 1 and kp.shape[0] == 1:
        sc = None if sc is None else sc[np.newaxis, ...]
    if sc is None or sc.shape != kp.shape[:2]:
        sc = _infer_scores(kp)
    return kp, np.asarray(sc, dtype=float)


def _split_persons_3d(kp3):
    if kp3 is None:
        return None
    if kp3.ndim == 2:
        return kp3[np.newaxis, ...]
    if kp3.ndim != 3 or kp3.shape[-1] < 3:
        return None
    return kp3


def _person_index(kp, sc, person):
    n = kp.shape[0]
    if n == 0:
        return None
    if isinstance(person, (int, np.integer)):
        idx = int(person)
        return idx if 0 <= idx < n else None
    if person == "largest":
        areas = []
        for p, s in zip(kp, sc):
            ok = (s > 0.05)[:, None] & ~np.all(np.isclose(p[:, :2], 0.0), axis=1)[:, None]
            pts = p[ok[:, 0], :2]
            if len(pts):
                span = np.max(pts, axis=0) - np.min(pts, axis=0)
                areas.append(float(span[0] * span[1]))
            else:
                areas.append(0.0)
        return int(np.argmax(areas))
    if person not in (None, "best"):
        raise AdapterError("person muss 'best', 'largest' oder ein Index sein")
    return int(np.argmax(np.mean(sc[:, :BODY_N], axis=1)))


def _infer_size(records, side):
    mx = my = 0.0
    for rec in records:
        kp, sc = _record_keypoints(rec, side)
        if kp is None:
            continue
        kp, sc = _split_persons(kp, sc)
        for p, s in zip(kp, sc):
            valid = s > 0.05
            if valid.any():
                mx = max(mx, float(np.max(p[valid, 0])))
                my = max(my, float(np.max(p[valid, 1])))
    return (mx if mx > 0 else 640.0), (my if my > 0 else 480.0)


def _landmarks(kp, sc, indices, W, H, aspect, world_scale, kp3=None):
    img = []
    world = []
    for idx in indices:
        x = float(kp[idx, 0]) / W
        y = float(kp[idx, 1]) / H
        conf = float(sc[idx])
        img.append([x, y, 0.0, conf])
        if kp3 is not None:
            world.append([float(kp3[idx, 0]), float(kp3[idx, 1]),
                          float(kp3[idx, 2]), conf])
        else:
            world.append([(x - 0.5) * aspect * world_scale,
                          (y - 0.5) * world_scale, 0.0, conf])
    return img, world


def _present(kp, sc, indices, threshold=0.05):
    return (float(np.max(sc[indices])) >= threshold
            and not np.all(np.isclose(kp[indices, :2], 0.0)))


def _frame_video(rec, index, frame_step):
    if isinstance(rec, dict):
        for key in ("frame_video", "frame_idx", "frame"):
            if rec.get(key) is not None:
                return int(rec[key])
    return int(index) * int(max(1, frame_step))


def _empty_frame(rec, index, frame_step):
    frame = {"frame_video": _frame_video(rec, index, frame_step)}
    frame.update(EMPTY_FRAME)
    return frame


def _build_frame(rec, index, width, height, side, person, frame_step, world_scale):
    kp, sc = _record_keypoints(rec, side)
    if kp is None:
        return _empty_frame(rec, index, frame_step)

    kp, sc = _split_persons(kp, sc)
    pidx = _person_index(kp, sc, person)
    if pidx is None:
        return _empty_frame(rec, index, frame_step)

    kp, sc = kp[pidx], sc[pidx]
    kp3 = _split_persons_3d(_record_3d(rec, side))
    if kp3 is not None:
        if kp3.shape[1] != kp.shape[0] or pidx >= kp3.shape[0]:
            kp3 = None
        else:
            kp3 = kp3[pidx]

    if not _present(kp, sc, np.arange(BODY_N)):
        return _empty_frame(rec, index, frame_step)

    aspect = width / height if height else 1.0
    frame = {"frame_video": _frame_video(rec, index, frame_step)}

    frame["pose"], frame["pose_world"] = _landmarks(
        kp, sc, range(BODY_N), width, height, aspect, world_scale, kp3)

    for side_name, base in (("left", LEFT_HAND_BASE), ("right", RIGHT_HAND_BASE)):
        idx = np.arange(base, base + HAND_N)
        if not _present(kp, sc, idx, 0.10):
            frame[side_name + "_hand"] = None
            frame[side_name + "_hand_world"] = None
            continue
        frame[side_name + "_hand"], frame[side_name + "_hand_world"] = _landmarks(
            kp, sc, idx, width, height, aspect, world_scale, kp3)
    return frame


def _looks_like_sequence(records):
    for rec in records:
        if isinstance(rec, dict) and rec.get("pose") is not None:
            return True
    return False


def sequence_from_rtmlib(data, width=None, height=None, fps=None, frame_step=None,
                         side="left", person="best", world_scale=1.0):
    """Baut aus einer RTMLib-Quelle eine :class:`Sequence`.

    Parameters
    ----------
    data:
        Pfad/JSON/Liste/PoseResult/VideoResult (siehe Modul-Docstring).
    width, height:
        Bildmasse in Pixeln. Fehlen sie, werden sie aus den groessten
        gefundenen Koordinaten geschaetzt.
    fps, frame_step:
        Bildrate des Originals und Abtastschritt der Frames. ``Sequence.fps``
        rechnet daraus die effektive Rate.
    side:
        Bei side-by-side-Aufnahmen ``"left"`` oder ``"right"``.
    person:
        ``"best"``, ``"largest"`` oder ein fester Personenindex.
    world_scale:
        Faktor fuer die synthetischen Weltkoordinaten. Nur das Verhaeltnis
        zueinander zaehlt; 1.0 genuegt.
    """
    records, meta = _records_and_meta(data)
    if not records:
        raise AdapterError("keine Frames in der Quelle")
    if _looks_like_sequence(records):
        raise AdapterError("die Quelle ist bereits eine Landmark-Sequenz "
                           "(Frames mit 'pose'); Adapter nicht noetig")

    if width is None:
        width = meta.get("width")
    if height is None:
        height = meta.get("height")
    if width is None or height is None:
        iw, ih = _infer_size(records, side)
        width = float(width if width is not None else iw)
        height = float(height if height is not None else ih)
    if width <= 0 or height <= 0:
        raise AdapterError("width/height muessen positiv sein")

    if fps is None:
        fps = meta.get("fps")
    fps = float(fps) if fps else 30.0

    if frame_step is None:
        frame_step = meta.get("frame_step")
    frame_step = int(frame_step) if frame_step else 1
    if frame_step < 1:
        raise AdapterError("frame_step muss >= 1 sein")

    frames = [_build_frame(rec, i, width, height, side, person, frame_step,
                           world_scale)
              for i, rec in enumerate(records)]

    return Sequence({
        "fps": fps,
        "width": int(round(width)),
        "height": int(round(height)),
        "frame_step": frame_step,
        "n": len(frames),
        "hands": "coco",
        "source": "rtmlib",
        "frames": frames,
    })


# --------------------------------------------------------------------- PEV

# Fuer die Skalierung nur den Oberkoerper (ohne Beine) auswerten. In den
# gsg-Aufnahmen sind Knie/Fussgelenke oft Ausreisser (z. B. y = -300000),
# die sonst die gesamte Normierung zusammendruecken.
PEV_BOUND_INDICES = tuple(range(13))
# Die alten gsg-Videos sind gespiegelt: COCO-links erscheint links im Bild.
PEV_SWAP_PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))


def looks_like_pev(data):
    """Erkennt das ``skeletonpoints``-Format aus ``extracted_videos/*.pev``."""
    if isinstance(data, (str, os.PathLike)):
        return str(data).lower().endswith(".pev")
    if isinstance(data, dict):
        data = data.get("frames")
    return (isinstance(data, list) and bool(data)
            and any(isinstance(rec, dict) and "skeletonpoints" in rec
                    for rec in data))


def _pev_records(data):
    if isinstance(data, (str, os.PathLike)):
        with open(data, "r", encoding="utf-8") as f:
            data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("frames"), list):
        return data["frames"]
    if isinstance(data, list):
        return data
    raise AdapterError("PEV-Daten brauchen eine Frame-Liste")


def _pev_arrays(records):
    n = len(records)
    xyz = np.full((n, 133, 3), np.nan)
    conf = np.zeros((n, 133))
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        points = rec.get("skeletonpoints")
        if isinstance(points, dict):
            points = [dict(value, id=key) for key, value in points.items()]
        if not isinstance(points, list):
            continue
        for point in points:
            try:
                idx = int(point["id"])
            except (KeyError, TypeError, ValueError):
                continue
            if not 0 <= idx < 133:
                continue
            try:
                xyz[i, idx] = (float(point.get("x", np.nan)),
                               float(point.get("y", np.nan)),
                               float(point.get("z", 0.0)))
                conf[i, idx] = float(point.get("confidence", 1.0))
            except (TypeError, ValueError):
                continue
    return xyz, conf


def _pev_bounds(xyz, conf, conf_min):
    """Robuste Mitte und einheitliche Skala; y bleibt vorerst nach oben."""
    finite = np.isfinite(xyz).all(axis=2) & (conf >= conf_min)
    for indices in (PEV_BOUND_INDICES, tuple(range(BODY_N)), tuple(range(133))):
        mask = finite[:, indices]
        if mask.any():
            xs = xyz[:, indices, 0][mask]
            ys = xyz[:, indices, 1][mask]
            break
    else:
        raise AdapterError("PEV-Frame ohne gueltige Koerperpunkte")

    x0, x1 = np.percentile(xs, (1.0, 99.0))
    y0, y1 = np.percentile(ys, (1.0, 99.0))
    if not (x1 > x0) or not (y1 > y0):
        x0, x1 = float(np.min(xs)), float(np.max(xs))
        y0, y1 = float(np.min(ys)), float(np.max(ys))
    scale = max(float(x1 - x0), float(y1 - y0))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return (float(x0 + x1) * 0.5, float(y0 + y1) * 0.5, scale)


def _pev_permutation(swap_sides):
    """Zielindex je Quellindex; tauscht bei Bedarf links/rechts."""
    perm = np.arange(133)
    if not swap_sides:
        return perm
    for a, b in PEV_SWAP_PAIRS:
        perm[a], perm[b] = b, a
    left = np.arange(LEFT_HAND_BASE, LEFT_HAND_BASE + HAND_N)
    right = np.arange(RIGHT_HAND_BASE, RIGHT_HAND_BASE + HAND_N)
    perm[left] = right
    perm[right] = left
    return perm


def _pev_empty(rec, index):
    frame = {"frame_video": _frame_video(rec, index, 1)}
    frame.update(EMPTY_FRAME)
    return frame


def _pev_frame(rec, index, center, scale, perm, conf_min):
    points = rec.get("skeletonpoints") if isinstance(rec, dict) else None
    if isinstance(points, dict):
        points = [dict(value, id=key) for key, value in points.items()]
    if not isinstance(points, list) or not points:
        return _pev_empty(rec, index)

    # [x, y, z, conf] im Bildsystem: x nach rechts, y nach unten, [0, 1].
    lm = np.zeros((133, 4))
    lm[:, 0] = lm[:, 1] = 0.5
    cx, cy = center
    for point in points:
        try:
            src = int(point["id"])
            x = float(point["x"])
            y = float(point["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= src < 133 or not (np.isfinite(x) and np.isfinite(y)):
            continue
        conf = float(point.get("confidence", 1.0))
        if not np.isfinite(conf) or conf < conf_min:
            continue
        dst = int(perm[src])
        lm[dst, 0] = 0.5 + (x - cx) / scale
        lm[dst, 1] = 0.5 - (y - cy) / scale      # PEV-y zeigt nach oben
        lm[dst, 2] = 0.0
        lm[dst, 3] = min(1.0, max(0.0, conf))

    if not np.any(lm[:BODY_N, 3] > 0.0):
        return _pev_empty(rec, index)

    frame = {"frame_video": _frame_video(rec, index, 1)}
    frame["pose"] = [list(map(float, lm[i])) for i in range(BODY_N)]
    frame["pose_world"] = [[float(lm[i, 0] - 0.5), float(lm[i, 1] - 0.5),
                            0.0, float(lm[i, 3])] for i in range(BODY_N)]
    for side, base in (("left", LEFT_HAND_BASE), ("right", RIGHT_HAND_BASE)):
        block = lm[base:base + HAND_N]
        if float(np.max(block[:, 3])) < 0.10:
            frame[side + "_hand"] = None
            frame[side + "_hand_world"] = None
        else:
            frame[side + "_hand"] = [list(map(float, row)) for row in block]
            frame[side + "_hand_world"] = [
                [float(row[0] - 0.5), float(row[1] - 0.5), 0.0, float(row[3])]
                for row in block]
    return frame


def sequence_from_pev(data, fps=25.0, swap_sides=True, conf_min=0.05):
    """Baut aus einer ``.pev``-Aufnahme (gsg) eine :class:`Sequence`.

    Die Punkte werden mit *einer* Skala fuer x und y auf ``[0, 1]`` gebracht,
    damit die Winkel erhalten bleiben. ``swap_sides`` vertauscht links/rechts;
    das ist fuer die gespiegelten gsg-Videos richtig und entspricht dem alten
    ``bone_mapping.json``.
    """
    records = _pev_records(data)
    if not records:
        raise AdapterError("keine Frames in der PEV-Quelle")
    if any(isinstance(rec, dict) and rec.get("pose") is not None for rec in records):
        raise AdapterError("die Quelle ist bereits eine Landmark-Sequenz")

    xyz, conf = _pev_arrays(records)
    cx, cy, scale = _pev_bounds(xyz, conf, conf_min)
    perm = _pev_permutation(swap_sides)
    frames = [_pev_frame(rec, i, (cx, cy), scale, perm, conf_min)
              for i, rec in enumerate(records)]

    return Sequence({
        "fps": float(fps or 25.0),
        "width": 1000,
        "height": 1000,
        "frame_step": 1,
        "n": len(frames),
        "hands": "coco",
        "source": "pev",
        # Nach dem Seitentausch ist der Gebaerdearm (Original 5/7/9) der
        # rechte Arm des Avatars. Der andere Arm ist bei den gsg-Aufnahmen
        # oft verdeckt und wird deshalb nicht aus den Landmarks gedreht.
        "sign_arm": "r" if swap_sides else "l",
        "frames": frames,
    })
