"""
repair_pev.py

Prueft ``.pev``-Dateien und baut eine kaputte Tiefe (z) neu auf.

Hintergrund
-----------
``Extractor.py`` hebt die 2D-Keypoints per ONNX-Modell in den 3D-Raum.
Beim Zurueckrechnen skaliert
``pose_estimation_recognition_utils_rtmlib/rtm_lifting.py``
(``_denormalize_by_bounding_box``) nur x und y ueber die Bounding-Box;
die z-Komponente bleibt im Normbereich des Modells. Dadurch ist z rund
1000-mal zu klein: x/y liegen im Bereich 1e5, z im Bereich 1e2.

``main/Calculator.py`` liest die Punkte direkt als Three.js-Koordinaten.
Mit praktisch konstantem z liegen alle Knochen in der Bildebene, die
Arme kreuzen den Rumpf statt davor - genau das "Arme verschwinden im
Koerper"-Problem. Die Tiefe laesst sich nicht durch einen Faktor
reparieren (das z des Lifters ist auch in der Richtung nicht stabil),
deshalb wird sie aus der 2D-Verkürzung neu geschaetzt: pro Frame
``sqrt(L^2 - p^2)`` mit ``L`` = kanonische Segmentlaenge und ``p`` =
projizierte Laenge, plus einer Grundtiefe nach vorn. Dasselbe Verfahren
nutzt auch ``retarget/depth.py`` fuer die Live-Extraktion.

Befehle
-------
``check``   Struktur und Tiefe pruefen, Uebersicht ausgeben.
``repair``  z der betroffenen Dateien neu aufbauen (Backup als *.pev.bak).
``sync``    PEVs aus dem Datensatz in den Plugin-Ordner kopieren.

Beispiele
---------
    python DataLoaderDGS/repair_pev.py check
    python DataLoaderDGS/repair_pev.py check --verbose
    python DataLoaderDGS/repair_pev.py repair --dry-run
    python DataLoaderDGS/repair_pev.py repair
    python DataLoaderDGS/repair_pev.py sync --dry-run

Ohne ``--root`` wird ``main/plugins/gsg/extracted_videos`` relativ zum
Programmordner verwendet. ``repair`` aendert nur Dateien, die als
``flach`` oder ``kaputt`` erkannt wurden; alles andere bleibt unberuehrt.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

# --- Konstanten ------------------------------------------------------------

N_POINTS = 133            # COCO-WholeBody
BODY_POINTS = range(0, 17)  # Koerper inkl. Kopf, ohne Haende/Face

# Ab diesem Verhaeltnis (Tiefenausdehnung / Koerperhoehe) gilt die Tiefe
# als brauchbar. Echte 3D-Daten liegen bei 0.05..0.4, kaputte bei < 0.001.
Z_MIN_RATIO = 0.02

# Arme: Schulter -> Ellbogen -> Handgelenk, dazu die Handpunkte.
# COCO-WholeBody: 5/7/9 linke Kette, 6/8/10 rechte Kette,
# 91..111 linke Hand, 112..132 rechte Hand.
ARM_CHAINS = (
    (5, 7, 9, range(91, 112)),
    (6, 8, 10, range(112, 133)),
)

# Grundtiefe der Arme als Anteil der Segmentlaenge. Ohne diesen Anteil
# bleiben bildparallele Unterarme in der Rumpfebene.
ARM_BASE = {"upperarm": 0.20, "lowerarm": 0.45}

# Obergrenze der rekonstruierten Tiefe (schraegste noch plausible Lage).
MAX_OOP_DEG = 60.0

# Perzentil fuer die kanonische Segmentlaenge: die laengste Projektion im
# Clip entspricht ungefaehr der wahren Knochenlaenge.
LEN_PCT = 90.0

# Glaettung der Tiefe ueber die Zeit (Frames).
SMOOTH_FRAMES = 5


# --- Ein-/Ausgabe ----------------------------------------------------------

class PevError(Exception):
    """Datei ist kein lesbares PEV."""


def load_pev(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:                              # noqa: BLE001
        raise PevError(str(exc)) from exc
    if not isinstance(data, dict) or not isinstance(data.get("frames"), list):
        raise PevError("kein {'frames': [...]}-Objekt")
    if not data["frames"]:
        raise PevError("keine Frames")
    return data


def points_array(data):
    """Dichtes (Frames, 133, 3)-Array; fehlende Punkte bleiben NaN."""
    frames = data["frames"]
    arr = np.full((len(frames), N_POINTS, 3), np.nan, dtype=np.float64)
    for i, frame in enumerate(frames):
        for point in frame.get("skeletonpoints") or []:
            try:
                idx = int(point.get("id", -1))
            except (TypeError, ValueError):
                continue
            if 0 <= idx < N_POINTS:
                arr[i, idx, 0] = _num(point.get("x"))
                arr[i, idx, 1] = _num(point.get("y"))
                arr[i, idx, 2] = _num(point.get("z"))
    return arr


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def body_scale(arr):
    """Median der Koerperhoehe (y-Spanne) ueber alle Frames."""
    body = arr[:, list(BODY_POINTS), :]
    spans = _nan_ptp(body[:, :, 1])
    spans = spans[np.isfinite(spans) & (spans > 0)]
    return float(np.median(spans)) if len(spans) else 0.0


def depth_ratio(arr):
    """Tiefenausdehnung im Verhaeltnis zur Koerperhoehe (Median der Frames)."""
    scale = body_scale(arr)
    if scale <= 0:
        return 0.0
    spans = _nan_ptp(arr[:, :, 2])
    spans = spans[np.isfinite(spans)]
    return float(np.median(spans) / scale) if len(spans) else 0.0


def _nan_ptp(values):
    """ptp je Zeile, NaN werden ignoriert (NaN wenn nichts uebrig bleibt)."""
    out = np.full(values.shape[0], np.nan)
    for i, row in enumerate(values):
        row = row[np.isfinite(row)]
        if row.size:
            out[i] = row.max() - row.min()
    return out


# --- Diagnose --------------------------------------------------------------

def inspect(path):
    """Liefert ein Ergebnis-Dict: status = ok | flach | kaputt."""
    result = {"path": Path(path), "status": "ok", "grund": "",
              "frames": 0, "z_ratio": float("nan"), "punkte": 0}
    try:
        data = load_pev(path)
    except PevError as exc:
        result.update(status="kaputt", grund="nicht lesbar: %s" % exc)
        return result

    frames = data["frames"]
    result["frames"] = len(frames)
    counts = [len(f.get("skeletonpoints") or []) for f in frames]
    result["punkte"] = min(counts) if counts else 0
    if result["punkte"] < 17:
        result.update(status="kaputt",
                      grund="zu wenige Punkte je Frame (min %d)" % result["punkte"])
        return result

    arr = points_array(data)
    body = arr[:, list(BODY_POINTS), :2]
    if not np.isfinite(body).any():
        result.update(status="kaputt", grund="keine endlichen x/y-Werte")
        return result

    ratio = depth_ratio(arr)
    result["z_ratio"] = ratio
    if not np.isfinite(ratio) or ratio < Z_MIN_RATIO:
        result.update(status="flach",
                      grund="z kollabiert (Tiefe %.4f der Koerperhoehe)"
                            % (ratio if np.isfinite(ratio) else 0.0))
    return result


# --- Reparatur -------------------------------------------------------------

def _canonical_lengths(arr, i, j):
    """Kanonische 2D-Laenge einer Strecke ueber den ganzen Clip."""
    lens = np.linalg.norm(arr[:, j, :2] - arr[:, i, :2], axis=1)
    lens = lens[np.isfinite(lens)]
    if lens.size == 0:
        return 0.0
    return float(np.percentile(lens, LEN_PCT))


def _smooth(values, window=SMOOTH_FRAMES):
    if window < 3 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def reconstruct_depth(arr, front_sign=1.0):
    """Neues Array mit aus der 2D-Verkürzung geschaetzter Tiefe.

    x und y bleiben unveraendert. z wird fuer alle Punkte auf 0 gesetzt
    (der alte Wert ist Rauschen) und nur fuer die Armketten aufgebaut:
    +z zeigt nach vorn, wie in Three.js/glTF.
    """
    out = arr.copy()
    out[:, :, 2] = 0.0
    max_ratio = np.tan(np.radians(MAX_OOP_DEG))

    for shoulder, elbow, wrist, hand in ARM_CHAINS:
        len_upper = _canonical_lengths(arr, shoulder, elbow)
        len_lower = _canonical_lengths(arr, elbow, wrist)
        z_elbow = np.zeros(len(arr))
        z_wrist = np.zeros(len(arr))

        for i in range(len(arr)):
            if not np.isfinite(arr[i, [shoulder, elbow, wrist], :2]).all():
                continue
            proj_upper = np.linalg.norm(arr[i, elbow, :2] - arr[i, shoulder, :2])
            proj_lower = np.linalg.norm(arr[i, wrist, :2] - arr[i, elbow, :2])
            if proj_upper <= 0 or proj_lower <= 0:
                continue
            dz_upper = _segment_depth(proj_upper, len_upper, "upperarm", max_ratio)
            dz_lower = _segment_depth(proj_lower, len_lower, "lowerarm", max_ratio)
            z_elbow[i] = dz_upper
            z_wrist[i] = dz_upper + dz_lower

        z_elbow = _smooth(z_elbow)
        z_wrist = _smooth(z_wrist)
        out[:, shoulder, 2] = 0.0
        out[:, elbow, 2] = front_sign * z_elbow
        out[:, wrist, 2] = front_sign * z_wrist
        # Handpunkte liegen in der Ebene des Handgelenks.
        for idx in hand:
            out[:, idx, 2] = front_sign * z_wrist
    return out


def _segment_depth(projected, canonical, name, max_ratio):
    if canonical <= 0:
        return 0.0
    natural = np.sqrt(max(0.0, canonical * canonical - projected * projected))
    base = ARM_BASE.get(name, 0.0) * canonical
    return float(min(natural + base, max_ratio * projected))


def write_repaired(path, data, arr):
    """Schreibt die neuen z-Werte atomar zurueck, Rest der Datei bleibt."""
    for i, frame in enumerate(data["frames"]):
        for point in frame.get("skeletonpoints") or []:
            try:
                idx = int(point.get("id", -1))
            except (TypeError, ValueError):
                continue
            if 0 <= idx < N_POINTS and np.isfinite(arr[i, idx, 2]):
                point["z"] = float(arr[i, idx, 2])
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    os.replace(tmp, path)


def backup(path):
    """Original einmalig als ``<name>.pev.bak`` sichern (nie ueberschreiben)."""
    target = path.with_suffix(".pev.bak")
    if not target.exists():
        shutil.copy2(path, target)
    return target

# --- Pfade -----------------------------------------------------------------

def app_dir():
    """Ordner, in dem ``main.py`` liegt (eine Ebene ueber DataLoaderDGS)."""
    return Path(__file__).resolve().parent.parent


def default_root():
    return app_dir() / "main" / "plugins" / "gsg" / "extracted_videos"


def default_sync_source():
    return app_dir().parent.parent / "sign-language-avatar-gloss-dgs" / "extracted"


def pev_files(root):
    root = Path(root)
    if root.is_file():
        return [root] if root.suffix.lower() == ".pev" else []
    return sorted(p for p in root.rglob("*.pev"))


# --- Befehle ---------------------------------------------------------------

def cmd_check(args):
    files = pev_files(args.root)
    if not files:
        print("Keine .pev-Dateien in %s" % args.root)
        return 1

    results = [inspect(p) for p in files]
    counts = {"ok": 0, "flach": 0, "kaputt": 0}
    shown = 0
    for r in results:
        counts[r["status"]] += 1
        if args.verbose or r["status"] != "ok":
            if shown < args.limit:
                print("%-10s %-28s %s" % (r["status"], r["path"].name, r["grund"]
                                          or "Tiefe %.2f" % r["z_ratio"]))
            elif shown == args.limit:
                print("... (weitere mit --limit oder --json)")
            shown += 1

    print("\n%d Dateien: %d ok, %d flach, %d kaputt"
          % (len(files), counts["ok"], counts["flach"], counts["kaputt"]))
    if counts["flach"] or counts["kaputt"]:
        print("Reparatur: python %s repair" % _self())
    if args.json:
        payload = [{"datei": r["path"].name, "status": r["status"],
                    "grund": r["grund"], "frames": r["frames"],
                    "z_ratio": None if not np.isfinite(r["z_ratio"]) else round(r["z_ratio"], 6)}
                   for r in results]
        Path(args.json).write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        print("Bericht: %s" % args.json)
    return 0 if counts["kaputt"] == 0 else 2


def cmd_repair(args):
    files = pev_files(args.root)
    if not files:
        print("Keine .pev-Dateien in %s" % args.root)
        return 1

    done = skipped = broken = 0
    for path in files:
        info = inspect(path)
        if info["status"] == "ok" and not args.force:
            skipped += 1
            continue
        if info["status"] == "kaputt":
            print("uebersprungen %-28s %s" % (path.name, info["grund"]))
            broken += 1
            continue
        try:
            data = load_pev(path)
        except PevError as exc:
            print("uebersprungen %-28s %s" % (path.name, exc))
            broken += 1
            continue
        arr = points_array(data)
        fixed = reconstruct_depth(arr, front_sign=args.front_sign)
        before = depth_ratio(arr)
        after = depth_ratio(fixed)
        if args.dry_run:
            print("wuerde reparieren %-24s Tiefe %.4f -> %.3f" % (path.name, before, after))
            done += 1
            continue
        if not args.no_backup:
            backup(path)
        write_repaired(path, data, fixed)
        print("repariert %-28s Tiefe %.4f -> %.3f" % (path.name, before, after))
        done += 1

    verb = "wuerden repariert" if args.dry_run else "repariert"
    print("\n%d %s, %d unveraendert, %d uebersprungen (kaputt)"
          % (done, verb, skipped, broken))
    return 0 if broken == 0 else 2


def cmd_sync(args):
    source = Path(args.source)
    dest = Path(args.dest)
    if not source.is_dir():
        print("Quellordner fehlt: %s" % source)
        return 1
    dest.mkdir(parents=True, exist_ok=True)

    copied = repaired = skipped = missing = 0
    for src in pev_files(source):
        info = inspect(src)
        if info["status"] == "kaputt":
            print("uebersprungen %-28s %s" % (src.name, info["grund"]))
            skipped += 1
            continue
        target = dest / src.name
        if target.exists() and not args.force:
            skipped += 1
            continue
        if args.dry_run:
            print("wuerde kopieren %s -> %s" % (src.name, target))
            copied += 1
            continue
        shutil.copy2(src, target)
        copied += 1
        if args.repair and info["status"] == "flach":
            data = load_pev(target)
            write_repaired(target, data, reconstruct_depth(points_array(data),
                                                           args.front_sign))
            repaired += 1

    # Dateien, die im Ziel liegen, aber nicht in der Quelle vorkommen.
    src_names = {p.name for p in pev_files(source)}
    for target in pev_files(dest):
        if target.name not in src_names:
            missing += 1

    verb = "wuerden kopiert" if args.dry_run else "kopiert"
    print("\n%d %s, %d Tiefe repariert, %d uebersprungen"
          % (copied, verb, repaired, skipped))
    if missing:
        print("Hinweis: %d Dateien im Ziel haben keine Quelle" % missing)
    return 0


def _self():
    return os.path.relpath(__file__, Path.cwd())


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="PEV-Dateien pruefen, reparieren und synchronisieren.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--root", default=str(default_root()),
                       help="Ordner oder Datei mit .pev-Dateien "
                            "(Standard: main/plugins/gsg/extracted_videos)")
        p.add_argument("--front-sign", type=float, default=1.0,
                       help="+1: Arme zeigen nach vorn (+z), -1: nach hinten")
        p.add_argument("--force", action="store_true",
                       help="auch bereits brauchbare Dateien bearbeiten")

    check = sub.add_parser("check", help="Struktur und Tiefe pruefen")
    add_common(check)
    check.add_argument("--verbose", action="store_true",
                       help="auch brauchbare Dateien auflisten")
    check.add_argument("--json", default=None, help="Bericht als JSON schreiben")
    check.add_argument("--limit", type=int, default=40,
                       help="hoechstens so viele auffaellige Dateien auflisten")
    check.set_defaults(func=cmd_check)

    repair = sub.add_parser("repair", help="Tiefe aus der 2D-Verkürzung neu aufbauen")
    add_common(repair)
    repair.add_argument("--dry-run", action="store_true",
                        help="nur anzeigen, nichts schreiben")
    repair.add_argument("--no-backup", action="store_true",
                        help="kein *.pev.bak anlegen")
    repair.set_defaults(func=cmd_repair)

    sync = sub.add_parser("sync", help="PEVs aus dem Datensatz uebernehmen")
    sync.add_argument("--source", default=str(default_sync_source()))
    sync.add_argument("--dest", default=str(default_root()))
    sync.add_argument("--front-sign", type=float, default=1.0)
    sync.add_argument("--force", action="store_true",
                      help="vorhandene Zieldateien ueberschreiben")
    sync.add_argument("--repair", action="store_true",
                      help="flache Tiefe direkt beim Kopieren reparieren")
    sync.add_argument("--dry-run", action="store_true")
    sync.set_defaults(func=cmd_sync)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
