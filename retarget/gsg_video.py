"""Echte gsg-Videos laden und mit RTMLib auswerten.

Die ``.pev``-Dateien aus ``plugins/gsg/extracted_videos`` sind der Fallback,
wenn das Video fehlt; mit ``DataLoaderDGS/Extractor.py`` neu erzeugt, enthalten
sie dieselbe 2D-Pose wie der Videopfad (frueher: 3D-Lifting mit kaputter
linker Armkette).

Ablauf je Wort:

* Wort in ``wordlist.csv`` nachschlagen (id + ``videoUrl``),
* Video herunterladen (der jwplayer-Link liefert 403, die SignDict-Seite
  bettet dieselbe Datei ueber ``cdn.wishlephant.com`` ein),
* alle Frames mit ``rtmlib.Wholebody`` (COCO-WholeBody, 133 Punkte) auswerten,
* Keypoints als JSON neben dem Video ablegen und beim naechsten Mal direkt
  laden (kein erneuter Download, keine erneute Inferenz).

RTMLib laeuft standardmaessig ueber sein OpenCV-DNN-Backend
(``AVATAR_RTMLIB_BACKEND=opencv``); damit wird das Paket ``onnxruntime``
nicht gebraucht. Die Gewichte selbst sind bei RTMLib immer ``.onnx``-Dateien.
``AVATAR_RTMLIB_BACKEND=onnxruntime`` schaltet die alte Engine wieder ein,
``AVATAR_RTMLIB_DEVICE`` setzt das Geraet (Standard ``cpu``).

Der Cache liegt unter ``main/plugins/gsg/video_cache`` und ist in
``.gitignore`` eingetragen.
"""

import csv
import json
import os
import re
import urllib.request
from pathlib import Path

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
MIRROR = "https://cdn.wishlephant.com/signdict/videos/%s"


class GsgVideoError(RuntimeError):
    pass


def _norm(text):
    return " ".join((text or "").split()).upper()


def lookup(wordlist, word):
    """Zeile der ``wordlist.csv`` fuer ein Wort (oder ``None``)."""
    key = _norm(word)
    if not key:
        return None
    with open(wordlist, newline="", encoding="utf-8-sig", errors="replace") as f:
        for row in csv.DictReader(f):
            if _norm(row.get("text")) == key:
                return row
    return None


def row_by_id(wordlist, ident):
    """Zeile der ``wordlist.csv`` fuer eine Id (oder ``None``)."""
    key = str(ident or "").strip()
    if not key:
        return None
    with open(wordlist, newline="", encoding="utf-8-sig", errors="replace") as f:
        for row in csv.DictReader(f):
            if (row.get("id") or "").strip() == key:
                return row
    return None


def cache_path(cache_dir, ident):
    """Pfad der RTMLib-Keypoints im Cache (ohne Download/Inferenz)."""
    slug = re.sub(r"[^0-9A-Za-z_.-]", "_", str(ident))
    return Path(cache_dir) / ("%s.rtmlib.json" % slug)


def video_urls(row):
    """Kandidaten in absteigender Erfolgswahrscheinlichkeit.

    ``videoUrl`` zeigt auf jwplayer und antwortet ohne gueltigen Token mit
    403. SignDict selbst spielt dieselbe Datei ueber ``cdn.wishlephant.com``
    aus; der Dateiname bleibt gleich.
    """
    urls = []
    original = (row.get("videoUrl") or "").strip()
    if original:
        urls.append(original)
        name = os.path.basename(original.split("?")[0])
        if name:
            urls.append(MIRROR % name)
    return urls


def _download(url, dest, referer):
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Referer": referer,
        "Accept": "*/*",
    })
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
        f.write(resp.read())


def fetch_video(row, dest, ident=None):
    """Video fuer eine Wortlisten-Zeile nach ``dest`` laden (mit Fallback)."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 1024:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    entry = "https://signdict.org/entry/%s" % (ident if ident is not None
                                               else row.get("id", ""))
    errors = []
    for url in video_urls(row):
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            _download(url, tmp, entry)
        except Exception as exc:                       # noqa: BLE001
            errors.append("%s: %s" % (url, exc))
            tmp.unlink(missing_ok=True)
            continue
        if tmp.stat().st_size < 1024:
            errors.append("%s: Datei zu klein" % url)
            tmp.unlink(missing_ok=True)
            continue
        tmp.replace(dest)
        return dest
    raise GsgVideoError("Video fuer '%s' nicht ladbar:\n  %s"
                        % (row.get("text"), "\n  ".join(errors)))


def rtmlib_backend(default="opencv"):
    """Backend fuer ``rtmlib.Wholebody`` (``opencv`` braucht kein onnxruntime)."""
    value = (os.environ.get("AVATAR_RTMLIB_BACKEND") or default or "opencv")
    value = value.strip().lower()
    return value if value in ("opencv", "onnxruntime", "openvino") else "opencv"


def rtmlib_device(default="cpu"):
    return ((os.environ.get("AVATAR_RTMLIB_DEVICE") or default or "cpu")
            .strip().lower())


def make_wholebody(backend=None, device=None, mode=None):
    """RTMLib-Modell laden; ohne ``onnxruntime``-Zwang.

    Faellt auf ``onnxruntime`` zurueck, wenn OpenCV das Modell nicht laden
    kann (z. B. aeltere OpenCV-Version) und das Paket installiert ist.
    """
    try:
        from rtmlib import Wholebody
    except ImportError as exc:                         # noqa: BLE001
        raise GsgVideoError(
            "RTMLib fehlt (pip install rtmlib opencv-python-headless): %s" % exc)

    backend = backend or rtmlib_backend()
    device = device or rtmlib_device()
    kwargs = {"to_openpose": False, "backend": backend, "device": device}
    if mode:
        kwargs["mode"] = mode
    try:
        return Wholebody(**kwargs)
    except Exception as exc:                           # noqa: BLE001
        if backend != "opencv":
            raise GsgVideoError("RTMLib laedt nicht (%s): %s" % (backend, exc))
        print("[retarget] OpenCV-Backend nicht nutzbar (%s) - versuche "
              "onnxruntime" % exc)
        kwargs["backend"] = "onnxruntime"
        try:
            return Wholebody(**kwargs)
        except Exception as exc2:                      # noqa: BLE001
            raise GsgVideoError(
                "RTMLib laedt weder mit OpenCV noch mit onnxruntime: %s" % exc2)


def extract_keypoints(video_path, out_path=None, backend=None, device=None):
    """Alle Frames mit RTMLib (COCO-WholeBody) auswerten."""
    try:
        import cv2
    except ImportError as exc:                         # noqa: BLE001
        raise GsgVideoError(
            "OpenCV fehlt (pip install opencv-python-headless): %s" % exc)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise GsgVideoError("Video nicht lesbar: %s" % video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    model = make_wholebody(backend=backend, device=device)

    frames = []
    while True:
        ok, image = cap.read()
        if not ok:
            break
        kp, sc = model(image)
        if kp is None or len(kp) == 0:
            frames.append(None)
            continue
        # Bei mehreren Personen die mit der besten mittleren Koerperkonfidenz.
        scores = [float(s[:17].mean()) for s in sc]
        best = int(max(range(len(scores)), key=scores.__getitem__))
        frames.append({"keypoints": kp[best].tolist(),
                       "scores": sc[best].tolist()})
    cap.release()

    if not any(f is not None for f in frames):
        raise GsgVideoError("keine Person im Video gefunden: %s" % video_path)

    data = {"width": width, "height": height, "fps": fps,
            "frame_step": 1, "source": "gsg-video", "frames": frames}
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    return data


def recording(wordlist, word, cache_dir, ident=None):
    """Aufnahme (Keypoints) fuer ein Wort - aus dem Cache oder frisch."""
    row = lookup(wordlist, word)
    if row is None:
        raise GsgVideoError("Wort '%s' steht nicht in %s" % (word, wordlist))
    return _recording(row, cache_dir,
                      ident if ident is not None else row.get("id", ""))


def recording_by_id(wordlist, ident, cache_dir):
    """Wie :func:`recording`, aber die Id ist bereits bekannt (Satzplanung)."""
    row = row_by_id(wordlist, ident)
    if row is None:
        raise GsgVideoError("Id '%s' steht nicht in %s" % (ident, wordlist))
    return _recording(row, cache_dir, ident)


def _recording(row, cache_dir, ident):
    slug = re.sub(r"[^0-9A-Za-z_.-]", "_", str(ident or row.get("text", "")))
    cache_dir = Path(cache_dir)
    keypoints = cache_path(cache_dir, ident or slug)
    if keypoints.exists():
        with open(keypoints, "r", encoding="utf-8") as f:
            return json.load(f)
    video = fetch_video(row, cache_dir / ("%s.mp4" % slug), ident)
    return extract_keypoints(video, keypoints)
