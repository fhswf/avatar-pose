"""
retarget.py

License: Apache License 2.0

Description:
Replay-Plugin fuer den Webviewer. Es liest Retarget-Kurven (oder direkt eine
RTMLib-Aufnahme), rechnet sie in glTF-Node-Rotationen um und schiebt die
Frames ueber ``add_send_object`` in den bestehenden WebSocket-Puffer.

Der Webviewer selbst bleibt unveraendert: Er bekommt weiterhin
``{knochen: {"rotation": [x, y, z, w]}}`` in Three.js-Reihenfolge.

Pfade lassen sich ueber Umgebungsvariablen setzen:

    AVATAR_PLUGIN=retarget
    AVATAR_RETARGET_CURVES=pfad/zu/kurven.json   # optional, Standard fuer Enter
    AVATAR_RETARGET_RIG=pfad/zu/rig_kim.json
    AVATAR_RETARGET_GLB=pfad/zu/avatar.glb

Im Textfeld des Webviewers:

    <Enter>              Standarddatei abspielen (oder letzte wiederholen)
    replay / play        letzte Datei erneut abspielen
    <wort>               Wort aus plugins/gsg/wordlist.csv nachschlagen und
                         die Pose aus dem echten Video ableiten (z. B. "Auto")
    <pfad>               Kurven-JSON abspielen
    <pfad-zu-rtmlib>     RTMLib-JSON durch die ganze Kette schicken

Die Wortsuche ist nicht Gross-/Kleinschreibung-abhaengig. "Auto" findet in
der wordlist.csv die id 61, laedt das zugehoerige Video (SignDict-CDNs),
wertet es mit RTMLib aus und legt die Keypoints als Cache ab. Die
``.pev``-Dateien aus ``extracted_videos`` sind nur noch der Notnagel: sie
stammen aus einem 3D-Lifting-Modell und sind fuer viele Woerter kaputt.

Die Wortwiedergabe beginnt und endet in der T-Pose (Ruhepose des glTF): Aus
dem Video wird der Abschnitt genommen, in dem die Haende oben sind, davor und
dahinter blendet ``webviewer.anchor_rest`` aus der T-Pose hinein und zurueck.
Mit ``AVATAR_RETARGET_ANCHOR=none`` laesst sich die Blende abschalten.

Wichtig: ``AVATAR_RETARGET_CURVES`` ist ein Dateipfad, kein Schalter. Der
Wert ``auto`` wird deshalb verworfen und mit einem Hinweis gemeldet.

Die PEV-Dateien der gsg-Videos sind gespiegelt (y zeigt nach oben). Der
Adapter dreht das zurueck und tauscht links/rechts; mit
``AVATAR_RETARGET_PEV_SWAP=0`` laesst sich der Tausch abschalten.
Bei ``.pev`` folgt nur der Gebaerdearm den Landmarks, weil der zweite Arm in
den Aufnahmen oft verdeckt und stark verkuerzt ist. Der andere Arm haengt
entspannt herab. ``AVATAR_RETARGET_ARM_MODE=both`` schaltet wieder beide Arme
ein, ``=sign`` erzwingt den Gebaerdearm auch fuer andere Quellen.

Der Video-Cache liegt unter ``main/plugins/gsg/video_cache`` (gitignoriert).
Mit ``AVATAR_RETARGET_CACHE`` laesst sich ein anderer Ordner setzen.

Der WebSocket-Puffer existiert erst nach dem Konstruktor des Plugins,
deshalb startet die Wiedergabe mit der ersten Eingabe und nicht schon beim
Serverstart.
"""

import csv
import json
import os
import sys
from pathlib import Path

from ..LangPlugin import LangPlugin

HERE = Path(__file__).resolve()
APP_DIR = HERE.parents[3]        # .../avatar-pose
PROJECT_DIR = HERE.parents[4]    # Repository-Wurzel

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

DEFAULT_RIG = PROJECT_DIR / "retarget" / "data" / "rig_kim.json"
DEFAULT_GLB = APP_DIR / "webviewer" / "media" / "avatar.glb"

# Wortliste und bereits extrahierte Posen aus dem gsg-Plugin.
GSG_DIR = APP_DIR / "main" / "plugins" / "gsg"
WORDLIST = GSG_DIR / "wordlist.csv"
PEV_DIR = GSG_DIR / "extracted_videos"
# Videos + RTMLib-Keypoints der Woerter (gitignoriert).
CACHE_DIR = GSG_DIR / "video_cache"

# Der Server sendet mit fester Rate; die Kurven werden darauf umgerechnet.
PLAYBACK_FPS = 30.0
# Diese Eingaben sind Befehle, keine Dateinamen. "auto" ist bewusst *kein*
# Befehl mehr, sondern das Wort aus der gsg-Wortliste.
REPLAY_COMMANDS = ("", "replay", "play", "start")
# Nur fuer AVATAR_RETARGET_CURVES: dieser Wert ist kein Pfad.
NOT_A_PATH = REPLAY_COMMANDS + ("auto",)
HELP = ("[retarget] <Enter>/replay spielt die Standarddatei, ein Wort wie "
        "'Auto' wird in plugins/gsg/wordlist.csv gesucht. Sonst den Pfad zu "
        "einer Kurven-, RTMLib- oder PEV-JSON eingeben. "
        "AVATAR_RETARGET_CURVES muss auf eine Datei zeigen.")


def _resolve(path):
    """Absoluter Pfad; relative Angaben zuerst im cwd, dann im Projekt."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (PROJECT_DIR / p).resolve()


def _looks_like_curves(data):
    return isinstance(data, dict) and isinstance(data.get("bones"), dict)


def _looks_like_sequence(data):
    frames = data.get("frames") if isinstance(data, dict) else None
    return isinstance(frames, list) and any(
        isinstance(fd, dict) and fd.get("pose") is not None for fd in frames)


class retarget(LangPlugin):
    """Spielt Retarget-Kurven im Webviewer ab."""

    def __init__(self, unified_web_socket):
        super().__init__(unified_web_socket)

        self.rig_path = _resolve(os.environ.get("AVATAR_RETARGET_RIG", DEFAULT_RIG))
        self.glb_path = _resolve(os.environ.get("AVATAR_RETARGET_GLB", DEFAULT_GLB))

        self.default_path = (os.environ.get("AVATAR_RETARGET_CURVES") or "").strip()
        if self.default_path and self.default_path.lower() in NOT_A_PATH:
            print("[retarget] AVATAR_RETARGET_CURVES=%r ist kein Dateipfad und "
                  "wird ignoriert." % self.default_path)
            self.default_path = None
        self.pev_swap = (os.environ.get("AVATAR_RETARGET_PEV_SWAP", "1")
                         .strip().lower() not in ("0", "false", "no", "off"))
        # "auto" (nur Gebaerdearm bei .pev), "sign" oder "both".
        self.arm_mode = (os.environ.get("AVATAR_RETARGET_ARM_MODE", "auto")
                         .strip().lower())
        self.cache_dir = _resolve(os.environ.get("AVATAR_RETARGET_CACHE", CACHE_DIR))
        # Wortwiedergabe beginnt/endet in der T-Pose (Ruhepose).
        self.anchor = (os.environ.get("AVATAR_RETARGET_ANCHOR", "rest")
                       .strip().lower() not in ("0", "none", "off", "false", "no"))
        self._rig = None
        self._last_path = None
        self._last_data = None
        self._last_label = None
        self._words = None

    # ------------------------------------------------------------------ API

    def input(self, string: str):
        """Reagiert auf die Eingabe aus dem Textfeld des Webviewers."""
        command = (string or "").strip().strip('"').strip("'")
        try:
            if command.lower() in REPLAY_COMMANDS:
                if self._last_data is not None:
                    self._play_data(self._last_data, self._last_label)
                    return
                target = self._last_path or self.default_path
                if not target:
                    print(HELP)
                    return
            elif not _resolve(command).exists():
                # Kein existierender Pfad? Dann als Wort in der gsg-Liste suchen.
                data = self._word_recording(command)
                if data is not None:
                    self._play_data(data, command)
                    return
                target = self._word_pev(command)
                if target is None:
                    target = command
            else:
                target = command
            self._play(target)
        except Exception as exc:                          # noqa: BLE001
            print("[retarget] %s: %s" % (command, exc))

    # ------------------------------------------------------------- intern

    def _wordlist(self):
        """text (gross) -> id aus plugins/gsg/wordlist.csv, einmal geladen."""
        if self._words is None:
            words = {}
            if WORDLIST.exists():
                with open(WORDLIST, newline="", encoding="utf-8-sig",
                          errors="replace") as f:
                    for row in csv.DictReader(f):
                        text = " ".join((row.get("text") or "").split()).upper()
                        ident = (row.get("id") or "").strip()
                        if text and ident:
                            words.setdefault(text, ident)
            else:
                print("[retarget] Wortliste fehlt: %s" % WORDLIST)
            self._words = words
        return self._words

    def _word_pev(self, text):
        """Pfad zur .pev-Datei fuer ein Wort aus der gsg-Wortliste."""
        key = " ".join((text or "").split()).upper()
        if not key:
            return None
        ident = self._wordlist().get(key)
        if ident is None:
            return None
        path = PEV_DIR / ("%s.pev" % ident)
        if not path.exists():
            raise FileNotFoundError(
                "Wort '%s' gefunden (id %s), aber die PEV-Datei fehlt: %s"
                % (text, ident, path))
        return path

    def _word_recording(self, text):
        """Keypoints aus dem echten Video eines Wortes - oder ``None``."""
        try:
            from retarget import gsg_video
        except Exception as exc:                          # noqa: BLE001
            print("[retarget] Video-Modul nicht ladbar (%s) - nutze PEV" % exc)
            return None
        if gsg_video.lookup(WORDLIST, text) is None:
            return None
        print("[retarget] werte Video fuer '%s' aus (nur beim ersten Mal "
              "langsam) ..." % text)
        try:
            return gsg_video.recording(WORDLIST, text, self.cache_dir)
        except Exception as exc:                          # noqa: BLE001
            print("[retarget] Video fuer '%s' nicht nutzbar (%s) - nutze PEV"
                  % (text, exc))
            return None

    def _rig_data(self):
        if self._rig is None:
            from retarget.rig import Rig
            self._rig = Rig.load(str(self.rig_path))
        return self._rig

    def _frames_from_curves(self, curves, fps):
        from retarget import webviewer
        curves = webviewer.resample(curves, fps, PLAYBACK_FPS)
        return webviewer.frames(self._rig_data(), curves, str(self.glb_path))

    def _frames_from_recording(self, data, gesture=False):
        """RTMLib-, PEV- oder Landmark-Aufnahme einmal durch die ganze Kette."""
        from retarget import landmarks, webviewer
        from retarget import retarget as pipeline
        from retarget.rtm_adapter import (looks_like_pev, sequence_from_pev,
                                          sequence_from_rtmlib)
        from retarget.sequence import Sequence
        from retarget.window import raised_arms_window

        if looks_like_pev(data):
            seq = sequence_from_pev(data, swap_sides=self.pev_swap)
        elif _looks_like_sequence(data):
            seq = Sequence(data)
        else:
            seq = sequence_from_rtmlib(data)
        landmarks.process(seq)
        curves = pipeline.solve(self._rig_data(), seq, arm_mode=self.arm_mode)
        if not curves:
            raise ValueError("kein Knochen gesetzt - Konfidenz oder Rig pruefen")

        if gesture:
            win = raised_arms_window(seq)
            if win is not None:
                # Nur der Abschnitt mit erhobenen Haenden; Anfang und Ende
                # werden spaeter aus der T-Pose geblendet.
                print("[retarget] Bewegungsfenster: Frames %d-%d von %d"
                      % (win[0], win[1], len(seq.frames)))
                curves = {name: qs[win[0]:win[1] + 1]
                          for name, qs in curves.items()}

        curves = webviewer.resample(curves, seq.fps, PLAYBACK_FPS)
        frames = webviewer.frames(self._rig_data(), curves, str(self.glb_path))
        if gesture and self.anchor and frames:
            rest = webviewer.rest_frame(str(self.glb_path), list(frames[0]))
            frames = webviewer.anchor_rest(frames, PLAYBACK_FPS, rest=rest)
        return frames

    def _send(self, frames, label):
        if not frames:
            raise ValueError("keine Frames erzeugt")
        for frame in frames:
            self.output_hook.add_send_object(frame)
        print("[retarget] %d Frames aus %s" % (len(frames), label))
        return frames

    def _play_data(self, data, label):
        """Aufnahme (Keypoint-Dict) direkt abspielen, inkl. T-Pose-Blende."""
        self._send(self._frames_from_recording(data, gesture=True), label)
        self._last_data = data
        self._last_label = label
        self._last_path = None

    def _play(self, path):
        target = _resolve(path)
        if not target.exists():
            raise FileNotFoundError(
                "Datei nicht gefunden: %s - AVATAR_RETARGET_CURVES oder den "
                "Pfad im Textfeld pruefen" % target)
        if target.is_dir():
            raise IsADirectoryError("das ist ein Ordner, keine JSON-Datei: %s" % target)
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)

        if _looks_like_curves(data):
            from retarget import webviewer
            curves = {name: qs for name, qs in data["bones"].items()}
            fps = float(data.get("fps") or PLAYBACK_FPS)
            frames = self._frames_from_curves(curves, fps)
        else:
            frames = self._frames_from_recording(data)

        self._send(frames, target)
        self._last_path = str(target)
        self._last_data = None
        self._last_label = None
