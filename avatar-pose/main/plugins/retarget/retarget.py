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
                         die vorab extrahierte PEV abspielen (z. B. "Auto")
    <satz>               mehrere Woerter/Buchstaben, z. B. "Der Hund laeuft"
                         oder "H A U S"; siehe unten
    <pfad>               Kurven-JSON abspielen
    <pfad-zu-rtmlib>     RTMLib-JSON durch die ganze Kette schicken

Die Wortsuche ist nicht Gross-/Kleinschreibung-abhaengig. "Auto" findet in
der wordlist.csv die id 61; gespielt wird ausschliesslich
``extracted_videos/61.pev``. Diese Dateien werden vorab mit
``DataLoaderDGS/Extractor.py`` aus den zugeschnittenen gsg-Videos erzeugt;
zur Laufzeit werden weder Videos geladen noch Keypoints berechnet und es wird
nicht auf Signdict zugegriffen. Fehlt ein Wort als Gebaerde (oder die PEV),
wird es wie in ``gsg.py`` Buchstabe fuer Buchstabe gespielt (``SCH`` zusammen).

Mehrere Woerter werden ueber ``text_to_gloss`` (Deutsch -> DGS-Gloss)
uebersetzt und dann Wort fuer Wort abgespielt; unbekannte Woerter werden
buchstabiert. Das Modell ist auf Wettertexten trainiert, deshalb bleibt die
Wortliste der Rueckfall. Ein einzelnes Wort, das in der Wortliste steht,
geht weiterhin direkt durch (kein Modell). Ein-Blendungen zur T-Pose und eine
kurze Pause trennen die Gebaerden. Abschalten mit ``AVATAR_GLOSS=0``.

    AVATAR_GLOSS_EPOCH   Standard 20 (bester BLEU auf data/test.txt)
    AVATAR_GLOSS_DIR     Ordner von text_to_gloss
    AVATAR_RETARGET_SIGN_PAUSE  Pause zwischen Gebaerden in Sekunden (0.2)

Die zugeschnittenen PEVs enthalten bereits die ganze Gebaerde, deshalb werden
sie ohne weiteres Bewegungsfenster abgespielt. Die Wiedergabe beginnt und
endet in der T-Pose (Ruhepose des glTF): davor und dahinter blendet
``webviewer.anchor_rest`` hinein und zurueck. Mit
``AVATAR_RETARGET_ANCHOR=none`` laesst sich die Blende abschalten.

Wichtig: ``AVATAR_RETARGET_CURVES`` ist ein Dateipfad, kein Schalter. Der
Wert ``auto`` wird deshalb verworfen und mit einem Hinweis gemeldet.

Die PEV-Dateien der gsg-Videos sind gespiegelt (y zeigt nach oben). Der
Adapter dreht das zurueck und tauscht links/rechts; mit
``AVATAR_RETARGET_PEV_SWAP=0`` laesst sich der Tausch abschalten.
Seit der Neuerzeugung mit ``DataLoaderDGS/Extractor.py`` folgen bei ``.pev``
beide Arme den Landmarks (``auto`` = ``both``). ``AVATAR_RETARGET_ARM_MODE=sign``
laesst weiterhin nur den Gebaerdearm folgen, der andere haengt entspannt herab.

Der WebSocket-Puffer existiert erst nach dem Konstruktor des Plugins,
deshalb startet die Wiedergabe mit der ersten Eingabe und nicht schon beim
Serverstart.
"""

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
# Uebersetzungsmodul (Deutsch -> DGS-Gloss) mit den vorhandenen Checkpoints.
GLOSS_DIR = APP_DIR / "text_to_gloss"
DEFAULT_GLOSS_EPOCH = 20

# Wortliste und bereits extrahierte Posen aus dem gsg-Plugin.
GSG_DIR = APP_DIR / "main" / "plugins" / "gsg"
WORDLIST = GSG_DIR / "wordlist.csv"
PEV_DIR = GSG_DIR / "extracted_videos"
# Der Server sendet mit fester Rate; die Kurven werden darauf umgerechnet.
PLAYBACK_FPS = 30.0
# Diese Eingaben sind Befehle, keine Dateinamen. "auto" ist bewusst *kein*
# Befehl mehr, sondern das Wort aus der gsg-Wortliste.
REPLAY_COMMANDS = ("", "replay", "play", "start")
# Nur fuer AVATAR_RETARGET_CURVES: dieser Wert ist kein Pfad.
NOT_A_PATH = REPLAY_COMMANDS + ("auto",)
HELP = ("[retarget] <Enter>/replay wiederholt die letzte Ausgabe. Ein Wort wie "
        "'Auto' wird in plugins/gsg/wordlist.csv gesucht und die PEV aus "
        "extracted_videos gespielt; fehlt sie, wird Buchstabe fuer Buchstabe "
        "gespielt. Mehrere Woerter wie 'Der Hund laeuft' werden uebersetzt und "
        "Gebaerden einzeln gespielt. Sonst den Pfad zu einer Kurven-, RTMLib- "
        "oder PEV-JSON eingeben. AVATAR_RETARGET_CURVES muss auf eine Datei "
        "zeigen.")


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


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        print("[retarget] %s ist keine Zahl - nutze %s" % (name, default))
        return float(default)


def _looks_like_sequence(data):
    frames = data.get("frames") if isinstance(data, dict) else None
    return isinstance(frames, list) and any(
        isinstance(fd, dict) and fd.get("pose") is not None for fd in frames)


def _looks_like_path(text):
    """Dateinamen auch dann als Pfad melden, wenn sie nicht existieren."""
    return Path(text).suffix.lower() in (".json", ".pev")


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
        # "auto" = beide Arme; "sign" erzwingt den entspannten zweiten Arm.
        self.arm_mode = (os.environ.get("AVATAR_RETARGET_ARM_MODE", "auto")
                         .strip().lower())
        # Wortwiedergabe beginnt/endet in der T-Pose (Ruhepose).
        self.anchor = (os.environ.get("AVATAR_RETARGET_ANCHOR", "rest")
                       .strip().lower() not in ("0", "none", "off", "false", "no"))
        # Satz-Eingabe: Uebersetzung und Pause zwischen den Gebaerden.
        self.gloss_dir = _resolve(os.environ.get("AVATAR_GLOSS_DIR", GLOSS_DIR))
        self.gloss_epoch = int(_env_float("AVATAR_GLOSS_EPOCH",
                                          DEFAULT_GLOSS_EPOCH))
        self.sign_pause = max(0.0, _env_float("AVATAR_RETARGET_SIGN_PAUSE", 0.2))
        self._rig = None
        self._last_path = None
        self._last_data = None
        self._last_label = None
        self._last_sentence = None
        self._frame_cache = {}
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
                if self._last_sentence is not None:
                    self._play_sentence(self._last_sentence)
                    return
                target = self._last_path or self.default_path
                if not target:
                    print(HELP)
                    return
            elif not (_looks_like_path(command) or _resolve(command).exists()):
                from retarget import sentence as sentence_mod
                tokens = sentence_mod.split_tokens(command)
                # Ein Mehrwort-Eintrag der Wortliste ("Auto fahren") hat
                # Vorrang; mehrere freie Woerter werden als Satz geplant.
                # "H A U S" ist dagegen eine Buchstabenfolge, kein Eintrag.
                if (len(tokens) > 1
                        and sentence_mod.normalize(command)
                        not in self._wordlist()):
                    # Gloss -> Wortliste -> Buchstaben.
                    self._play_sentence(command)
                    return
                signs = self._signs_for_text(command)
                if not signs:
                    raise ValueError(
                        "'%s': kein Gebaerden-Eintrag und keine Buchstaben in "
                        "plugins/gsg/wordlist.csv" % command)
                if len(signs) == 1 and not signs[0].spelled:
                    self._play_data(self._load_pev(signs[0].ident), command)
                else:
                    label = " ".join(sign.label for sign in signs)
                    self._play_signs(signs, command,
                                     "Buchstabieren: " + label)
                return
            else:
                target = command
            self._play(target)
        except Exception as exc:                          # noqa: BLE001
            print("[retarget] %s: %s" % (command, exc))

    # ------------------------------------------------------------- intern

    def _wordlist(self):
        """text (gross) -> id aus plugins/gsg/wordlist.csv, einmal geladen."""
        if self._words is None:
            from retarget import sentence as sentence_mod
            if WORDLIST.exists():
                self._words = sentence_mod.load_wordlist(WORDLIST)
            else:
                print("[retarget] Wortliste fehlt: %s" % WORDLIST)
                self._words = {}
        return self._words

    @staticmethod
    def _pev_path(ident):
        """Pfad der vorab extrahierten PEV zu einer Wortlisten-Id."""
        return PEV_DIR / ("%s.pev" % str(ident).strip())

    def _load_pev(self, ident):
        path = self._pev_path(ident)
        if not path.exists():
            raise FileNotFoundError("PEV fehlt: %s" % path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _signs_for_text(self, text):
        """Ein Wort (oder eine Buchstabenfolge) -> Zeichen mit PEV.

        Fehlt die Gebaerde oder die PEV-Datei, wird wie in ``gsg.py``
        Buchstabe fuer Buchstabe gespielt.
        """
        from retarget import sentence as sentence_mod
        words = self._wordlist()
        key = sentence_mod.normalize(text)
        ident = words.get(key)
        if ident is not None and self._pev_path(ident).exists():
            return [sentence_mod.Sign(key, ident)]
        return [sign for sign in sentence_mod.spell_word(key, words)
                if self._pev_path(sign.ident).exists()]

    def _play_signs(self, signs, text, announce):
        """Mehrere Zeichen mit Ruhepose-Pause an den WebSocket schicken."""
        frames = self._sentence_frames(signs)
        self._send(frames, announce)
        self._last_sentence = text
        self._last_data = None
        self._last_path = None
        self._last_label = None

    def _play_sentence(self, text):
        """Mehrere Woerter/Buchstaben: Gloss -> Wortliste -> Buchstabieren.

        Jedes Zeichen muss eine PEV haben; fehlt sie, wird das Wort
        buchstabiert. Ist ein eingegebenes Wort unbekannt (kein Eintrag oder
        keine PEV), darf die Gloss-Uebersetzung es nicht ersetzen: dann werden
        die eingegebenen Woerter benutzt und das unbekannte Wort buchstabiert.
        Damit kommt zur Laufzeit ausschliesslich die Extraktion aus
        ``DataLoaderDGS/Extractor.py`` zum Einsatz.
        """
        from retarget import sentence as sentence_mod

        tokens = sentence_mod.split_tokens(text)
        words = self._wordlist()
        missing = []
        for token in tokens:
            key = sentence_mod.normalize(token)
            ident = words.get(key)
            if ident is None or not self._pev_path(ident).exists():
                missing.append(key)
        if missing:
            print("[retarget] unbekannt: %s - buchstabiere; Gloss wird "
                  "nicht genutzt" % ", ".join(missing))
            gloss_words = None
        else:
            gloss_words = self._gloss(text, tokens)
        plan = sentence_mod.plan(tokens, words, gloss=gloss_words)

        signs = []
        skipped = list(plan.skipped)
        for sign in plan.signs:
            if self._pev_path(sign.ident).exists():
                signs.append(sign)
                continue
            spelled = [item for item in sentence_mod.spell_word(sign.label, words)
                       if self._pev_path(item.ident).exists()]
            if spelled:
                print("[retarget] '%s' (id %s) ohne PEV - buchstabiere %s"
                      % (sign.label, sign.ident,
                         " ".join(item.label for item in spelled)))
                signs.extend(spelled)
            else:
                skipped.append(sign.label)
        if not signs:
            raise ValueError(
                "Satz '%s': kein Wort aus plugins/gsg/wordlist.csv abbildbar"
                % text)

        label = " ".join(sign.label for sign in signs)
        print("[retarget] Satz '%s' -> %s (%s)"
              % (text, label, plan.source))
        if skipped:
            print("[retarget] nicht abbildbar: %s" % ", ".join(skipped))
        self._play_signs(signs, text, "Satz: " + label)

    def _gloss(self, text, tokens):
        """Gloss-Reihenfolge vom Uebersetzungsmodul (oder ``None``)."""
        from retarget import sentence as sentence_mod
        if sentence_mod.is_letter_sequence(tokens):
            return None                         # "H A U S" nicht uebersetzen
        try:
            from retarget import gloss
        except Exception as exc:                          # noqa: BLE001
            print("[retarget] Uebersetzungsmodul nicht ladbar (%s)" % exc)
            return None
        translator = gloss.get_translator(self.gloss_dir, self.gloss_epoch)
        if translator is None:
            return None
        try:
            words = translator.translate(text)
        except Exception as exc:                          # noqa: BLE001
            print("[retarget] Uebersetzung fehlgeschlagen (%s)" % exc)
            return None
        if words:
            print("[retarget] Gloss: %s" % " ".join(w.upper() for w in words))
        return words or None

    def _frames_for_id(self, sign):
        """Frames einer Gebaerden-Einheit aus der PEV, mit Cache pro Id."""
        ident = str(sign.ident)
        if ident in self._frame_cache:
            return self._frame_cache[ident]

        path = self._pev_path(ident)
        if not path.exists():
            raise FileNotFoundError("PEV fehlt: %s" % path)
        with open(path, "r", encoding="utf-8") as f:
            frames = self._frames_from_recording(json.load(f), anchor=True)
        if not frames:
            raise ValueError("keine Frames aus %s" % path)
        if len(self._frame_cache) < 128:
            self._frame_cache[ident] = frames
        return frames

    def _sentence_frames(self, signs):
        """Frames aller Zeichen mit T-Pose-Pause dazwischen verketten."""
        from retarget import webviewer

        segments = []
        for sign in signs:
            frames = self._frames_for_id(sign)
            if frames:
                segments.append(frames)
        if not segments:
            return []

        names = sorted({name for segment in segments for name in segment[0]})
        rest = webviewer.rest_frame(str(self.glb_path), names)
        pause = int(round(self.sign_pause * PLAYBACK_FPS))
        out = []
        for segment in segments:
            if out and pause > 0:
                out.extend(rest for _ in range(pause))
            out.extend(segment)
        return out

    def _rig_data(self):
        if self._rig is None:
            from retarget.rig import Rig
            self._rig = Rig.load(str(self.rig_path))
        return self._rig

    def _frames_from_curves(self, curves, fps):
        from retarget import webviewer
        curves = webviewer.resample(curves, fps, PLAYBACK_FPS)
        return webviewer.frames(self._rig_data(), curves, str(self.glb_path))

    def _frames_from_recording(self, data, anchor=False):
        """Aufnahme einmal durch die ganze Kette.

        Die PEVs sind bereits auf die Gebaerde zugeschnitten; es gibt kein
        zusaetzliches Bewegungsfenster mehr. ``anchor`` blendet Anfang und
        Ende aus der T-Pose (nur fuer Wort-/Satzwiedergabe).
        """
        from retarget import landmarks, webviewer
        from retarget import retarget as pipeline
        from retarget.rtm_adapter import (looks_like_pev, sequence_from_pev,
                                          sequence_from_rtmlib)
        from retarget.sequence import Sequence

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

        curves = webviewer.resample(curves, seq.fps, PLAYBACK_FPS)
        frames = webviewer.frames(self._rig_data(), curves, str(self.glb_path))
        if anchor and self.anchor and frames:
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
        self._send(self._frames_from_recording(data, anchor=True), label)
        self._last_data = data
        self._last_label = label
        self._last_path = None
        self._last_sentence = None

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
        self._last_sentence = None
