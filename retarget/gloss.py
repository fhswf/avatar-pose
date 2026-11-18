"""Deutsch -> DGS-Gloss mit dem vorhandenen ``text_to_gloss``-Modell.

Das Modul liegt unter ``avatar-pose/text_to_gloss`` und besteht aus flachen
Skripten (``from constant import ...``). Es wird deshalb beim ersten Satz
einmal geladen, mit dem Projektordner als Arbeitsverzeichnis; danach werden
die Modulnamen wieder aus ``sys.modules`` entfernt, damit sie nicht mit
``plugins/gsg`` kollidieren.

Die Uebersetzung ist nur ein Vorschlag: das Modell wurde auf Wettertexten
trainiert (``Text2Gloss.csv``) und liefert ausserhalb dieses Bereichs oft
unbrauchbare Glossen. Der Aufrufer muss deshalb immer auf die Wortliste
zurueckfallen koennen.

Standard-Epoche ist 20: auf 300 Testsätzen aus ``data/test.txt`` erreicht sie
den besten BLEU-Wert (0.2108; Epoche 40/50: 0.2087, Epoche 10: 0.1947).
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve()
PROJECT_DIR = HERE.parents[1]                       # Repository-Wurzel
DEFAULT_DIR = PROJECT_DIR / "avatar-pose" / "text_to_gloss"

HIDDEN_SIZE = 512
DEFAULT_EPOCH = 20
# Reihenfolge wie in text_to_gloss/main.py; "models" ist ein Namensraum-Paket.
_FLAT_MODULES = ("constant", "lang", "preprocessing", "dataset", "evaluation",
                 "data_processing")
_TTG_MODULES = _FLAT_MODULES + ("models", "models.encoder", "models.decoder",
                                "models.attention")


class GlossError(RuntimeError):
    """Uebersetzungsmodul nicht ladbar oder Modell unbrauchbar."""


def disabled():
    """``AVATAR_GLOSS=0`` schaltet die Uebersetzung ab."""
    return os.environ.get("AVATAR_GLOSS", "1").strip().lower() in (
        "0", "off", "false", "no", "none")


def _import_ttg(directory):
    """Module aus ``directory`` laden, ohne globale Namen zu hinterlassen."""
    import importlib

    directory = str(Path(directory))
    if not Path(directory, "models", "encoder.py").exists():
        raise GlossError("text_to_gloss nicht gefunden: %s" % directory)

    saved = {name: sys.modules.pop(name) for name in _TTG_MODULES
             if name in sys.modules}
    saved_path = list(sys.path)
    try:
        sys.path.insert(0, directory)
        importlib.invalidate_caches()
        namespace = type(sys)("text_to_gloss_vendor")
        for name in _FLAT_MODULES:
            setattr(namespace, name, importlib.import_module(name))
        namespace.encoder = importlib.import_module("models.encoder")
        namespace.decoder = importlib.import_module("models.decoder")
        return namespace
    finally:
        sys.path[:] = saved_path
        for name in _TTG_MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


class GlossTranslator:
    """Uebersetzt deutsche Saetze in DGS-Gloss (Liste klein geschriebener Woerter)."""

    def __init__(self, directory=None, epoch=None, device=None):
        self.directory = Path(directory or os.environ.get("AVATAR_GLOSS_DIR")
                              or DEFAULT_DIR)
        self.epoch = int(epoch or os.environ.get("AVATAR_GLOSS_EPOCH")
                         or DEFAULT_EPOCH)
        self.device = (device or os.environ.get("AVATAR_GLOSS_DEVICE")
                       or "cpu").strip()
        self._modules = None
        self._encoder = None
        self._decoder = None
        self._input_lang = None
        self._output_lang = None
        self._cache = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------- oeffentlich

    def available(self):
        """Lazy laden; ``False`` statt Ausnahme, wenn etwas fehlt."""
        try:
            self._ensure_loaded()
            return True
        except Exception as exc:                          # noqa: BLE001
            print("[retarget] Uebersetzungsmodul nicht verfuegbar (%s) - "
                  "nutze Wortliste" % exc)
            return False

    def translate(self, text):
        """Gloss-Woerter fuer ``text`` (klein) oder ``[]``."""
        sentence = " ".join((text or "").split())
        if not sentence:
            return []
        if sentence.lower() in self._cache:
            return list(self._cache[sentence.lower()])

        self._ensure_loaded()
        preprocessing = self._modules.preprocessing
        lemmatize = self._modules.data_processing.lemmatize_verbs
        # Wie beim Training: erst Lemmatisieren, dann normalisieren.
        prepared = preprocessing.normalize_string(lemmatize(sentence))
        if not prepared:
            return []

        words = self._modules.evaluation.evaluate(
            self._encoder, self._decoder, prepared, self._input_lang,
            self._output_lang, self.device)
        special = {"SOS", "EOS", "UNK", "PAD"}
        words = [w for w in words if w and w.upper() not in special]
        self._cache[sentence.lower()] = list(words)
        return list(words)

    # ------------------------------------------------------------------ intern

    def _ensure_loaded(self):
        if self._encoder is not None:
            return
        with self._lock:
            if self._encoder is not None:
                return
            import torch

            encoder_path = self.directory / "data" / ("encoder_epoch%d.pt"
                                                      % self.epoch)
            decoder_path = self.directory / "data" / ("decoder_epoch%d.pt"
                                                      % self.epoch)
            for path in (encoder_path, decoder_path):
                if not path.exists():
                    raise GlossError("Checkpoint fehlt: %s" % path)

            modules = _import_ttg(self.directory)
            # prepare_data liest relativ zu "data" und druckt Emojis, die auf
            # einer cp1252-Konsole sonst einen UnicodeError ausloesen.
            cwd = os.getcwd()
            try:
                os.chdir(self.directory)
                sink = io.StringIO()
                with contextlib.redirect_stdout(sink):
                    input_lang, output_lang, _ = modules.preprocessing.prepare_data(
                        "dgs", "de")
            finally:
                os.chdir(cwd)

            encoder = modules.encoder.EncoderRNN(input_lang.n_words, HIDDEN_SIZE)
            decoder = modules.decoder.AttnDecoderRNN(HIDDEN_SIZE,
                                                     output_lang.n_words)
            encoder.load_state_dict(torch.load(str(encoder_path),
                                               map_location="cpu"))
            decoder.load_state_dict(torch.load(str(decoder_path),
                                               map_location="cpu"))
            encoder.eval()
            decoder.eval()
            encoder.to(self.device)
            decoder.to(self.device)

            self._modules = modules
            self._encoder = encoder
            self._decoder = decoder
            self._input_lang = input_lang
            self._output_lang = output_lang
            print("[retarget] Uebersetzungsmodul geladen (Epoche %d, "
                  "Vokabular %d -> %d)"
                  % (self.epoch, input_lang.n_words, output_lang.n_words))


_TRANSLATORS = {}


def get_translator(directory=None, epoch=None, device=None):
    """Gecachter Uebersetzer oder ``None``, wenn nicht nutzbar/abgeschaltet."""
    if disabled():
        return None
    translator = GlossTranslator(directory, epoch, device)
    key = (str(translator.directory), translator.epoch, translator.device)
    if key not in _TRANSLATORS:
        _TRANSLATORS[key] = translator
    cached = _TRANSLATORS[key]
    return cached if cached.available() else None
