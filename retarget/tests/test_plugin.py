"""Pruefungen fuer das Webviewer-Plugin (ohne laufenden WebSocket).

Die Plugin-Tests benutzen ausschliesslich kuenstliche PEVs: zur Laufzeit
duerfen keine Videos geladen und keine Keypoints berechnet werden.
"""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from fixtures import pev_frames, rtm_frames
from retarget import rotations as rot
from retarget.retarget import save_curves
from retarget.rig import Rig

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "avatar-pose" / "main"
RIG = ROOT / "retarget" / "data" / "rig_kim.json"
GLB = ROOT / "avatar-pose" / "webviewer" / "media" / "avatar.glb"

# 12 PEV-Frames bei 25 fps -> 14 Frames bei 30 fps, dazu je 10 T-Pose-Frames.
GESTURE = 14 + 2 * 10
PAUSE = int(round(0.2 * 30))


class FakeSocket:
    def __init__(self):
        self.frames = []

    def get_t_pose(self):
        return {}

    def add_send_object(self, obj):
        self.frames.append(obj)


@pytest.fixture(scope="module")
def plugin_class():
    if str(MAIN) not in sys.path:
        sys.path.insert(0, str(MAIN))
    module = importlib.import_module("plugins.retarget.retarget")
    return module.retarget


def fake_pev_dir(tmp_path, idents, frames=12):
    """PEV-Dateien fuer die gegebenen Ids in einem temporaeren Ordner."""
    target = tmp_path / "extracted"
    target.mkdir(exist_ok=True)
    data = {"fps": 25.0, "frames": pev_frames(frames)}
    for ident in idents:
        (target / ("%s.pev" % ident)).write_text(
            json.dumps(data), encoding="utf-8")
    return target


def test_plugin_findet_rig_und_glb(plugin_class, monkeypatch):
    monkeypatch.delenv("AVATAR_RETARGET_RIG", raising=False)
    monkeypatch.delenv("AVATAR_RETARGET_GLB", raising=False)
    plugin = plugin_class(FakeSocket())
    assert Path(plugin.rig_path) == RIG
    assert Path(plugin.glb_path) == GLB


def test_plugin_spielt_kurven_und_rohdaten(plugin_class, monkeypatch, tmp_path):
    rig = Rig.load(str(RIG))
    curves = {name: np.tile(rot.IDENTITY, (13, 1)) for name in rig.rest}
    curves_path = tmp_path / "kurven.json"
    save_curves(curves, list(range(1, 14)), str(curves_path), 25.0)

    monkeypatch.setenv("AVATAR_RETARGET_CURVES", str(curves_path))
    socket = FakeSocket()
    plugin = plugin_class(socket)
    plugin.input("")
    # 13 Bilder bei 25 fps auf 30 fps: (13-1)*30/25 = 14,4 -> 15 Bilder
    assert len(socket.frames) == 15
    assert set(socket.frames[0]) == set(rig.rest)
    assert all(len(f["upperarm_l"]["rotation"]) == 4 for f in socket.frames)

    raw_path = tmp_path / "rtmlib.json"
    raw_path.write_text(json.dumps({"fps": 25.0, "frames": rtm_frames(n=6)}),
                        encoding="utf-8")
    raw_socket = FakeSocket()
    raw_plugin = plugin_class(raw_socket)
    raw_plugin.input(str(raw_path))
    assert len(raw_socket.frames) == 7
    assert set(raw_socket.frames[0]) <= set(rig.rest)
    assert "upperarm_l" in raw_socket.frames[0]


def test_plugin_meldet_fehlende_datei_ohne_absturz(plugin_class, capsys):
    plugin = plugin_class(FakeSocket())
    plugin.input("gibt-es-nicht.json")
    out = capsys.readouterr().out
    assert "gibt-es-nicht.json" in out
    assert "Datei nicht gefunden" in out


def test_plugin_spielt_wort_aus_der_gsg_liste(plugin_class, monkeypatch,
                                               tmp_path, capsys):
    from plugins.retarget import retarget as plugin_module
    from retarget import webviewer

    ident = plugin_class(FakeSocket())._wordlist().get("AUTO")
    if ident is None:
        pytest.skip("gsg-Wortliste mit 'Auto' fehlt")
    monkeypatch.setattr(plugin_module, "PEV_DIR",
                        fake_pev_dir(tmp_path, [ident]))

    socket = FakeSocket()
    plugin = plugin_class(socket)
    plugin.input("Auto")
    capsys.readouterr()
    assert plugin._last_data is not None
    assert len(socket.frames) == GESTURE
    assert len(socket.frames[0]["upperarm_r"]["rotation"]) == 4

    # Anfang und Ende stehen in der T-Pose (Ruhepose des glTF).
    rest = webviewer.rest_frame(str(GLB), list(socket.frames[0]))
    for name, value in rest.items():
        q = np.asarray(socket.frames[-1][name]["rotation"], float)
        r = np.asarray(value["rotation"], float)
        # Quaternionen sind bis aufs Vorzeichen gleich; das GLB speichert
        # in float32, daher die grosszuegige Toleranz.
        assert min(np.abs(q - r).max(), np.abs(q + r).max()) < 1e-6


def test_plugin_replay_befehl_wiederholt_das_letzte_wort(plugin_class,
                                                         monkeypatch, tmp_path):
    from plugins.retarget import retarget as plugin_module

    ident = plugin_class(FakeSocket())._wordlist().get("AUTO")
    if ident is None:
        pytest.skip("gsg-Wortliste mit 'Auto' fehlt")
    monkeypatch.setattr(plugin_module, "PEV_DIR",
                        fake_pev_dir(tmp_path, [ident]))

    socket = FakeSocket()
    plugin = plugin_class(socket)
    plugin.input("Auto")
    first = len(socket.frames)
    assert first == GESTURE
    plugin.input("replay")
    assert len(socket.frames) == 2 * first


def test_plugin_buchstabiert_unbekanntes_wort(plugin_class, monkeypatch,
                                               tmp_path, capsys):
    from plugins.retarget import retarget as plugin_module

    words = plugin_class(FakeSocket())._wordlist()
    if "QUX" in words or not all(char in words for char in "QUX"):
        pytest.skip("Buchstaben Q/U/X oder 'Qux' in der Wortliste vorhanden")
    monkeypatch.setattr(plugin_module, "PEV_DIR",
                        fake_pev_dir(tmp_path, [words[c] for c in "QUX"]))

    socket = FakeSocket()
    plugin = plugin_class(socket)
    plugin.input("Qux")
    out = capsys.readouterr().out
    assert "Buchstabieren: Q U X" in out
    assert len(socket.frames) == 3 * GESTURE + 2 * PAUSE


def test_plugin_buchstabiert_wenn_pev_fehlt(plugin_class, monkeypatch,
                                             tmp_path, capsys):
    """Fehlt die Gebaerden-PEV, wird das Wort Buchstabe fuer Buchstabe gespielt."""
    from plugins.retarget import retarget as plugin_module

    words = plugin_class(FakeSocket())._wordlist()
    letters = "AUTO"
    if not all(char in words for char in letters):
        pytest.skip("Einzelbuchstaben fehlen in der Wortliste")
    # Bewusst keine PEV fuer 'AUTO' (id 61) anlegen.
    monkeypatch.setattr(plugin_module, "PEV_DIR",
                        fake_pev_dir(tmp_path, [words[c] for c in letters]))

    socket = FakeSocket()
    plugin = plugin_class(socket)
    plugin.input("Auto")
    out = capsys.readouterr().out
    assert "Buchstabieren: A U T O" in out
    assert len(socket.frames) == 4 * GESTURE + 3 * PAUSE


def test_plugin_ignoriert_auto_als_standardpfad(plugin_class, monkeypatch, capsys):
    monkeypatch.setenv("AVATAR_RETARGET_CURVES", "auto")
    plugin = plugin_class(FakeSocket())
    assert plugin.default_path is None
    assert "kein Dateipfad" in capsys.readouterr().out

    plugin.input("")
    assert "replay" in capsys.readouterr().out


def test_plugin_ohne_standarddatei_zeigt_hilfe(plugin_class, monkeypatch, capsys):
    monkeypatch.delenv("AVATAR_RETARGET_CURVES", raising=False)
    plugin = plugin_class(FakeSocket())
    plugin.input("")
    assert "replay" in capsys.readouterr().out


def test_plugin_mehrwort_eintrag_hat_vorrang(plugin_class, monkeypatch,
                                              tmp_path, capsys):
    from plugins.retarget import retarget as plugin_module

    ident = plugin_class(FakeSocket())._wordlist().get("AUTO FAHREN")
    if ident is None:
        pytest.skip("Mehrwort-Eintrag 'Auto fahren' fehlt")
    monkeypatch.setattr(plugin_module, "PEV_DIR",
                        fake_pev_dir(tmp_path, [ident]))
    monkeypatch.setenv("AVATAR_GLOSS", "0")

    socket = FakeSocket()
    plugin = plugin_class(socket)
    plugin.input("Auto fahren")
    out = capsys.readouterr().out
    assert "Satz" not in out
    assert plugin._last_data is not None
    assert len(socket.frames) == GESTURE


def test_plugin_spielt_satz_aus_mehreren_woertern(plugin_class, monkeypatch,
                                                   tmp_path, capsys):
    from plugins.retarget import retarget as plugin_module

    words = plugin_class(FakeSocket())._wordlist()
    if not all(key in words for key in ("AUTO", "HUND")):
        pytest.skip("gsg-Wortliste ohne Auto/Hund")
    idents = [words["AUTO"], words["HUND"]]
    monkeypatch.setattr(plugin_module, "PEV_DIR",
                        fake_pev_dir(tmp_path, idents))
    monkeypatch.setenv("AVATAR_GLOSS", "0")

    socket = FakeSocket()
    plugin = plugin_class(socket)
    plugin.input("Auto Hund")
    out = capsys.readouterr().out
    assert "Satz" in out and "(words)" in out
    assert len(socket.frames) == 2 * GESTURE + PAUSE

    played = len(socket.frames)
    plugin.input("replay")
    assert len(socket.frames) == 2 * played


def test_plugin_buchstabiert_unbekanntes_wort_trotz_gloss(plugin_class,
                                                           monkeypatch,
                                                           tmp_path, capsys):
    """Ein unbekanntes Wort darf nicht durch die Gloss-Uebersetzung entfallen."""
    from plugins.retarget import retarget as plugin_module

    words = plugin_class(FakeSocket())._wordlist()
    if "QUX" in words or not all(char in words for char in "QUX"):
        pytest.skip("Buchstaben Q/U/X oder 'Qux' in der Wortliste vorhanden")
    monkeypatch.setattr(
        plugin_module, "PEV_DIR",
        fake_pev_dir(tmp_path, [words["AUTO"]] + [words[c] for c in "QUX"]))

    socket = FakeSocket()
    plugin = plugin_class(socket)
    # Das Modell wuerde 'Auto Qux' auf 'AUTO' reduzieren; das ist nicht
    # zulaessig, solange 'Qux' unbekannt ist.
    monkeypatch.setattr(plugin, "_gloss", lambda text, tokens: ["AUTO"])
    plugin.input("Auto Qux")
    out = capsys.readouterr().out
    assert "unbekannt: QUX" in out
    assert "(words)" in out
    assert "AUTO Q U X" in out
    # AUTO + drei Buchstaben, dazwischen jeweils eine Ruhepose-Pause.
    assert len(socket.frames) == 4 * GESTURE + 3 * PAUSE


def test_plugin_spielt_buchstabenfolge(plugin_class, monkeypatch, tmp_path,
                                        capsys):
    from plugins.retarget import retarget as plugin_module

    words = plugin_class(FakeSocket())._wordlist()
    if not all(char in words for char in "HAUS"):
        pytest.skip("Einzelbuchstaben fehlen in der Wortliste")
    monkeypatch.setattr(plugin_module, "PEV_DIR",
                        fake_pev_dir(tmp_path, [words[c] for c in "HAUS"]))
    monkeypatch.setenv("AVATAR_GLOSS", "0")

    socket = FakeSocket()
    plugin = plugin_class(socket)
    plugin.input("H A U S")
    out = capsys.readouterr().out
    assert "Satz" in out and "(letters)" in out
    assert len(socket.frames) == 4 * GESTURE + 3 * PAUSE
