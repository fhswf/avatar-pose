"""Pruefungen fuer das Webviewer-Plugin (ohne laufenden WebSocket)."""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from fixtures import rtm_frames
from retarget import rotations as rot
from retarget.retarget import save_curves
from retarget.rig import Rig

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "avatar-pose" / "main"
RIG = ROOT / "retarget" / "data" / "rig_kim.json"
GLB = ROOT / "avatar-pose" / "webviewer" / "media" / "avatar.glb"


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


def test_plugin_spielt_wort_aus_der_gsg_liste(plugin_class, monkeypatch, tmp_path,
                                              capsys):
    from plugins.retarget import retarget as plugin_module
    from retarget import gsg_video, webviewer

    if gsg_video.lookup(plugin_module.WORDLIST, "Auto") is None:
        pytest.skip("gsg-Wortliste mit 'Auto' fehlt")
    # Aufnahme aus dem Cache, damit der Test nicht ins Netz geht.
    rec = {"width": 640, "height": 480, "fps": 30.0, "frame_step": 1,
           "frames": rtm_frames(n=12)}
    (tmp_path / "61.rtmlib.json").write_text(json.dumps(rec), encoding="utf-8")
    monkeypatch.setenv("AVATAR_RETARGET_CACHE", str(tmp_path))

    socket = FakeSocket()
    plugin = plugin_class(socket)
    plugin.input("Auto")
    capsys.readouterr()
    assert plugin._last_data is not None
    # 12 Bilder + je 10 Frames T-Pose-Blende davor/dahinter
    assert len(socket.frames) == 12 + 2 * 10
    assert len(socket.frames[0]["upperarm_r"]["rotation"]) == 4

    # Anfang und Ende stehen in der T-Pose (Ruhepose des glTF).
    rest = webviewer.rest_frame(str(GLB), list(socket.frames[0]))
    for frame in (socket.frames[-1],):
        for name, value in rest.items():
            q = np.asarray(frame[name]["rotation"], float)
            r = np.asarray(value["rotation"], float)
            # Quaternionen sind bis aufs Vorzeichen gleich; das GLB speichert
            # in float32, daher die grosszuegige Toleranz.
            assert min(np.abs(q - r).max(), np.abs(q + r).max()) < 1e-6


def test_plugin_replay_befehl_wiederholt_das_letzte_wort(plugin_class, monkeypatch,
                                                         tmp_path):
    from plugins.retarget import retarget as plugin_module
    from retarget import gsg_video

    if gsg_video.lookup(plugin_module.WORDLIST, "Auto") is None:
        pytest.skip("gsg-Wortliste mit 'Auto' fehlt")
    (tmp_path / "61.rtmlib.json").write_text(
        json.dumps({"width": 640, "height": 480, "fps": 30.0,
                    "frames": rtm_frames(n=12)}), encoding="utf-8")
    monkeypatch.setenv("AVATAR_RETARGET_CACHE", str(tmp_path))

    socket = FakeSocket()
    plugin = plugin_class(socket)
    plugin.input("Auto")
    first = len(socket.frames)
    assert first == 12 + 2 * 10
    plugin.input("replay")
    assert len(socket.frames) == 2 * first


def test_plugin_meldet_unbekanntes_wort(plugin_class, capsys):
    plugin = plugin_class(FakeSocket())
    plugin.input("gibt-es-nicht-als-wort")
    out = capsys.readouterr().out
    assert "Datei nicht gefunden" in out


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
