"""Pruefungen fuer die Bruecke Retarget-Kurven -> glTF/Webviewer."""

import math
from pathlib import Path

import numpy as np
import pytest

from retarget import filters
from retarget import rotations as rot
from retarget import webviewer
from retarget.rig import Rig

PROJECT = Path(__file__).resolve().parents[2]
RIG = PROJECT / "retarget" / "data" / "rig_kim.json"
GLB = PROJECT / "avatar-pose" / "webviewer" / "media" / "avatar.glb"


@pytest.fixture(scope="module")
def rig():
    return Rig.load(str(RIG))


def test_identity_kurven_geben_die_gltf_ruhelage(rig):
    curves = {name: np.tile(rot.IDENTITY, (2, 1)) for name in rig.rest}
    frames = webviewer.frames(rig, curves, str(GLB))
    rest = webviewer.GltfRest(str(GLB))
    assert len(frames) == 2
    worst = 0.0
    for name, payload in frames[0].items():
        q = rot.from_three(payload["rotation"])
        delta = rot.quat_angle(rot.positive(rot.quat_mul(
            q, rot.quat_conj(rest.local_quat(name)))))
        worst = max(worst, math.degrees(delta))
    assert worst < 0.1


def test_alle_rig_knochen_finden_einen_gltf_node(rig):
    rest = webviewer.GltfRest(str(GLB))
    missing = sorted(set(rig.rest) - set(rest.nodes))
    assert missing == []


def test_drehung_landet_auf_derselben_knochenrichtung(rig):
    turn = rot.axis_angle((0.0, 0.0, 1.0), math.radians(40.0))
    curves = {name: np.tile(rot.IDENTITY, (1, 1)) for name in rig.rest}
    curves["upperarm_l"] = np.array([turn])
    mapped = webviewer.node_curves(rig, curves, str(GLB))

    want = rig.forward(curves, 0)
    global_rot = {}
    for name in rig.order:
        if name not in mapped:
            continue
        parent = rig.parent.get(name)
        local = rot.quat_to_mat(mapped[name][0])
        global_rot[name] = (local if not parent or parent not in global_rot
                            else global_rot[parent] @ local)
    wrong = []
    for name, mat in global_rot.items():
        a = webviewer.COORD @ want[name][:, 1]
        b = mat[:, 1]
        err = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b))))))
        if err > 0.01:
            wrong.append((name, err))
    assert wrong == []


def test_drei_js_reihenfolge_und_laenge(rig):
    curves = {name: np.tile(rot.IDENTITY, (4, 1)) for name in rig.rest}
    frames = webviewer.frames(rig, curves, str(GLB))
    assert len(frames) == 4
    for frame in frames:
        for payload in frame.values():
            assert len(payload["rotation"]) == 4
            assert abs(float(np.linalg.norm(payload["rotation"])) - 1.0) < 1e-9


def test_resample_haelt_konstante_kurven_und_die_enden():
    curves = {"bone": np.tile(rot.IDENTITY, (25, 1))}
    out = webviewer.resample(curves, 25.0, 30.0)
    assert len(out["bone"]) == 30
    assert np.allclose(out["bone"], rot.IDENTITY, atol=1e-9)

    start = rot.axis_angle((0.0, 0.0, 1.0), 0.0)
    end = rot.axis_angle((0.0, 0.0, 1.0), math.radians(25.0))
    ramp = np.array([rot.axis_angle((0.0, 0.0, 1.0), math.radians(i))
                     for i in range(26)])
    out = webviewer.resample({"bone": ramp}, 25.0, 50.0)
    assert len(out["bone"]) == 51
    assert filters.qangle(out["bone"][0], start) < 1e-5
    assert filters.qangle(out["bone"][-1], end) < 1e-5
    assert filters.qangle(out["bone"][2], ramp[1]) < 1e-5


def test_resample_ohne_aenderung_ist_eine_kopie():
    curves = {"bone": np.tile(rot.IDENTITY, (3, 1))}
    out = webviewer.resample(curves, 30.0, 30.0)
    assert np.allclose(out["bone"], curves["bone"])


def test_rest_frame_ist_die_lokale_gltf_ruherotation():
    rest = webviewer.GltfRest(str(GLB))
    frame = webviewer.rest_frame(str(GLB), ["spine_01", "upperarm_l"])
    assert sorted(frame) == ["spine_01", "upperarm_l"]
    for name, payload in frame.items():
        q = rot.from_three(payload["rotation"])
        r = rest.local_quat(name)
        assert min(np.abs(q - r).max(), np.abs(q + r).max()) < 1e-12


def test_anchor_rest_beginnt_und_endet_in_der_tpose(rig):
    rest = webviewer.rest_frame(str(GLB), list(rig.rest))
    turn = rot.axis_angle((0.0, 0.0, 1.0), math.radians(40.0))
    frame = {name: {"rotation": rot.to_three(rot.IDENTITY).tolist()}
             for name in rig.rest}
    frame["upperarm_l"] = {"rotation": rot.to_three(turn).tolist()}

    out = webviewer.anchor_rest([frame, frame], fps=30.0, fade_sec=0.2,
                                rest=rest)
    n = 6
    assert len(out) == 2 + 2 * n
    # Die Bewegung bleibt unveraendert in der Mitte.
    assert out[n:n + 2] == [frame, frame]
    # Der letzte Frame ist exakt die Ruhepose (bis aufs Quaternion-Vorzeichen).
    for name, payload in rest.items():
        q = np.asarray(out[-1][name]["rotation"], float)
        r = np.asarray(payload["rotation"], float)
        assert min(np.abs(q - r).max(), np.abs(q + r).max()) < 1e-6


def test_kurven_datei_geht_hin_und_zurueck(rig, tmp_path):
    curves = {name: np.tile(rot.IDENTITY, (3, 1)) for name in rig.rest}
    from retarget.retarget import save_curves
    path = tmp_path / "kurven.json"
    save_curves(curves, [0, 1, 2], str(path), 30.0)
    loaded, fps, frames = webviewer.load_curves(str(path))
    assert fps == pytest.approx(30.0)
    assert frames == [0, 1, 2]
    for name, qs in loaded.items():
        assert np.allclose(qs, curves[name])
