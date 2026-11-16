"""Pruefungen fuer Filter, Haende und Metriken (ohne Blender und Video)."""

import math

import numpy as np
import pytest

from retarget import filters, hands, metrics
from retarget.sequence import Sequence


def rot_x(deg):
    a = math.radians(deg) * 0.5
    return np.array([math.cos(a), math.sin(a), 0.0, 0.0])


def test_savgol_haelt_gerade_und_verschiebt_nicht():
    x = np.linspace(0.0, 10.0, 200)
    out = filters.savgol(x.copy(), 21)
    assert np.allclose(out[10:-10], x[10:-10], atol=1e-9)


def test_hampel_ersetzt_einzelnen_ausreisser():
    v = np.zeros(50)
    v[25] = 5.0
    out = filters.hampel(v.copy(), floor=0.01)
    assert abs(out[25]) < 1e-9
    assert np.allclose(np.delete(out, 25), 0.0)


def test_qangle_kennt_das_vorzeichen_nicht():
    a, b = rot_x(0), rot_x(90)
    assert math.degrees(filters.qangle(a, b)) == pytest.approx(90.0, abs=1e-6)
    assert math.degrees(filters.qangle(a, -b)) == pytest.approx(90.0, abs=1e-6)


def test_jerk_ist_null_bei_gleichmaessiger_drehung():
    qs = np.array([rot_x(2.0 * t) for t in range(60)])
    r = metrics.jerk(qs)
    assert r["zappel"] == pytest.approx(0.0, abs=1e-6)
    assert r["median_step"] == pytest.approx(2.0, abs=1e-6)
    assert r["jumps"] == 0


def test_jerk_zaehlt_spruenge():
    qs = np.array([rot_x(0.0) for _ in range(20)])
    qs[10] = rot_x(90.0)
    r = metrics.jerk(qs, jump_deg=30.0)
    assert r["jumps"] == 2
    assert r["zappel"] > 25.0


def test_quat_filter_entfernt_den_ausreisser():
    qs = np.array([rot_x(0.0) for _ in range(40)])
    qs[20] = rot_x(60.0)
    valid = np.ones(len(qs), bool)
    out = filters.filter_quat_series(qs.copy(), valid, 30.0, window=9, clean=True)
    assert math.degrees(filters.qangle(out[20], rot_x(0.0))) < 5.0


def test_gestreckter_finger_hat_beugung_null():
    world = [[0.0, 0.0, 0.0, 1.0]] * 21
    world[0] = [0.0, 0.0, 0.0, 1.0]
    world[5] = [0.03, 0.09, 0.0, 1.0]
    world[9] = [0.0, 0.09, 0.0, 1.0]
    world[17] = [-0.03, 0.09, 0.0, 1.0]
    for k, y in ((6, 0.13), (7, 0.16), (8, 0.18)):
        world[k] = [0.03, y, 0.0, 1.0]
    ang = hands.finger_joint_angles(world, "l")["index"]
    assert all(abs(math.degrees(c)) < 1.0 for c in ang["curl"])


def test_sequence_rechnet_frame_step_in_die_bildrate():
    seq = Sequence({"fps": 30.0, "frame_step": 2, "width": 100, "height": 200,
                    "frames": []})
    assert seq.fps == pytest.approx(15.0)
    assert seq.aspect == pytest.approx(0.5)


def _reach_hand(reach, palm=0.09):
    """Hand mit vorgegebenem Spitzenabstand (in Handflaechen) zur Handwurzel."""
    world = [[0.0, 0.0, 0.0, 1.0] for _ in range(21)]
    world[9] = [0.0, palm, 0.0, 1.0]
    for x, tip in ((0.03, 4), (-0.02, 8), (0.0, 12), (0.02, 16), (0.03, 20)):
        world[tip] = [x, reach * palm, 0.0, 1.0]
    return world


def test_faust_hat_kurzen_reach_und_offene_hand_langen():
    assert hands.fist_reach(_reach_hand(0.8)) < 1.0
    assert hands.fist_reach(_reach_hand(2.6)) > 2.4
    assert hands.fist_closure(0.8) == pytest.approx(1.0)
    assert hands.fist_closure(2.6) == pytest.approx(0.0)
    assert 0.0 < hands.fist_closure(1.8) < 1.0


def test_finger_straight_unterscheidet_gestreckt_und_eingerollt():
    world = [[0.0, 0.0, 0.0, 1.0] for _ in range(21)]
    world[5] = [0.03, 0.09, 0.0, 1.0]
    for k, y in ((6, 0.13), (7, 0.16), (8, 0.18)):
        world[k] = [0.03, y, 0.0, 1.0]
    gestreckt = hands.finger_straight(world)["index"]
    assert gestreckt == pytest.approx(1.0, abs=1e-6)

    # Spitze zurueck auf den Mittelhandknochen: stark eingerollt.
    world[6] = [0.03, 0.12, 0.0, 1.0]
    world[7] = [0.02, 0.10, 0.0, 1.0]
    world[8] = [0.02, 0.09, 0.0, 1.0]
    eingerollt = hands.finger_straight(world)["index"]
    assert eingerollt < 0.8


def test_faust_huellkurve_schliesst_schnell_und_oeffnet_langsam():
    raw = np.array([0.0] * 5 + [1.0] * 5 + [0.0] * 30)
    out = hands.fist_envelope(raw, 30.0)
    assert out[9] > 0.9          # nach fuenf Bildern fast geschlossen
    assert out[15] > 0.6         # Oeffnen braucht laenger als Schliessen
    assert out[25] < 0.5
    assert out[25] < out[15]
