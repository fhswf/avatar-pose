"""Pruefungen fuer Rig und Retargeting."""

import math

import numpy as np
import pytest

from fixtures import toy_rig, toy_sequence
from retarget import Sequence, landmarks, retarget
from retarget import rotations as rot
from retarget.config import (FINGER_CURL_MAX, FINGER_MAP, NECK_SHARE,
                             RELAXED_ARM_UP)
from retarget.rig import Rig

OPTS = {"upper_only": True, "arm_depth": 1.0, "spine_depth": 0.2,
        "min_conf": 0.5, "smooth_sec": 0.2, "clean": True}


def winkel(a, b):
    return math.degrees(rot.quat_angle(rot.positive(rot.mat_to_quat(a @ b.T))))


@pytest.fixture(scope="module")
def geloest():
    rig = Rig(toy_rig()["bones"], "toy")
    seq = Sequence(toy_sequence(n=60, elbow=(0.78, 0.45), wrist=(0.90, 0.25),
                                head_tilt=0.06))
    landmarks.process(seq)
    targets, _n, _h = retarget.build_targets(rig, seq, OPTS)
    return rig, seq, targets, retarget.solve(rig, seq)


def test_matrix_und_quaternion_gehen_hin_und_zurueck():
    rng = np.random.default_rng(7)
    for _ in range(50):
        v = rng.normal(size=4)
        q = rot.positive(v / np.linalg.norm(v))
        M = rot.quat_to_mat(q)
        assert np.allclose(M @ M.T, np.eye(3), atol=1e-12)
        assert np.allclose(q, rot.positive(rot.mat_to_quat(M)), atol=1e-9)


def test_rotation_difference_legt_a_auf_b():
    rng = np.random.default_rng(3)
    for _ in range(50):
        a, b = rng.normal(size=3), rng.normal(size=3)
        got = rot.quat_to_mat(rot.rotation_difference(a, b)) @ (a / np.linalg.norm(a))
        assert np.allclose(got, b / np.linalg.norm(b), atol=1e-9)
    q = rot.rotation_difference((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
    assert np.allclose(rot.quat_to_mat(q) @ (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
                       atol=1e-9)


def test_rig_misst_seine_ruhelage_ab():
    rig = Rig(toy_rig()["bones"], "toy")
    assert rig.find("spine") == "spine_01"
    assert np.allclose(rig.direction({}, "upperarm_l"), (1.0, 0.0, 0.0), atol=1e-9)
    assert np.allclose(rig.direction({}, "head"), (0.0, 0.0, 1.0), atol=1e-9)
    assert len(rig.wrist_bones("l")) == 5
    assert set(rig.rest_palms) == {"l", "r"}
    assert "spine" in rig.rest_frames and "hand_l" in rig.rest_frames


def test_aim_richtet_die_knochenachse_auf_das_ziel(geloest):
    rig, _seq, targets, curves = geloest
    for name in ("upperarm_l", "lowerarm_l", "upperarm_r", "lowerarm_r"):
        for i in (5, 20, 40, 55):
            want = rot.unit(targets[name]["along"][i])
            got = rig.direction(curves, name, i)
            assert float(np.dot(want, got)) > math.cos(math.radians(0.01))


def test_kopf_und_hals_treffen_ihre_weltdrehung(geloest):
    rig, _seq, targets, curves = geloest
    for i in (10, 25, 40):
        M = rig.forward(curves, i)
        for bname in ("head", "neck_01"):
            want = rot.quat_to_mat(targets[bname]["quat"][i]) @ rig.rest[bname]
            assert winkel(M[bname], want) < 0.01


def test_hals_geht_einen_bruchteil_der_kopfdrehung_mit(geloest):
    _rig, _seq, targets, _curves = geloest
    bewegt = 0
    for i in range(len(targets["head"]["quat"])):
        kopf = rot.quat_angle(rot.positive(targets["head"]["quat"][i]))
        if kopf < math.radians(2.0):
            continue
        hals = rot.quat_angle(rot.positive(targets["neck_01"]["quat"][i]))
        assert hals / kopf == pytest.approx(NECK_SHARE, abs=1e-4)
        bewegt += 1
    assert bewegt > 10


def test_alle_fuenf_mittelhandknochen_teilen_eine_drehung(geloest):
    rig, _seq, _targets, curves = geloest
    for side in ("l", "r"):
        bones = rig.wrist_bones(side)
        assert bones
        for i in (12, 30, 50):
            Ks = [rig.rest[b] @ rot.quat_to_mat(curves[b][i]) @ rig.rest[b].T
                  for b in bones]
            for K in Ks[1:]:
                assert winkel(K, Ks[0]) < 1e-3


def test_offene_hand_streckt_die_langen_finger():
    offen = retarget.finger_quaternions(None, "l", 4, preset="open_hand")
    locker = retarget.finger_quaternions(None, "l", 4, preset="relaxed")
    for name, qs in offen.items():
        if name.startswith("thumb"):
            continue
        grad = math.degrees(rot.quat_angle(rot.positive(qs[0])))
        if "_01_" in name:
            assert grad < 20.0
        else:
            assert grad == pytest.approx(0.0, abs=1e-9)
            assert rot.quat_angle(rot.positive(locker[name][0])) > 0.0


def test_entspannte_hand_rollt_die_finger_ein():
    fq = retarget.finger_quaternions(None, "l", 4, preset="relaxed")
    assert math.degrees(rot.quat_angle(rot.positive(fq["pinky_03_l"][0]))) > 30.0


def _finger_data(fist_values, curls=0.0, straight=None):
    n = len(fist_values)
    straight = straight or {}
    return {"l": {
        "curl": {j: np.full((n, 3), curls) for j, _, _ in FINGER_MAP},
        "spread": {j: np.zeros(n) for j, _, _ in FINGER_MAP},
        "straight": {j: np.full(n, float(straight.get(j, 0.0)))
                     for j, _, _ in FINGER_MAP},
        "fist": np.asarray(fist_values, float),
    }}


def test_faust_krallt_alle_fingerglieder():
    fq = retarget.finger_quaternions(_finger_data([1.0, 1.0]), "l", 2)
    for jname, cmax in FINGER_CURL_MAX.items():
        for k in (1, 2, 3):
            deg = math.degrees(rot.quat_angle(
                rot.positive(fq["%s_%02d_l" % (jname, k)][0])))
            assert deg == pytest.approx(cmax[k - 1], abs=1e-4)


def test_gestreckter_zeigefinger_bleibt_trotz_faust_offen():
    data = _finger_data([1.0], straight={"index": 1.0})
    fq = retarget.finger_quaternions(data, "l", 1)
    for k in (1, 2, 3):
        deg = math.degrees(rot.quat_angle(
            rot.positive(fq["index_%02d_l" % k][0])))
        assert deg == pytest.approx(0.0, abs=1e-6)
    # Die uebrigen Finger gehoeren weiterhin zur Faust.
    assert math.degrees(rot.quat_angle(
        rot.positive(fq["middle_01_l"][0]))) > 30.0


def test_ohne_faust_bleibt_die_hand_offen():
    fq = retarget.finger_quaternions(_finger_data([0.0, 0.0]), "l", 2)
    for qs in fq.values():
        assert math.degrees(rot.quat_angle(rot.positive(qs[0]))) < 1e-6


def test_kurven_ueberstehen_den_weg_durch_die_datei(geloest, tmp_path):
    import json
    _rig, seq, _targets, curves = geloest
    p = tmp_path / "kurven.json"
    retarget.save_curves(curves, retarget.frame_numbers(seq), str(p), seq.fps)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["frames"][0] == 1 and len(data["frames"]) == len(seq)
    for name, qs in data["bones"].items():
        assert np.allclose(np.asarray(qs), curves[name], atol=1e-12)


def _sign_sequence():
    seq = Sequence(toy_sequence(n=40))
    seq["source"] = "pev"
    seq["sign_arm"] = "r"
    landmarks.process(seq)
    return seq


def test_sign_modus_haelt_zweiten_arm_entspannt():
    rig = Rig(toy_rig()["bones"], "toy")
    seq = _sign_sequence()
    # "auto" laesst inzwischen beide Arme folgen (neue PEVs mit korrekter
    # 2D-Pose); der entspannte Arm ist nur noch explizit erzwingbar.
    assert retarget.arm_mode_for(seq) == "both"
    assert retarget.arm_mode_for(seq, "sign") == "sign"
    assert retarget.arm_mode_for(Sequence(toy_sequence())) == "both"

    # Standard: beide Arme kommen aus den Landmarks (der linke Arm des
    # Spielzeug-Skeletts haengt gerade herab, nicht in der entspannten Haltung).
    curves_both = retarget.solve(rig, seq)
    got_l = rig.direction(curves_both, "upperarm_l", 0)
    relaxed_l = rot.unit(RELAXED_ARM_UP["l"])
    assert float(np.dot(rot.unit(got_l), relaxed_l)) < math.cos(math.radians(5.0))

    curves = retarget.solve(rig, seq, arm_mode="sign")
    want = rot.unit(RELAXED_ARM_UP["l"])
    for i in (0, 10, 20, 30, 39):
        got = rig.direction(curves, "upperarm_l", i)
        assert float(np.dot(rot.unit(got), want)) > math.cos(math.radians(0.5))

    # Der Gebaerdearm folgt weiter den Landmarks statt der entspannten Haltung.
    got = rig.direction(curves, "upperarm_r", 0)
    assert abs(float(got[0])) < 0.05


def test_pev_modus_laesst_rumpf_und_kopf_in_ruhe():
    rig = Rig(toy_rig()["bones"], "toy")
    seq = _sign_sequence()
    opts = dict(OPTS, arm_mode="sign", sign_side="r")
    targets, normals, _hands = retarget.build_targets(rig, seq, opts)
    assert "spine_01" not in targets and "head" not in targets
    assert "upperarm_l" in targets and "upperarm_r" in targets
    assert targets["upperarm_l"]["valid"].all()
    assert "l" not in normals      # keine verrauschten Handdaten fuer den Arm
