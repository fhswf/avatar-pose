"""Pruefungen fuer den RTMLib/COCO-Adapter."""

import numpy as np
import pytest

from fixtures import rtm_frames, toy_sequence
from retarget.rtm_adapter import (AdapterError, looks_like_pev, sequence_from_pev,
                                  sequence_from_rtmlib)


def test_adapter_baut_koerper_und_haende():
    seq = sequence_from_rtmlib({"fps": 25.0, "frames": rtm_frames(n=5)},
                               width=640, height=480)
    assert len(seq) == 5
    assert seq.size == (640, 480)
    assert seq.fps == pytest.approx(25.0)
    assert seq["hands"] == "coco"

    fd = seq.frames[0]
    assert len(fd["pose"]) == 17 and len(fd["pose_world"]) == 17
    assert len(fd["left_hand"]) == 21 and len(fd["right_hand"]) == 21
    assert fd["pose"][0] == pytest.approx([320 / 640, 100 / 480, 0.0, 0.9])
    assert fd["left_hand"][0][3] == pytest.approx(0.8)
    # synthetische Welt: x mit Seitenverhaeltnis, y nach unten, z = 0
    aspect = 640 / 480
    assert fd["pose_world"][0][0] == pytest.approx((320 / 640 - 0.5) * aspect)
    assert fd["pose_world"][0][1] == pytest.approx(100 / 480 - 0.5)
    assert fd["pose_world"][0][2] == pytest.approx(0.0)


def test_adapter_nimmt_3d_punkte_wenn_vorhanden():
    seq = sequence_from_rtmlib({"fps": 30.0, "frames": rtm_frames(n=3, with_3d=True)},
                               width=640, height=480, world_scale=2.0)
    wp = seq.frames[0]["pose_world"][0]
    assert wp == pytest.approx([320 / 640 - 0.5, 100 / 480 - 0.5, 0.1, 0.9])


def test_adapter_nimmt_die_beste_person():
    frames = rtm_frames(n=2, persons=2)
    frames[0]["scores"][1] = [0.99] * 133          # Person 1 wird besser
    seq = sequence_from_rtmlib({"fps": 30, "frames": frames}, width=640, height=480)
    assert seq.frames[0]["pose"][1][0] == pytest.approx((310 + 40) / 640)


def test_adapter_fester_personenindex():
    frames = rtm_frames(n=1, persons=2)
    frames[0]["scores"][1] = [0.99] * 133
    seq = sequence_from_rtmlib({"fps": 30, "frames": frames}, width=640, height=480,
                               person=0)
    assert seq.frames[0]["pose"][1][0] == pytest.approx(310 / 640)


def test_adapter_side_by_side_waehlt_die_seite():
    rec = {"frame_idx": 0, "left": rtm_frames(n=1)[0], "right": rtm_frames(n=1)[0]}
    rec["right"]["keypoints"][0][0][0] += 100
    flags = {}
    for side in ("left", "right"):
        seq = sequence_from_rtmlib({"fps": 30, "frames": [rec]},
                                   width=640, height=480, side=side)
        flags[side] = seq.frames[0]["pose"][0][0]
    assert flags["right"] - flags["left"] == pytest.approx(100 / 640)


def test_adapter_behaelt_leere_frames():
    frames = rtm_frames(n=3)
    frames[1]["keypoints"] = np.zeros((1, 133, 2)).tolist()
    frames[1]["scores"] = np.zeros((1, 133)).tolist()
    seq = sequence_from_rtmlib({"fps": 30, "frames": frames}, width=640, height=480)
    assert len(seq) == 3
    assert seq.frames[1]["pose"] is None
    assert seq.frames[1]["left_hand"] is None
    assert seq.frames[2]["pose"] is not None


def test_adapter_lehnt_bereits_konvertierte_sequenz_ab():
    with pytest.raises(AdapterError):
        sequence_from_rtmlib(toy_sequence(n=2), width=1920, height=1080)


def test_adapter_schaetzt_bildgroesse():
    seq = sequence_from_rtmlib({"fps": 30, "frames": rtm_frames(n=2)})
    assert seq.size[0] >= 420
    assert seq.size[1] >= 450


def pev_points(x_offset=0, hands=True):
    """PEV-Punkte aus den RTMLib-Spielzeugdaten, y wie im gsg-Video nach oben."""
    rec = rtm_frames(n=1)[0]
    kp = rec["keypoints"][0]
    sc = rec["scores"][0]
    points = []
    for i in range(133):
        if not hands and i >= 91:
            continue
        points.append({"id": i, "x": float(kp[i][0]) + x_offset,
                       "y": -float(kp[i][1]), "z": 0.0,
                       "confidence": float(sc[i])})
    return points


def test_pev_adapter_normiert_und_dreht_y():
    frames = [{"frame": i, "skeletonpoints": pev_points()}
              for i in range(3)]
    assert looks_like_pev({"frames": frames})
    seq = sequence_from_pev({"frames": frames}, swap_sides=False)

    assert len(seq) == 3
    assert seq.size == (1000, 1000)
    assert seq.fps == pytest.approx(25.0)
    assert seq["source"] == "pev"
    assert seq["sign_arm"] == "l"
    fd = seq.frames[0]
    assert len(fd["pose"]) == 17
    assert fd["left_hand"] is not None and fd["right_hand"] is not None
    # Nase liegt ueber den Fuessen (y nach unten, also kleinerer Wert).
    assert fd["pose"][0][1] < fd["pose"][16][1]


def test_pev_adapter_tauscht_gespiegelte_seiten():
    points = [{"id": i, "x": float(i * 100), "y": -float(i * 10),
               "z": 0.0, "confidence": 1.0} for i in range(17)]
    seq = sequence_from_pev({"frames": [{"skeletonpoints": points}]},
                            swap_sides=False)
    assert seq.frames[0]["pose"][6][0] > seq.frames[0]["pose"][5][0]

    swapped = sequence_from_pev({"frames": [{"skeletonpoints": points}]},
                                swap_sides=True)
    assert swapped.frames[0]["pose"][6][0] < swapped.frames[0]["pose"][5][0]


def test_pev_adapter_behaelt_leere_frames():
    good = {"skeletonpoints": pev_points()}
    seq = sequence_from_pev({"frames": [good, {}, good]})
    assert len(seq) == 3
    assert seq.frames[1]["pose"] is None
    assert seq.frames[1]["left_hand"] is None
    assert seq.frames[2]["pose"] is not None
