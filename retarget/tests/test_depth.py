"""Pruefungen fuer die Tiefenschaetzung aus 2D-Landmarks."""

import math

import numpy as np

from retarget import depth
from retarget.config import MAX_OOP_DEG

W, H = 640, 480


def _pose(elbow_y, wrist_y):
    """COCO-Koerperpunkte; nur die Arme werden variiert."""
    pose = [[0.5, 0.5, 0.0, 1.0] for _ in range(17)]
    pose[5] = [0.30, 0.40, 0.0, 1.0]     # linke Schulter
    pose[6] = [0.60, 0.40, 0.0, 1.0]     # rechte Schulter
    pose[7] = [0.45, elbow_y, 0.0, 1.0]  # linker Ellbogen
    pose[8] = [0.70, elbow_y, 0.0, 1.0]  # rechter Ellbogen
    pose[9] = [0.45, wrist_y, 0.0, 1.0]  # linkes Handgelenk
    pose[10] = [0.70, wrist_y, 0.0, 1.0]
    return pose


def _frames(wrist_y_series):
    out = []
    for i, wy in enumerate(wrist_y_series):
        pose = _pose(0.55, wy)
        world = [list(p) for p in pose]       # 2D: z bleibt 0
        out.append({"frame_video": i, "pose": pose, "pose_world": world})
    return out


def test_2d_arme_bekommen_tiefe_nach_vorn():
    # Gestreckt (0.75) und verkuerzt (0.60) im Wechsel: die Verkuerzung ist
    # der einzige Hinweis auf eine Bewegung aus der Bildebene.
    ys = [0.75 if i % 3 else 0.60 for i in range(20)]
    frames = _frames(ys)
    lengths = depth.estimate_px_lengths(frames, W, H)
    series = depth.depth_series(frames, W, H, lengths, fps=30.0)

    lower = series["lowerarm_l"]
    valid = lower[~np.isnan(lower)]
    assert len(valid) > 0
    # Ohne 3D-Daten ist die Richtung bekannt: vor den Koerper (negatives y).
    assert valid.min() < -1.0
    assert valid.max() <= 0.0

    # Nicht-Arm-Strecken bleiben ohne Tiefe, sonst kippt der Rumpf.
    assert np.allclose(series["shoulders"], 0.0, atol=1e-12)


def test_bildparalleler_arm_bleibt_vor_dem_rumpf():
    # Arme exakt bildparallel: die Verkuerzung ist null. Ohne die Grundtiefe
    # laegen die Haende in der Ebene des Rumpfes und verschwaenden im Mesh.
    ys = [0.70] * 20
    frames = _frames(ys)
    lengths = depth.estimate_px_lengths(frames, W, H)
    series = depth.depth_series(frames, W, H, lengths, fps=30.0)

    lower = series["lowerarm_l"][~np.isnan(series["lowerarm_l"])]
    assert lower.max() < -1.0
    # Der Anteil bleibt an die Segmentlaenge gekoppelt, nicht beliebig gross.
    limit = math.tan(math.radians(MAX_OOP_DEG)) * lengths["lowerarm_l"]
    assert lower.min() >= -limit


def test_segmentrichtungen_uebernehmen_die_tiefe():
    ys = [0.75 if i % 3 else 0.60 for i in range(20)]
    frames = _frames(ys)
    lengths = depth.estimate_px_lengths(frames, W, H)
    series = depth.depth_series(frames, W, H, lengths, fps=30.0)

    vec, valid = depth.segment_directions(frames, W, H, series, "lowerarm_l")
    assert valid.any()
    # x, y, z im Armature-Raum: y ist die Tiefe.
    assert vec[valid, 1].min() < -1.0
