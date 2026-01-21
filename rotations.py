# rotations.py

"""
rotations.py

Mathematische Hilfsfunktionen für Rotationsberechnungen.
Da Rotationen auf verschiedene Arten dargestellt werden können (Quaternionen, Axis-Angle, Matrizen),
werden hier zentrale Konvertierungsfunktionen bereitgestellt, die von `dataset.py`, `train.py` und `infer.py` genutzt werden.

Funktionen:
- quat_to_axis_angle: Konvertiert Quaternion zu Axis-Angle Vektor.
- axis_angle_to_quat: Konvertiert Axis-Angle Vektor zu Quaternion (für Model Output -> Blender).
- quat_mul: Multiplikation zweier Quaternionen (Verkettung von Rotationen).
"""

import torch, numpy as np

def quat_to_axis_angle(q):
    """
    Numpy-basierte Konvertierung: Quaternion (x,y,z,w) -> Axis-Angle (3,).
    """
    q = np.array(q, dtype=np.float32)
    v, w = q[:3], q[3]
    angle = 2 * np.arccos(np.clip(w, -1, 1))
    n = np.linalg.norm(v) + 1e-8
    return v / n * angle

def axis_angle_to_quat(a):
    """
    PyTorch-basierte Konvertierung: Axis-Angle (..., 3) -> Quaternion (..., 4) [x,y,z,w].
    Wichtig für den Modell-Output, um diesen wieder in Quaternionen für Blender zu wandeln.
    """
    angle = torch.norm(a, dim=-1, keepdim=True)
    axis = a / (angle + 1e-8)
    half = angle * 0.5
    return torch.cat([
        axis * torch.sin(half),
        torch.cos(half)
    ], dim=-1)

def quat_mul(q1, q2):
    """
    PyTorch-basierte Quaternion-Multiplikation.
    q1 * q2 entspricht: Erst Rotation q2, dann q1 (bei Standardnotation).
    """
    x1,y1,z1,w1 = q1.T
    x2,y2,z2,w2 = q2.T
    return torch.stack([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], dim=1)
