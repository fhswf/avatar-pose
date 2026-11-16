"""Drehungen ohne mathutils: Dreibeine, Quaternionen, Matrizen.

Alles, wofür das Add-on Blenders Vector/Quaternion/Matrix benutzt, steht hier
noch einmal in numpy. Quaternionen sind durchgehend (w, x, y, z), Matrizen 3x3
mit x, y und z als Spalten - dieselbe Lesart wie in Blender.
"""

import math

import numpy as np

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def unit(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return None if n < 1e-9 else v / n


# Dreibein aus einer Richtung und einer groben Referenz: y zeigt entlang der
# Knochenachse - so liegen Blender-Knochen -, z steht senkrecht dazu in der
# Ebene der Referenz. Zurück kommt die Matrix mit x, y und z als Spalten.
def ortho_frame(along, ref):
    y = unit(along)
    if y is None:
        return None
    ref = np.asarray(ref, dtype=float)
    z = unit(ref - float(np.dot(ref, y)) * y)
    if z is None:
        return None
    return np.column_stack((np.cross(y, z), y, z))


def orthonormalize(M):
    # Blenders to_quaternion() wirft Skalierung vorher raus, das hier auch.
    M = np.asarray(M, dtype=float)[:3, :3]
    cols = []
    for c in range(3):
        v = unit(M[:, c])
        cols.append(np.zeros(3) if v is None else v)
    return np.column_stack(cols)


# Shepperds Verfahren: die vier Formeln sind mathematisch gleichwertig, aber
# jede wird nahe einer bestimmten Drehung numerisch unbrauchbar. Deshalb wird
# die genommen, deren Nenner am größten ist.
def mat_to_quat(M):
    m = orthonormalize(M)
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        q = (0.25 * s, (m[2, 1] - m[1, 2]) / s,
             (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = ((m[2, 1] - m[1, 2]) / s, 0.25 * s,
             (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
             0.25 * s, (m[1, 2] + m[2, 1]) / s)
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
             (m[1, 2] + m[2, 1]) / s, 0.25 * s)
    q = np.array(q, dtype=float)
    return q / (np.linalg.norm(q) + 1e-12)


def quat_to_mat(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - z*w),       2.0*(x*z + y*w)],
        [2.0*(x*y + z*w),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - x*w)],
        [2.0*(x*z - y*w),       2.0*(y*z + x*w),       1.0 - 2.0*(x*x + y*y)],
    ])


def quat_mul(a, b):
    aw, ax, ay, az = (float(v) for v in a)
    bw, bx, by, bz = (float(v) for v in b)
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def positive(q):
    # q und -q sind dieselbe Drehung, aber nur mit w >= 0 meinen Winkel und
    # Slerp den kurzen Weg.
    q = np.asarray(q, dtype=float)
    return -q if q[0] < 0.0 else q


def quat_angle(q):
    return 2.0 * math.acos(max(-1.0, min(1.0, float(q[0]))))


def axis_angle(axis, angle):
    a = unit(axis)
    if a is None:
        return IDENTITY.copy()
    h = 0.5 * float(angle)
    return np.concatenate(([math.cos(h)], math.sin(h) * a))


# Kürzeste Drehung, die a auf b legt. Zeigen beide genau gegeneinander, ist die
# Achse nicht bestimmt - dann tut es irgendeine, die senkrecht auf a steht.
def rotation_difference(a, b):
    a, b = unit(a), unit(b)
    if a is None or b is None:
        return IDENTITY.copy()
    axis = np.cross(a, b)
    d = max(-1.0, min(1.0, float(np.dot(a, b))))
    if float(np.linalg.norm(axis)) < 1e-9:
        if d > 0.0:
            return IDENTITY.copy()
        helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        return axis_angle(np.cross(a, helper), math.pi)
    return axis_angle(axis, math.acos(d))


def cap(q, limit_rad):
    # Drehung auf einen Höchstwinkel zurücknehmen, Richtung bleibt erhalten.
    from .filters import qslerp
    q = positive(q)
    ang = quat_angle(q)
    return q if ang <= limit_rad else qslerp(IDENTITY, q, limit_rad / ang)


def to_three(q):
    """(w,x,y,z) -> (x,y,z,w) für Three.js/WebSocket."""
    q = np.asarray(q, dtype=float)
    return np.array([q[1], q[2], q[3], q[0]], dtype=float)


def from_three(q):
    """(x,y,z,w) -> (w,x,y,z) für interne Rechnung."""
    q = np.asarray(q, dtype=float)
    return np.array([q[3], q[0], q[1], q[2]], dtype=float)