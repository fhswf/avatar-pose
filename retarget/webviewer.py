"""Retarget-Kurven in das Format des Webviewers bringen.

Die Retarget-Kette liefert je Knochen eine Quaternion im Blender-Basis-System
(w, x, y, z). Der Webviewer erwartet dagegen lokale glTF-Node-Rotationen in
Three.js-Reihenfolge (x, y, z, w).

Die Umrechnung ist in der Projekt-Diskussion hergeleitet und numerisch
geprueft: Mit

    C   = [[1,0,0],[0,0,1],[0,-1,0]]      Blender -> glTF
    G_b = Ruhe-Globalrotation des Knochens im Rig (Blender)
    H_b = Ruhe-Globalrotation des Nodes im glTF
    D_b = (C * G_b)^T * H_b               Drehung um die Knochenachse

gilt fuer eine Retarget-Basisrotation ``B_b``:

    q_node = Q_rest_gltf,b * (D_b^T * B_b * D_b)

``Q_rest_gltf`` ist die lokale Ruherotation des Nodes aus der GLB-Datei.
"""

import json

import numpy as np

from . import rotations as rot

COORD = np.array([[1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0],
                  [0.0, -1.0, 0.0]])


class GltfRest:
    """Ruherotationen und Hierarchie aus einer GLB-Datei."""

    def __init__(self, glb_path):
        from pygltflib import GLTF2

        self.gltf = GLTF2().load(str(glb_path))
        self.nodes = {}
        self.parent = {}
        for i, node in enumerate(self.gltf.nodes):
            if node.name is not None and node.name not in self.nodes:
                self.nodes[node.name] = node
        for node in self.gltf.nodes:
            for child_index in (node.children or []):
                child = self.gltf.nodes[child_index]
                if child.name is not None:
                    self.parent[child.name] = node.name

        self._local = {
            name: rot.from_three(np.asarray(node.rotation or [0.0, 0.0, 0.0, 1.0],
                                            dtype=float))
            for name, node in self.nodes.items()
        }
        self._global = {}

    def local_quat(self, name):
        return self._local[name]

    def local_mat(self, name):
        return rot.quat_to_mat(self._local[name])

    def global_mat(self, name, _stack=None):
        """Produkt der lokalen Rotationen von der Wurzel bis ``name``."""
        if name in self._global:
            return self._global[name]
        if name not in self.nodes:
            return None
        stack = set() if _stack is None else _stack
        if name in stack:
            return None
        stack.add(name)
        parent = self.parent.get(name)
        parent_mat = self.global_mat(parent, stack) if parent else None
        local_mat = rot.quat_to_mat(self._local[name])
        mat = local_mat if parent_mat is None else parent_mat @ local_mat
        self._global[name] = mat
        return mat


def node_curves(rig, curves, glb_path):
    """Wandelt Retarget-Kurven in lokale glTF-Node-Quaternionen (w, x, y, z)."""
    rest = GltfRest(glb_path)
    out = {}
    for name, series in curves.items():
        if name not in rest.nodes or name not in rig.rest:
            continue
        qs = np.asarray(series, dtype=float)
        if qs.ndim != 2 or qs.shape[1] != 4:
            raise ValueError("Kurve fuer %s muss (n, 4) sein, nicht %s"
                             % (name, qs.shape))
        global_rest = rest.global_mat(name)
        if global_rest is None:
            continue
        basis = (COORD @ rig.rest[name]).T @ global_rest
        local_rest = rest.local_mat(name)
        node_qs = np.zeros_like(qs)
        for i in range(qs.shape[0]):
            delta = basis.T @ rot.quat_to_mat(qs[i]) @ basis
            node_qs[i] = rot.mat_to_quat(local_rest @ delta)
        out[name] = node_qs
    return out


def frames(rig, curves, glb_path):
    """Liste von Webviewer-Frames ``{knochen: {"rotation": [x, y, z, w]}}``."""
    mapped = node_curves(rig, curves, glb_path)
    if not mapped:
        return []
    n = max(len(qs) for qs in mapped.values())
    out = []
    for i in range(n):
        frame = {}
        for name, qs in mapped.items():
            if i < len(qs):
                frame[name] = {"rotation": rot.to_three(qs[i]).tolist()}
        out.append(frame)
    return out


IDENTITY_THREE = (0.0, 0.0, 0.0, 1.0)


def rest_frame(glb_path, names):
    """Ruhepose (T-Pose) als Webviewer-Frame: die lokalen Rest-Rotationen.

    Die Knoten im GLB stehen nicht auf Identitaet - ihre Ruherotation *ist*
    die T-Pose. Genau diese Werte muessen am Anfang und Ende gesendet werden.
    """
    rest = GltfRest(glb_path)
    out = {}
    for name in names:
        if name in rest.nodes:
            out[name] = {"rotation": rot.to_three(rest.local_quat(name)).tolist()}
    return out


def _smoothstep(t):
    t = max(0.0, min(1.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def _blend_frame(a, b, t):
    """Slerp zwischen zwei Webviewer-Frames (Three.js-Reihenfolge)."""
    from .filters import qslerp

    out = {}
    for name in set(a) | set(b):
        qa = rot.from_three(a[name]["rotation"] if name in a else IDENTITY_THREE)
        qb = rot.from_three(b[name]["rotation"] if name in b else IDENTITY_THREE)
        out[name] = {"rotation": rot.to_three(qslerp(qa, qb, _smoothstep(t))).tolist()}
    return out


def anchor_rest(frames_list, fps=30.0, fade_sec=0.35, rest=None):
    """Blendet aus der T-Pose in die Bewegung und wieder zurueck.

    ``rest`` ist der Ruhe-Frame aus :func:`rest_frame`. Ohne Angabe wird die
    Identitaet benutzt - das ist nur fuer Rigs richtig, deren Ruhepose die
    Identitaet ist.
    """
    if not frames_list or fade_sec <= 0.0:
        return list(frames_list)
    n = max(1, int(round(fade_sec * float(fps or 30.0))))
    if rest is None:
        rest = {name: {"rotation": list(IDENTITY_THREE)} for name in frames_list[0]}
    lead = [_blend_frame(rest, frames_list[0], k / (n + 1.0))
            for k in range(1, n + 1)]
    tail = [_blend_frame(frames_list[-1], rest, k / float(n))
            for k in range(1, n + 1)]
    return lead + list(frames_list) + tail


def resample(curves, src_fps, dst_fps=30.0):
    """Kurven auf eine andere Bildrate umrechnen (slerp, Enden bleiben stehen)."""
    from .filters import qslerp

    src_fps = float(src_fps or 0.0)
    dst_fps = float(dst_fps or 0.0)
    if src_fps <= 0 or dst_fps <= 0 or abs(src_fps - dst_fps) < 1e-6:
        return {name: np.asarray(qs, dtype=float) for name, qs in curves.items()}

    out = {}
    for name, series in curves.items():
        qs = np.asarray(series, dtype=float)
        n = len(qs)
        if n < 2:
            out[name] = qs
            continue
        count = max(2, int(round((n - 1) * dst_fps / src_fps)) + 1)
        t = np.arange(count, dtype=float) * (src_fps / dst_fps)
        i0 = np.clip(np.floor(t).astype(int), 0, n - 2)
        frac = np.clip(t - i0, 0.0, 1.0)
        out[name] = np.array([qslerp(qs[i], qs[i + 1], f)
                              for i, f in zip(i0, frac)])
    return out


def load_curves(path):
    """Liest eine mit ``retarget.save_curves`` geschriebene Datei."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    bones = {name: np.asarray(qs, dtype=float)
             for name, qs in data.get("bones", {}).items()}
    return bones, float(data.get("fps", 30.0)), data.get("frames")


def save_frames(frames_list, path, indent=None):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(frames_list, f, indent=indent)
    return path
