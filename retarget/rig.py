"""Das Rig, so viel davon wie zum Rechnen nötig ist."""

import json

import numpy as np

from . import rotations as rot
from .config import BONE_VARIANTS, FINGER_MAP


class Rig:

    def __init__(self, bones, name=""):
        self.name = name
        self.parent = {}
        self.head = {}
        self.tail = {}
        self.rest = {}
        self.rel = {}
        order_in = [b["name"] for b in bones]
        local = {}
        for b in bones:
            m = np.asarray(b["matrix_local"], dtype=float).reshape(4, 4)
            local[b["name"]] = m
            self.parent[b["name"]] = b.get("parent") or None
            self.head[b["name"]] = np.asarray(b["head_local"], dtype=float)
            self.tail[b["name"]] = np.asarray(b["tail_local"], dtype=float)
            self.rest[b["name"]] = rot.quat_to_mat(rot.mat_to_quat(m))
        for n in order_in:
            p = self.parent[n]
            if p not in local:
                self.parent[n] = p = None
            if p:
                m = np.linalg.inv(local[p]) @ local[n]
                self.rel[n] = rot.quat_to_mat(rot.mat_to_quat(m))
            else:
                self.rel[n] = self.rest[n]

        def depth(n):
            d, p = 0, self.parent[n]
            while p:
                d += 1
                p = self.parent[p]
            return d

        self.order = sorted(order_in, key=depth)
        self.rest_frames, self.rest_palms = self._anatomy()

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["bones"], data.get("name", ""))

    def find(self, canonical):
        exact = self.find_exact(canonical)
        if exact:
            return exact
        cl = canonical.replace("_", "").replace(".", "").lower()
        for n in self.order:
            if cl == n.replace("_", "").replace(".", "").lower():
                return n
        return None

    def find_exact(self, canonical):
        for n in BONE_VARIANTS.get(canonical, [canonical]):
            if n in self.rest:
                return n
        return canonical if canonical in self.rest else None

    def _anatomy(self):
        frames = {}
        palms = {}
        sl, sr = self.find_exact("upperarm_l"), self.find_exact("upperarm_r")
        across = (self.head[sl] - self.head[sr]) if (sl and sr) else None
        sp = self.find_exact("spine")
        if sp and across is not None:
            frames["spine"] = rot.ortho_frame(self.tail[sp] - self.head[sp], across)
        for side in ("l", "r"):
            low = self.find_exact("lowerarm_" + side)
            mc = "metacarpal_mid_" + side
            im = "metacarpal_index_" + side
            pm = "metacarpal_pinky_" + side
            if not (low and mc in self.rest and im in self.rest and pm in self.rest):
                continue
            forward = self.tail[mc] - self.head[mc]
            acr = self.tail[im] - self.tail[pm]
            nor = rot.unit(np.cross(forward, acr))
            if nor is None:
                continue
            palms[side] = self.rest[low].T @ nor
            frames["hand_" + side] = rot.ortho_frame(forward, nor)
        return frames, palms

    def wrist_bones(self, side):
        out = []
        for _jname, mcname, _idx in FINGER_MAP:
            n = self.find("%s_%s" % (mcname, side))
            if n is not None:
                out.append(n)
        return out

    def chain(self, targets):
        needed = set()
        for name in targets:
            cur = name
            while cur and cur not in needed:
                needed.add(cur)
                cur = self.parent.get(cur)
        return [b for b in self.order if b in needed]

    def basis(self, name, A, t, i):
        try:
            Ainv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            return None
        mode = t["mode"]
        if mode == "quat":
            R = rot.quat_to_mat(t["quat"][i])
            return rot.mat_to_quat(Ainv @ R @ self.rest[name])
        if mode == "frame" and t["valid"][i]:
            F = rot.ortho_frame(t["along"][i], t["ref"][i])
            if F is not None:
                R = F @ t["rest"].T
                return rot.mat_to_quat(Ainv @ R @ self.rest[name])
        v = rot.unit(t["along"][i])
        if v is None:
            return None
        yL = rot.unit(Ainv @ v)
        if yL is None:
            return None
        return rot.rotation_difference((0.0, 1.0, 0.0), yL)

    def forward(self, curves, i=0):
        M = {}
        for name in self.order:
            p = self.parent[name]
            A = (M[p] @ self.rel[name]) if p else self.rel[name]
            q = curves.get(name)
            M[name] = (A @ rot.quat_to_mat(q[i])) if q is not None else A
        return M

    def direction(self, curves, name, i=0):
        return self.forward(curves, i)[name][:, 1]