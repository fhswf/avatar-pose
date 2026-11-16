"""Zugriff auf eine Landmark-Aufnahme, also auf das JSON aus der Pose-Extraktion."""

import json


class Sequence(dict):

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dict(self), f)

    @property
    def frames(self):
        return self["frames"]

    @property
    def size(self):
        return int(self.get("width") or 0), int(self.get("height") or 0)

    @property
    def fps(self):
        # Bildrate der ausgewerteten Bilder, nicht die des Videos
        return float(self.get("fps", 30.0)) / max(1, int(self.get("frame_step", 1)))

    @property
    def aspect(self):
        w, h = self.size
        return (w / h) if (w and h) else 1.0

    def __len__(self):
        return len(self.get("frames", []))


def P(arr, i):
    if not arr or i >= len(arr):
        return None
    return arr[i]


def mid(a, b):
    if a is None or b is None:
        return None
    return [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5]


def n_landmarks(frames, key):
    for fd in frames:
        lst = fd.get(key)
        if lst:
            return len(lst)
    return None


def in_frame(img_lm, margin):
    return (-margin <= img_lm[0] <= 1.0 + margin
            and -margin <= img_lm[1] <= 1.0 + margin)