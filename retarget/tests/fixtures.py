"""Spielzeug-Rig und Spielzeug-Aufnahme im COCO-WholeBody-Layout."""

import numpy as np

from retarget.config import FINGER_MAP


def bone(name, parent, head, tail, along, up):
    y = np.asarray(along, float)
    y /= np.linalg.norm(y)
    z = np.asarray(up, float)
    z = z - np.dot(z, y) * y
    z /= np.linalg.norm(z)
    m = np.eye(4)
    m[:3, 0] = np.cross(y, z)
    m[:3, 1] = y
    m[:3, 2] = z
    m[:3, 3] = head
    return {"name": name, "parent": parent, "matrix_local": m.tolist(),
            "head_local": list(head), "tail_local": list(tail)}


def toy_rig():
    up = (0.0, 0.0, 1.0)
    front = (0.0, -1.0, 0.0)
    bones = [
        bone("spine_01", None, (0, 0, 1.0), (0, 0, 1.4), up, front),
        bone("neck_01", "spine_01", (0, 0, 1.4), (0, 0, 1.5), up, front),
        bone("head", "neck_01", (0, 0, 1.5), (0, 0, 1.7), up, front),
    ]
    for side, sx in (("l", 1.0), ("r", -1.0)):
        out = (sx, 0.0, 0.0)
        bones += [
            bone("upperarm_" + side, "spine_01",
                 (0.2 * sx, 0, 1.4), (0.5 * sx, 0, 1.4), out, up),
            bone("lowerarm_" + side, "upperarm_" + side,
                 (0.5 * sx, 0, 1.4), (0.8 * sx, 0, 1.4), out, up),
        ]
        for k, (jname, mcname, _idx) in enumerate(FINGER_MAP):
            spread = (2 - k) * 0.02
            head = (0.8 * sx, 0.0, 1.4)
            tail = (0.88 * sx, spread * sx, 1.4)
            bones.append(bone("%s_%s" % (mcname, side), "lowerarm_" + side,
                              head, tail, np.subtract(tail, head), up))
            parent = "%s_%s" % (mcname, side)
            base = np.asarray(tail, float)
            for j in range(1, 4):
                nxt = base + np.array((0.03 * sx, 0.0, 0.0))
                bones.append(bone("%s_%02d_%s" % (jname, j, side), parent,
                                  tuple(base), tuple(nxt), (sx, 0.0, 0.0), up))
                parent = "%s_%02d_%s" % (jname, j, side)
                base = nxt
    return {"name": "toy", "bones": bones}


W, H = 1920, 1080


def flat_hand_world():
    """Flache Hand in MediaPipe-Weltachsen (x rechts, y unten, z zur Kamera)."""
    pts = [[0.0, 0.0, 0.0, 1.0]]
    across = {"thumb": 0.045, "index": 0.025, "middle": 0.0,
              "ring": -0.022, "pinky": -0.042}
    base = {"thumb": -0.035, "index": -0.085, "middle": -0.090,
            "ring": -0.085, "pinky": -0.078}
    for jname, _mc, _idx in FINGER_MAP:
        x, y = across[jname], base[jname]
        for k in range(4):
            pts.append([x, y - 0.028 * k, 0.0, 1.0])
    return pts


def toy_sequence(n=40, elbow=(0.60, 0.59), wrist=(0.60, 0.83), hands=True,
                 head_tilt=0.0):
    """COCO-WholeBody-Body-Indizes 0..16, Rest auf 0."""
    frames = []
    for i in range(n):
        tilt = head_tilt * np.sin(2.0 * np.pi * i / max(1, n))
        pose = [[0.5, 0.5, 0.0, 0.0] for _ in range(17)]

        def put(idx, x, y, z=0.0):
            pose[idx] = [x, y, z, 1.0]

        put(0, 0.50 + tilt, 0.25)                      # Nase
        put(1, 0.53, 0.24); put(2, 0.47, 0.24)         # Augen
        put(3, 0.56 + tilt, 0.25); put(4, 0.44 + tilt, 0.25)   # Ohren
        put(5, 0.60, 0.35); put(6, 0.40, 0.35)         # Schultern
        put(7, elbow[0], elbow[1]); put(8, 0.40, 0.59)  # Ellbogen
        put(9, wrist[0], wrist[1]); put(10, 0.40, 0.83)  # Handgelenke
        put(11, 0.57, 0.72); put(12, 0.43, 0.72)       # Hueften
        put(13, 0.57, 0.90); put(14, 0.43, 0.90)       # Knie
        put(15, 0.57, 0.99); put(16, 0.43, 0.99)       # Fussgelenke

        world = [[(p[0] - 0.5) * 1.4, (p[1] - 0.72) * 1.4, 0.0, p[3]]
                 for p in pose]

        fd = {"frame_video": i, "pose": pose, "pose_world": world,
              "left_hand": None, "right_hand": None,
              "left_hand_world": None, "right_hand_world": None}
        if hands:
            for side, (wx, wy) in (("left", wrist), ("right", (0.40, 0.83))):
                img = [[wx + p[0], wy + p[1], 0.0, 1.0]
                       for p in flat_hand_world()]
                fd[side + "_hand"] = img
                fd[side + "_hand_world"] = [list(p) for p in flat_hand_world()]
        frames.append(fd)
    return {"fps": 30.0, "width": W, "height": H, "frame_step": 1,
            "n": n, "hands": "holistic_labeled", "frames": frames}


def pev_frames(n=12):
    """PEV-Frames (``skeletonpoints``) aus den RTMLib-Spielzeugdaten."""
    out = []
    for index, rec in enumerate(rtm_frames(n=n)):
        points = []
        for idx, (x, y) in enumerate(rec["keypoints"][0]):
            points.append({
                "id": idx,
                "x": float(x),
                "y": float(y),
                "z": 0.0,
                "confidence": float(rec["scores"][0][idx]),
            })
        out.append({"frame_video": index, "skeletonpoints": points})
    return out


def rtm_frames(n=12, persons=1, with_3d=False, hands=True):
    """Rohe RTMLib-Frames ``keypoints``/``scores`` im COCO-Layout."""
    body = {0: (320, 100), 1: (310, 95), 2: (330, 95), 3: (300, 100),
            4: (340, 100), 5: (280, 160), 6: (360, 160), 7: (250, 220),
            8: (390, 220), 9: (220, 280), 10: (420, 280), 11: (290, 300),
            12: (350, 300), 13: (285, 380), 14: (355, 380), 15: (280, 450),
            16: (360, 450)}
    frames = []
    for i in range(n):
        kp = np.zeros((persons, 133, 2))
        sc = np.zeros((persons, 133))
        for p in range(persons):
            offset = p * 40
            for k, (x, y) in body.items():
                kp[p, k] = (x + offset, y)
                sc[p, k] = 0.9 - 0.1 * p
            if hands:
                for base in (91, 112):
                    for k in range(21):
                        kp[p, base + k] = (200 + offset + 5 * k, 280 + 5 * k)
                        sc[p, base + k] = 0.8 - 0.1 * p
        rec = {"frame_idx": i, "keypoints": kp.tolist(), "scores": sc.tolist()}
        if with_3d:
            kp3 = np.zeros((persons, 133, 3))
            kp3[..., 0] = kp[..., 0] / 640.0 - 0.5
            kp3[..., 1] = kp[..., 1] / 480.0 - 0.5
            kp3[..., 2] = 0.1
            rec["keypoints_3d"] = kp3.tolist()
        frames.append(rec)
    return frames
