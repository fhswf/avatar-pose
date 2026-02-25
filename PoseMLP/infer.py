# infer.py

""" 
infer.py

Verantwortlich für die Inferenz des trainierten Modells auf neuen Daten (.pev Dateien).

Workflow:
1. Lädt trainiertes PoseMLP Modell (pose_mlp.pt).
2. Liest .pev Dateien aus dem Ordner 'PevData/' (Enthält Keypoints pro Frame).
3. Bereitet Input-Vektoren für jeden Frame vor (Single Frame).
4. Führt Inferenz durch (Vorhersage von Bone-Rotationen).
5. Konvertiert Vorhersagen in ein für Blender kompatibles JSON-Format (inkl. Matrix Basis).
6. Speichert Ergebnisse in 'InferOut/<source_file>/pose_frame_######.json'.

Besonderheiten:
- Das Modell sagt Axis-Angle Rotationen vorher.
- Für Blender wird dies in Quaternionen und 4x4 Matrizen umgerechnet.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch

from config import DEVICE, NUM_KEYPOINTS, BONES
from model import PoseMLP
from rotations import axis_angle_to_quat


# -------------------------
# IO helpers
# -------------------------

def list_pev_files(folder: str = "PevData") -> List[Path]:
    """Listet alle .pev Dateien im angegebenen Ordner auf."""
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f"Folder not found: {p.resolve()}")
    files = sorted(p.glob("*.pev"))
    if not files:
        raise FileNotFoundError(f"No .pev files found in: {p.resolve()}")
    return files


def read_pev_json(path: Path) -> Dict[str, Any]:
    """Liest eine .pev Datei (JSON Format) ein und behandelt Fehler."""
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        raise ValueError(f"Empty file: {path.name}")
    try:
        return json.loads(text)
    except Exception as e:
        raise ValueError(f"Invalid JSON in {path.name}: {e}")


def frames_from_pev(pev: Dict[str, Any]) -> np.ndarray:
    """
    Extrahiert Keypoints- Baut pro Frame einen Input-Vektor (NUM_KEYPOINTS * 3)
  -> Simple MLP (Frame-by-Frame).
                    Enthält x, y, z Koordinaten für jeden Keypoint.
    """
    frames_list = pev.get("frames", [])
    if not isinstance(frames_list, list):
        raise ValueError("PEV JSON missing 'frames' list")

    # Bestimme maximale Frame-ID um Array-Größe festzulegen (T)
    max_frame = -1
    for fr in frames_list:
        if isinstance(fr, dict) and "frame" in fr:
            try:
                max_frame = max(max_frame, int(fr["frame"]))
            except Exception:
                pass
    if max_frame < 0:
        raise ValueError("No valid frame indices")

    T = max_frame + 1
    out = np.zeros((T, NUM_KEYPOINTS, 3), dtype=np.float32)

    # Fülle das Array mit Koordinaten
    for fr in frames_list:
        if not isinstance(fr, dict):
            continue
        fidx = int(fr.get("frame", 0))
        pts = fr.get("skeletonpoints", [])
        if not isinstance(pts, list):
            continue
        for p in pts:
            if not isinstance(p, dict):
                continue
            kid = p.get("id", None)
            if kid is None:
                continue
            kid = int(kid)
            # Nur Keypoints innerhalb des definierten Bereichs beachten
            if 0 <= kid < NUM_KEYPOINTS:
                out[fidx, kid, 0] = float(p.get("x", 0.0))
                out[fidx, kid, 1] = float(p.get("y", 0.0))
                out[fidx, kid, 2] = float(p.get("z", 0.0))

    return out


# -------------------------
# Build PoseMLP input (match training)
# -------------------------

# -------------------------
# Build PoseMLP input (match training)
# -------------------------

def build_single_frame_input(frames: np.ndarray, t: int) -> np.ndarray:
    """
    Erstellt den Eingabe-Vektor für einen bestimmten Frame t (Single Frame).
    Shape: (NUM_KEYPOINTS * 3, ) - geflattet.
    """
    T = frames.shape[0]
    t = min(max(t, 0), T - 1) # Clamp index just in case
    
    frame_data = frames[t] # (N, 3)
    return frame_data.astype(np.float32).reshape(-1)


# -------------------------
# Quaternion -> matrix_basis
# -------------------------

def quat_xyzw_to_rotmat3(q: np.ndarray) -> np.ndarray:
    """Konvertiert Quaternion [x,y,z,w] in 3x3 Rotationsmatrix."""
    x, y, z, w = map(float, q)
    n = (x*x + y*y + z*z + w*w) ** 0.5
    if n < 1e-8:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = x/n, y/n, z/n, w/n

    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),     2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),         1 - 2*(xx + yy)]
    ], dtype=np.float32)


def make_matrix_basis(loc=(0.0, 0.0, 0.0), quat_xyzw=(0.0, 0.0, 0.0, 1.0), scl=(1.0, 1.0, 1.0)):
    """Erstellt eine 4x4 Transformationsmatrix (Matrix Basis) für Blender."""
    R = quat_xyzw_to_rotmat3(np.asarray(quat_xyzw, dtype=np.float32))
    sx, sy, sz = map(float, scl)
    S = np.diag([sx, sy, sz]).astype(np.float32)

    RS = R @ S
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = RS
    m[0, 3], m[1, 3], m[2, 3] = map(float, loc)

    return [[float(m[r, c]) for c in range(4)] for r in range(4)]


# -------------------------
# Blender pose JSON builder
# -------------------------

def make_blender_pose_json(quats_xyzw: np.ndarray, frame_index: int, source_file: str, armature_name: str):
    """
    Erstellt das Dictionary für die JSON-Ausgabe eines einzelnen Frames,
    kompatibel mit dem Blender Import-Skript.
    """
    bones_dict: Dict[str, Any] = {}

    # optional root identity (hilft Stabilität)
    bones_dict["Root"] = {
        "name": "Root",
        "local": {
            "location": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation_quaternion": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
            "matrix_basis": make_matrix_basis((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0)),
            "rotation_mode": "QUATERNION",
        }
    }

    for i, b in enumerate(BONES):
        x, y, z, w = map(float, quats_xyzw[i])  # xyzw
        bones_dict[b] = {
            "name": b,
            "local": {
                "location": {"x": 0.0, "y": 0.0, "z": 0.0},
                "rotation_quaternion": {"w": w, "x": x, "y": y, "z": z},
                "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
                "matrix_basis": make_matrix_basis((0.0, 0.0, 0.0), (x, y, z, w), (1.0, 1.0, 1.0)),
                "rotation_mode": "QUATERNION",
            }
        }

    return {
        "metadata": {
            "armature_name": armature_name,
            "export_time": "PREDICTED",
            "blender_version": "4.x",
            "bone_count": len(bones_dict),
            "end_bone_count": 0,
            "spaces_included": ["LOCAL_CHANNELS"],
            "note": "Predicted pose. Contains local.matrix_basis for Blender import (Use Matrix Basis).",
            "source_file": source_file,
            "frame_index": int(frame_index),
        },
        "bones": bones_dict,
    }


# -------------------------
# Main
# -------------------------

def main():
    pev_files = list_pev_files("PevData")

    model_path = Path("pose_mlp.pt")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path.resolve()}")

    input_dim = NUM_KEYPOINTS * 3
    output_dim = len(BONES) * 3

    print(f"[infer] Loading model from {model_path}")
    print(f"[infer] Input Dim: {input_dim}, Output Dim: {output_dim}")

    model = PoseMLP(input_dim=input_dim, output_dim=output_dim).to(DEVICE)
    sd = torch.load(str(model_path), map_location=DEVICE)
    model.load_state_dict(sd)
    model.eval()

    out_root = Path("InferOut")
    out_root.mkdir(exist_ok=True)

    for pev_path in pev_files:
        print(f"[infer] reading {pev_path.name}")
        pev = read_pev_json(pev_path)
        frames = frames_from_pev(pev)  # (T,N,3)
        T = frames.shape[0]

        out_dir = out_root / pev_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for t in range(T):
                # 1. Input erstellen
                x_np = build_single_frame_input(frames, t)  # (input_dim,)
                x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)

                # 2. Inferenz
                aa = model(x).squeeze(0).cpu().numpy().reshape(len(BONES), 3)

                # 3. Konvertierung Axis-Angle -> Quaternion
                aa_t = torch.tensor(aa, dtype=torch.float32)
                q_xyzw = axis_angle_to_quat(aa_t).numpy()  # (B,4) xyzw

                # 4. JSON erstellen
                pose_json = make_blender_pose_json(
                    quats_xyzw=q_xyzw,
                    frame_index=t,
                    source_file=pev_path.name,
                    armature_name="KIM_caucasian_male",
                )

                # 5. Speichern
                out_path = out_dir / f"pose_frame_{t:06d}.json"
                out_path.write_text(json.dumps(pose_json, indent=2), encoding="utf-8")

        # Manifest Datei für Übersicht
        manifest = {
            "source_file": pev_path.name,
            "frames_total": int(T),
            "num_keypoints": int(NUM_KEYPOINTS),
            "bones": BONES,
            "model": "PoseMLP",
            "model_file": str(model_path.name),
            "output_folder": str(out_dir),
            "file_pattern": "pose_frame_######.json",
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(f"[infer] wrote {T} frames -> {out_dir}")

    print("[infer] done")


if __name__ == "__main__":
    main()
