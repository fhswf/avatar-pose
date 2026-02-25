# dataset.py

"""
dataset.py

Verantwortlich für das Laden und Aufbereiten der Trainingsdaten.

Kernkomponenten:
- blender_to_synthetic_coco_keypoints: Konvertiert 3D-Positionen aus Blender-Daten in ein flaches COCO-Keypoint Format.
- PoseDataset: Ein PyTorch Dataset, das Einzel-Frames lädt und Eingabe (X) sowie Zielvektoren (Y) bereitstellt.
- Nutzt `rotations.py` für die Umrechnung von Quaternionen (Blender) zu Axis-Angle (Model Target).

Ziel:
- X: Eingabe-Features (Keypoints eines Frames).
- Y: Ground-Truth (Blender Bone-Rotationen desselben Frames).
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from config import BONES
from rotations import quat_to_axis_angle

# Mapping von Blender-Bones zu COCO-Keypoints (Index)
# Wir verwenden Proxies (Head/Tail von Bones), um COCO-Joints zu simulieren.
def blender_to_synthetic_coco_keypoints(d):
    """
    Erstellt synthetische COCO-Keypoints aus Blender-Bone-Daten.

    Args:
        d (dict): Ein Frame-Dictionary aus der Blender JSON-Datei.

    Returns:
        np.array: (133, 3) Array mit x,y,z Koordinaten für jeden Keypoint.
    """
    # Output: (133, 3) COCO wholebody slots, default 0
    kp = np.zeros((133, 3), dtype=np.float32)

    bones = d["bones"]

    # --- Mapping Logik ---
    
    # Beispiel: Nose proxy = head head-position
    if "head" in bones:
        h = bones["head"]["world_space"]["head"]
        kp[0] = [h["x"], h["y"], h["z"]]

    # Schultern/Elbows/Wrists proxy über Arm-Bones (head/tail)
    if "upperarm_l" in bones:
        p = bones["upperarm_l"]["world_space"]["head"]; kp[5] = [p["x"], p["y"], p["z"]]
    if "upperarm_r" in bones:
        p = bones["upperarm_r"]["world_space"]["head"]; kp[6] = [p["x"], p["y"], p["z"]]
    if "lowerarm_l" in bones:
        p = bones["lowerarm_l"]["world_space"]["head"]; kp[7] = [p["x"], p["y"], p["z"]]      # left_elbow
        p = bones["lowerarm_l"]["world_space"]["tail"]; kp[9] = [p["x"], p["y"], p["z"]]      # left_wrist
        kp[91] = kp[9]  # left_hand_wrist (Double mapping)
    if "lowerarm_r" in bones:
        p = bones["lowerarm_r"]["world_space"]["head"]; kp[8] = [p["x"], p["y"], p["z"]]      # right_elbow
        p = bones["lowerarm_r"]["world_space"]["tail"]; kp[10] = [p["x"], p["y"], p["z"]]     # right_wrist
        kp[112] = kp[10]  # right_hand_wrist (Double mapping)

    # Hände: wir nehmen jeweils Bone head oder tail als Proxy für die COCO Hand-Joints
    # Hilfsfunktion zum Setzen von Hand-Joints
    def set_from_bone(coco_idx, bone_name, which="head"):
        if bone_name in bones:
            p = bones[bone_name]["world_space"][which]
            kp[coco_idx] = [p["x"], p["y"], p["z"]]

    # LEFT hand joints (92..111) mapping proxies
    set_from_bone(92, "metacarpal_thumb_l", "head")
    set_from_bone(93, "thumb_01_l", "head")
    set_from_bone(94, "thumb_02_l", "head")
    set_from_bone(95, "thumb_03_l", "tail")

    set_from_bone(96, "metacarpal_index_l", "head")
    set_from_bone(97, "index_01_l", "head")
    set_from_bone(98, "index_02_l", "head")
    set_from_bone(99, "index_03_l", "tail")

    set_from_bone(100, "metacarpal_mid_l", "head")
    set_from_bone(101, "middle_01_l", "head")
    set_from_bone(102, "middle_02_l", "head")
    set_from_bone(103, "middle_03_l", "tail")

    set_from_bone(104, "metacarpal_ring_l", "head")
    set_from_bone(105, "ring_01_l", "head")
    set_from_bone(106, "ring_02_l", "head")
    set_from_bone(107, "ring_03_l", "tail")

    set_from_bone(108, "metacarpal_pinky_l", "head")
    set_from_bone(109, "pinky_01_l", "head")
    set_from_bone(110, "pinky_02_l", "head")
    set_from_bone(111, "pinky_03_l", "tail")

    # RIGHT hand joints (113..132) mapping proxies
    set_from_bone(113, "metacarpal_thumb_r", "head")
    set_from_bone(114, "thumb_01_r", "head")
    set_from_bone(115, "thumb_02_r", "head")
    set_from_bone(116, "thumb_03_r", "tail")

    set_from_bone(117, "metacarpal_index_r", "head")
    set_from_bone(118, "index_01_r", "head")
    set_from_bone(119, "index_02_r", "head")
    set_from_bone(120, "index_03_r", "tail")

    set_from_bone(121, "metacarpal_mid_r", "head")
    set_from_bone(122, "middle_01_r", "head")
    set_from_bone(123, "middle_02_r", "head")
    set_from_bone(124, "middle_03_r", "tail")

    set_from_bone(125, "metacarpal_ring_r", "head")
    set_from_bone(126, "ring_01_r", "head")
    set_from_bone(127, "ring_02_r", "head")
    set_from_bone(128, "ring_03_r", "tail")

    set_from_bone(129, "metacarpal_pinky_r", "head")
    set_from_bone(130, "pinky_01_r", "head")
    set_from_bone(131, "pinky_02_r", "head")
    set_from_bone(132, "pinky_03_r", "tail")

    return kp





class PoseDataset(Dataset):
    """
    Dataset-Klasse für Frame-by-Frame Posen-Vorhersage (Simple MLP).
    Lädt JSON-Dateien und generiert Input (X) und Target (Y) für jeweils einen Frame.
    """
    def __init__(self, blender_files):
        """
        Args:
            blender_files (list): Liste von Pfaden zu den JSON-Trainingsdateien.
        """
        self.blender = blender_files

    def __len__(self):
        return len(self.blender)

    def _load_blender_json(self, path):
        """Lädt eine einzelne JSON Datei."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _make_input(self, blender_json):
        """Erzeugt den Input-Vektor für einen Frame (flattened Keypoints)."""
        # synthetic COCO (133,3) -> flatten
        kp = blender_to_synthetic_coco_keypoints(blender_json)
        return kp.astype(np.float32)

    def _make_target(self, blender_json):
        """Erzeugt den Target-Vektor (Axis-Angle Rotationen für alle relevanten Bones)."""
        # axis-angle for each bone in BONES -> flatten (len(BONES)*3)
        aa = []
        for b in BONES:
            q = blender_json["bones"][b]["local"]["rotation_quaternion"]
            aa.append(quat_to_axis_angle([q["x"], q["y"], q["z"], q["w"]]))
        return np.concatenate(aa).astype(np.float32)

    def __getitem__(self, idx):
        """
        Gibt ein Paar (Input, Target) zurück für Frame idx.
        """
        d = self._load_blender_json(self.blender[idx])
        
        # Input: (133 * 3)
        x_in = self._make_input(d).flatten()
        
        # Target: (len(BONES) * 3)
        y_target = self._make_target(d)

        return torch.tensor(x_in), torch.tensor(y_target)
