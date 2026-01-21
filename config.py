# config.py

"""
config.py

Zentrale Konfiguration für das Projekt "COCOModel".
Diese Datei definiert Konstanten, die über das gesamte Projekt hinweg verwendet werden,
um Konsistenz zwischen Training (train.py), Datensatz-Erstellung (dataset.py) und Inferenz (infer.py) zu gewährleisten.

Wichtige Parameter:
- DEVICE: Berechnungseinheit (CPU oder CUDA).
- BONES: Liste der Knochen, für die Rotationen vorhergesagt werden.
"""

import torch

# Fenstergröße entfernt für Simple MLP (Frame-by-Frame)
# WINDOW = 1

# Anzahl der Keypoints im COCO-Format (WholeBody)
NUM_KEYPOINTS = 133

# Automatische Wahl der Hardware-Beschleunigung
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Definition der Ziel-Knochen für die Vorhersage (Sign Language Subset)
# Die Reihenfolge hier bestimmt die Reihenfolge im Output-Vektor des Modells.
# Output-Dim = len(BONES) * 3 (Axis-Angle)
BONES = [
    "head",
    "upperarm_l", "lowerarm_l",
    "upperarm_r", "lowerarm_r",

    # Linke Hand
    "metacarpal_thumb_l", "thumb_01_l", "thumb_02_l", "thumb_03_l",
    "metacarpal_index_l", "index_01_l", "index_02_l", "index_03_l",
    "metacarpal_mid_l",   "middle_01_l", "middle_02_l", "middle_03_l",
    "metacarpal_ring_l",  "ring_01_l",   "ring_02_l",   "ring_03_l",
    "metacarpal_pinky_l", "pinky_01_l",  "pinky_02_l",  "pinky_03_l",

    # Rechte Hand
    "metacarpal_thumb_r", "thumb_01_r", "thumb_02_r", "thumb_03_r",
    "metacarpal_index_r", "index_01_r", "index_02_r", "index_03_r",
    "metacarpal_mid_r",   "middle_01_r", "middle_02_r", "middle_03_r",
    "metacarpal_ring_r",  "ring_01_r",   "ring_02_r",   "ring_03_r",
    "metacarpal_pinky_r", "pinky_01_r",  "pinky_02_r",  "pinky_03_r",
]
