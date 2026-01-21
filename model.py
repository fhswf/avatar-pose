# model.py

"""
model.py

Definiert das neuronale Netz (PoseMLP).
Es handelt sich um ein einfaches Multi-Layer Perceptron (MLP),
das die Positionen von Keypoints (eines Frames) auf die Rotationen von Knochen abbildet.
"""

import torch
import torch.nn as nn


class PoseMLP(nn.Module):
    """
    PoseMLP: Ein Feed-Forward Netzwerk für Pose Correction / Animation Prediction.
    
    Architektur:
    - Input: Flattened Vector aller Keypoints eines Frames (NUM_KEYPOINTS * 3).
    - LayerNorm am Eingang zur Stabilisierung.
    - Hidden Layer 1: 512 Neuronen (Standard) + ReLU
    - Hidden Layer 2: 256 Neuronen (Hidden/2) + ReLU
    - Output: Axis-Angle Rotationen für jeden Zielknochen (len(BONES) * 3).
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        # Normalisierung des Eingabevektors ist wichtig, da Koordinaten (x,y,z) variieren können.
        self.norm = nn.LayerNorm(input_dim)

        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        ]
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers += [
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        ]
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Output Layer (keine Aktivierung, da Regression auf Axis-Angle Werte)
        layers += [
            nn.Linear(hidden_dim // 2, output_dim),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.net(x)
