"""retarget - Landmark-zu-Knochenwinkel-Pipeline fuer Avatar-Pose.

Basiert auf VideoPose/signpose, angepasst auf RTMLib/COCO-WholeBody-Landmarks.
Quaternionen sind durchgehend (w, x, y, z), Matrizen 3x3 mit x, y, z als
Spalten (dieselbe Lesart wie in Blender). Am WebSocket/JSON-Ausgang wird nach
(x, y, z, w) umgeordnet, weil Three.js diese Reihenfolge erwartet.

    from retarget import Rig, Sequence, landmarks, retarget, rtm_adapter, webviewer

    seq = rtm_adapter.sequence_from_rtmlib("rtmlib.json")   # COCO -> Sequenz
    landmarks.process(seq)
    rig = Rig.load("retarget/data/rig_kim.json")
    kurven = retarget.solve(rig, seq)
    frames = webviewer.frames(rig, kurven, "avatar.glb")     # Webviewer-Format
"""

from . import (config, depth, filters, gloss, hands, landmarks, metrics,
               retarget, rotations, rtm_adapter, sentence, webviewer)
from .rig import Rig
from .sequence import Sequence

__version__ = "0.1.0"

__all__ = ["Rig", "Sequence", "config", "depth", "filters", "gloss", "hands",
           "landmarks", "metrics", "retarget", "rotations", "rtm_adapter",
           "sentence", "webviewer"]
