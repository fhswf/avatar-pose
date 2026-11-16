"""Konstanten der Kette – angepasst auf RTMLib/COCO-WholeBody.

COCO-WholeBody-Indizes (133 Keypoints):
  0-16   Körper (COCO-17)
  17-22  Füße
  23-90  Gesicht
  91-111 linke Hand (21 Punkte)
  112-132 rechte Hand (21 Punkte)

RTMLib und (später) Sapiens1 liefern genau dieses Format.
"""

# Landmark-Indizes (COCO-Body). Links/rechts aus Sicht der gefilmten Person.
POSE = {
    "nose": 0,
    "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}

# Hand-Basis-Indizes in COCO-WholeBody
LEFT_HAND_BASE = 91
RIGHT_HAND_BASE = 112
HAND_N = 21

# Finger, sein Mittelhandknochen im Rig, und die fünf Handpunkte von der
# Handwurzel bis zur Spitze. Die 0 steht bei jedem Finger vorn: alle fünf
# hängen am selben Handgelenk. Indizes relativ zur Hand-Basis.
FINGER_MAP = [
    ("thumb",  "metacarpal_thumb", [0, 1, 2, 3, 4]),
    ("index",  "metacarpal_index", [0, 5, 6, 7, 8]),
    ("middle", "metacarpal_mid",   [0, 9, 10, 11, 12]),
    ("ring",   "metacarpal_ring",  [0, 13, 14, 15, 16]),
    ("pinky",  "metacarpal_pinky", [0, 17, 18, 19, 20]),
]

# Strecken, deren Tiefe aus der 2D-Verkürzung rekonstruiert wird. Die beiden
# Zahlen sind der plausible Bereich der Strecke in Schulterbreiten; sie fangen
# eine verunglückte Längen-Kalibrierung ab (siehe estimate_px_lengths).
DEPTH_SEGMENTS = {
    "upperarm_l": (POSE["left_shoulder"],  POSE["left_elbow"],    0.65, 1.05),
    "lowerarm_l": (POSE["left_elbow"],     POSE["left_wrist"],    0.60, 0.95),
    "upperarm_r": (POSE["right_shoulder"], POSE["right_elbow"],   0.65, 1.05),
    "lowerarm_r": (POSE["right_elbow"],    POSE["right_wrist"],   0.60, 0.95),
    "thigh_l":    (POSE["left_hip"],       POSE["left_knee"],     0.90, 1.50),
    "calf_l":     (POSE["left_knee"],      POSE["left_ankle"],    0.90, 1.45),
    "thigh_r":    (POSE["right_hip"],      POSE["right_knee"],    0.90, 1.50),
    "calf_r":     (POSE["right_knee"],     POSE["right_ankle"],   0.90, 1.45),
    "shoulders":  (POSE["right_shoulder"], POSE["left_shoulder"], 0.95, 1.05),
    "ears":       (POSE["right_ear"],      POSE["left_ear"],      0.25, 0.55),
}
ARM_SEGMENTS = ("upperarm_l", "lowerarm_l", "upperarm_r", "lowerarm_r")

LEN_PCT = 95
LEN_MIN_SAMPLES = 8
MAX_OOP_DEG = 55.0
DEPTH_SIGN_SMOOTH_S = 0.5
DEPTH_MAG_SMOOTH_S = 0.25
DEPTH_SIGN_SCALE = 0.35

# 2D-Quellen (RTMLib ohne 3D-Lifting) liefern keine Tiefe: pose_world hat
# z = 0. Die Verkuerzung sagt dann nur, *wie weit* ein Segment aus der
# Bildebene kippt, nicht in welche Richtung. Fuer Arme ist die Richtung
# bekannt - Gebaerden finden vor dem Koerper statt. -1 heisst "nach vorn"
# im Armature-Raum (x rechts, y Tiefe, z oben); +1 waere hinter dem Rumpf.
# Nur fuer Arme; Rumpf/Kopf bleiben ohne Tiefe, sonst kippt der Oberkoerper.
DEPTH_2D_FORWARD = -1.0

# Daumen-Regel gegen zu grossen Tiefenausschlag: der Oberarm steht fast
# immer senkrecht, seine Verkuerzung ist meist Perspektive/Rauschen. Die
# Handfuehrung kommt aus dem Unterarm, deshalb wird dort voll gerechnet.
DEPTH_WEIGHT = {"upperarm_l": 0.5, "upperarm_r": 0.5,
                "lowerarm_l": 1.0, "lowerarm_r": 1.0}

# Auch eine gestreckte Hand liegt vor dem Rumpf, nicht in seiner Ebene. Ohne
# 3D-Daten fehlt diese absolute Tiefe komplett: die Verkuerzung kennt nur den
# Anteil *senkrecht* zur Bildebene. Bei "Ablauf" stehen die Unterarme fast
# bildparallel, die Haende landen dadurch mitten im Brustkorb. Dieser Anteil
# der Segmentlaenge wird deshalb immer nach vorn addiert (nur 2D, nur Arme).
DEPTH_2D_BASE = {"upperarm_l": 0.20, "upperarm_r": 0.20,
                 "lowerarm_l": 0.45, "lowerarm_r": 0.45}

# Faust aus der 2D-Hand: Abstand Handgelenk -> Fingerspitze geteilt durch die
# Handflaechenlaenge. Offen ist das Verhaeltnis gross, in der Faust klein.
# Die Werte sind Erfahrungswerte aus den gsg-Videos (RTMLib/COCO-Hand).
FIST_REACH_OPEN = 2.4
FIST_REACH_CLOSED = 1.15

# Gestrecktheit je Finger (Sehne / Fingerpolygon, siehe hands.finger_straight).
# Ab dem oberen Wert gilt ein Finger als ganz gestreckt und wird nicht mehr
# eingerollt, auch wenn die globale Faust geschlossen ist. Das trennt z. B.
# bei "Ablauf" den zeigenden Zeigefinger von den eingerollten uebrigen Fingern.
FINGER_STRAIGHT_LO = 0.90
FINGER_STRAIGHT_HI = 0.985

# Damit ein einzelner gestreckter Finger nicht nur ein Messfehler ist, muss er
# ueber die Zeit bestehen: Anteil der Frames mit Gestrecktheit > SURE in einem
# Fenster von etwa einer halben Sekunde. Erst ab FRAC wird der Finger wirklich
# gestreckt gehalten - sonst wuerde z. B. bei "Auto" jeder kurze Ausreisser als
# ausgestreckter Finger durchschlagen.
FINGER_STRAIGHT_SURE = 0.95
FINGER_STRAIGHT_FRAC = 0.40
FINGER_STRAIGHT_FRAC_SPAN = 0.25

TINY_GAP_S = 0.12
GAP_FULL_S = 0.40
GAP_BLEND_S = 0.25
HAMPEL_WIN = 7
QUAT_SPIKE_DEG = 20.0
VMAX_DEG_S = 1200.0
JUMP_SEG_S = 0.25
WEAK_CONF = 0.02
FRAME_MARGIN = 0.05

SMOOTH_SEC = 0.20

# Welche Arme dem Bild folgen.
#   "both"  beide Arme (Standard fuer echte 3D-/RTMLib-Aufnahmen)
#   "sign"  nur der Gebaerdearm; der andere Arm haengt entspannt herab
#   "auto"  "sign" fuer .pev (dort ist der zweite Arm stark verdeckt/verrauscht),
#           sonst "both"
ARM_MODE = "auto"

# Entspannte Armhaltung im Armature-Raum (x rechts, y Tiefe, z oben). Der
# Gebaerdearm zeigt beim gsg-Material nach links, der andere haengt herab.
RELAXED_ARM_UP = {"l": (0.22, 0.0, -0.975), "r": (-0.22, 0.0, -0.975)}
RELAXED_ARM_LOW = {"l": (0.10, 0.0, -0.995), "r": (-0.10, 0.0, -0.995)}

BONE_VARIANTS = {
    "spine":      ["spine_01", "spine", "Spine", "DEF-spine", "chest", "spine_02"],
    "neck":       ["neck_01", "neck", "Neck", "DEF-neck"],
    "head":       ["head", "Head", "DEF-head"],
    "upperarm_l": ["upperarm_l", "upper_arm_l", "UpperArm_L", "DEF-upper_arm.L"],
    "lowerarm_l": ["lowerarm_l", "lower_arm_l", "forearm_l", "DEF-forearm.L"],
    "upperarm_r": ["upperarm_r", "upper_arm_r", "UpperArm_R", "DEF-upper_arm.R"],
    "lowerarm_r": ["lowerarm_r", "lower_arm_r", "forearm_r", "DEF-forearm.R"],
    "thigh_l":    ["thigh_l", "upper_leg_l", "Thigh_L", "DEF-thigh.L"],
    "calf_l":     ["calf_l", "lower_leg_l", "shin_l", "DEF-shin.L"],
    "foot_l":     ["foot_l", "Foot_L", "DEF-foot.L"],
    "thigh_r":    ["thigh_r", "upper_leg_r", "Thigh_R", "DEF-thigh.R"],
    "calf_r":     ["calf_r", "lower_leg_r", "shin_r", "DEF-shin.R"],
    "foot_r":     ["foot_r", "Foot_R", "DEF-foot.R"],
}

LEG_SEGMENTS = ("thigh_l", "calf_l", "thigh_r", "calf_r")

NECK_SHARE = 0.35
HEAD_MAX_DEG = 55.0
ROLL_SMOOTH_FACTOR = 2.0
ROLL_MIN_SIN = 0.25
ROLL_VMAX_DEG_S = 600.0
WRIST_MAX_DEG = 65.0
WRIST_VMAX_DEG_S = 900.0
HIP_MIN_FRAC = 0.5

FINGER_CURL_MAX = {
    "thumb": (55, 65, 65), "index": (95, 110, 80), "middle": (95, 110, 80),
    "ring": (95, 110, 80), "pinky": (95, 100, 80),
}

FINGER_SPREAD_MAX = {"thumb": 35, "index": 25, "middle": 14, "ring": 22, "pinky": 30}

FINGER_PRESETS = {
    "relaxed":   {"thumb": (0.30, 0.20, 0.10), "index": (0.10, 0.30, 0.50),
                  "middle": (0.10, 0.40, 0.60), "ring": (0.20, 0.50, 0.70),
                  "pinky":  (0.30, 0.60, 0.80)},
    "open_hand": {"thumb": (0.05, 0.05, 0.05), "index": (0.0, 0.0, 0.0),
                  "middle": (0.0, 0.0, 0.0), "ring": (0.0, 0.0, 0.0),
                  "pinky":  (0.0, 0.0, 0.0)},
}

FINGER_BEND = {
    "thumb":  [(-20, 80), (0, 70), (0, 60)],
    "index":  [(-20, 90), (0, 110), (0, 90)],
    "middle": [(-20, 90), (0, 110), (0, 90)],
    "ring":   [(-20, 90), (0, 110), (0, 90)],
    "pinky":  [(-20, 90), (0, 100), (0, 80)],
}
