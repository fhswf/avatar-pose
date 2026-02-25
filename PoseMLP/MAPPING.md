# Blender to COCO Keypoint Mapping

[English](#english) | [Deutsch](#deutsch)

---

<a name="english"></a>
## 🇬🇧 English

This document describes how 3D bone positions from Blender are mapped to the synthetic COCO-WholeBody keypoint format used for training.

Since COCO keypoints represent joints (points) and Blender bones are vectors (Head to Tail), we use proxies (Head or Tail positions) to simulate the keypoints.

> [!NOTE]
> **Sparse Usage**: Although the input vector follows the full **COCO-WholeBody** format (133 Keypoints), we only populate the keypoints relevant for Sign Language (Upper Body + Hands) that are available in the Blender rig. The remaining keypoints (e.g., lower body, face details, feet) remain **0.0**. This maintains compatibility with standard pose estimators.

### Body & Arms

| Blender Bone (Point) | COCO Index | Description |
| :--- | :--- | :--- |
| `head` (head) | **0** | Nose |
| `upperarm_l` (head) | **5** | Left Shoulder |
| `upperarm_r` (head) | **6** | Right Shoulder |
| `lowerarm_l` (head) | **7** | Left Elbow |
| `lowerarm_r` (head) | **8** | Right Elbow |
| `lowerarm_l` (tail) | **9** | Left Wrist |
| `lowerarm_r` (tail) | **10** | Right Wrist |

### Hands

The mapping follows the pattern: `Metacarpal (Head)` -> `_01 (Head)` -> `_02 (Head)` -> `_03 (Tail)`.

#### Left Hand (Indices 92-111)

| Finger | Joint 1 (CMC) | Joint 2 (MCP) | Joint 3 (IP/DIP) | Joint 4 (Tip) |
| :--- | :--- | :--- | :--- | :--- |
| **Thumb** | 92 (`metacarpal`) | 93 (`thumb_01`) | 94 (`thumb_02`) | 95 (`thumb_03` tail) |
| **Index** | 96 | 97 | 98 | 99 |
| **Middle** | 100 | 101 | 102 | 103 |
| **Ring** | 104 | 105 | 106 | 107 |
| **Pinky** | 108 | 109 | 110 | 111 |

#### Right Hand (Indices 113-132)

| Finger | Joint 1 (CMC) | Joint 2 (MCP) | Joint 3 (IP/DIP) | Joint 4 (Tip) |
| :--- | :--- | :--- | :--- | :--- |
| **Thumb** | 113 | 114 | 115 | 116 |
| **Index** | 117 | 118 | 119 | 120 |
| **Middle** | 121 | 122 | 123 | 124 |
| **Ring** | 125 | 126 | 127 | 128 |
| **Pinky** | 129 | 130 | 131 | 132 |

### Notes

- **Proxies**: We map `bone.head` for most joints, but `bone.tail` for the finger tips and wrists to get the correct end-effector position.
- **Double Mapping**: Wrists are mapped to both the body wrist index (9/10) and the hand root index (91/112) for consistency.

### Usage Statistics

| Category | Count | Indices |
| :--- | :--- | :--- |
| **Face** (Nose) | 1 | 0 |
| **Body** (Shoulders, Elbows, Wrists) | 6 | 5, 6, 7, 8, 9, 10 |
| **Hands Root** (Proxies for Wrist) | 2 | 91, 112 |
| **Fingers** (Left & Right) | 40 | 92-111, 113-132 |
| **Total Active** | **49** | |
| **Total Unused** (Zeroed) | 84 | |
| **Total Slots** | **133** | |

**Explanation:**
- **49 Active Keypoints**: Only these indices contain actual data derived from the Blender rig. They cover the upper body (nose, shoulders, arms) and detailed hand movements (fingers).
- **84 Unused Slots**: The remaining indices (legs, feet, detailed face mesh, mouth, eyes) are filled with `0.0`. This sparse usage is intentional as the Sign Language model focuses on manual features and the upper body. The input vector size remains constant (133 * 3) to be compatible with standard COCO-WholeBody architectures.

### Data Logic & Coordinate Systems

#### 1. Input Features (X)
**Source:** Blender `world_space` coordinates.
- **Why?** This simulates the output of real-world pose estimators (like MediaPipe) which provide absolute 3D coordinates relative to the camera or world origin. The model learns **Inverse Kinematics**: deducing internal joint angles from external spatial positions.
- **Example:** A "Head" bone input is simply `[x, y, z]` (e.g., `[0.001, 0.063, 0.948]`). The model sees only this point cloud, not the bone structure.
- **Summary:**
    - **Data:** Absolute 3D positions (Point Cloud).
    - **Goal:** Emulate Webcam/Tracker input.

#### 2. Target Labels (Y)
**Source:** Blender `local` rotation quaternions (converted to Axis-Angle).
- **Why?** We need to reconstruct the pose inside a 3D engine. Since bones are hierarchical (a hand moves because the arm moves), we cannot predict world rotatations directly. We must predict the **local rotation** relative to the parent bone.
- **Process:**
    1. Extract `rotation_quaternion` (w,x,y,z) from JSON.
    2. Convert to **Axis-Angle** vector (3 floats).
    3. Model predicts these 3 floats per bone.
- **Summary:**
    - **Data:** Local rotations (Axis-Angle).
    - **Goal:** Drive 3D Skeletal Animation.

### Final Summary

The training pipeline is designed to solve a specific **Inverse Kinematics (IK)** problem for Sign Language:

1.  **Input:** A "sparse point cloud" of the upper body (49 points), simulating a webcam's view. Coordinates are absolute (`World Space`).
2.  **Model:** A robust, single-frame MLP (`512 -> 256`) that maps these spatial points to internal skeletal angles.
3.  **Output:** A set of bone rotations (`Local Space`) that can be applied to a 3D Rig (in Blender, Unity, or Unreal) to perfectly recreate the pose.

This separation of **World Space Input** and **Local Space Output** ensures the model is usable with any standard tracking camera while producing professional-grade animation data.

---

<a name="deutsch"></a>
## 🇩🇪 Deutsch

Dieses Dokument beschreibt, wie 3D-Knochenpositionen aus Blender auf das synthetische COCO-WholeBody Keypoint-Format abgebildet werden, das für das Training verwendet wird.

Da COCO-Keypoints Gelenke (Punkte) darstellen und Blender-Knochen Vektoren sind (Head bis Tail), verwenden wir Proxies (Head- oder Tail-Positionen), um die Keypoints zu simulieren.

> [!NOTE]
> **Sparse Usage (Dünnbesetzte Nutzung)**: Obwohl der Eingabevektor dem vollständigen **COCO-WholeBody** Format (133 Keypoints) folgt, befüllen wir nur die für die Gebärdensprache relevanten Keypoints (Oberkörper + Hände), die im Blender-Rig verfügbar sind. Die restlichen Keypoints (z.B. Unterkörper, Gesichtsdetails, Füße) bleiben **0.0**. Dies gewährleistet die Kompatibilität mit Standard-Pose-Estimators.

### Körper & Arme

| Blender Bone (Punkt) | COCO Index | Beschreibung |
| :--- | :--- | :--- |
| `head` (head) | **0** | Nase |
| `upperarm_l` (head) | **5** | Linke Schulter |
| `upperarm_r` (head) | **6** | Rechte Schulter |
| `lowerarm_l` (head) | **7** | Linker Ellbogen |
| `lowerarm_r` (head) | **8** | Rechter Ellbogen |
| `lowerarm_l` (tail) | **9** | Linkes Handgelenk |
| `lowerarm_r` (tail) | **10** | Rechtes Handgelenk |

### Hände

Das Mapping folgt dem Muster: `Metacarpal (Head)` -> `_01 (Head)` -> `_02 (Head)` -> `_03 (Tail)`.

#### Linke Hand (Indizes 92-111)

| Finger | Gelenk 1 (CMC) | Gelenk 2 (MCP) | Gelenk 3 (IP/DIP) | Gelenk 4 (Tip) |
| :--- | :--- | :--- | :--- | :--- |
| **Daumen** | 92 (`metacarpal`) | 93 (`thumb_01`) | 94 (`thumb_02`) | 95 (`thumb_03` tail) |
| **Zeigefinger** | 96 | 97 | 98 | 99 |
| **Mittelfinger** | 100 | 101 | 102 | 103 |
| **Ringfinger** | 104 | 105 | 106 | 107 |
| **Kleiner Finger** | 108 | 109 | 110 | 111 |

#### Rechte Hand (Indizes 113-132)

| Finger | Gelenk 1 (CMC) | Gelenk 2 (MCP) | Gelenk 3 (IP/DIP) | Gelenk 4 (Tip) |
| :--- | :--- | :--- | :--- | :--- |
| **Daumen** | 113 | 114 | 115 | 116 |
| **Zeigefinger** | 117 | 118 | 119 | 120 |
| **Mittelfinger** | 121 | 122 | 123 | 124 |
| **Ringfinger** | 125 | 126 | 127 | 128 |
| **Kleiner Finger** | 129 | 130 | 131 | 132 |

### Hinweise

- **Proxies**: Wir mappen `bone.head` für die meisten Gelenke, aber `bone.tail` für die Fingerspitzen und Handgelenke, um die korrekte End-Effektor-Position zu erhalten.
- **Doppelte Zuweisung**: Handgelenke werden sowohl auf den Körper-Handgelenk-Index (9/10) als auch auf den Hand-Wurzel-Index (91/112) gemappt, um Konsistenz zu gewährleisten.

### Nutzungsstatistik

| Kategorie | Anzahl | Indizes |
| :--- | :--- | :--- |
| **Gesicht** (Nase) | 1 | 0 |
| **Körper** (Schultern, Ellbogen, Handgelenke) | 6 | 5, 6, 7, 8, 9, 10 |
| **Handwurzel** (Proxies für Handgelenk) | 2 | 91, 112 |
| **Finger** (Links & Rechts) | 40 | 92-111, 113-132 |
| **Total Aktiv** | **49** | |
| **Total Ungenutzt** (Genullt) | 84 | |
| **Total Slots** | **133** | |

**Erklärung:**
- **49 Aktive Keypoints**: Nur diese Indizes enthalten tatsächliche Daten aus dem Blender-Rig. Sie decken den Oberkörper (Nase, Schultern, Arme) und detaillierte Handbewegungen (Finger) ab.
- **84 Ungenutzte Slots**: Die restlichen Indizes (Beine, Füße, detailliertes Gesichtsnetz, Mund, Augen) sind mit `0.0` gefüllt. Diese "sparse" (spärliche) Nutzung ist beabsichtigt, da sich das Gebärdensprachmodell auf manuelle Merkmale und den Oberkörper konzentriert. Die Größe des Eingabevektors bleibt konstant (133 * 3), um mit Standard-COCO-WholeBody-Architekturen kompatibel zu sein.

### Datenlogik & Koordinatensysteme

#### 1. Eingabe-Features (Input X)
**Quelle:** Blender `world_space` Koordinaten.
- **Warum?** Dies simuliert die Ausgabe von realen Pose-Estimators (wie MediaPipe), die absolute 3D-Koordinaten relativ zur Kamera oder zum Weltursprung liefern. Das Modell lernt **Inverse Kinematik**: Ableitung interner Gelenkwinkel aus externen räumlichen Positionen.
- **Beispiel:** Ein "Kopf"-Knochen-Input ist einfach `[x, y, z]` (z.B. `[0.001, 0.063, 0.948]`). Das Modell sieht nur diese Punktwolke, nicht die Knochenstruktur.
- **Zusammenfassung:**
    - **Daten:** Absolute 3D-Positionen (Punktwolke).
    - **Ziel:** Webcam-/Tracker-Input emulieren.

#### 2. Ziel-Label (Target Y)
**Quelle:** Blender `local` Rotations-Quaternionen (umgewandelt in Axis-Angle).
- **Warum?** Wir müssen die Pose in einer 3D-Engine rekonstruieren. Da Knochen hierarchisch sind (eine Hand bewegt sich, weil sich der Arm bewegt), können wir Weltrotationen nicht direkt vorhersagen. Wir müssen die **lokale Rotation** relativ zum Elternknochen vorhersagen.
- **Prozess:**
    1. Extrahiere `rotation_quaternion` (w,x,y,z) aus JSON.
    2. Wandle um in **Axis-Angle** Vektor (3 Floats).
    3. Modell sagt diese 3 Floats pro Knochen vorher.
- **Zusammenfassung:**
    - **Daten:** Lokale Rotationen (Axis-Angle).
    - **Ziel:** 3D-Skelett-Animation steuern.

### Finale Zusammenfassung

Die Trainings-Pipeline wurde entwickelt, um ein spezifisches Problem der **Inversen Kinematik (IK)** für Gebärdensprache zu lösen:

1.  **Input:** Eine "dünne Punktwolke" des Oberkörpers (49 Punkte), die die Sicht einer Webcam simuliert. Die Koordinaten sind absolut (`World Space`).
2.  **Modell:** Ein robustes, Frame-für-Frame MLP (`512 -> 256`), das diese räumlichen Punkte auf interne Skelett-Winkel abbildet.
3.  **Output:** Ein Satz von Knochenrotationen (`Local Space`), der auf ein 3D-Rig (in Blender, Unity oder Unreal) angewendet werden kann, um die Pose perfekt zu rekonstruieren.

Diese Trennung von **World Space Input** und **Local Space Output** stellt sicher, dass das Modell mit jeder Standard-Tracking-Kamera verwendet werden kann und gleichzeitig professionelle Animationsdaten liefert.
