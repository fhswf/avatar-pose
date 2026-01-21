# Projektplan: Einzeldatei-Test & Vorhersage / Project Plan: Single File Test & Prediction

[Deutsch](#deutsch) | [English](#english)

---

<a name="deutsch"></a>
## 🇩🇪 Deutsch

Dieses Dokument beschreibt den Plan, um das Pose-Modell mit einer echten Blender-Datei zu testen. Dabei werden die originalen Rotationsdaten entfernt, um eine echte Vorhersage zu simulieren.

### Ziel
Verifizierung des Modells durch Vergleich zwischen Original und Vorhersage in Blender.

### Ablauf

#### 1. Quelldatei wählen
Wir verwenden eine spezifische Datei aus dem Datensatz:
`DatasetBlender/pose_0001_seed1000_open_hand_20260113_202427.json`

#### 2. Quaternionen entfernen
Ein Skript (`debug_single_inference.py`) lädt die Datei und löscht die `rotation_quaternion` Einträge. Dies stellt sicher, dass das Modell keine Ground-Truth-Daten sieht.

#### 3. Modell-Input generieren
Aus den verbleibenden Positionsdaten des Skeletts (World Space) werden die Eingabe-Vektoren für das neuronale Netz berechnet.

#### 4. Inferenz & Vorhersage
Das trainierte Modell (`pose_mlp.pt`) berechnet die Gelenkrotationen neu.

#### 5. Ausgabe speichern
Das Ergebnis wird als neue JSON-Datei gespeichert:
`testprediction_pose_0001_seed1000_open_hand_20260113_202427.json`

### Verifizierung

**Manuelle Prüfung:**
Laden Sie sowohl die **Originaldatei** als auch die **`testprediction_...` Datei** in Blender. Vergleichen Sie die Posen visuell.

### Benutzung

Führen Sie folgendes Kommando im Terminal aus:

```bash
python debug_single_inference.py
```

Falls Abhängigkeiten fehlen (z.B. `torch`), installieren Sie diese vorher:

```bash
pip install -r requirements.txt
```

---

<a name="english"></a>
## 🇬🇧 English

This document outlines the plan to test the pose model using a real Blender file. The original rotation data is removed to simulate a real prediction scenario.

### Goal
Verify the model by comparing the original and predicted poses in Blender.

### Workflow

#### 1. Select Source File
We use a specific file from the dataset:
`DatasetBlender/pose_0001_seed1000_open_hand_20260113_202427.json`

#### 2. Strip Quaternions
A script (`debug_single_inference.py`) loads the file and removes `rotation_quaternion` entries. This ensures the model does not see ground truth data.

#### 3. Generate Model Input
The input vectors for the neural network are calculated from the remaining skeleton position data (world space).

#### 4. Inference & Prediction
The trained model (`pose_mlp.pt`) recalculates the joint rotations.

#### 5. Save Output
The result is saved as a new JSON file:
`testprediction_pose_0001_seed1000_open_hand_20260113_202427.json`

### Verification

**Manual Check:**
Load both the **original file** and the **`testprediction_...` file** in Blender. Visually compare the poses.

### Usage

Run the following command in the terminal:

```bash
python debug_single_inference.py
```

If dependencies are missing (e.g., `torch`), install them first:

```bash
pip install -r requirements.txt
```
