# COCOModel - PoseMLP

[English](#english) | [Deutsch](#deutsch)

---

<a name="english"></a>
## 🇬🇧 English

### Description

This project enables the training of a neural network (`PoseMLP`) that maps 3D keypoints (in COCO format) to local bone rotations (Axis-Angle / Quaternion) for a 3D character (e.g. for Blender).

The approach is **Frame-by-Frame** (Simple MLP):
- **Input**: Flattened vector of 133 Keypoints * 3 coordinates (x, y, z).
- **Output**: Flattened vector of Axis-Angle rotations for a defined set of bones.

For details on the data mapping, see [MAPPING.md](MAPPING.md).

### Project Structure

- `model.py`: Architecture of the MLP (Input -> 512 -> 256 -> Output).
- `train.py`: Training script (Split 80/20, Validation, Early Stopping).
- `dataset.py`: Data loading and processing (`PoseDataset`).
- `infer.py`: Inference script for new `.pev` data.
- `config.py`: Central configuration (Keypoints, Bones, Device).
- `rotations.py`: Math helpers (Quaternion <-> Axis-Angle).
- `paths.py`: Path helpers.
- `MAPPING.md`: Documentation of the Blender-to-COCO keypoint mapping.

### Usage

#### 1. Training

Ensure your training data (JSON) is in the `DatasetBlender` folder.

```bash
python train.py
```

This will:
- Train the model.
- Save the best model to `pose_mlp.pt`.
- Generate a loss plot `loss_plot.png`.

#### 2. Inference

Ensure your input data (`.pev` files) is in the `PevData` folder.

```bash
python infer.py
```

This will:
- Load `pose_mlp.pt`.
- Process all `.pev` files in `PevData`.
- Save the predicted poses as JSON files in `InferOut/<filename>/`.

### Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib

Install dependencies via:
```bash
pip install -r requirements.txt
```

---

<a name="deutsch"></a>
## 🇩🇪 Deutsch

### Beschreibung

Dieses Projekt ermöglicht das Training eines neuronalen Netzes (`PoseMLP`), das 3D-Keypoints (im COCO-Format) auf lokale Knochenrotationen (Axis-Angle / Quaternion) für einen 3D-Charakter (z.B. für Blender) abbildet.

Der Ansatz ist **Frame-für-Frame** (Einfaches MLP):
- **Eingabe (Input)**: Flacher Vektor aus 133 Keypoints * 3 Koordinaten (x, y, z).
- **Ausgabe (Output)**: Flacher Vektor aus Axis-Angle Rotationen für eine definierte Menge an Knochen.

Details zum Daten-Mapping finden Sie in [MAPPING.md](MAPPING.md).

### Projektstruktur

- `model.py`: Architektur des MLP (Input -> 512 -> 256 -> Output).
- `train.py`: Trainingsskript (Split 80/20, Validierung, Early Stopping).
- `dataset.py`: Datenladen und -verarbeitung (`PoseDataset`).
- `infer.py`: Inferenz-Skript für neue `.pev` Daten.
- `config.py`: Zentrale Konfiguration (Keypoints, Knochen, Gerät).
- `rotations.py`: Mathematische Hilfsfunktionen (Quaternion <-> Axis-Angle).
- `paths.py`: Pfad-Hilfsfunktionen.
- `MAPPING.md`: Dokumentation des Blender-zu-COCO Keypoint Mappings.

### Verwendung

#### 1. Training

Stellen Sie sicher, dass Ihre Trainingsdaten (JSON) im Ordner `DatasetBlender` liegen.

```bash
python train.py
```

Dies wird:
- Das Modell trainieren.
- Das beste Modell als `pose_mlp.pt` speichern.
- Einen Loss-Plot `loss_plot.png` erstellen.

#### 2. Inferenz (Vorhersage)

Stellen Sie sicher, dass Ihre Eingabedaten (`.pev` Dateien) im Ordner `PevData` liegen.

```bash
python infer.py
```

Dies wird:
- Das Modell `pose_mlp.pt` laden.
- Alle `.pev` Dateien in `PevData` verarbeiten.
- Die vorhergesagten Posen als JSON-Dateien im Ordner `InferOut/<filename>/` speichern.

### Voraussetzungen

- Python 3.x
- PyTorch
- NumPy
- Matplotlib

Installation der Abhängigkeiten via:
```bash
pip install -r requirements.txt
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
