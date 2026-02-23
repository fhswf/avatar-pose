# Avatar Pose & JSON Replay Tool (Blender 4.x)

[Deutsch](#deutsch) | [English](#english)

---

<a name="english"></a>
## 🇬🇧 English

This repository contains a Blender tool for **sign-language avatar posing**:

- ✅ Generate **random upper-body poses**
- ✅ **Export** bone transforms to JSON (**local + world**)
- ✅ **Import / replay** a JSON pose back onto the rig (high fidelity via `matrix_basis`)
- ✅ Simple UI panel in the 3D Viewport (**N-panel → “Avatar” tab**)

---

### Contents

- Blender avatar mesh/rig file (`KIM_avatar.blend`)
- `blender_avatar_pose.py` (the script you paste/run in Blender)

> **Default armature name:** `KIM_caucasian_male`  
> If your rig has a different name, you can change it in the UI or edit `DEFAULT_ARMATURE_NAME` in the script.

---

### Requirements

- **Blender 4.x**
- A `.blend` file containing the avatar mesh + **armature** (rig)

---

### Quick Start

#### 1) Open the avatar file
1. Open **Blender**
2. `File → Open…`
3. Select `KIM_avatar.blend`

#### 2) Add the script

**Option A — Paste into Blender Text Editor**
1. Switch to the **Scripting** workspace
2. Click `New`
3. Paste the full content of `blender_avatar_pose.py`
4. Press **Run Script** (▶)

**Option B — Open script file**
1. Go to **Scripting**
2. Text Editor → `Open…`
3. Select `blender_avatar_pose.py`
4. Press **Run Script**

After running, Blender prints:
- `✅ Avatar Pose & JSON Replay Tool registered.`
- `Open: Viewport → N panel → Avatar tab`

### Where to find the UI Panel

1. Go to the **Layout** workspace
2. In the 3D Viewport press **N**
3. Open the **Avatar** tab
4. You will see:

**“Avatar Pose & JSON (Replay)”**

---

### Using the Tool

#### A) Select the armature
In **Avatar → Armature**:
- Set **Name** to your armature object name (e.g. `KIM_caucasian_male`)

If unknown:
- Check the **Outliner**
- Copy the armature object name

---

#### B) Generate a random pose

Options:

- **Upper Body Only** – Limits to upper body (recommended)
- **Allow Small Pelvis Motion** – Subtle root/hips movement
- **Randomness (0.0–2.0)** – Higher = more variation
- **Blend (0.0–1.0)**  
  - `1.0` = fully replace  
  - lower = softer changes
- **Include Fingers**
- **Finger Preset**: `random`, `relaxed`, `fist`, `pointing`, `open_hand`, `grasping`
- **Finger Intensity**

Click:
- **Apply Random Pose**

> Tip: If the pose becomes unrealistic, reduce **Randomness** or increase **Blend** gradually.

---

#### C) Export bone data to JSON

In **Avatar → Export**:

- **Export Dir** – Target folder  
  If empty → `~/Desktop/Blender_Bone_Exports`

Click:
- **Export JSON**

Result:
- File like: `bone_data_full_YYYYMMDD_HHMMSS.json`
- Includes:
  - Metadata
  - Local channels (location/rotation/scale + `matrix_basis`)
  - Armature-space data
  - World-space data

---

#### D) Import / replay a pose

In **Avatar → Import / Replay Pose from JSON**:

- **Import JSON Path**
- **Use Matrix Basis (best)** ✅ recommended
- **Apply Upper Body Only**
- **Ignore Missing Bones**

Click:
- **Apply Pose from JSON**

Status example:
- `Pose applied: 135 bones (missing:0, skipped:0)`

---

### E) Reset pose
Click:
- **Reset Pose**  
Clears pose transforms (same as *Pose → Clear Transform → All*).

---

## Expected Workflow (Typical)

1. Open avatar `.blend`
2. Run script
3. Generate a pose → export JSON
4. Later: open the same avatar (or a compatible rig)
5. Import JSON → pose is replayed

---

## Notes on Compatibility

- JSON replay assumes:
  - same (or mostly similar) bone names
  - same rig structure is ideal
- If you changed rigs:
  - enable **Ignore Missing Bones**
  - optionally enable **Apply Upper Body Only**

---

## Troubleshooting

### “Armature not found”
- The tool looks for the armature name in:
  1) the UI field **Avatar Armature Name**
  2) fallback: `KIM_caucasian_male`
- Fix: set the correct name in the UI.

### The “Avatar” tab doesn’t appear
- Make sure you pressed **Run Script** in the Text Editor.
- If still missing:
  - run the script again
  - restart Blender and run again

### Import fails: “JSON path not found”
- Ensure **Import JSON Path** points to an existing file on disk.
- If you copied the JSON from elsewhere, save it locally and reselect it.

### Pose looks off after import
- Ensure **Use Matrix Basis (best)** is enabled.
- Ensure the rig matches the exported rig.
- If the rig differs, try:
  - **Apply Upper Body Only**
  - **Ignore Missing Bones**


  <a name="deutsch"></a>
## 🇩🇪 Deutsch

Dieses Repository enthält ein Blender-Tool für **Gebärdensprach-Avatare**:

- ✅ Erzeugt **zufällige Oberkörper-Posen**
- ✅ **Exportiert** Knochen-Transformationen als JSON (**lokal + Weltkoordinaten**)
- ✅ **Import / Replay** einer JSON-Pose auf das Rig (`matrix_basis`)
- ✅ UI-Panel im 3D-Viewport (**N-Panel → Tab „Avatar“**)

---

## Inhalt

- Blender-Datei mit Avatar-Mesh/Rig (`KIM_avatar.blend`)
- `blender_avatar_pose.py` (das Script, das du in Blender einfügst/ausführst)

> **Standard-Armature-Name:** `KIM_caucasian_male`  
> Wenn dein Rig anders heißt, kannst du den Namen im UI ändern oder `DEFAULT_ARMATURE_NAME` im Script anpassen.

---

## Voraussetzungen

- **Blender 4.x** (getestet mit Blender 4.x API)
- Eine `.blend`-Datei mit Avatar-Mesh + **Armature** (Rig)

---

## Quick Start

### 1) Avatar-Datei öffnen
1. **Blender** öffnen
2. `Datei → Öffnen…`
3. Die Avatar-`.blend` auswählen (z. B. `KIM_avatar.blend`)

### 2) Script hinzufügen
Du hast zwei Optionen:

#### Option A — In den Text-Editor kopieren
1. In den Workspace **Scripting** wechseln (oben)
2. Im **Text Editor**:
   - `Neu`
   - den kompletten Inhalt von `blender_avatar_pose.py` einfügen
3. **Run Script** (▶) klicken

#### Option B — Script-Datei in Blender öffnen
1. Workspace **Scripting**
2. Text Editor → `Öffnen…`
3. `blender_avatar_pose.py` auswählen
4. **Run Script** (▶) klicken

Nach dem Ausführen zeigt Blender im Output:
- `✅ Avatar Pose & JSON Replay Tool registered.`
- `Open: Viewport → N panel → Avatar tab`

---

## Wo finde ich das UI-Panel?

1. In den Workspace **Layout** wechseln (oder in einer 3D-Ansicht bleiben)
2. Im 3D-Viewport **N** drücken (Sidebar ein-/ausblenden)
3. Den Tab **Avatar** öffnen
4. Dort erscheint das Panel:

**„Avatar Pose & JSON (Replay)“**

---

## Tool benutzen

### A) Armature (Rig) auswählen
Unter **Avatar → Armature**:
- **Name** auf den Armature-Objektnamen setzen (z. B. `KIM_caucasian_male`)

Wenn du den Namen nicht kennst:
- Im **Outliner** (oben rechts) die Armature suchen (Strichmännchen-Icon)
- Den Objektnamen kopieren und ins Feld einfügen

---

### B) Zufällige Pose erzeugen
Unter **Avatar → Random Pose**:

- **Upper Body Only**  
  Beschränkt das Posing auf den Oberkörper (empfohlen für Gebärdensprache).
- **Allow Small Pelvis Motion**  
  Erlaubt leichte Bewegungen in Root/Hüfte (wirkt natürlicher).
- **Randomness (0.0–2.0)**  
  Höher = mehr Variation.
- **Blend (0.0–1.0)**  
  Wie stark die neue Pose die aktuelle ersetzt:  
  - `1.0` = komplett ersetzen  
  - kleinere Werte = weichere Änderungen
- **Include Fingers**  
  Fügt Finger-Posing hinzu.
- **Finger Preset**  
  `random`, `relaxed`, `fist`, `pointing`, `open_hand`, `grasping`
- **Finger Intensity**  
  Stärke von Fingerkrümmung/-spreizung.

Klicken:
- **Apply Random Pose**

> Tipp: Wenn die Pose unnatürlich wird, **Randomness** reduzieren und/oder **Blend** schrittweise erhöhen.

---

### C) Knochen-Daten als JSON exportieren
Unter **Avatar → Export**:

- **Export Dir**  
  Zielordner für die JSON-Dateien.  
  Wenn leer, wird exportiert nach:
  - `~/Desktop/Blender_Bone_Exports`

Klicken:
- **Export JSON**

Du erhältst:
- Eine Datei z. B.: `bone_data_full_YYYYMMDD_HHMMSS.json`
- Enthält:
  - Metadaten (Blender-Version, Armature-Name, Bone-Anzahl)
  - pro Bone:
    - lokale Kanäle (Location/Rotation/Scale + `matrix_basis`)
    - Armature-Space Head/Tail
    - World-Space Head/Tail + World-Rotation + World-Matrix

---

### D) Pose aus JSON importieren / replayen
Unter **Avatar → Import / Replay Pose from JSON**:

- **Import JSON Path**  
  Exportierte JSON-Datei auswählen.
- **Use Matrix Basis (best)** ✅  
  Empfohlen: bestes Replay mit höchster Genauigkeit.
- **Apply Upper Body Only**  
  Wendet nur Oberkörper-Bones an (praktisch bei abweichendem Unterkörper).
- **Ignore Missing Bones**  
  Macht weiter, auch wenn Bones in JSON fehlen oder im Rig anders heißen.

Klicken:
- **Apply Pose from JSON**

Du siehst eine Meldung wie:
- `Pose applied: 135 bones (missing:0, skipped:0)`

---

### E) Pose zurücksetzen
Klicken:
- **Reset Pose**  
Setzt Pose-Transforms zurück (entspricht *Pose → Transform löschen → Alle*).

---

## Typischer Workflow

1. Avatar `.blend` öffnen
2. Script ausführen
3. Pose erzeugen → JSON exportieren
4. Später: Avatar/kompatibles Rig öffnen
5. JSON importieren → Pose wird reproduziert

---

## Hinweise zur Kompatibilität

- Replay funktioniert am besten bei:
  - gleichen (oder sehr ähnlichen) Bone-Namen
  - gleicher Rig-Struktur
- Wenn du ein anderes Rig nutzt:
  - **Ignore Missing Bones** aktivieren
  - optional **Apply Upper Body Only** aktivieren

---

## Troubleshooting

### „Armature not found“
- Das Tool sucht den Armature-Namen in:
  1) UI-Feld **Avatar Armature Name**
  2) Fallback: `KIM_caucasian_male`
- Lösung: richtigen Namen im UI eintragen.

### Der Tab „Avatar“ erscheint nicht
- Prüfe, ob du **Run Script** (▶) geklickt hast.
- Falls weiterhin nichts erscheint:
  - Script erneut ausführen
  - Blender neu starten und Script erneut ausführen

### Import schlägt fehl: „JSON path not found“
- Prüfe, ob **Import JSON Path** wirklich auf eine existierende Datei zeigt.
- Wenn du die JSON kopiert hast: lokal speichern und neu auswählen.

### Pose sieht nach Import „komisch“ aus
- **Use Matrix Basis (best)** aktivieren.
- Prüfen, ob Rig/Armature zum Export passt.
- Bei abweichendem Rig:
  - **Apply Upper Body Only**
  - **Ignore Missing Bones**