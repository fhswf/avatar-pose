# Avatar-Pose – Open-Source Sign Language Avatar / Open-Source Gebärdenavatar

**Avatar-Pose** is an open-source project aimed at creating a fully controllable 3D avatar for **German Sign Language (DGS)**. The goal is to provide tools that allow Deaf users to generate and use their own personalized avatars. The project is developed in collaboration between **Cobtras** and **FH Südwestfalen**.

**Avatar-Pose** ist ein Open-Source-Projekt, das einen vollständig steuerbaren 3D-Avatar für die **Deutsche Gebärdensprache (DGS)** erstellt. Ziel ist es, der Gehörlosengemeinschaft Werkzeuge zur Verfügung zu stellen, mit denen sie eigene personalisierte Avatare erzeugen und nutzen können. Das Projekt wird in Zusammenarbeit zwischen **Cobtras** und der **FH Südwestfalen** entwickelt.

---

## Features / Funktionen

- Extraction of human pose and gesture data for avatar animation
- Support for German Sign Language (DGS) glosses
- Integration with datasets for gloss-aligned avatar generation
- Open-source and community-driven development

- Extraktion von Körperpose- und Gestendaten für Avatar-Animationen
- Unterstützung von Glossen der Deutschen Gebärdensprache (DGS)
- Integration mit Datensätzen für glossenbasierte Avatar-Generierung
- Open-Source-Entwicklung mit Community-Fokus

---

## Related Resources / Verwandte Ressourcen

- **Cobtras** – https://www.cobtras.com
- **FH Südwestfalen** – https://www.fh-swf.de/de/index.php
- **DGS Gloss Dataset (Huggingface)** – [fhswf/sign-language-avatar-gloss-dgs](https://huggingface.co/datasets/fhswf/sign-language-avatar-gloss-dgs)

---

## Usage / Nutzung

The avatar pipeline can be used for:
- Generating sign language animations from glosses
- Research on sign language representation
- Educational tools for the Deaf community

Die Avatar-Pipeline kann verwendet werden für:
- Erstellung von Gebärdenanimationen aus Glossen
- Forschung zur Darstellung von Gebärdensprache
- Bildungswerkzeuge für die Gehörlosengemeinschaft

---

## Pose-Retargeting / Pose retargeting

RTMLib/COCO-WholeBody keypoints can drive the avatar without MediaPipe:

- `retarget/` – landmarks to bone rotations, CLI, tests; details in `retarget/README.md`
- `python -m retarget retarget aufnahme.json -o kurven.json --frames frames.json`
- `AVATAR_PLUGIN=retarget AVATAR_RETARGET_CURVES=kurven.json python main.py` (in `avatar-pose/main`)
- In the webviewer text field press Enter or type the path to a curves/RTMLib JSON.
- Type a word from the gsg list (e.g. `Auto`) to look it up in
  `plugins/gsg/wordlist.csv` and play its pre-extracted
  `plugins/gsg/extracted_videos/<id>.pev`. Playback uses only those PEVs: at
  runtime no videos are downloaded, no RTMLib inference runs and Signdict is
  never contacted. Both arms follow the landmarks; playback starts and ends in
  the avatar's T-pose (the glTF rest pose). If a sign or its PEV is missing,
  the word is fingerspelled following `gsg.py` (`SCH` as one sign).
- Multiple words or letters (`Der Hund läuft`, `H A U S`) can be translated to
  a DGS gloss sequence with `text_to_gloss` and are played sign by sign with a
  short T-pose pause. The model was trained on weather text, so its output is
  only used when **every input token** has a wordlist entry and a PEV and at
  least half of the gloss words are in the wordlist. Otherwise the original
  input words are played: unknown words or words without a PEV are
  fingerspelled (e.g. `Guten Abend Jonas` becomes `G U T E N ABEND J O N A S`).
  This prevents the model from silently dropping a name. `AVATAR_GLOSS=0`
  disables the translation entirely.
- The PEV files are generated offline from the trimmed `videos-cut` clips via
  `DataLoaderDGS/Extractor.py`; there is no runtime video/keypoint cache.
  `AVATAR_RETARGET_ANCHOR=none` disables the T-pose blend.
- `AVATAR_RETARGET_ARM_MODE=sign|both|auto` selects which arms follow the
  landmarks (default `auto`: both for every source, including `.pev`).

### Creating and repairing `.pev` files

`DataLoaderDGS/Extractor.py` converts `videos-cut/**/*.mp4` into
`extracted/**/*.pev`:

```bash
python DataLoaderDGS/Extractor.py --dataset_path /path/to/sign-language-avatar-gloss-dgs
python DataLoaderDGS/Extractor.py --dataset_path ... --limit 3 --overwrite --backup
python DataLoaderDGS/Extractor.py --dataset_path ... --backend official
```

The default backend `rtmlib` uses the installed `rtmlib.Wholebody`
(COCO-WholeBody, 133 points) and needs no 3D lifting. It runs through RTMLib's
OpenCV-DNN engine by default (`--rtmlib-backend opencv`), so the `onnxruntime`
package is not required; `--rtmlib-backend onnxruntime` switches back. The
RTMLib weights themselves are always `.onnx` files. `--backend official`
switches to `RTMPoseEstimator3D`, which additionally requires
`pose_estimation_recognition_utils`. Both backends mirror x, turn y upwards and
rebuild z from the 2D foreshortening, so newly written files do not have the
collapsed depth described below. `--device auto` uses CUDA when the chosen
engine offers it, `--front-sign -1` flips the arm depth.

Older files created by the previous extractor were lifted to 3D, but
`_denormalize_by_bounding_box` only rescales x and y. The z component stays in
the lifting model's normalised range, so it is about 1000× too small
(x/y ≈ 1e5, z ≈ 1e2). `Calculator.compute_final_rotations` reads the points as
Three.js coordinates, so with a constant z every bone lies in the image plane
and the arms disappear inside the torso. A factor cannot fix this, because the
lifted z is not even consistent in direction.

`DataLoaderDGS/repair_pev.py` rebuilds z from the 2D foreshortening
(`sqrt(L² - p²)` plus a forward base, the same idea as `retarget/depth.py`):

```bash
python DataLoaderDGS/repair_pev.py check            # status per file
python DataLoaderDGS/repair_pev.py repair --dry-run # show what would change
python DataLoaderDGS/repair_pev.py repair           # backup *.pev.bak, then rewrite z
python DataLoaderDGS/repair_pev.py sync --repair    # copy from the dataset + fix
```

`x`/`y` and all other fields stay untouched; `--front-sign -1` flips the
reconstructed depth if the arms should point backwards. The Extractor applies
this reconstruction itself, so freshly generated PEVs are ready for the
retarget plugin.

### `.pev`-Dateien erzeugen und reparieren

`DataLoaderDGS/Extractor.py` wandelt `videos-cut/**/*.mp4` in
`extracted/**/*.pev` um:

```bash
python DataLoaderDGS/Extractor.py --dataset_path /pfad/zu/sign-language-avatar-gloss-dgs
python DataLoaderDGS/Extractor.py --dataset_path ... --limit 3 --overwrite --backup
python DataLoaderDGS/Extractor.py --dataset_path ... --backend official
```

Das Standard-Backend `rtmlib` nutzt das installierte `rtmlib.Wholebody`
(COCO-WholeBody, 133 Punkte) und braucht kein 3D-Lifting. Es läuft standardmäßig
über RTMLibs OpenCV-DNN-Engine (`--rtmlib-backend opencv`), dadurch wird das
Paket `onnxruntime` nicht gebraucht; `--rtmlib-backend onnxruntime` schaltet
zurück. Die RTMLib-Gewichte selbst sind immer `.onnx`-Dateien. `--backend
official` schaltet auf `RTMPoseEstimator3D` um, das zusätzlich
`pose_estimation_recognition_utils` voraussetzt. Beide Backends spiegeln x,
drehen y nach oben und bauen z aus der 2D-Verkürzung neu auf; neu erzeugte
Dateien haben also nicht mehr die unten beschriebene plattgedrückte Tiefe.
`--device auto` nutzt CUDA, wenn die gewählte Engine es anbietet,
`--front-sign -1` dreht die Arm-Tiefe.

Ältere Dateien aus dem früheren Extractor wurden in 3D geliftet, aber
`_denormalize_by_bounding_box` skaliert nur x und y zurück. Die z-Komponente
bleibt im Normbereich des Lifters und ist damit rund 1000-mal zu klein
(x/y ≈ 1e5, z ≈ 1e2). `Calculator.compute_final_rotations` liest die Punkte
als Three.js-Koordinaten; mit konstantem z liegt jeder Knochen in der
Bildebene und die Arme verschwinden im Rumpf. Ein Faktor hilft nicht, weil das
geliftete z auch in der Richtung nicht stabil ist.

`DataLoaderDGS/repair_pev.py` baut z deshalb aus der 2D-Verkürzung neu auf
(`sqrt(L² - p²)` plus Grundtiefe nach vorn, dieselbe Idee wie
`retarget/depth.py`). `x`/`y` und alle übrigen Felder bleiben unverändert,
das Original wird als `*.pev.bak` gesichert. `--front-sign -1` dreht die
Tiefe, falls die Arme nach hinten zeigen sollen. Der Extractor wendet die
Korrektur beim Schreiben selbst an; die PEVs sind damit direkt abspielbar.

RTMLib/COCO-WholeBody-Keypoints steuern den Avatar ohne MediaPipe. Die
Kette liegt in `retarget/`, Einzelheiten stehen in `retarget/README.md`.
Ein Wort wie `Auto` wird über `plugins/gsg/wordlist.csv` gesucht und die
vorab mit `DataLoaderDGS/Extractor.py` aus dem zugeschnittenen Video
erzeugte PEV aus `plugins/gsg/extracted_videos` abgespielt. Zur Laufzeit
werden ausschließlich diese Dateien gelesen: kein Video-Download, keine
RTMLib-Inferenz, kein Signdict-Zugriff und kein Keypoint-Cache. Beide Arme
folgen den Landmarks. Die Wiedergabe beginnt und endet in der T-Pose
(Ruhepose des glTF); ein zusätzliches Bewegungsfenster gibt es nicht, weil
die Clips bereits zugeschnitten sind. Fehlt ein Wort als Gebärde oder seine
PEV, wird es nach dem Vorbild von `gsg.py` buchstabiert (`SCH` zusammen).

Mehrere Wörter oder Buchstaben (z. B. `Der Hund läuft`, `H A U S`) können
über `text_to_gloss` (Deutsch → DGS-Gloss) in eine Glossfolge übersetzt und
Gebärde für Gebärde mit kurzer T-Pose-Pause abgespielt werden. Das Modell ist
auf Wettertexten trainiert; deshalb wird seine Übersetzung nur benutzt, wenn
**alle eingegebenen Wörter** in der Wortliste stehen und eine PEV haben und
mindestens die Hälfte der Gloss-Wörter in der Wortliste steht. Andernfalls
werden die ursprünglichen Wörter gespielt: unbekannte Wörter oder Wörter ohne
PEV werden buchstabiert (z. B. `Guten Abend Jonas` wird zu
`G U T E N ABEND J O N A S`). So kann das Modell keinen Namen einfach
fallen lassen. `AVATAR_GLOSS=0` schaltet die Übersetzung ganz ab.

--

##Contribution / Mitwirkung

Contributions are welcome. We aim for a community-driven project that Deaf users and researchers can maintain and extend.
Beiträge sind willkommen. Wir streben ein Community-getriebenes Projekt an, das von Gehörlosen und Forschenden gepflegt und erweitert werden kann.

---

##License / Lizenz

This project is open-source under the MIT License.
Dieses Projekt ist Open-Source unter der MIT-Lizenz.

---

##Acknowledgements / Danksagung

Developed in collaboration between Cobtras and FH Südwestfalen, leveraging the DGS gloss dataset to support personalized avatar generation for the Deaf community.
Entwickelt in Zusammenarbeit zwischen Cobtras und der FH Südwestfalen, unter Nutzung des DGS-Glossendatensatzes zur Unterstützung personalisierter Avatar-Erstellung für die Gehörlosengemeinschaft.

---

## Contributors / Beteiligte Personen

This project is a collaborative effort between **Cobtras**, **FH Südwestfalen**, and individual contributors. Special thanks to:

Dieses Projekt ist eine Zusammenarbeit zwischen **Cobtras**, der **FH Südwestfalen** und einzelnen Mitwirkenden. Besonderer Dank an:

- Jonas David Stephan
- Prof. Dr. Christian Gawron
- Ann-Kristin Schulte
- Carolin Gottschalk
- Galiya Yelubayeva
- Dirk Lewanczeck

If you contributed to the project and your name is missing, please contact us to be added.

Wenn du zum Projekt beigetragen hast und dein Name fehlt, kontaktiere uns bitte, damit wir dich hinzufügen können.
