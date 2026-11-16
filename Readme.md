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
  `plugins/gsg/wordlist.csv`, load its SignDict video, extract
  COCO-WholeBody keypoints with RTMLib and play the result. Both arms follow
  the landmarks; the clip starts and ends in the avatar's T-pose (the glTF
  rest pose). `.pev` files are only a fallback because many of them are
  corrupted by 3D lifting.
- The video/keypoint cache lives in `main/plugins/gsg/video_cache` (ignored by
  git). `AVATAR_RETARGET_CACHE` points it elsewhere,
  `AVATAR_RETARGET_ANCHOR=none` disables the T-pose blend.
- `AVATAR_RETARGET_ARM_MODE=sign|both|auto` selects which arms follow the
  landmarks (default `auto`: both for RTMLib video, signing arm for `.pev`).

### Creating and repairing `.pev` files

`DataLoaderDGS/Extractor.py` converts `videos-cut/**/*.mp4` into
`extracted/**/*.pev`:

```bash
python DataLoaderDGS/Extractor.py --dataset_path /path/to/sign-language-avatar-gloss-dgs
python DataLoaderDGS/Extractor.py --dataset_path ... --limit 3 --overwrite
python DataLoaderDGS/Extractor.py --dataset_path ... --backend official
```

The default backend `rtmlib` uses the installed `rtmlib.Wholebody`
(COCO-WholeBody, 133 points) and needs no 3D lifting. `--backend official`
switches to `RTMPoseEstimator3D`, which additionally requires
`pose_estimation_recognition_utils`. Both backends mirror x, turn y upwards and
rebuild z from the 2D foreshortening, so newly written files do not have the
collapsed depth described below. `--device auto` uses CUDA when onnxruntime
offers it, `--front-sign -1` flips the arm depth.

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
reconstructed depth if the arms should point backwards. The retarget plugin
does not need this step because it reads the keypoints straight from the video.

### `.pev`-Dateien erzeugen und reparieren

`DataLoaderDGS/Extractor.py` wandelt `videos-cut/**/*.mp4` in
`extracted/**/*.pev` um:

```bash
python DataLoaderDGS/Extractor.py --dataset_path /pfad/zu/sign-language-avatar-gloss-dgs
python DataLoaderDGS/Extractor.py --dataset_path ... --limit 3 --overwrite
python DataLoaderDGS/Extractor.py --dataset_path ... --backend official
```

Das Standard-Backend `rtmlib` nutzt das installierte `rtmlib.Wholebody`
(COCO-WholeBody, 133 Punkte) und braucht kein 3D-Lifting. `--backend official`
schaltet auf `RTMPoseEstimator3D` um, das zusätzlich
`pose_estimation_recognition_utils` voraussetzt. Beide Backends spiegeln x,
drehen y nach oben und bauen z aus der 2D-Verkürzung neu auf; neu erzeugte
Dateien haben also nicht mehr die unten beschriebene plattgedrückte Tiefe.
`--device auto` nutzt CUDA, wenn onnxruntime es anbietet, `--front-sign -1`
dreht die Arm-Tiefe.

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
Tiefe, falls die Arme nach hinten zeigen sollen. Das Retarget-Plugin braucht
den Schritt nicht, weil es die Keypoints direkt aus dem Video liest.

RTMLib/COCO-WholeBody-Keypoints steuern den Avatar ohne MediaPipe. Die
Kette liegt in `retarget/`, Einzelheiten stehen in `retarget/README.md`.
Ein Wort wie `Auto` wird über `plugins/gsg/wordlist.csv` gesucht; das
zugehörige SignDict-Video wird geladen, mit RTMLib ausgewertet und
abgespielt. Beide Arme folgen den Landmarks. Die Wiedergabe beginnt und
endet in der T-Pose (Ruhepose des glTF); aus dem Video wird nur der
Abschnitt mit erhobenen Händen verwendet. Die `.pev`-Dateien dienen nur
noch als Notnagel, weil viele durch das 3D-Lifting unbrauchbar sind.
Video- und Keypoint-Cache: `main/plugins/gsg/video_cache` (gitignoriert).

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
