# retarget – RTMLib-Landmarks zu Knochendrehungen

Portierung der Winkel- und Retarget-Logik aus `VideoPose/signpose` auf
**COCO-WholeBody (133 Punkte)**, wie sie RTMLib und später Sapiens1 liefern.
Die Kette rechnet ohne Blender und ohne MediaPipe: Eingang sind Keypoints,
Ausgang sind Quaternionen je Knochen und daraus die Node-Rotationen für den
bestehenden Webviewer. Für Wörter aus der gsg-Liste kommen die Keypoints
ausschließlich aus vorab extrahierten `.pev`-Dateien; die Retarget-Kette selbst
sieht nur noch Keypoints. Zur Laufzeit werden keine Videos geladen, keine
Keypoints berechnet und Signdict nicht kontaktiert.

```
videos-cut/*.mp4 ──DataLoaderDGS/Extractor.py──> extracted_videos/*.pev
    ──rtm_adapter──> Landmark-Sequenz ──landmarks.process──> ──retarget.solve──>
    Kurven (w,x,y,z) ──webviewer.frames/anchor_rest──>
    {knochen: {"rotation": [x,y,z,w]}} ──WebSocket──> webviewer/index.js
```

Der Webviewer bleibt unverändert. Sein Vertrag ist
`{knochen: {"rotation": [x, y, z, w]}}` in Three.js-Reihenfolge; intern
rechnet das Paket durchgehend mit `(w, x, y, z)` wie Blender.

## Installation

Nur `numpy` wird zum Rechnen gebraucht, `pygltflib` für den Brückenschritt zur
GLB-Datei:

```bash
pip install numpy pygltflib
```

Für die Pose-Extraktion selbst ist RTMLib nötig (`avatar-pose/main/pose_estimator_2d.py`).
Tests laufen mit `pytest`.

## Kommandozeile

Alle Befehle laufen aus dem Verzeichnis `avatar-pose/` (dort liegt das Paket
`retarget/`).

```bash
# 1. RTMLib-JSON in eine Landmark-Sequenz umwandeln
python -m retarget prepare rtmlib.json -o aufnahme.json --width 1280 --height 720 --fps 25

# 2. Güte der Aufnahme ansehen
python -m retarget metrics aufnahme.json --process

# 3. Retargeting inklusive Webviewer-Frames
python -m retarget retarget rtmlib.json -o kurven.json --frames frames.json \
    --width 1280 --height 720 --fps 25

# 4. Nur Kurven -> Webviewer-Frames
python -m retarget bridge kurven.json -o frames.json
```

`retarget` erkennt selbst, ob die Eingabe eine rohe RTMLib-Aufnahme oder eine
bereits konvertierte Sequenz ist. `--full-body` nimmt die Beine mit, sonst
bleiben sie in Ruhe. Weitere Schalter: `--arm-depth`, `--spine-depth`,
`--min-conf`, `--hand-min`, `--smooth-sec`, `--finger-fallback`,
`--arm-mode`, `--no-snap-wrist`, `--raw`.

Standard-Rig ist `retarget/data/rig_kim.json` (56 Knochen, KIM), Standard-GLB
ist `avatar-pose/webviewer/media/avatar.glb`. Beide lassen sich mit `--rig`
und `--glb` überschreiben.

`pose_estimator_2d.export_to_json` schreibt keine Bildmaße. Ohne
`--width/--height` schätzt der Adapter sie aus den größten Keypoint-Koordinaten;
für exakte Pixel-Längen besser die Werte des Videos mitgeben.

## Unterstützte Eingaben

`rtm_adapter.sequence_from_rtmlib` nimmt:

* `PoseResult` / `VideoResult` aus `pose_estimator_2d.py` direkt,
* das JSON von `export_to_json` (`{"frames": [...]}`),
* die Liste aus `process_side_by_side_video` (`left`/`right`),
* eine nackte Liste von Frame-Dicts,
* einen Pfad zu einer dieser JSON-Dateien.

Wichtige Optionen: `person="best"` (höchste Körperkonfidenz), `"largest"`
(größte Bounding-Box) oder ein fester Index; `side="left"/"right"` bei
Side-by-Side-Aufnahmen. Leere Frames bleiben als Lücken erhalten.

RTMLib liefert 2D. `pose_world` und die Hand-Weltpunkte werden deshalb aus
den normierten Bildkoordinaten erzeugt (x mit Seitenverhältnis, y nach unten,
z = 0) – dieselbe Achsenlage wie bei MediaPipe. Liefert eine Quelle echte
3D-Punkte (`keypoints_3d`, `keypoints_world`, `landmarks_3d` oder `world`),
werden diese bevorzugt. Damit ist der Weg für Sapiens1 offen.

Für Wörter aus `plugins/gsg/wordlist.csv` spielt das Plugin ausschließlich die
passende Datei aus `plugins/gsg/extracted_videos` ab. Sie wird vorab mit
`DataLoaderDGS/Extractor.py` aus den **zugeschnittenen** `videos-cut`-Clips
erzeugt, sodass nur die eigentliche Gebärde enthalten ist. Der Explorer baut
dabei z aus der 2D-Verkürzung neu auf (`repair_pev.reconstruct_depth`); der
alte Lifting-Stand (linke Armkette weit außerhalb des Körpers, Konfidenzen 1.0)
ist damit überholt. Der Adapter `rtm_adapter.sequence_from_pev` liest die Dateien
(y zeigt nach oben, Videos gespiegelt). Fehlt ein Wort als Gebärde oder seine
PEV, wird es wie in `gsg.py` Buchstabe für Buchstabe gespielt (`SCH` zusammen).

`--arm-mode auto` (Standard) lässt bei allen Quellen beide Arme den Landmarks
folgen – auch bei den `.pev`-Dateien. `--arm-mode sign` erzwingt den Gebärdearm,
der zweite Arm hängt dann entspannt herab.

## Tiefe aus 2D

RTMLib liefert keine 3D-Punkte (`pose_world.z = 0`). Die Verkürzung eines
Segments sagt dann nur, *wie weit* es aus der Bildebene kippt, nicht in welche
Richtung. Für die Arme ist die Richtung bekannt – Gebärden finden vor dem
Körper statt –, deshalb setzt `depth.py` dort `DEPTH_2D_FORWARD = -1`
(Armature-y zeigt nach vorn). Rumpf und Kopf bleiben ohne Tiefe, sonst kippt
der Oberkörper. Der Oberarm wird nur halb gewichtet (`DEPTH_WEIGHT`), weil
seine Verkürzung meist Perspektive ist; die Handführung kommt aus dem
Unterarm. Liefert eine Quelle echtes `z`, wird wie bisher das Vorzeichen
daraus bestimmt.

Die Verkürzung allein kennt nur den Anteil *senkrecht* zur Bildebene. Ein
gestreckter Arm, der im Bild parallel zur Kamera steht, hätte damit Tiefe 0
und läge mitten im Rumpf – bei „Ablauf“ verschwanden die Hände genau deshalb
im Mesh. `DEPTH_2D_BASE` addiert deshalb für 2D-Arme eine Grundtiefe
(20 % der Oberarm-, 45 % der Unterarmlänge) nach vorn. Erst danach greift die
Begrenzung auf `MAX_OOP_DEG`. Rumpf, Kopf und Beine bekommen weiterhin keine
Grundtiefe, damit der Oberkörper stabil bleibt.

## Faust aus 2D

RTMLib-Handpunkte sind ebenfalls 2D. Zeigen die Finger in die Kamera, sind
die Fingergliedwinkel im Bild fast 0° – die Hand sieht offen aus, obwohl sie
geschlossen ist. `hands.fist_reach` misst stattdessen den mittleren Abstand
Handgelenk → Fingerspitze, geteilt durch die Handflächenlänge: offen ≈ 2–3,
Faust ≈ 0,8–1,1. `FIST_REACH_OPEN` / `FIST_REACH_CLOSED` bilden daraus einen
Faustanteil zwischen 0 und 1. Eine Hüllkurve (`fist_envelope`) schließt
schnell und öffnet langsam, weil die erkannten Punkte zwischen Faust und
offener Hand springen; so bleibt z. B. bei „Auto“ die Faust stehen. Der
Faustanteil hebt die Beugung je Fingerglied bis `FINGER_CURL_MAX` an und
zieht die Spreizung auf null. Ohne Handdaten bleibt der bisherige Weg über
`FINGER_PRESETS` (`relaxed` / `open_hand`).

### Einzelne gestreckte Finger

Eine globale Faust reicht nicht: bei „Ablauf“ zeigt der Zeigefinger, während
Mittel-, Ring- und kleiner Finger eingerollt sind. `hands.finger_straight`
misst deshalb je Finger die Sehne (Fingerwurzel → Spitze) geteilt durch die
Länge des Fingerpolygons: gestreckt ≈ 1, eingerollt deutlich darunter. Ein
Finger, der über die Zeit gestreckt bleibt, wird aus der Faust ausgenommen.

Die Rohwerte springen frame-weise (die 2D-Punkte kippen zwischen gestreckt
und eingerollt), deshalb läuft die Reihe durch:

1. `median_filter` über ~0,1 s,
2. `rolling_max` über ~0,35 s (kurz halten),
3. Smoothstep-Sättigung zwischen `FINGER_STRAIGHT_LO` und `FINGER_STRAIGHT_HI`,
4. Anteilsfilter: nur wenn der Finger in einem Fenster von ~0,5 s in
   mindestens `FINGER_STRAIGHT_FRAC` der Frames „sicher gestreckt“ war
   (> `FINGER_STRAIGHT_SURE`), wird er wirklich gestreckt gehalten,
5. `savgol`.

Der so entstandene Faktor 0..1 skaliert in `finger_quaternions` den
Faustanteil (`cl *= 1 - straight`). Ergebnis: „Auto“ bleibt eine Faust,
„Ablauf“ behält den zeigenden Zeigefinger bei eingerollten übrigen Fingern.

## T-Pose

Die `videos-cut`-Clips und damit die PEVs sind bereits auf die Gebärde
zugeschnitten; das Plugin wendet kein zusätzliches Bewegungsfenster mehr an
(`window.py` bleibt als Werkzeug für ungeschnittene Aufnahmen erhalten). Die
Wiedergabe beginnt und endet in der T-Pose: davor und dahinter blendet das
Plugin per Slerp aus der glTF-Ruhepose ein und wieder zurück
(`webviewer.anchor_rest`). Die Ruhepose sind die lokalen Rest-Rotationen der
GLB-Knoten (`webviewer.rest_frame`), **nicht** die Identität: mit Identität
zeigen die Arme im KIM-Rig senkrecht nach oben.

## Webviewer-Plugin

`avatar-pose/main/plugins/retarget/` spielt Kurven über den vorhandenen
`UnifiedWebSocket` ab. Plugin-Auswahl über eine Umgebungsvariable, Standard
bleibt `gsg`:

```bash
AVATAR_PLUGIN=retarget AVATAR_RETARGET_CURVES=kurven.json python main.py
```

Das läuft aus `avatar-pose/main/`. Weitere Variablen:

| Variable | Bedeutung |
| --- | --- |
| `AVATAR_RETARGET_CURVES` | Pfad zur Standard-Kurven-JSON (kein Schalter wie `auto`) |
| `AVATAR_RETARGET_RIG` | Rig-JSON, Standard `retarget/data/rig_kim.json` |
| `AVATAR_RETARGET_GLB` | Avatar-GLB, Standard `avatar-pose/webviewer/media/avatar.glb` |
| `AVATAR_RETARGET_PEV_SWAP` | `0` schaltet den Links/Rechts-Tausch für `.pev` ab |
| `AVATAR_RETARGET_ARM_MODE` | `auto` (Standard), `sign` oder `both`; siehe oben |
| `AVATAR_RTMLIB_BACKEND` | Engine für RTMLib: `opencv` (Standard, ohne `onnxruntime`), `onnxruntime` oder `openvino` |
| `AVATAR_RTMLIB_DEVICE` | Gerät für RTMLib, Standard `cpu` |
| `AVATAR_RETARGET_ANCHOR` | `none` schaltet die T-Pose-Blende für Wortwiedergaben ab |
| `AVATAR_RETARGET_SIGN_PAUSE` | Pause zwischen Gebärden in Sekunden, Standard `0.2` |
| `AVATAR_GLOSS` | `0` schaltet die Übersetzung ab (nur Wortliste/Buchstabieren) |
| `AVATAR_GLOSS_EPOCH` | Checkpoint-Epoche, Standard `20` |
| `AVATAR_GLOSS_DIR` | Ordner von `text_to_gloss`, Standard `avatar-pose/text_to_gloss` |

Im Textfeld des Webviewers (`Enter` sendet):

* leer / `replay` / `play` / `start` – letzte Aufnahme erneut abspielen,
* ein Wort wie `Auto` – wird in `plugins/gsg/wordlist.csv` gesucht und die
  vorab extrahierte PEV aus `extracted_videos` gespielt, eingerahmt von der
  T-Pose; fehlt sie, wird das Wort buchstabiert,
* mehrere Wörter oder Buchstaben wie `Der Hund läuft` oder `H A U S` – siehe
  [Sätze und Übersetzung](#sätze-und-übersetzung),
* ein Pfad zu einer Kurven-JSON – direkt abspielen,
* ein Pfad zu einer RTMLib-/Landmark-/PEV-JSON – einmal durch die ganze Kette.

Die Wortsuche ignoriert Groß-/Kleinschreibung und funktioniert auch für
Mehrwort-Einträge wie `Auto fahren`. Sie greift nur, wenn die Eingabe kein
vorhandener Dateipfad ist; Dateinamen mit `.json`/`.pev` werden grundsätzlich
als Pfad behandelt. Es gibt keinen Video-Download und keinen Keypoint-Cache
mehr: zur Laufzeit wird ausschließlich `extracted_videos/*.pev` gelesen.

### Sätze und Übersetzung

Mehrere Wörter nimmt das Plugin als Satz: die Eingabe wird an Leerzeichen,
Kommas, Semikolons, `|` und `/` getrennt (mindestens zwei Teile). Dann läuft
sie durch `text_to_gloss` (GRU-Encoder + Attention-Decoder in
`avatar-pose/text_to_gloss/`), das eine DGS-Glossfolge liefert. Jedes
Gloss-Wort wird in `plugins/gsg/wordlist.csv` nachgeschlagen; fehlt es oder die
PEV, wird es nach dem Vorbild von `gsg.py` Buchstabe für Buchstabe gespielt
(`SCH` als eine Gebärde, Umlaute über `AE`/`OE`/`UE`).

Das Modell wurde auf Wettertexten trainiert (`Text2Gloss.csv`, 9162 Zeilen)
und ist außerhalb dieses Bereichs schwach (BLEU 0,21 auf den Testsätzen).
Deshalb wird die Übersetzung nur benutzt, wenn **alle** eingegebenen Wörter
als ganzes Wort in der Wortliste stehen und eine PEV haben und mindestens die
Hälfte der Gloss-Wörter in der Wortliste steht. Fehlt ein Wort oder seine PEV,
wird es buchstabiert und die Gloss-Übersetzung wird nicht verwendet – sonst
könnte das Modell ein unbekanntes Wort einfach fallen lassen (z. B. `Jonas`
in `Guten Abend Jonas`). `AVATAR_GLOSS=0` schaltet das Modell ganz ab. Ein
einzelnes Wort, das in der Wortliste steht, geht weiterhin direkt durch und
lädt das Modell gar nicht.

Zwischen zwei Gebärden steht der Avatar für `AVATAR_RETARGET_SIGN_PAUSE`
Sekunden in der T-Pose (Standard 0,2 s). Jede Gebärde beginnt und endet dort
ohnehin durch `anchor_rest`. Ein Buchstabe ist eine eigene Gebärde, d. h.
`H A U S` ergibt vier Abschnitte; `H|A|U|S` funktioniert genauso. Glossen
werden auf `MAX_SIGNS = 24` Zeichen begrenzt, damit der Frame-Puffer reicht
(der Ringpuffer hält jetzt 120 s statt 30 s).

Die Modulnamen von `text_to_gloss` sind flach (`from constant import ...`).
`gloss.py` lädt sie deshalb mit einem temporären `sys.path`-Eintrag und räumt
`sys.modules` danach wieder auf, damit sie nicht mit `plugins/gsg`
kollidieren. Epoche 20 ist Standard, weil sie auf 300 Testsätzen den besten
BLEU-Wert erreicht (0,2108; Epoche 40/50: 0,2087, Epoche 10: 0,1947).

Die Wiedergabe startet mit der ersten Eingabe, weil der WebSocket-Puffer erst
nach dem Plugin-Konstruktor existiert. Der Server sendet mit fester Rate von
30 Bildern/s; Kurven mit anderer `fps` werden per Slerp umgerechnet.

`UnifiedWebSocket` hält die zuletzt erzeugten Frames in einem Ringpuffer, und
jede Verbindung liest mit einem eigenen Zeiger. Dadurch bekommen mehrere
gleichzeitig offene Webviewer (oder ein alter Browser-Tab) jeweils die ganze
Animation; vorher hat ein Client die Frames der anderen verbraucht. Ein neu
verbundener Client beginnt mit dem aktuellen Frame, nicht mit dem Anfang der
Historie.

## Aufbau

| Modul | Aufgabe |
| --- | --- |
| `rtm_adapter.py` | RTMLib/COCO-WholeBody bzw. gsg-PEV -> Landmark-Sequenz |
| `gsg_video.py` | alter Video-Download/Keypoint-Pfad (vom Plugin nicht mehr genutzt) |
| `sentence.py` | Satz in Gebärden-Ids: Gloss, Wortliste, Buchstabieren |
| `gloss.py` | `text_to_gloss` laden und Deutsch -> DGS-Gloss übersetzen |
| `window.py` | Bewegungsfenster für ungeschnittene Aufnahmen (nicht mehr im Plugin) |
| `landmarks.py` | Handseiten, Mikro-Lücken, Ausreißer, Naht Pose/Hand |
| `depth.py` | Tiefe aus perspektivischer Verkürzung, 2D-Vorzeichen und Grundtiefe für Arme |
| `hands.py` | Beugung, Spreizung und Faustanteil aus der 2D-Handgeometrie |
| `retarget.py` | Knochendrehungen, Filter, Kurven schreiben |
| `webviewer.py` | Blender-Basis -> glTF-Node-Rotation, Resampling, T-Pose-Blende |
| `metrics.py` | Abdeckung, Konfidenz, Zappeln |
| `cli.py` | Kommandozeile |
| `data/rig_kim.json` | Rig des KIM-Avatars |

Die Brücke zwischen interner Blender-Lesart und glTF ist in `webviewer.py`
hergeleitet: mit `C = [[1,0,0],[0,0,1],[0,-1,0]]`, Ruhe-Globalrotation `G_b`
des Rig-Knochens und `H_b` des glTF-Nodes gilt für eine Retarget-Rotation
`B_b`:

```
D_b    = (C * G_b)^T * H_b
q_node = Q_rest_gltf * (D_b^T * B_b * D_b)
```

Numerisch geprüft: identische Kurven reproduzieren die GLB-Ruhelage auf
0,03° genau; eine 40°-Drehung des Oberarms landet mit 0,0003° Abweichung auf
derselben Knochenrichtung.

## Tests

```bash
python -m pytest retarget/tests -q
```

Die Tests decken Filter, Fingerwinkel, Rig, Retargeting, Adapter und die
glTF-Brücke ab und brauchen weder Video noch RTMLib noch Blender.
