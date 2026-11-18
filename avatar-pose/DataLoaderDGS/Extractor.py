"""
Extractor.py

Authors: Carolin Gottschalk, Jonas D. Stephan, Manuela Wittmann
License: Apache License 2.0

Description:
Wandelt die Videos aus ``<dataset>/videos-cut`` in ``.pev``-Dateien um.

Ein ``.pev`` ist JSON mit ``{"frames": [{"frame": i, "skeletonpoints": [...]}]}``
und 133 COCO-WholeBody-Punkten je Frame. Koordinaten liegen in Pixeln:
x gespiegelt (``-bild_x``), y nach oben (``bild_hoehe - bild_y``), z nach
vorn positiv und aus der 2D-Verkuerzung rekonstruiert.

Zwei Backends stehen zur Verfuegung:

``rtmlib``   (Standard) ``rtmlib.Wholebody`` liefert die 2D-Punkte direkt.
             Standard-Engine ist ``opencv`` (cv2.dnn), dafuer wird das Paket
             ``onnxruntime`` nicht gebraucht; ``--rtmlib-backend onnxruntime``
             schaltet die alte Engine ein. Die RTMLib-Gewichte sind immer
             ``.onnx``-Dateien.
``official`` ``pose_estimation_recognition_utils_rtmlib.RTMPoseEstimator3D``
             (2D + ONNX-Lifting). Optional, falls das Paket installiert ist,
             und nur noch fuer Vergleichszwecke gedacht.

Die Tiefe (z) wird in beiden Faellen mit ``repair_pev.reconstruct_depth``
neu aufgebaut: das Lifting-Modell skaliert beim Zurueckrechnen nur x/y, das
gelieferte z ist rund 1000-mal zu klein und in der Richtung instabil (siehe
``repair_pev.py``). Ohne den Neuaufbau verschwinden die Arme im Rumpf.

Beispiele
---------
    python DataLoaderDGS/Extractor.py --dataset_path C:\\data\\sign-language-avatar-gloss-dgs
    python DataLoaderDGS/Extractor.py --dataset_path ... --limit 3 --overwrite
    python DataLoaderDGS/Extractor.py --dataset_path ... --backend official
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from repair_pev import (N_POINTS, depth_ratio, points_array,  # noqa: E402
                        reconstruct_depth)

# Voreinstellungen des offiziellen Backends (bleiben wie im Original).
DEFAULT_DET_MODEL = ('https://download.openmmlab.com/mmpose/v1/projects/'
                     'rtmposev1/onnx_sdk/'
                     'yolox_m_8xb8-300e_humanart-c2c7a14a.zip')
DEFAULT_POSE_MODEL = 'https://cobtras.com/data/model.onnx'
DEFAULT_SPECIAL_MODEL = (
    'fhswf/rtm133lifting-finetuning-signs-fingerspelling-dgs/'
    'rtm133lifting-finetuning-sings-fingerspelling-dgs-best.onnx')


class ExtractionError(RuntimeError):
    """Video konnte nicht ausgewertet werden."""


# --- Geraet und Modelle ----------------------------------------------------

def detect_device(preferred='auto', backend='opencv'):
    """``cpu``/``cuda``; ``auto`` nimmt CUDA nur mit passender Engine."""
    if preferred and preferred != 'auto':
        return preferred
    if backend != 'onnxruntime':
        # OpenCV-CUDA gibt es nur mit selbst gebautem OpenCV.
        return 'cpu'
    try:
        import onnxruntime as ort
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            return 'cuda'
    except Exception:                                      # noqa: BLE001
        pass
    return 'cpu'


def make_rtmlib_estimator(device, mode='balanced', backend='opencv'):
    """``rtmlib.Wholebody`` (COCO-WholeBody, 133 Punkte).

    ``opencv`` (Standard) nutzt ``cv2.dnn`` und braucht das Paket
    ``onnxruntime`` nicht; die Modellgewichte bleiben ``.onnx``-Dateien.
    """
    try:
        from rtmlib import Wholebody
    except ImportError as exc:                             # noqa: BLE001
        raise ExtractionError(
            'RTMLib fehlt (pip install rtmlib opencv-python): %s' % exc) from exc
    return Wholebody(to_openpose=False, mode=mode,
                     backend=backend, device=device)


def make_official_estimator(device, mode, cache_dir=None):
    """Offizieller 2D+3D-Estimator; optional, Paket muss installiert sein."""
    try:
        from pose_estimation_recognition_utils_rtmlib import RTMPoseEstimator3D
    except ImportError as exc:                             # noqa: BLE001
        raise ExtractionError(
            'Backend "official" braucht pose_estimation_recognition_utils_rtmlib '
            'und pose_estimation_recognition_utils. Fuer den Normalbetrieb '
            '--backend rtmlib verwenden. (%s)' % exc) from exc
    return RTMPoseEstimator3D(
        det_model_path=DEFAULT_DET_MODEL,
        pose_model_path=DEFAULT_POSE_MODEL,
        device=device,
        mode=mode,
        lifting_mode='ai',
        special_model=DEFAULT_SPECIAL_MODEL,
        cache_dir=cache_dir,
    )


# --- Extraktion ------------------------------------------------------------

def extract_rtmlib(video_path, estimator):
    """Alle Frames mit RTMLib auswerten -> (frames, width, height, fps)."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ExtractionError('Video nicht lesbar: %s' % video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0

    frames = []
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            keypoints, scores = estimator(image)
            if keypoints is None or len(keypoints) == 0:
                frames.append(None)
                continue
            # Bei mehreren Personen die mit der besten Koerperkonfidenz.
            body = [float(np.asarray(s)[:17].mean()) for s in scores]
            best = int(np.argmax(body))
            frames.append({'keypoints': np.asarray(keypoints[best],
                                                   dtype=np.float64).tolist(),
                           'scores': np.asarray(scores[best],
                                                dtype=np.float64).tolist()})
    finally:
        cap.release()

    if not any(frame is not None for frame in frames):
        raise ExtractionError('keine Person im Video gefunden: %s' % video_path)
    return frames, width, height, fps


def extract_official(video_path, estimator):
    """Offizielles Backend -> Liste fertiger Frame-Dicts (mit Konfidenz)."""
    from pose_estimation_recognition_utils_rtmlib import (
        video3d_result_to_video_skeleton_data_with_confidence)

    result = estimator.process_video(str(video_path))
    items = video3d_result_to_video_skeleton_data_with_confidence(result)
    frames = [json.loads(item.to_json()) for item in items]
    if not frames:
        raise ExtractionError('keine Frames geliefert: %s' % video_path)
    return frames


# --- Umrechnung in PEV-Koordinaten -----------------------------------------

def frames_to_arrays(frames, height):
    """RTMLib-Frames -> dichtes (F, 133, 3)-Array und (F, 133)-Konfidenzen.

    x wird gespiegelt und y nach oben gedreht, damit die Punkte zu
    ``main/data/bone_mapping.json`` passen.
    """
    count = len(frames)
    arr = np.full((count, N_POINTS, 3), np.nan, dtype=np.float64)
    conf = np.zeros((count, N_POINTS), dtype=np.float64)

    for i, frame in enumerate(frames):
        if not frame:
            continue
        keypoints = np.asarray(frame['keypoints'], dtype=np.float64)
        if keypoints.ndim != 2 or keypoints.shape[0] < N_POINTS:
            continue
        scores = np.asarray(frame.get('scores', np.ones(len(keypoints))),
                            dtype=np.float64)
        arr[i, :, 0] = -keypoints[:N_POINTS, 0]
        arr[i, :, 1] = (height - keypoints[:N_POINTS, 1]) if height > 0 \
            else -keypoints[:N_POINTS, 1]
        conf[i] = scores[:N_POINTS]
    return arr, conf


def fill_missing(arr, conf):
    """Frames ohne Person mit dem naechsten gueltigen Frame auffuellen.

    Der Avatar braucht eine lueckenlose Folge; aufgefuellte Frames bekommen
    Konfidenz 0, damit sie spaeter erkennbar bleiben.
    """
    valid = np.flatnonzero(np.isfinite(arr[:, :, 0]).any(axis=1))
    if valid.size == 0:
        raise ExtractionError('keine verwertbaren Keypoints im Video')
    missing = 0
    for i in range(len(arr)):
        if np.isfinite(arr[i, :, 0]).any():
            continue
        nearest = int(valid[np.argmin(np.abs(valid - i))])
        arr[i] = arr[nearest]
        conf[i] = 0.0
        missing += 1
    return arr, conf, missing


def skeleton_frames(arr, conf):
    """Dichtes Array -> Liste ``{"frame": i, "skeletonpoints": [...]}``."""
    frames = []
    for i in range(len(arr)):
        points = []
        for idx in range(N_POINTS):
            x, y, z = arr[i, idx]
            points.append({
                'id': idx,
                'x': round(float(x), 3),
                'y': round(float(y), 3),
                'z': round(float(z), 3),
                'confidence': round(float(conf[i, idx]), 6),
            })
        frames.append({'frame': i, 'skeletonpoints': points})
    return frames


def apply_depth(data, front_sign):
    """z aus der 2D-Verkuerzung neu aufbauen, Rest der Datei bleibt."""
    arr = points_array(data)
    before = depth_ratio(arr)
    fixed = reconstruct_depth(arr, front_sign=front_sign)
    for i, frame in enumerate(data['frames']):
        for point in frame.get('skeletonpoints') or []:
            try:
                idx = int(point.get('id', -1))
            except (TypeError, ValueError):
                continue
            if 0 <= idx < N_POINTS and np.isfinite(fixed[i, idx, 2]):
                point['z'] = float(fixed[i, idx, 2])
    return before, depth_ratio(fixed)


def build_pev(video_path, backend, estimator, front_sign):
    """Liefert (PEV-Dict, Info-Dict) fuer ein Video."""
    if backend == 'rtmlib':
        frames, width, height, fps = extract_rtmlib(video_path, estimator)
        arr, conf = frames_to_arrays(frames, height)
        arr, conf, missing = fill_missing(arr, conf)
        before = depth_ratio(arr)
        arr = reconstruct_depth(arr, front_sign=front_sign)
        info = {'frames': len(arr), 'fps': fps, 'width': width,
                'height': height, 'missing': missing,
                'depth': (before, depth_ratio(arr))}
        return {'frames': skeleton_frames(arr, conf)}, info

    frames = extract_official(video_path, estimator)
    data = {'frames': frames}
    before, after = apply_depth(data, front_sign)
    info = {'frames': len(frames), 'fps': float('nan'), 'width': 0,
            'height': 0, 'missing': 0, 'depth': (before, after)}
    return data, info


# --- Dateien ---------------------------------------------------------------

def output_path(video_path, videos_root, output_root):
    """``videos-cut/<x>/<id>.mp4`` -> ``extracted/<x>/<id>.pev``."""
    video_path = Path(video_path)
    try:
        relative = video_path.relative_to(videos_root)
    except ValueError:
        relative = Path(video_path.name)
    return Path(output_root) / relative.with_suffix('.pev')


def write_pev(path, data, backup=False):
    """Atomar schreiben, damit kein halbes PEV liegen bleibt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if backup and path.exists():
        original = path.with_suffix('.pev.bak')
        if not original.exists():
            shutil.copy2(path, original)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    os.replace(tmp, path)


def find_videos(videos_root, ids=None):
    videos = sorted(Path(videos_root).rglob('*.mp4'))
    if ids:
        wanted = {str(i).strip().lower() + '.mp4' for i in ids if str(i).strip()}
        videos = [v for v in videos if v.name.lower() in wanted]
    return videos


def parse_ids(value):
    return [part.strip() for part in (value or '').split(',') if part.strip()]


# --- Ablauf ----------------------------------------------------------------

def run(args):
    videos_root = Path(args.videos or Path(args.dataset_path) / 'videos-cut')
    output_root = Path(args.output or Path(args.dataset_path) / 'extracted')
    if not videos_root.is_dir():
        print('Videoordner fehlt: %s' % videos_root)
        return 1

    videos = find_videos(videos_root, parse_ids(args.ids))
    if args.limit:
        videos = videos[:args.limit]
    if not videos:
        print('Keine .mp4-Dateien in %s' % videos_root)
        return 1

    todo = [v for v in videos
            if args.overwrite or not output_path(v, videos_root, output_root).exists()]
    print('%d Videos gefunden, %d zu verarbeiten -> %s'
          % (len(videos), len(todo), output_root))
    if args.dry_run:
        for video in todo:
            print('wuerde schreiben %s' % output_path(video, videos_root, output_root))
        print('\n%d Dateien wuerden erzeugt (--dry-run).' % len(todo))
        return 0
    if not todo:
        print('Alles vorhanden (mit --overwrite neu erzeugen).')
        return 0

    device = detect_device(args.device,
                           'onnxruntime' if args.backend == 'official'
                           else args.rtmlib_backend)
    mode = args.mode or ('individual' if args.backend == 'official'
                         else 'balanced')
    engine = '2D+Lifting' if args.backend == 'official' else args.rtmlib_backend
    print('Backend %s (%s), Geraet %s, Modus %s'
          % (args.backend, engine, device, mode))
    try:
        if args.backend == 'official':
            estimator = make_official_estimator(device, mode, args.cache_dir)
        else:
            estimator = make_rtmlib_estimator(device, mode,
                                              args.rtmlib_backend)
    except ExtractionError as exc:
        print('FEHLER: %s' % exc)
        return 1

    done = failed = 0
    started = time.time()
    for number, video in enumerate(todo, start=1):
        target = output_path(video, videos_root, output_root)
        prefix = '[%d/%d] %s' % (number, len(todo), video.name)
        try:
            data, info = build_pev(video, args.backend, estimator,
                                   args.front_sign)
        except Exception as exc:                           # noqa: BLE001
            failed += 1
            print('%s FEHLER: %s' % (prefix, exc), flush=True)
            continue
        write_pev(target, data, backup=args.backup)
        done += 1
        before, after = info['depth']
        extra = '' if not info['missing'] else ' (%d Frames aufgefuellt)' \
            % info['missing']
        print('%s -> %s  %d Frames @ %.1f fps, Tiefe %.4f -> %.3f%s'
              % (prefix, target.name, info['frames'], info['fps'],
                 before, after, extra), flush=True)

    minutes = (time.time() - started) / 60.0
    print('\n%d Dateien geschrieben, %d fehlgeschlagen (%.1f min).'
          % (done, failed, minutes))
    if failed:
        print('Hinweis: fehlgeschlagene Videos erneut mit --overwrite und '
              '--limit pruefen.')
    return 0 if failed == 0 else 1


def build_parser():
    parser = argparse.ArgumentParser(
        description='MP4-Videos in PEV-Skelettdateien umwandeln.')
    parser.add_argument('--dataset_path', '--dataset-path', dest='dataset_path',
                        required=True,
                        help='Wurzel des Datensatzes (enthaelt videos-cut/ '
                             'und extracted/)')
    parser.add_argument('--videos', default=None,
                        help='Videoordner (Standard: <dataset_path>/videos-cut)')
    parser.add_argument('--output', default=None,
                        help='Zielordner (Standard: <dataset_path>/extracted)')
    parser.add_argument('--backend', choices=('rtmlib', 'official', 'auto'),
                        default='rtmlib',
                        help='rtmlib (Standard) oder official (2D+Lifting)')
    parser.add_argument('--rtmlib-backend',
                        choices=('opencv', 'onnxruntime', 'openvino'),
                        default=os.environ.get('AVATAR_RTMLIB_BACKEND', 'opencv'),
                        help='Inferenz-Engine fuer rtmlib; opencv braucht kein '
                             'onnxruntime (Standard: opencv bzw. '
                             'AVATAR_RTMLIB_BACKEND)')
    parser.add_argument('--device', default='auto',
                        help='auto, cpu oder cuda (Standard: auto)')
    parser.add_argument('--mode', default=None,
                        help='Modellvariante des 2D-Estimators (rtmlib: '
                             'balanced/performance/lightweight, Standard '
                             'balanced; official: individual, Standard '
                             'individual)')
    parser.add_argument('--front-sign', type=float, default=1.0,
                        help='+1: Arme zeigen nach vorn (+z), -1: nach hinten')
    parser.add_argument('--overwrite', action='store_true',
                        help='vorhandene PEVs neu erzeugen')
    parser.add_argument('--backup', action='store_true',
                        help='vorhandene PEVs vorher als *.pev.bak sichern')
    parser.add_argument('--limit', type=int, default=0,
                        help='nur die ersten N Videos verarbeiten')
    parser.add_argument('--ids', default=None,
                        help='nur diese Video-IDs verarbeiten (z. B. '
                             '"861,61,841")')
    parser.add_argument('--cache_dir', default=None,
                        help='Modellcache des offiziellen Backends')
    parser.add_argument('--dry-run', action='store_true',
                        help='nur anzeigen, was geschrieben wuerde')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.backend == 'auto':
        args.backend = 'rtmlib'
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
