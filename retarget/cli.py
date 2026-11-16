"""Kommandozeile der Retarget-Kette fuer RTMLib/COCO-WholeBody.

    python -m retarget prepare aufnahme_rtm.json -o aufnahme.json
    python -m retarget retarget aufnahme_rtm.json --rig rig_kim.json \
        -o kurven.json --glb avatar.glb --frames webviewer_frames.json
    python -m retarget bridge kurven.json --rig rig_kim.json \
        --glb avatar.glb -o webviewer_frames.json
    python -m retarget metrics aufnahme.json --process
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .metrics import hand_coverage, signal_quality
from .sequence import Sequence

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
DEFAULT_RIG = PACKAGE_DIR / "data" / "rig_kim.json"
DEFAULT_GLB = PROJECT_DIR / "avatar-pose" / "webviewer" / "media" / "avatar.glb"


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_sequence_data(data):
    if not isinstance(data, dict):
        return False
    frames = data.get("frames")
    if not isinstance(frames, list):
        return False
    return any(isinstance(fd, dict) and ("pose" in fd or "pose_world" in fd)
               for fd in frames)


def _sequence_from_args(args):
    data = _load_json(args.input)
    if _is_sequence_data(data):
        return Sequence(data)
    from .rtm_adapter import sequence_from_rtmlib
    return sequence_from_rtmlib(
        data, width=args.width, height=args.height, fps=args.fps,
        frame_step=args.frame_step, side=args.side, person=args.person)


def _adapter_options(p, default_side=True):
    p.add_argument("--width", type=float, help="Bildbreite in Pixeln (sonst geschaetzt)")
    p.add_argument("--height", type=float, help="Bildhoehe in Pixeln (sonst geschaetzt)")
    p.add_argument("--fps", type=float, help="Bildrate des Originals")
    p.add_argument("--frame-step", type=int, help="Abtastschritt der Frames")
    if default_side:
        p.add_argument("--side", choices=["left", "right"], default="left",
                       help="Seite bei side-by-side-Aufnahmen")
    p.add_argument("--person", default="best",
                   help="'best', 'largest' oder fester Personenindex")


def _cmd_prepare(args):
    seq = _sequence_from_args(args)
    out = args.out or (str(Path(args.input).with_suffix("")) + "_aufnahme.json")
    seq.save(out)
    print("%d Frames, %dx%d, %.2f fps -> %s"
          % (len(seq), seq.size[0], seq.size[1], seq.fps, out))
    return 0


def _cmd_retarget(args):
    from . import landmarks
    from . import retarget as rt
    from .rig import Rig

    seq = _sequence_from_args(args)
    landmarks.process(seq, clean=not args.raw, snap_wrist=not args.no_snap_wrist)
    rig = Rig.load(args.rig)
    curves = rt.solve(rig, seq, upper_only=not args.full_body,
                      arm_depth=args.arm_depth, spine_depth=args.spine_depth,
                      min_conf=args.min_conf,
                      finger_fallback=args.finger_fallback,
                      hand_min=args.hand_min, smooth_sec=args.smooth_sec,
                      clean=not args.raw, arm_mode=args.arm_mode)
    if not curves:
        print("kein Knochen gesetzt - Rig-Namen pruefen oder Konfidenz senken")
        return 1

    out = args.out or (str(Path(args.input).with_suffix("")) + "_kurven.json")
    rt.save_curves(curves, rt.frame_numbers(seq, args.start_frame), out, seq.fps)
    print("%d Knochen, %d Frames -> %s" % (len(curves), len(seq), out))

    if args.frames:
        from . import webviewer
        glb = args.glb or str(DEFAULT_GLB)
        frames = webviewer.frames(rig, curves, glb)
        webviewer.save_frames(frames, args.frames, indent=args.frames_indent)
        print("%d Webviewer-Frames -> %s" % (len(frames), args.frames))
    return 0


def _cmd_bridge(args):
    from . import webviewer
    from .rig import Rig

    curves, _fps, _frames = webviewer.load_curves(args.curves)
    rig = Rig.load(args.rig)
    frames = webviewer.frames(rig, curves, args.glb)
    out = args.out or (str(Path(args.curves).with_suffix(""))
                       + "_webviewer.json")
    webviewer.save_frames(frames, out, indent=args.indent)
    print("%d Knochen, %d Frames -> %s" % (len(curves), len(frames), out))
    return 0


def _cmd_metrics(args):
    data = _load_json(args.input)
    seq = Sequence(data) if _is_sequence_data(data) else _sequence_from_args(args)
    if args.process:
        from .landmarks import process
        process(seq)
    report = {"file": args.input, "frames": len(seq), "fps": round(seq.fps, 3),
              "size": list(seq.size), "coverage": hand_coverage(seq),
              "signal": signal_quality(seq)}
    text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="retarget")
    sub = p.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare", help="RTMLib-JSON -> Landmark-Sequenz")
    prep.add_argument("input")
    prep.add_argument("-o", "--out")
    _adapter_options(prep)
    prep.set_defaults(func=_cmd_prepare)

    r = sub.add_parser("retarget", help="Landmarks -> Knochendrehungen")
    r.add_argument("input", help="RTMLib-JSON oder bereits konvertierte Sequenz")
    r.add_argument("--rig", default=str(DEFAULT_RIG))
    r.add_argument("-o", "--out")
    r.add_argument("--start-frame", type=int, default=1)
    r.add_argument("--full-body", action="store_true", help="auch die Beine")
    r.add_argument("--arm-depth", type=float, default=1.0)
    r.add_argument("--spine-depth", type=float, default=0.2)
    r.add_argument("--min-conf", type=float, default=0.5)
    r.add_argument("--hand-min", type=float, default=0.7)
    r.add_argument("--smooth-sec", type=float, default=0.2)
    r.add_argument("--arm-mode", choices=["auto", "both", "sign"], default="auto",
                   help="auto: bei .pev nur der Gebaerdearm; both: beide Arme")
    r.add_argument("--finger-fallback", default="relaxed",
                   choices=["relaxed", "open_hand", "none"])
    r.add_argument("--no-snap-wrist", action="store_true")
    r.add_argument("--raw", action="store_true",
                   help="ohne Ausreisserbereinigung und Sprungfilter")
    _adapter_options(r)
    r.add_argument("--glb", help="GLB fuer den Webviewer-Brueckenschritt")
    r.add_argument("--frames", help="optional: Webviewer-Frames mitschreiben")
    r.add_argument("--frames-indent", type=int, default=None)
    r.set_defaults(func=_cmd_retarget)

    b = sub.add_parser("bridge", help="Kurven -> Webviewer-Frames")
    b.add_argument("curves")
    b.add_argument("--rig", default=str(DEFAULT_RIG))
    b.add_argument("--glb", default=str(DEFAULT_GLB))
    b.add_argument("-o", "--out")
    b.add_argument("--indent", type=int, default=None)
    b.set_defaults(func=_cmd_bridge)

    m = sub.add_parser("metrics", help="Zuverlaessigkeit der Aufnahme messen")
    m.add_argument("input")
    m.add_argument("--process", action="store_true")
    m.add_argument("--json-out")
    _adapter_options(m)
    m.set_defaults(func=_cmd_metrics)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
