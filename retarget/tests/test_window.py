"""Pruefungen fuer das Bewegungsfenster (Haende oben)."""

from retarget.window import raised_arms_window


class _Seq:
    def __init__(self, frames, size=(640, 480)):
        self.frames = frames
        self.size = size


def _pose(elbow_y, wrist_y):
    pose = [[0.5, 0.5, 0.0, 1.0] for _ in range(17)]
    pose[5] = [0.60, 0.35, 0.0, 1.0]
    pose[6] = [0.40, 0.35, 0.0, 1.0]
    pose[7] = [0.60, elbow_y, 0.0, 1.0]
    pose[8] = [0.40, elbow_y, 0.0, 1.0]
    pose[9] = [0.60, wrist_y, 0.0, 1.0]
    pose[10] = [0.40, wrist_y, 0.0, 1.0]
    return pose


def test_fenster_findet_die_erhobenen_haende():
    frames = []
    for i in range(20):
        # Frames 5..14: Unterarme zeigen nach oben, davor/danach haengen sie.
        up = 5 <= i <= 14
        frames.append({"pose": _pose(0.60, 0.48 if up else 0.75)})
    start, end = raised_arms_window(_Seq(frames), pad=2)
    assert (start, end) == (3, 16)


def test_fenster_ohne_erhobene_haende_ist_none():
    frames = [{"pose": _pose(0.60, 0.80)} for _ in range(20)]
    assert raised_arms_window(_Seq(frames)) is None


def test_fenster_ignoriert_kurze_ausreisser():
    frames = []
    for i in range(20):
        # Nur drei Frames oben - zu kurz fuer eine Gebaerde.
        up = 8 <= i <= 10
        frames.append({"pose": _pose(0.60, 0.48 if up else 0.75)})
    assert raised_arms_window(_Seq(frames), min_span=5) is None


def test_fenster_braucht_beide_arme():
    frames = []
    for i in range(20):
        pose = _pose(0.60, 0.48)
        if i >= 10:
            # Rechter Unterarm haengt wieder herab.
            pose[10] = [0.40, 0.80, 0.0, 1.0]
        frames.append({"pose": pose})
    start, end = raised_arms_window(_Seq(frames), pad=0)
    assert (start, end) == (0, 9)
