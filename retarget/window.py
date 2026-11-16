"""Bewegungsfenster einer Aufnahme: der Abschnitt, in dem die Haende oben sind.

Die gsg-Videos zeigen meist erst ein Herabhaengen der Arme, dann die eigentliche
Gebaerde und am Ende wieder Herabhaengen. Fuer den Avatar ist nur der mittlere
Teil interessant; Anfang und Ende werden durch eine Blende aus der Ruhepose
(T-Pose) ersetzt. Dieses Modul findet den mittleren Abschnitt anhand der
Landmarks, ohne ihn fest zu verdrahten.
"""

from .sequence import P


def _above_elbow(p2, sho, elb, wri, W, H, min_frac=0.25):
    """Liegt das Handgelenk deutlich ueber dem Ellbogen?

    Bezugsgroesse ist die projizierte Oberarmlaenge, damit die Schwelle mit
    der Bildgroesse skaliert. y zeigt im Bild nach unten.
    """
    s, e, w = P(p2, sho), P(p2, elb), P(p2, wri)
    if s is None or e is None or w is None:
        return False
    upper = ((e[0] - s[0]) * W) ** 2 + ((e[1] - s[1]) * H) ** 2
    if upper <= 1e-9:
        return False
    rise = (e[1] - w[1]) * H
    return rise > min_frac * upper ** 0.5


def raised_arms_window(seq, pad=2, min_span=5, min_frac=0.25):
    """(start, end) des laengsten Abschnitts mit erhobenen Unterarmen.

    Beide Handgelenke muessen ueber ihren Ellbogen liegen. Gibt ``None``
    zurueck, wenn kein solcher Abschnitt lang genug ist - dann ist die
    Aufnahme vermutlich keine Armbewegung und wird unveraendert gespielt.
    """
    W, H = seq.size
    flags = []
    for fd in seq.frames:
        p2 = fd.get("pose")
        if not p2:
            flags.append(False)
            continue
        flags.append(_above_elbow(p2, 5, 7, 9, W, H, min_frac)
                     and _above_elbow(p2, 6, 8, 10, W, H, min_frac))

    best = None
    start = None
    for i, ok in enumerate(flags + [False]):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i - 1)
            start = None
    if best is None or best[1] - best[0] + 1 < min_span:
        return None

    n = len(flags)
    return max(0, best[0] - pad), min(n - 1, best[1] + pad)
