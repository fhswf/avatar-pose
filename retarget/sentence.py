"""Satzplanung fuer das retarget-Plugin.

Aus einer Eingabe wie ``"Der Hund laeuft"`` oder ``"H A U S"`` wird eine
Reihenfolge von Gebaerden. Ein Eintrag ist ein Wort aus ``wordlist.csv``
(``Sign``) oder ein einzelner Buchstabe, wenn das Wort fehlt.

Die Uebersetzung (``retarget.gloss``) liefert die Gloss-Reihenfolge; ohne
Uebersetzung werden die eingegebenen Woerter direkt abgespielt. Buchstabieren
ist immer der letzte Ausweg.
"""

from __future__ import annotations

import csv
import re
from typing import NamedTuple

# Mehr Zeichen passen nicht sinnvoll in den Frame-Puffer (30 fps).
MAX_SIGNS = 24
# Anteil der Gloss-Woerter, die als ganzes Wort in der Wortliste stehen
# muessen, damit die Uebersetzung benutzt wird. Das Modell kennt nur
# Wettertexte; sonst ist die eingegebene Wortfolge naeher an der Gebaerde.
GLOSS_MIN_HITS = 0.5
_SPLIT = re.compile(r"[\s,;|/]+")
_PARTS = re.compile(r"[-_]+")
_CLEAN = re.compile(r"[^0-9A-ZÄÖÜ]")
_UMLAUTE = (("AE", "Ä"), ("OE", "Ö"), ("UE", "Ü"))
# Rueckrichtung: einzelne Umlautbuchstaben werden in der gsg-Wortliste meist
# als AE/OE/UE gefuehrt (Vorlage: gsg.py behaelt Umlaute beim Buchstabieren).
_UMLAUTE_BACK = (("Ä", "AE"), ("Ö", "OE"), ("Ü", "UE"))
_UMLAUTE_SPREAD = dict(_UMLAUTE_BACK)


class Sign(NamedTuple):
    """Eine Gebaerden-Einheit: Anzeigetext und Id in der Wortliste."""

    label: str
    ident: str
    spelled: bool = False


class Plan(NamedTuple):
    """Ergebnis der Planung: abspielbare Zeichen und was weggefallen ist."""

    signs: tuple
    skipped: tuple
    source: str                      # "gloss", "words" oder "letters"


def load_wordlist(path):
    """``TEXT (gross) -> id`` aus ``wordlist.csv``."""
    words = {}
    with open(path, newline="", encoding="utf-8-sig",
              errors="replace") as handle:
        for row in csv.DictReader(handle):
            text = normalize(row.get("text"))
            ident = (row.get("id") or "").strip()
            if text and ident:
                words.setdefault(text, ident)
    return words


def normalize(text):
    """Leerzeichen vereinheitlichen, Grossschreibung (``ß`` wird zu ``SS``)."""
    return " ".join((text or "").split()).upper()


def split_tokens(text):
    """Eingabe an Leerzeichen und ueblichen Trennern zerlegen."""
    return [token for token in _SPLIT.split((text or "").strip()) if token]


def is_letter_sequence(tokens):
    """``True``, wenn alle Teile einzelne Buchstaben sind (z. B. ``H A U S``)."""
    letters = [letters_only(token) for token in tokens]
    return bool(letters) and all(len(item) == 1 for item in letters)


def letters_only(text):
    return [char for char in normalize(text) if char.isalpha()]


def _candidates(token):
    key = normalize(token)
    yield key
    for source, target in _UMLAUTE + _UMLAUTE_BACK:
        if source in key:
            yield key.replace(source, target)
    compact = _CLEAN.sub("", key)
    if compact and compact != key:
        yield compact


def lookup_id(words, token):
    """Id eines Wortes; beruecksichtigt ``AE/OE/UE``-Schreibweise."""
    for candidate in _candidates(token):
        ident = words.get(candidate)
        if ident:
            return ident
    return None


def spell_word(word, words):
    """Buchstabieren wie in ``gsg.py``: ein Zeichen pro Gebaerde, ``SCH`` zusammen.

    Umlaute werden zuerst als ``AE/OE/UE``-Gebaerde versucht. Gibt es dafuer
    keinen Eintrag, zerfallen sie in zwei Buchstaben (``Ä -> A, E``), damit der
    Laut nicht verloren geht. ``ß`` wird beim Grossschreiben zu ``SS``.
    Nicht-Buchstaben (Bindestriche, Satzzeichen, Ziffern) fallen weg. Fehlt ein
    Buchstabe in der Wortliste, wird er ausgelassen; der Aufrufer kann die
    uebrigen Zeichen trotzdem abspielen.
    """
    key = normalize(word)
    letters = [char for char in key if char.isalpha()]
    chunks = []
    i = 0
    while i < len(letters):
        char = letters[i]
        if char in _UMLAUTE_SPREAD and lookup_id(words, char) is None:
            chunks.extend(_UMLAUTE_SPREAD[char])
            i += 1
            continue
        if "".join(letters[i:i + 3]) == "SCH":
            chunks.append("SCH")
            i += 3
        else:
            chunks.append(char)
            i += 1
    signs = []
    for chunk in chunks:
        ident = lookup_id(words, chunk)
        if ident is not None:
            signs.append(Sign(chunk, ident, spelled=True))
    return signs


def signs_for_token(token, words):
    """Ein Token als Wort, zusammengesetzte Teile oder buchstabiert."""
    key = normalize(token)
    if not key:
        return []
    ident = lookup_id(words, key)
    if ident is not None:
        return [Sign(key, ident)]

    # Glossen wie "IN-KOMMEND" in ihre Teile zerlegen.
    parts = [part for part in _PARTS.split(key) if part]
    if len(parts) > 1:
        out = []
        for part in parts:
            signs = signs_for_token(part, words)
            if not signs:
                return []
            out.extend(signs)
        return out

    return spell_word(key, words)


def plan(tokens, words, gloss=None, max_signs=MAX_SIGNS):
    """Abspielreihenfolge fuer ``tokens`` bauen.

    ``gloss`` ist die Ausgabe des Uebersetzungsmoduls (Liste klein
    geschriebener Gloss-Woerter). Fehlt sie, laesst sich daraus nichts
    abbilden oder ist zu wenig davon in der Wortliste, werden die
    eingegebenen Woerter benutzt.
    """
    tokens = list(tokens or [])
    if gloss and _hits(gloss, words) >= GLOSS_MIN_HITS:
        signs, skipped = _resolve(gloss, words, max_signs)
        if signs:
            return Plan(tuple(signs), tuple(skipped), "gloss")

    signs, skipped = _resolve(tokens, words, max_signs)
    source = "letters" if is_letter_sequence(tokens) else "words"
    return Plan(tuple(signs), tuple(skipped), source)


def _hits(tokens, words):
    """Anteil der Tokens, die als ganzes Wort in der Wortliste stehen."""
    tokens = list(tokens or [])
    if not tokens:
        return 0.0
    found = sum(1 for token in tokens if lookup_id(words, token) is not None)
    return found / float(len(tokens))


def _resolve(tokens, words, max_signs):
    signs = []
    skipped = []
    for token in tokens:
        found = signs_for_token(token, words)
        if not found:
            skipped.append(normalize(token))
            continue
        signs.extend(found)
        if len(signs) >= max_signs:
            return signs[:max_signs], skipped
    return signs, skipped
