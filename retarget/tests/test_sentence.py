"""Satzplanung: Wortliste, Gloss-Rueckfall und Buchstabieren."""

import csv

from retarget import sentence


def write_wordlist(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "text"])
        writer.writeheader()
        for ident, text in rows:
            writer.writerow({"id": ident, "text": text})
    return path


def test_load_wordlist_und_grossschreibung(tmp_path):
    path = write_wordlist(tmp_path / "words.csv",
                          [("1", "Auto"), ("2", "  Haus "), ("3", "Straße")])
    words = sentence.load_wordlist(path)
    assert words["AUTO"] == "1"
    assert words["HAUS"] == "2"
    # ß wird beim Grossschreiben zu SS (wie in der Wortliste).
    assert words["STRASSE"] == "3"


def test_split_tokens_und_buchstabenfolge():
    assert sentence.split_tokens("  Auto, Hund; Haus ") == ["Auto", "Hund",
                                                            "Haus"]
    assert sentence.split_tokens("H|A|U|S") == ["H", "A", "U", "S"]
    assert sentence.is_letter_sequence(["H", "A", "u", "S"])
    assert not sentence.is_letter_sequence(["H", "Auto"])
    assert not sentence.is_letter_sequence([])


def test_umlaut_schreibweise_wird_gefunden():
    words = {"BRÜCKE": "1", "STRASSE": "2"}
    assert sentence.lookup_id(words, "BRUECKE") == "1"
    assert sentence.lookup_id(words, "Brücke") == "1"
    assert sentence.lookup_id(words, "Strasse") == "2"
    assert sentence.lookup_id(words, "gibt-es-nicht") is None


def test_signs_fuer_wort_teile_und_buchstaben():
    words = {"AUTO": "61", "IN": "7", "KOMMEND": "8",
             "H": "10", "A": "11", "U": "12", "S": "13"}

    assert sentence.signs_for_token("Auto", words) == [
        sentence.Sign("AUTO", "61")]

    parts = sentence.signs_for_token("in-kommend", words)
    assert [sign.ident for sign in parts] == ["7", "8"]

    spelled = sentence.signs_for_token("Haus", words)
    assert [sign.label for sign in spelled] == ["H", "A", "U", "S"]
    assert all(sign.spelled for sign in spelled)

    # Fehlt ein Buchstabe, wird er ausgelassen; die bekannten Zeichen bleiben.
    assert sentence.signs_for_token("HÄuser", {"H": "10"}) == [
        sentence.Sign("H", "10", spelled=True)]

    # Einzelne Umlaute nutzen die AE/OE/UE-Eintraege der gsg-Wortliste.
    assert sentence.spell_word("Ä", {"AE": "99"}) == [
        sentence.Sign("Ä", "99", spelled=True)]

    # Ohne AE/OE/UE-Eintrag zerfaellt der Umlaut in zwei Buchstaben (wie gsg).
    spread = {"A": "11", "E": "12", "R": "13", "G": "14",
              "U": "15", "B": "16"}
    assert [sign.label for sign in sentence.spell_word("Ärger", spread)] == [
        "A", "E", "R", "G", "E", "R"]
    assert [sign.label for sign in sentence.spell_word("Über", spread)] == [
        "U", "E", "B", "E", "R"]

    # SCH wird wie in gsg.py zu einer Gebaerde zusammengefasst.
    assert [sign.label for sign in sentence.spell_word(
        "Schule", {"SCH": "1", "U": "2", "L": "3", "E": "4"})] == [
            "SCH", "U", "L", "E"]


def test_plan_nutzt_gloss_nur_bei_genug_treffern():
    words = {"SONNE": "1", "SCHEINEN": "2", "HUND": "3", "LAUFEN": "4",
             "S": "5", "C": "6", "H": "7", "I": "9", "N": "10"}

    # Beide Gloss-Woerter bekannt -> Uebersetzung gewinnt.
    plan = sentence.plan(["Die", "Sonne", "scheint"], words,
                         gloss=["sonne", "scheinen"])
    assert plan.source == "gloss"
    assert [sign.ident for sign in plan.signs] == ["1", "2"]

    # Weniger als die Haelfte bekannt -> eingegebene Woerter.
    gloss = ["region", "gefahr", "wald", "berg"]
    plan = sentence.plan(["Der", "Hund", "laeuft"], words, gloss=gloss)
    assert plan.source == "words"
    assert [sign.label for sign in plan.signs] == ["HUND"]
    assert "DER" in plan.skipped and "LAEUFT" in plan.skipped

    # Ohne Gloss direkt die Eingabe.
    plan = sentence.plan(["Sonne"], words)
    assert plan.source == "words"
    assert plan.signs == (sentence.Sign("SONNE", "1"),)


def test_plan_meldet_unbekannte_und_begrenzt():
    words = {"A": "1", "B": "2", "HAUS": "3"}
    plan = sentence.plan(["Haus", "xyz"], words)
    assert [sign.label for sign in plan.signs] == ["HAUS"]
    assert plan.skipped == ("XYZ",)

    # Buchstabenfolge als Quelle kennzeichnen.
    plan = sentence.plan(["A", "B"], words)
    assert plan.source == "letters"
    assert [sign.ident for sign in plan.signs] == ["1", "2"]

    # Sehr lange Glossen werden auf MAX_SIGNS begrenzt.
    words = {"A": "1"}
    plan = sentence.plan(["A" * 30], words, max_signs=5)
    assert len(plan.signs) == 5
