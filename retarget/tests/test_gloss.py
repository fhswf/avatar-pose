"""Uebersetzungsmodul (langsam: laedt torch, spacy und das Vokabular).

Standardmaessig uebersprungen; mit ``AVATAR_SLOW_TESTS=1`` ausfuehren.
"""

import os

import pytest

from retarget import gloss, sentence

TTG = gloss.DEFAULT_DIR
CHECKPOINTS = TTG / "data" / "encoder_epoch20.pt"

pytestmark = pytest.mark.skipif(
    os.environ.get("AVATAR_SLOW_TESTS") != "1" or not CHECKPOINTS.exists(),
    reason="langsam oder Checkpoints fehlen (AVATAR_SLOW_TESTS=1)")


def test_uebersetzt_wettertext_und_plant_gebaerden():
    translator = gloss.GlossTranslator(epoch=20)
    assert translator.available()

    words = translator.translate("Die Sonne scheint.")
    assert "sonne" in words
    assert all(word.upper() not in ("SOS", "EOS", "UNK") for word in words)

    plan = sentence.plan(sentence.split_tokens("Die Sonne scheint."),
                         {"SONNE": "672"}, gloss=words)
    assert plan.signs[0] == sentence.Sign("SONNE", "672")


def test_get_translator_cachet_und_abschalten(monkeypatch):
    monkeypatch.delenv("AVATAR_GLOSS", raising=False)
    first = gloss.get_translator(epoch=20)
    second = gloss.get_translator(epoch=20)
    assert first is not None and first is second

    monkeypatch.setenv("AVATAR_GLOSS", "0")
    assert gloss.get_translator(epoch=20) is None
