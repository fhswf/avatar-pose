"""
TestLang.py

Authors: Ann-Kristin Schulte, Carolin Gottschalk, Jonas D. Stephan
License: Apache License 2.0

Description:
This module defines the TestLang class, which extends LangPlugin to process pose estimation data.
It reads JSON data, computes final rotations using the Calculator class, and sends processed
frames to the output hook.
"""
import json
import torch
import spacy
import sys
import os
import re
from pathlib import Path

import pandas as pd

from .models.decoder import AttnDecoderRNN
from .models.encoder import EncoderRNN
from ..LangPlugin import LangPlugin
from .dataset import get_dataloader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from Calculator import Calculator

HERE = Path(__file__).resolve().parent
WORDLIST = HERE / "wordlist.csv"
KNOWN_NAMES = HERE / "known_names.txt"
PEV_DIR = HERE / "extracted_videos"
TTG_DATA = HERE.parents[2] / "text_to_gloss" / "data"
# Kandidaten fuer zusammenhängende Buchstabier-Gebärden (längste zuerst).
SPELLING_GROUPS = ("SCH", "AA", "EE", "SS", "AE", "OE", "UE")


class gsg(LangPlugin):
    """
    A language processing plugin for handling skeleton-based pose estimation.
    It computes final rotations for pose data and sends processed frames to the output hook.
    """

    def __init__(self, output_hook):
        """
        Initializes the TestLang plugin with an output hook and a Calculator instance.

        :param output_hook: An object responsible for sending processed data.
        :type output_hook: object
        """
        super().__init__(output_hook)

        hidden_size = 512
        batch_size = 64
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_lang, output_lang, train_dataloader, pairs = get_dataloader("dgs", "de", batch_size, device)
        self.encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
        self.decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

        # Vorhandene Checkpoints laden, damit der Encoder initialisiert ist.
        epoch = (os.environ.get("AVATAR_GLOSS_EPOCH", "20").strip() or "20")
        encoder_path = TTG_DATA / f"encoder_epoch{epoch}.pt"
        decoder_path = TTG_DATA / f"decoder_epoch{epoch}.pt"
        disabled = os.environ.get("AVATAR_GLOSS", "1").strip().lower() in (
            "0", "off", "false", "no", "none")
        if not disabled and encoder_path.exists() and decoder_path.exists():
            try:
                self.load_model(self.encoder, self.decoder, str(encoder_path),
                                str(decoder_path), device)
            except Exception as exc:                      # noqa: BLE001
                print(f"[gsg] Modell nicht geladen: {exc}")
        elif not disabled:
            print(f"[gsg] Checkpoints fehlen: {encoder_path.name}, "
                  f"{decoder_path.name}")

        self.nlp = spacy.load("de_core_news_sm")

        # Die Wortliste waechst weiter: DataFrame statt Dict.
        self.data = pd.read_csv(WORDLIST, dtype=str, keep_default_na=False,
                                encoding="utf-8-sig")
        self.data["text"] = self.data["text"].str.upper()
        self.data = (self.data.drop_duplicates(subset="text", keep="first")
                     .set_index("text"))
        with open(KNOWN_NAMES, "r", encoding="utf-8") as file:
            self.known_names = [line.strip() for line in file]
        # Gruppen nur dann als eine Gebaerde, wenn Wordlist UND PEV vorhanden
        # sind. Die Wortliste waechst weiter - dann greifen sie automatisch.
        self.spelling_groups = {
            group for group in SPELLING_GROUPS
            if group in self.data.index
            and pev_exists(self.data.at[group, "id"])
        }
        self.calculator = Calculator(output_hook.get_t_pose())

    def input(self, string: str):
        """
        Processes an input command and computes pose rotations based on provided JSON data.

        :param string: Input command to determine the processing mode.
        :type string: str
        """
        doc = self.nlp(string)
        names = [ent.text for ent in doc.ents if ent.label_ in ["PER", "LOC", "ORG", "MISC"]]
        names = [name for name in names if name not in self.known_names]
        for name in names:
            string = string.replace(name, preprocess_unknown_signs(
                name, self.spelling_groups))

        # Pro Token arbeiten, statt den ganzen Satz mehrfach zu ersetzen.
        signs = []
        for item in re.split(r"[\s|]+", string):
            if not item:
                continue
            key = item.upper()
            if key in self.data.index:
                signs.append(key)
            else:
                signs.extend(part for part in preprocess_unknown_signs(
                    item, self.spelling_groups).split("|") if part)

        # Ganzes Wort mit PEV bevorzugen, sonst buchstabieren (wie retarget).
        resolved = []
        for sign in signs:
            ident = self.data.at[sign, "id"] if sign in self.data.index else None
            if ident and pev_exists(ident):
                resolved.append((sign, ident))
                continue
            for chunk in preprocess_unknown_signs(sign, self.spelling_groups).split("|"):
                if not chunk:
                    continue
                chunk_id = (self.data.at[chunk, "id"]
                            if chunk in self.data.index else None)
                if chunk_id and pev_exists(chunk_id):
                    resolved.append((chunk, chunk_id))
        if not resolved:
            raise FileNotFoundError("Keine PEV fuer: %s" % ", ".join(signs))

        for sign, ident in resolved:
            # PEV laden und Rotationen an den WebSocket senden.
            with open(PEV_DIR / f"{str(ident).strip()}.pev", encoding="utf-8") as file:
                data = json.load(file)

            for i in range(len(data["frames"])):
                self.output_hook.add_send_object(
                    self.calculator.compute_final_rotations(data["frames"][i].get("skeletonpoints"))
                )    
            

    def load_model(self, encoder, decoder, encoder_path, decoder_path, device):
        """
        Loads the saved model weights into the given encoder and decoder models.

        :param encoder: The encoder model.
        :type encoder: torch.nn.Module
        :param decoder: The decoder model.
        :type decoder: torch.nn.Module
        :param encoder_path: Path to the saved encoder checkpoint file.
        :type encoder_path: str
        :param decoder_path: Path to the saved decoder checkpoint file.
        :type decoder_path: str
        :param device: The device on which the models should be loaded (CPU or GPU).
        :type device: torch.device
        """
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))

        encoder.to(device)
        decoder.to(device)

        encoder.eval()  # Set the model to evaluation mode
        decoder.eval()  # Set the model to evaluation mode

def preprocess_unknown_signs(name, groups=None):
    """Buchstabier-Einheiten fuer ein unbekanntes Wort.

    ``groups`` enthaelt nur Kombinationen, fuer die Wordlist und PEV vorhanden
    sind. Ohne Angabe werden alle Kandidaten benutzt (z. B. fuer Helfer-Tests).
    """
    processed_name = ""
    name = name.upper()
    # Wie in retarget/sentence.py: Umlaute und ß in Buchstabier-Einheiten.
    for src, dst in (("Ä", "AE"), ("Ö", "OE"), ("Ü", "UE"), ("ß", "SS")):
        name = name.replace(src, dst)
    name = re.sub(r"[^a-zA-Z ]", "", name)
    candidates = SPELLING_GROUPS if groups is None else groups
    i = 0
    while i < len(name):
        for group in candidates:
            if name.startswith(group, i):
                processed_name += group
                i += len(group)
                break
        else:
            processed_name += name[i]
            i += 1

        if i < len(name):
            processed_name += "|"

    return processed_name

def pev_exists(number) -> bool:
    """Prueft die vorab extrahierte PEV in 'plugins/gsg/extracted_videos/'."""
    return (PEV_DIR / f"{str(number).strip()}.pev").is_file()
