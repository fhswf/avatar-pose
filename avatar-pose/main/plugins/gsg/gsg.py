"""
TestLang.py

Authors: Ann-Kristin Schulte, Carolin Gottschalk, Jonas D. Stephan
License: Apache License 2.0

Description:
This module defines the TestLang class, which extends LangPlugin to process pose estimation data.
It reads JSON data, computes final rotations using the Calculator class, and sends processed
frames to the output hook.
"""
import csv
import torch
import spacy
import sys
import os
import re

from .models.decoder import AttnDecoderRNN
from .models.encoder import EncoderRNN
from ..LangPlugin import LangPlugin
from .dataset import get_dataloader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from Calculator import Calculator


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
        #TODO Hier Quelltext einfügen

        self.nlp = spacy.load("de_core_news_sm")
        self.data={}
        with open("plugins/gsg/wordlist.csv", newline='', encoding='utf-8') as csvfile:
            reader=csv.DictReader(csvfile)
            for row in reader:
                key=row["text"].upper()  # 'text' als Schlüssel in Großbuchstaben
                self.data[key]=row
        with open("plugins/gsg/known_names.txt", "r") as file:
            self.known_names = [line.strip() for line in file]
        self.calculator = Calculator(output_hook.get_t_pose())

    def input(self, string: str):
        """
        Processes an input command and computes pose rotations based on provided JSON data.

        :param string: Input command to determine the processing mode.
        :type string: str
        """
        doc = self.nlp(string)
        names = [ent.text for ent in doc.ents if ent.label_ in ["PER", "LOC", "ORG", "MISC"]]
        # TODO
        print(string)
        names = [name for name in names if name not in self.known_names]

        for name in names:
            string = string.replace(name, preprocess_unknown_signs(name))

        words=string.split()
        for word in words:
            if word not in self.data:
                string=string.replace(word, preprocess_unknown_signs(word))

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

def preprocess_unknown_signs(name):
    processed_name = ""
    name = name.upper()
    name = re.sub(r"[^a-zA-ZäöüÄÖÜß ]", "", name)
    i = 0
    while i < len(name):
        # TODO AA, EE, SS
        if name[i:i + 3] == "SCH":
            processed_name += "SCH"
            i += 3
        else:
            processed_name += name[i]
            i += 1

        if i < len(name):
            processed_name += "|"

    return processed_name

def pev_exists(number: int) -> bool:
    """Checks, if an file to the id is existing in 'plugins/gsg/videos/'."""
    filename = f"plugins/gsg/videos/{number}.pev"
    return os.path.isfile(filename)
