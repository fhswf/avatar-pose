"""
preprocessing.py

Authors: Peter Spanke, Carolin Gottschalk
License: Apache License 2.0

Description:
This script provides preprocessing functions for text data used in machine learning models.
It includes normalization, reading bilingual sentence pairs, filtering based on length,
and vocabulary preparation.
"""

from lang import Lang
from constant import MAX_LENGTH
import re

def normalize_string(s):
    """
    Normalizes a string by converting it to lowercase, removing extra spaces, and formatting punctuation.

    :param s: Input string to be normalized.
    :type s: str
    :return: Normalized string.
    :rtype: str
    """
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-ZÄÖÜäöüß?]+", r" ", s)
    return s.strip()

def read_langs(data_path, lang1, lang2, reverse=False):
    """
    Reads bilingual sentence pairs from a text file, normalizes them, and optionally reverses the language order.

    :param data_path: Path to the directory containing language data.
    :type data_path: str
    :param lang1: Source language.
    :type lang1: str
    :param lang2: Target language.
    :type lang2: str
    :param reverse: Whether to reverse input and output languages, defaults to False.
    :type reverse: bool, optional
    :return: Tuple containing input language object, output language object, and list of sentence pairs.
    :rtype: tuple
    """
    print(f"Processing training file: {data_path}/{lang1}-{lang2}.txt")

    # Read lines and split by tab
    lines = open(f"{data_path}/{lang1}-{lang2}.txt", encoding='utf-8').read().strip().split('\n')
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    print(f"🔍 Number of sentences read: {len(pairs)}")
    print(f"🔍 First sentence pair before reversal: {pairs[0] if pairs else 'No data'}")

    # Reverse language order if needed
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        print(f"🔍 First sentence pair after reversal: {pairs[0] if pairs else 'No data'}")
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def filter_pairs(pairs):
    """
    Filters sentence pairs based on maximum length constraints.

    :param pairs: List of sentence pairs.
    :type pairs: list
    :return: Filtered list of sentence pairs.
    :rtype: list
    """
    return [pair for pair in pairs if (len(pair[0].split()) < MAX_LENGTH and len(pair[1].split()) < MAX_LENGTH)]

def prepare_data(lang1, lang2, reverse=True):
    """
    Prepares training data by reading, normalizing, filtering, and building vocabulary.

    :param lang1: Source language.
    :type lang1: str
    :param lang2: Target language.
    :type lang2: str
    :param reverse: Whether to reverse input and output languages, defaults to True.
    :type reverse: bool, optional
    :return: Tuple containing input language object, output language object, and filtered sentence pairs.
    :rtype: tuple
    """
    input_lang, output_lang, pairs = read_langs("data", lang1, lang2, reverse=reverse)
    print(f"{len(pairs)} data rows processed.")
    pairs = filter_pairs(pairs)
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Vocabulary:", input_lang.name + "(lemmatized)", input_lang.n_words)
    print("Vocabulary:", output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

