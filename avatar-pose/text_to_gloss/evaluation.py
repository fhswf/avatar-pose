"""
evaluation.py

Authors: Peter Spanke, Carolin Gottschalk
License: Apache License 2.0

Description:
This module provides evaluation functions for sequence-to-sequence models.
It includes methods for generating translations, computing BLEU scores,
and evaluating model performance on test data.
"""

import torch
from preprocessing import normalize_string
from dataset import tensor_from_sentence
from torcheval.metrics import BLEUScore
import random

from constant import EOS_token

def evaluate(encoder, decoder, sentence, input_lang, output_lang, device):
    """
    Generates a translation for a given input sentence using the trained encoder-decoder model.

    :param encoder: Trained encoder model.
    :type encoder: torch.nn.Module
    :param decoder: Trained decoder model.
    :type decoder: torch.nn.Module
    :param sentence: Input sentence to translate.
    :type sentence: str
    :param input_lang: Source language vocabulary.
    :type input_lang: object
    :param output_lang: Target language vocabulary.
    :type output_lang: object
    :param device: Computation device (CPU/GPU).
    :type device: torch.device
    :return: Translated sentence as a list of words.
    :rtype: list
    """
    with torch.no_grad():
        input_tensor = tensor_from_sentence(input_lang, sentence, device)
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)   # Get most likely word indexes
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                break
            decoded_words.append(output_lang.index2word[idx.item()])

    return decoded_words

def evaluate_BLEU(encoder, decoder, test_file, input_lang, output_lang, n_gram=2, device=None):
    """
    Computes the BLEU score for a trained model on a test dataset.

    :param encoder: Trained encoder model.
    :type encoder: torch.nn.Module
    :param decoder: Trained decoder model.
    :type decoder: torch.nn.Module
    :param test_file: Path to the test dataset.
    :type test_file: str
    :param input_lang: Source language vocabulary.
    :type input_lang: object
    :param output_lang: Target language vocabulary.
    :type output_lang: object
    :param n_gram: The n-gram size for BLEU score computation, defaults to 2.
    :type n_gram: int, optional
    :param device: Computation device (CPU/GPU), defaults to None.
    :type device: torch.device, optional
    :return: Average BLEU score for the dataset.
    :rtype: float
    """
    bleu_sum = 0
    test_lines = open(test_file, encoding="utf-8").read().strip().split("\n")
    test_pairs = [[normalize_string(s) for s in l.split("\t")] for l in test_lines]
    n = len(test_pairs)

    for i in range(n - 1):
        test_pair = test_pairs[i]

        output_words = evaluate(encoder, decoder, test_pair[1], input_lang, output_lang, device)
        output_sentence = " ".join(output_words)

        if len(test_pair[0].split()) < n_gram or len(output_sentence.split()) < n_gram:
            bleu_sentence = 0
        else:
            metric = BLEUScore(n_gram=n_gram)
            metric.update(test_pair[0], [output_sentence])
            bleu_sentence = float(metric.compute())

        bleu_sum += bleu_sentence

    return bleu_sum / n

def evaluate_random_BLEU(encoder, decoder, pairs, input_lang, output_lang, n=10, n_gram=2, device=None):
    """
    Computes the BLEU score for a randomly selected subset of sentence pairs.

    :param encoder: Trained encoder model.
    :type encoder: torch.nn.Module
    :param decoder: Trained decoder model.
    :type decoder: torch.nn.Module
    :param pairs: List of sentence pairs.
    :type pairs: list
    :param input_lang: Source language vocabulary.
    :type input_lang: object
    :param output_lang: Target language vocabulary.
    :type output_lang: object
    :param n: Number of random sentences to evaluate, defaults to 10.
    :type n: int, optional
    :param n_gram: The n-gram size for BLEU score computation, defaults to 2.
    :type n_gram: int, optional
    :param device: Computation device (CPU/GPU), defaults to None.
    :type device: torch.device, optional
    :return: Average BLEU score for the selected sentences.
    :rtype: float
    """
    bleu_sum = 0

    for _ in range(n):
        pair = random.choice(pairs)
        print(f"Input (lemmatized): {pair[0]}")
        print(f"Target: {pair[1]}")

        output_words = evaluate(encoder, decoder, pair[0], input_lang, output_lang, device)
        output_sentence = " ".join(output_words)
        print(f"Output: {output_sentence}")

        if len(pair[1].split()) < n_gram or len(output_sentence.split()) < n_gram:
            bleu_score = 0
        else:
            metric = BLEUScore(n_gram=n_gram)
            metric.update(pair[1], [output_sentence])
            bleu_score = metric.compute().item()

        bleu_sum += bleu_score
        print(f"BLEU-Score: {bleu_score:.4f}")

    return bleu_sum / n
