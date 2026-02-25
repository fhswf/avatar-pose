"""
dataset.py

Authors: Peter Spanke, Carolin Gottschalk
License: Apache License 2.0

Description:
This module provides functions for preparing datasets for training sequence-to-sequence models.
It includes functions for tokenizing sentences, converting tokens into tensors,
and creating DataLoader objects for training.
"""

from preprocessing import prepare_data
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from constant import EOS_token,MAX_LENGTH,UNK_token

def indexes_from_sentence(lang, sentence):
    """
    Converts a sentence into a list of token indices based on the language vocabulary.
    If a word is not found in the vocabulary, it is replaced with the UNK token.

    :param lang: Language object containing the vocabulary.
    :type lang: object
    :param sentence: The sentence to convert into token indices.
    :type sentence: str
    :return: List of token indices representing the sentence.
    :rtype: list
    """
    result = []
    words = sentence.split()
    for word in words:
        if word in lang.word2index:
            result.append(lang.word2index[word])
        else:
            result.append(UNK_token)
    return result

def tensor_from_sentence(lang, sentence, device):
    """
    Converts a sentence into a PyTorch tensor of token indices.
    Appends an EOS token at the end of the sentence.

    :param lang: Language object containing the vocabulary.
    :type lang: object
    :param sentence: The sentence to convert into a tensor.
    :type sentence: str
    :param device: The computation device (CPU/GPU).
    :type device: torch.device
    :return: Tensor containing token indices for the sentence.
    :rtype: torch.Tensor
    """
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def get_dataloader(lang1, lang2, batch_size, device):
    """
    Prepares a DataLoader for training by tokenizing and batching sentence pairs.

    :param lang1: Source language code.
    :type lang1: str
    :param lang2: Target language code.
    :type lang2: str
    :param batch_size: Number of sentence pairs per batch.
    :type batch_size: int
    :param device: The computation device (CPU/GPU).
    :type device: torch.device
    :return: Tuple containing input language, output language, DataLoader, and sentence pairs.
    :rtype: tuple
    """
    input_lang, output_lang, pairs = prepare_data(lang1, lang2)

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        #print(tgt)
        inp_ids = indexes_from_sentence(input_lang, inp)
        tgt_ids = indexes_from_sentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device), torch.LongTensor(target_ids).to(device))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    return input_lang, output_lang, train_dataloader, pairs
