"""
attention.py

Authors: Peter Spanke, Carolin Gottschalk
License: Apache License 2.0

Description:
This module implements Bahdanau attention, a soft attention mechanism used in sequence-to-sequence models.
It calculates context vectors based on attention scores computed from encoder outputs and the decoder query.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """
    Implements Bahdanau (Additive) Attention mechanism.
    """

    def __init__(self, hidden_size: int):
        """
        Initializes the attention mechanism.

        :param hidden_size: The size of the hidden layers in the encoder and decoder.
        :type hidden_size: int
        """
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)  # Linear transformation for query
        self.Ua = nn.Linear(hidden_size, hidden_size)  # Linear transformation for keys
        self.Va = nn.Linear(hidden_size, 1)  # Linear transformation for attention scores

    def forward(self, query: torch.Tensor, keys: torch.Tensor):
        """
        Computes attention scores and context vectors.

        :param query: The decoder hidden state (query vector).
        :type query: torch.Tensor
        :param keys: The encoder outputs (key vectors).
        :type keys: torch.Tensor
        :return: A tuple containing the computed context vector and attention weights.
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)
        return context, weights
