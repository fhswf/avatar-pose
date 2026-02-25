"""
decoder.py

Authors: Peter Spanke, Carolin Gottschalk
License: Apache License 2.0

Description:
This module defines the AttnDecoderRNN class, which is an attention-based decoder
for sequence-to-sequence models. It uses Bahdanau attention, a GRU-based decoder,
and applies teacher forcing when available.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from .attention import BahdanauAttention

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from constant import MAX_LENGTH, SOS_token

class AttnDecoderRNN(nn.Module):
    """
    Attention-based Decoder RNN using Bahdanau attention.
    """
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        """
        Initializes the AttnDecoderRNN model.

        :param hidden_size: Size of the hidden layer in the GRU.
        :type hidden_size: int
        :param output_size: Vocabulary size of the target language.
        :type output_size: int
        :param dropout_p: Dropout probability, defaults to 0.1.
        :type dropout_p: float, optional
        """
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None, device=None):
        """
        Defines the forward pass of the decoder.

        :param encoder_outputs: The output sequence from the encoder.
        :type encoder_outputs: torch.Tensor
        :param encoder_hidden: The final hidden state from the encoder.
        :type encoder_hidden: torch.Tensor
        :param target_tensor: The target sequence for teacher forcing, defaults to None.
        :type target_tensor: torch.Tensor, optional
        :param device: The computation device (CPU/GPU), defaults to None.
        :type device: torch.device, optional
        :return: Tuple containing decoder outputs, hidden states, and attention weights.
        :rtype: tuple(torch.Tensor, torch.Tensor, torch.Tensor)
        """
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        """
        Performs a single decoding step.

        :param input: The input token tensor.
        :type input: torch.Tensor
        :param hidden: The hidden state of the decoder.
        :type hidden: torch.Tensor
        :param encoder_outputs: The encoder outputs for attention computation.
        :type encoder_outputs: torch.Tensor
        :return: Tuple containing decoder output, updated hidden state, and attention weights.
        :rtype: tuple(torch.Tensor, torch.Tensor, torch.Tensor)
        """

        embedded = self.dropout(self.embedding(input))
        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights