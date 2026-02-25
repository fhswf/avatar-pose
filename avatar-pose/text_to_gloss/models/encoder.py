"""
encoder.py

Authors: Peter Spanke, Carolin Gottschalk
License: Apache License 2.0

Description:
This module defines an EncoderRNN class for use in sequence-to-sequence models.
It utilizes an embedding layer and a GRU (Gated Recurrent Unit) network to encode input sequences.
"""

import torch.nn as nn


class EncoderRNN(nn.Module):
    """
    Encoder RNN that processes input sequences and returns encoded representations.
    """

    def __init__(self, input_size: int, hidden_size: int):
        """
        Initializes the Encoder RNN model.

        :param input_size: Size of the input vocabulary.
        :type input_size: int
        :param hidden_size: Size of the hidden state in the GRU.
        :type hidden_size: int
        """
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input):
        """
        Defines the forward pass of the encoder.

        :param input: Input sequence tensor.
        :type input: torch.Tensor
        :return: Tuple containing the output sequence and the final hidden state.
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded)
        return output, hidden
