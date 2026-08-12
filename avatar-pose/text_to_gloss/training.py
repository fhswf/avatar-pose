"""
training.py

Authors: Peter Spanke, Carolin Gottschalk
License: Apache License 2.0

Description:
This script implements the training loop for a sequence-to-sequence model using PyTorch.
It includes functions to train an encoder-decoder model, compute loss, update model weights,
and evaluate the model using BLEU scores.
"""
from torch import optim
import torch.nn as nn
import torch
from evaluation import evaluate_BLEU

def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device):
    #device ergänzt
    """
    Trains the model for one epoch.

    :param dataloader: DataLoader providing input-target pairs.
    :type dataloader: torch.utils.data.DataLoader
    :param encoder: Encoder model.
    :type encoder: torch.nn.Module
    :param decoder: Decoder model.
    :type decoder: torch.nn.Module
    :param encoder_optimizer: Optimizer for encoder parameters.
    :type encoder_optimizer: torch.optim.Optimizer
    :param decoder_optimizer: Optimizer for decoder parameters.
    :type decoder_optimizer: torch.optim.Optimizer
    :param criterion: Loss function for training.
    :type criterion: torch.nn.Module
    :return: Average loss for the epoch.
    :rtype: float
    """
    total_loss = 0
    for input_tensor, target_tensor in dataloader:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # Forward pass through the encoder
        encoder_outputs, encoder_hidden = encoder(input_tensor)

        # Forward pass through the decoder
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        # Compute loss
        loss = criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)), target_tensor.view(-1))
        loss.backward()

        # Update model parameters
        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

        #print(f"🔍 Batch-Loss: {loss.item()}")

    return total_loss / len(dataloader)

def train(dataloader, encoder, decoder, n_epochs, input_lang, output_lang, device, learning_rate=0.001):
    """
    Runs the training loop for multiple epochs.

    :param dataloader: DataLoader providing input-target pairs.
    :type dataloader: torch.utils.data.DataLoader
    :param encoder: Encoder model.
    :type encoder: torch.nn.Module
    :param decoder: Decoder model.
    :type decoder: torch.nn.Module
    :param n_epochs: Number of epochs for training.
    :type n_epochs: int
    :param input_lang: Input language vocabulary.
    :type input_lang: object
    :param output_lang: Output language vocabulary.
    :type output_lang: object
    :param device: Device to run the model on (CPU or GPU).
    :type device: torch.device
    :param learning_rate: Learning rate for the optimizer, defaults to 0.001.
    :type learning_rate: float, optional
    """
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        epoch_loss = train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device)
        # neu device ergänzt

        #print(f"Epoch {epoch}/{n_epochs} - Loss: {epoch_loss:.4f}")
        # Evaluate model performance using BLEU score
        print(evaluate_BLEU(encoder, decoder, "data/test.txt", input_lang, output_lang, device=device))

        # Save model checkpoints after each epoch
        torch.save(encoder.state_dict(), f"data/encoder_epoch{epoch}.pt")
        torch.save(decoder.state_dict(), f"data/decoder_epoch{epoch}.pt")


