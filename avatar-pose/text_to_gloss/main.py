"""
main.py

Authors: Open Source Contributor
License: Apache License 2.0

Description:
This script initializes and trains a sequence-to-sequence model for text-to-gloss translation.
It loads the dataset, prepares training and testing data, defines model parameters,
trains an encoder-decoder architecture, and evaluates the model using BLEU scores.
"""

from dataset import get_dataloader
from models.encoder import EncoderRNN
from models.decoder import AttnDecoderRNN
from training import train
from evaluation import evaluate_random_BLEU
import torch
from data_processing import prepare_train_test_data
from pathlib import Path # neu
import os # neu
import sys # neu

# 1. Standard-Streams erzwingen, UTF-8 zu nutzen #neu
sys.stdout.reconfigure(encoding='utf-8') #neu
sys.stderr.reconfigure(encoding='utf-8') #neu

# 2. Arbeitsverzeichnis im Projektordner erzwingen #neu
os.chdir(Path(__file__).resolve().parent) #neu

# Set device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
hidden_size = 512
batch_size = 64
n_epochs = 50
learning_rate = 0.001

# Skript-Ordner dynamisch ermitteln
script_dir = Path(__file__).resolve().parent #neu

# Absoluten Pfad zum data-Ordner erstellen
data_dir = script_dir / "data" #neu

# Prepare dataset mit absolutem Pfad aufrufen
prepare_train_test_data("Text2Gloss.csv", str(data_dir), retrain_model=True) #neu

# Prepare dataset for training and testing
#prepare_train_test_data("Text2Gloss.csv", "data", retrain_model=True)

# Load data and prepare vocabulary
input_lang, output_lang, train_dataloader, pairs = get_dataloader("dgs","de", batch_size, device)

print(f"Number of sentence pairs after `prepare_data`: {len(pairs)}")
print(f"First sentence pair after `prepare_data`: {pairs[0] if pairs else 'No data available'}")
print(f"Vocabulary size: {input_lang.name} = {input_lang.n_words}, {output_lang.name} = {output_lang.n_words}")

# Initialize encoder and decoder models
encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

# Train the model
train(train_dataloader, encoder, decoder, n_epochs, input_lang, output_lang, device, learning_rate)

# Evaluate the model using BLEU scores
bleu_score = evaluate_random_BLEU(encoder, decoder, pairs, input_lang, output_lang, n=10, device=device)
print(f"BLEU-Score: {bleu_score}")
