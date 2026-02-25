"""
data_processing.py

Authors: Peter Spanke, Carolin Gottschalk
License: Apache License 2.0

Description:
This module handles data preprocessing for training and testing text-to-gloss models.
It includes functions for lemmatizing verbs in sentences, filtering data, and generating training and test datasets.
"""

import spacy
import os
import pandas as pd

# Load the SpaCy pipeline for German language processing
pipeline = spacy.load("de_core_news_sm")

def lemmatize_verbs(sentence):
    """
     Lemmatizes all verbs in a given sentence while keeping other words unchanged.

     :param sentence: The input sentence to be processed.
     :type sentence: str
     :return: The lemmatized sentence.
     :rtype: str
     """
    lemmatized_sentence = []
    for token in pipeline(sentence):
        if token.pos_ == "VERB":
            lemmatized_sentence.append(token.lemma_)
        else:
            lemmatized_sentence.append(token.text)
    return " ".join(lemmatized_sentence)

def prepare_train_test_data(data_file_xlsx, data_path, input_lang_tag="dgs", output_lang_tag="de", retrain_model=False):
    """
    Processes the dataset by filtering, shuffling, and splitting it into training and test sets.
    Optionally writes the data to files if `retrain_model` is set to True.

    :param data_file_xlsx: Name of the CSV file containing the dataset.
    :type data_file_xlsx: str
    :param data_path: Path where the dataset is located.
    :type data_path: str
    :param input_lang_tag: Language tag for the input data, defaults to "dgs".
    :type input_lang_tag: str, optional
    :param output_lang_tag: Language tag for the output data, defaults to "de".
    :type output_lang_tag: str, optional
    :param retrain_model: If True, writes training and test files to disk, defaults to False.
    :type retrain_model: bool, optional
    """
    train_file = os.path.join(data_path, f"{input_lang_tag}-{output_lang_tag}.txt")
    test_file = os.path.join(data_path, "test.txt")

    data_xlsx = pd.read_csv(data_path + "/" + data_file_xlsx, sep=";")

    # Define entries to be excluded (if needed)
    skip_entries = []  # Placeholder list for filtering the dataset

    # Filter and shuffle data
    filtered_data = data_xlsx[~data_xlsx["origin"].isin(skip_entries)]
    filtered_data = filtered_data.sample(frac=1).reset_index(drop=True)

    # Split into training and test data
    train_test_split_ratio = 0.1
    test_size = int(len(filtered_data) * train_test_split_ratio)
    train_data = filtered_data[test_size:]
    test_data = filtered_data[:test_size]

    # Write training and test data to files if required
    if retrain_model:
        with open(train_file, "w", encoding="utf-8") as f_train:
            for index, row in train_data.iterrows():
                f_train.write(f"{row['target_gloss']}\t{lemmatize_verbs(row['source_text'])}\n")

        with open(test_file, "w", encoding="utf-8") as f_test:
            for index, row in test_data.iterrows():
                f_test.write(f"{row['target_gloss']}\t{lemmatize_verbs(row['source_text'])}\n")

    print(f"Training and test files created: \nTrain: {train_file}\nTest: {test_file}")
