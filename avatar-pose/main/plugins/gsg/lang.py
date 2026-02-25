"""
lang.py

Authors: Peter Spanke, Carolin Gottschalk
License: Apache License 2.0

Description:
This module provides a `Lang` class for handling vocabulary in a text dataset.
It supports token indexing, word frequency tracking, and conversion between indexes and words.
"""

from .constant import SOS_tag, SOS_token, EOS_tag, EOS_token, UNK_tag, UNK_token

class Lang:
    """
    A class to manage vocabulary for a language.
    It tracks word-to-index mappings, word frequencies, and index-to-word mappings.
    """

    def __init__(self, name: str):
        """
        Initializes the `Lang` class for a specific language.

        :param name: Name of the language.
        :type name: str
        """
        self.name = name
        self.word2index = {}  # Dictionary mapping words to unique indices
        self.word2count = {}  # Dictionary tracking word frequencies
        self.index2word = {SOS_token: SOS_tag, EOS_token: EOS_tag, UNK_token: UNK_tag}  # Predefine index mapping with special tokens
        self.n_words = len(self.index2word)  # Number of words, initialized with special tokens

    def addSentence(self, sentence: str):
        """
        Splits a sentence into words and adds them to the vocabulary.

        :param sentence: The input sentence to process.
        :type sentence: str
        """
        for word in sentence.split():  # Split sentence by spaces
            self.addWord(word)  # Add each word to the vocabulary

    def addWord(self, word: str):
        """
        Adds a word to the vocabulary or updates its count if it already exists.

        :param word: The word to add to the vocabulary.
        :type word: str
        """
        if word not in self.word2index:
            # Assign a new index, initialize count, and store in index2word
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            # Increment the word count if it already exists
            self.word2count[word] += 1
