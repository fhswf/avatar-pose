"""
constant.py

Authors: Peter Spanke, Carolin Gottschalk
License: Apache License 2.0

Description:
This module defines constants used throughout the project.
It includes special token IDs for sequence processing and the maximum allowed sequence length.
"""

# Define Start-of-Sentence (SOS) token
SOS_token = 0
SOS_tag = "SOS"

# Define End-of-Sentence (EOS) token
EOS_token = 1
EOS_tag = "EOS"

# Define Unknown (UNK) token
UNK_token = 2
UNK_tag = "UNK"

# Maximum sequence length constraint
MAX_LENGTH = 30