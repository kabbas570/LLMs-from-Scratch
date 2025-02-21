data = [
    {"en": "My name is Abbas", "ur": "میرا نام عباس ہے"},
    {"en": "How are you?", "ur": "آپ کیسے ہیں؟"},
    {"en": "I am fine", "ur": "میں ٹھیک ہوں"},
    {"en": "What is your name?", "ur": "آپ کا نام کیا ہے؟"},
    {"en": "Thank you", "ur": "شکریہ"},
]

import torch
import numpy as np

# Tokenize the data
def tokenize(text):
    return text.split()

# Create vocabularies
en_vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
ur_vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}

for pair in data:
    #print(pair['en'])
    for word in tokenize(pair["en"]):
        print(word)
        if word not in en_vocab:
            en_vocab[word] = len(en_vocab)
            

# Build vocabularies
for pair in data:
    for word in tokenize(pair["en"]):
        if word not in en_vocab:
            en_vocab[word] = len(en_vocab)
    for word in tokenize(pair["ur"]):
        if word not in ur_vocab:
            ur_vocab[word] = len(ur_vocab)

# Reverse vocabularies for decoding
en_idx2word = {idx: word for word, idx in en_vocab.items()}
ur_idx2word = {idx: word for word, idx in ur_vocab.items()}

# Convert sentences to sequences of indices
def sentence_to_seq(sentence, vocab):
    tokens = tokenize(sentence)
    return [vocab["<SOS>"]] + [vocab[word] for word in tokens] + [vocab["<EOS>"]]

# Prepare dataset
dataset = []
for pair in data:
    en_seq = sentence_to_seq(pair["en"], en_vocab)
    ur_seq = sentence_to_seq(pair["ur"], ur_vocab)
    dataset.append((en_seq, ur_seq))

# Pad sequences to the same length
def pad_sequence(seq, max_len, pad_idx):
    return seq + [pad_idx] * (max_len - len(seq))

max_len = max(max(len(en), len(ur)) for en, ur in dataset)
padded_dataset = [(pad_sequence(en, max_len, en_vocab["<PAD>"]), pad_sequence(ur, max_len, ur_vocab["<PAD>"])) for en, ur in dataset]

## Problem with this approach ###

# Idiomatic Expressions:
# Phrases like "What is your name?" translate to "آپ کا نام کیا ہے؟" as a whole, not word by word.
# The model might incorrectly learn mappings like "What" → "آپ" or "is" → "کیا", which are not accurate translations.
