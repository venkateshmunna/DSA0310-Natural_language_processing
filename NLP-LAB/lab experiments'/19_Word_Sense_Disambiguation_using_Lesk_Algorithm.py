"""Experiment 19: Word Sense Disambiguation using Lesk Algorithm"""

import nltk
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
 
# Run once if required:
# nltk.download('wordnet')
# nltk.download('omw-1.4')
 
sentence = "He deposited money in the bank"
sense = lesk(sentence.split(), "bank")
print("Selected sense:", sense)
if sense:
    print("Meaning:", sense.definition())
