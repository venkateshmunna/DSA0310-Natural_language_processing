"""Experiment 17: Accessing WordNet"""

import nltk
from nltk.corpus import wordnet as wn
 
# Run once if required:
# nltk.download('wordnet')
# nltk.download('omw-1.4')
 
word = "car"
for synset in wn.synsets(word):
    print("Synset:", synset.name())
    print("Definition:", synset.definition())
    print("Examples:", synset.examples())
    print()
