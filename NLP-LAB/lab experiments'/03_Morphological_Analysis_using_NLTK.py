"""Experiment 3: Morphological Analysis using NLTK"""

import nltk
from nltk.tokenize import word_tokenize
 
# Run once if required:
# nltk.download('punkt')
# nltk.download('punkt_tab')
 
text = "The boys are playing and the cats chased mice."
for word in word_tokenize(text):
    if word.isalpha():
        print(f"Word: {word:12} | Lowercase: {word.lower():12} | "
              f"Suffix: {word[-3:] if len(word) >= 3 else word}")
