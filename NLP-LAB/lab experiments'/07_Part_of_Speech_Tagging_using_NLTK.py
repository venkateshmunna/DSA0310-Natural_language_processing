"""Experiment 7: Part-of-Speech Tagging using NLTK"""

import nltk
from nltk.tokenize import word_tokenize
 
# Run once if required:
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')
 
sentence = "The intelligent student writes a program."
tokens = word_tokenize(sentence)
print(nltk.pos_tag(tokens))
