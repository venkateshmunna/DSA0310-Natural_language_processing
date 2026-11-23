"""Experiment 5: Porter Stemming Algorithm"""

from nltk.stem import PorterStemmer
 
stemmer = PorterStemmer()
words = ["playing", "played", "plays", "connected", "connection", "happiness"]
for word in words:
    print(word, "->", stemmer.stem(word))
