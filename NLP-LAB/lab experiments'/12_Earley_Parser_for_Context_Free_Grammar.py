"""Experiment 12: Earley Parser for Context-Free Grammar"""

import nltk
 
grammar = nltk.CFG.fromstring("""
S -> NP VP
NP -> Det N
VP -> V NP
Det -> 'the'
N -> 'girl' | 'book'
V -> 'likes'
""")
parser = nltk.ChartParser(grammar)
sentence = "the girl likes the book".split()
 
for tree in parser.parse(sentence):
    print(tree)
    tree.pretty_print()
