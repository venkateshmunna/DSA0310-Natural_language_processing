"""Experiment 15: Probabilistic Context-Free Grammar Parsing"""

import nltk
from nltk import PCFG
 
grammar = PCFG.fromstring("""
S -> NP VP [1.0]
NP -> Det N [0.8] | 'John' [0.2]
VP -> V NP [1.0]
Det -> 'the' [1.0]
N -> 'dog' [0.5] | 'cat' [0.5]
V -> 'sees' [1.0]
""")
parser = nltk.ViterbiParser(grammar)
for tree in parser.parse("John sees the dog".split()):
    print(tree)
    print("Probability:", tree.prob())
