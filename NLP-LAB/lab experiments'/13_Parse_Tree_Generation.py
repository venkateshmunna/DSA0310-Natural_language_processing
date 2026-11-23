"""Experiment 13: Parse Tree Generation"""

import nltk
 
grammar = nltk.CFG.fromstring("""
S -> NP VP
NP -> Det N
VP -> V
Det -> 'a'
N -> 'child'
V -> 'laughs'
""")
parser = nltk.ChartParser(grammar)
for tree in parser.parse("a child laughs".split()):
    print(tree)
    tree.pretty_print()
