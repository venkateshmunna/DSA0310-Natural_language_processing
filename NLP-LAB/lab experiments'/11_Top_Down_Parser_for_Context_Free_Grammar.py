"""Experiment 11: Top-Down Parser for Context-Free Grammar"""

import nltk
 
grammar = nltk.CFG.fromstring("""
S -> NP VP
NP -> Det N
VP -> V NP
Det -> 'the'
N -> 'student' | 'book'
V -> 'reads'
""")
parser = nltk.RecursiveDescentParser(grammar)
sentence = "the student reads the book".split()
 
for tree in parser.parse(sentence):
    print(tree)
    tree.pretty_print()
