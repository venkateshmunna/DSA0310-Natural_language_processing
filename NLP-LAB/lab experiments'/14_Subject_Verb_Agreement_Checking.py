"""Experiment 14: Subject–Verb Agreement Checking"""

import nltk
 
grammar = nltk.CFG.fromstring("""
S -> NP_S VP_S | NP_P VP_P
NP_S -> DET_S N_S
NP_P -> DET_P N_P
VP_S -> V_S
VP_P -> V_P
DET_S -> 'the'
DET_P -> 'the'
N_S -> 'boy' | 'girl'
N_P -> 'boys' | 'girls'
V_S -> 'runs' | 'plays'
V_P -> 'run' | 'play'
""")
parser = nltk.ChartParser(grammar)
 
for sentence in ["the boy runs", "the boys run", "the boy run"]:
    valid = any(parser.parse(sentence.split()))
    print(sentence, "->", "Correct agreement" if valid else "Incorrect agreement")
