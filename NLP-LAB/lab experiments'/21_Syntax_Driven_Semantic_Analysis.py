"""Experiment 21: Syntax-Driven Semantic Analysis"""

import nltk
from nltk import RegexpParser
 
# Run once if required:
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')
 
sentence = "The smart student reads a useful book."
tokens = nltk.word_tokenize(sentence)
tagged = nltk.pos_tag(tokens)
grammar = r"NP: {<DT>?<JJ.*>*<NN.*>+}"
tree = RegexpParser(grammar).parse(tagged)
 
print("Tagged words:", tagged)
print("Noun phrases:")
for subtree in tree.subtrees():
    if subtree.label() == "NP":
        print(" ".join(word for word, tag in subtree.leaves()))
