"""Experiment 6: Bigram N-gram Text Generation"""

import random
from collections import defaultdict
 
text = "we learn natural language processing we learn python"
tokens = text.split()
model = defaultdict(list)
 
for first, second in zip(tokens, tokens[1:]):
    model[first].append(second)
 
word = "we"
output = [word]
for _ in range(10):
    choices = model.get(word)
    if not choices:
        break
    word = random.choice(choices)
    output.append(word)
 
print("Generated text:", " ".join(output))
