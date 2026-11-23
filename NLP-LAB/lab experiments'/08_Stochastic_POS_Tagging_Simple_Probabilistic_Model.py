"""Experiment 8: Stochastic POS Tagging – Simple Probabilistic Model"""

from collections import defaultdict, Counter
 
training = [
    [("time", "N"), ("flies", "V"), ("quickly", "ADV")],
    [("fruit", "N"), ("flies", "N"), ("away", "ADV")],
    [("birds", "N"), ("fly", "V"), ("high", "ADV")]
]
counts = defaultdict(Counter)
for sentence in training:
    for word, tag in sentence:
        counts[word.lower()][tag] += 1
 
def tag_sentence(words):
    result = []
    for word in words:
        options = counts[word.lower()]
        tag = options.most_common(1)[0][0] if options else "UNK"
        result.append((word, tag))
    return result
 
print(tag_sentence(["Time", "flies", "high"]))
