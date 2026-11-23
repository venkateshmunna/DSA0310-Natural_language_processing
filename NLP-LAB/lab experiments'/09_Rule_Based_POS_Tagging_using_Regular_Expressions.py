"""Experiment 9: Rule-Based POS Tagging using Regular Expressions"""

import re
from nltk.tokenize import wordpunct_tokenize
 
patterns = [
    (r".*ing$", "VBG"),
    (r".*ed$", "VBD"),
    (r".*ly$", "RB"),
    (r".*ous$", "JJ"),
    (r".*(tion|ment)$", "NN"),
    (r"\d+", "CD"),
    (r".*", "NN")
]
tagger = re.compile("|".join(f"(?P<T{i}>{p})" for i, (p, _) in enumerate(patterns)))
 
def tag(word):
    for pattern, label in patterns:
        if re.fullmatch(pattern, word.lower()):
            return label
    return "NN"
 
sentence = "The quickly running dog completed 2 tasks"
print([(w, tag(w)) for w in wordpunct_tokenize(sentence)])
