"""Experiment 22: Reference Resolution in Text"""

import re
 
text = "Ravi bought a laptop. He uses the laptop every day."
sentences = re.split(r'(?<=[.!?])\s+', text)
last_person = None
 
for sentence in sentences:
    nouns = re.findall(r"\b[A-Z][a-z]+\b", sentence)
    if nouns:
        last_person = nouns[-1]
    if re.search(r"\b(He|She|They)\b", sentence) and last_person:
        sentence = re.sub(r"\b(He|She)\b", last_person, sentence, count=1)
    print(sentence)
