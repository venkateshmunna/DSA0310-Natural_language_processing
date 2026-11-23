"""Experiment 16: Named Entity Recognition using spaCy"""

import spacy
 
# Install model once: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")
doc = nlp("Sundar Pichai works at Google in California.")
for entity in doc.ents:
    print(entity.text, "->", entity.label_)
