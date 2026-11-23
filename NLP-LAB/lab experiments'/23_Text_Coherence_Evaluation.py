"""Experiment 23: Text Coherence Evaluation"""

import re
 
text = "NLP helps computers understand language. Language processing uses text data. Text data can be analyzed by models."
sentences = re.split(r'(?<=[.!?])\s+', text)
stop_words = {"the", "a", "an", "is", "by", "can", "be", "and"}
 
def words(sentence):
    return {w.lower() for w in re.findall(r"\b\w+\b", sentence)
            if w.lower() not in stop_words}
 
overlaps = []
for first, second in zip(sentences, sentences[1:]):
    a, b = words(first), words(second)
    score = len(a & b) / len(a | b) if a | b else 0
    overlaps.append(score)
    print(f"Adjacent sentence overlap: {score:.2f}")
 
print("Average coherence estimate:",
      round(sum(overlaps) / len(overlaps), 2) if overlaps else 0)
