"""Experiment 26: English-to-French Machine Translation"""

# Install: pip install transformers torch sentencepiece
from transformers import pipeline
 
translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-en-fr"
)
text = "Natural language processing is an interesting field."
result = translator(text, max_length=100)
print("English:", text)
print("French:", result[0]["translation_text"])
