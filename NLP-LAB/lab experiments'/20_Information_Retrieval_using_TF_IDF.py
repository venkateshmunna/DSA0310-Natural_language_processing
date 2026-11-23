"""Experiment 20: Information Retrieval using TF-IDF"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
documents = [
    "Python is used for machine learning",
    "Natural language processing works with text",
    "Python supports data analysis and visualization"
]
query = ["Python programming and machine learning"]
 
vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(documents + query)
scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
 
for index in scores.argsort()[::-1]:
    print(f"Document {index + 1}: score = {scores[index]:.3f}")
    print(documents[index])
