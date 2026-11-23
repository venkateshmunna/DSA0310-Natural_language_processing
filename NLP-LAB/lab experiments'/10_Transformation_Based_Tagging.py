"""Experiment 10: Transformation-Based Tagging"""

# Initial tags are deliberately simple; a transformation corrects 'book'
tokens = [("I", "PRP"), ("book", "NN"), ("tickets", "NNS"), ("today", "NN")]
 
# Rule: tag 'book' as a verb when it follows the pronoun 'I'
for i in range(1, len(tokens)):
    word, current_tag = tokens[i]
    previous_word = tokens[i - 1][0]
    if word.lower() == "book" and previous_word.lower() == "i":
        tokens[i] = (word, "VB")
 
print("After transformation:", tokens)
