"""Experiment 24: Dialog Act Recognition"""

def dialog_act(message):
    text = message.lower().strip()
    if text.endswith("?") or text.startswith(("what", "where", "when", "why", "how")):
        return "QUESTION"
    if any(word in text for word in ["hello", "hi", "good morning"]):
        return "GREETING"
    if any(word in text for word in ["thank", "thanks"]):
        return "THANKING"
    if any(word in text for word in ["please", "could you", "would you"]):
        return "REQUEST"
    return "STATEMENT"
 
messages = ["Hello there!", "Where is the lab?", "Please open the file.", "Thank you."]
for message in messages:
    print(message, "->", dialog_act(message))
