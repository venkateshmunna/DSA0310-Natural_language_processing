"""Experiment 4: Finite-State Morphological Parser – Plural Nouns"""

def pluralize(noun):
    lower = noun.lower()
    if lower.endswith(("s", "sh", "ch", "x", "z")):
        return noun + "es"
    if lower.endswith("y") and len(noun) > 1 and lower[-2] not in "aeiou":
        return noun[:-1] + "ies"
    return noun + "s"
 
for noun in ["cat", "bus", "box", "baby", "church"]:
    print(noun, "->", pluralize(noun))
