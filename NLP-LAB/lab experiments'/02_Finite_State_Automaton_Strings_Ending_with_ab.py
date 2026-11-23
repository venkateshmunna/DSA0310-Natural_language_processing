"""Experiment 2: Finite State Automaton – Strings Ending with 'ab'"""

def accepts(text):
    # States: q0=start, q1=last symbol was a, q2=string ends in ab
    state = "q0"
    for ch in text:
        if ch not in "ab":
            return False
        if state == "q0":
            state = "q1" if ch == "a" else "q0"
        elif state == "q1":
            state = "q1" if ch == "a" else "q2"
        else:  # q2
            state = "q1" if ch == "a" else "q0"
    return state == "q2"
 
for s in ["ab", "aab", "babab", "aba", "abb", "ba"]:
    print(s, "Accepted" if accepts(s) else "Rejected")
