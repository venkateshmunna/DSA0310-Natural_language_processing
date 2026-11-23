"""Experiment 18: FOPC Parser for Basic Logical Expressions"""

import re
 
TOKEN = re.compile(r"\s*(forall|exists|and|or|not|implies|[A-Za-z_]\w*|[(),.])")
 
def parse_tokens(expression):
    tokens, position = [], 0
    while position < len(expression):
        match = TOKEN.match(expression, position)
        if not match:
            raise ValueError(f"Invalid symbol at position {position}")
        tokens.append(match.group(1))
        position = match.end()
    return tokens
 
expression = "forall x (Human(x) implies Mortal(x))"
try:
    print("Tokens:", parse_tokens(expression))
except ValueError as error:
    print("Invalid expression:", error)
