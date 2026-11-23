"""Experiment 1: Regular Expressions in Python"""

import re
 
text = "Contact us at support@example.com or call 9876543210."
emails = re.findall(r'[\w.-]+@[\w.-]+\.\w+', text)
phone = re.search(r'\b\d{10}\b', text)
 
print("Email addresses:", emails)
print("Phone number:", phone.group() if phone else "Not found")
print("Contains 'Contact':", bool(re.search(r'Contact', text, re.I)))
