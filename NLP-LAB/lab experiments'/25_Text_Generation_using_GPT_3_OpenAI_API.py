"""Experiment 25: Text Generation using GPT-3 / OpenAI API"""

# Install: pip install openai
# Set environment variable OPENAI_API_KEY before running.
from openai import OpenAI
 
client = OpenAI()
prompt = "Explain natural language processing in simple terms."
 
response = client.responses.create(
    model="gpt-3.5-turbo",  # Use a model enabled for your account.
    input=prompt
)
print(response.output_text)
