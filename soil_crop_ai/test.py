import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

print("API key found:", bool(api_key))

if not api_key:
    raise ValueError("GOOGLE_API_KEY was not loaded from .env")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.5-flash", contents="Say hello in one short sentence."
)

print("Gemini response:")
print(response.text)
