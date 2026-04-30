from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Please set GOOGLE_API_KEY or GEMINI_API_KEY in your .env or environment.")

print("Available Gemini models:")
for model in genai.list_models(api_key=api_key):
    print(f"- {model.name}")
