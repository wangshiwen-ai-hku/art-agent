from google import genai
from google.genai.types import HttpOptions
import os
from dotenv import load_dotenv
load_dotenv()
client = genai.Client(http_options=HttpOptions(api_version="v1"))


# response = client.models.generate_content(
#     model="gemini-2.5-flash",
#     contents="How does AI work?",
# )