import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"),
    http_options={"api_version": "v1"}
)

models = client.models.list()

for m in models:
    print(m.name, "|", m.supported_actions)
