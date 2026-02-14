import os
import requests

API_KEY = os.getenv("GEMINI_API_KEY")

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"

payload = {
    "contents": [
        {
            "parts": [{"text": "Explain LSTM in simple terms"}]
        }
    ]
}

response = requests.post(url, json=payload)

print(response.json())