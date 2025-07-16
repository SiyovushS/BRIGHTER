import os
import requests

GEMINI_API_KEY = "AIzaSyBci7Tsl2Tpqcyv782vV_oyojgYimi8_ew"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=AIzaSyBci7Tsl2Tpqcyv782vV_oyojgYimi8_ew"

payload = {
    "contents": [{
        "role": "user",
        "parts": [{"text": "Tell me a joke about black holes."}]
    }]
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(GEMINI_ENDPOINT, headers=headers, json=payload)
print(response.json())
