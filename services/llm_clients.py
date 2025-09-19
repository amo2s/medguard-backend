import os
import requests
from openai import OpenAI
from google import genai

# === Load API keys ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# === Endpoints ===
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

# === LLM Clients ===
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# === GROQ ===
def call_groq(prompt: str, system_prompt: str):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-70b-versatile",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 800,
        "temperature": 0.2,
    }

    response = requests.post(GROQ_ENDPOINT, headers=headers, json=payload, timeout=20)

    if response.status_code == 200:
        data = response.json()
        return data["choices"][0]["message"]["content"]

    raise Exception(f"Groq API error: {response.status_code} {response.text}")


# === DEEPSEEK ===
def call_deepseek(prompt: str, system_prompt: str):
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )

    # DeepSeek sometimes returns as "message" or "content"
    return (
        response.choices[0].message.content
        if hasattr(response.choices[0], "message")
        else response.choices[0].text
    )


# === GEMINI ===
def call_gemini(prompt: str, system_prompt: str):
    payload = {
        "contents": [
            {"parts": [{"text": f"{system_prompt}\n\nUser: {prompt}"}]}
        ]
    }

    response = requests.post(
        f"{GEMINI_ENDPOINT}?key={GEMINI_API_KEY}",
        json=payload,
        timeout=20
    )

    if response.status_code == 200:
        data = response.json()

        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except:
            return data.get("output_text", "")  # fallback extraction

    raise Exception(f"Gemini API error: {response.status_code} {response.text}")


# === Smart fallback ===
def ask_any(prompt: str, system_prompt: str):
    """
    Tries Groq → DeepSeek → Gemini in order.
    Returns first successful response.
    """

    for provider in [call_groq, call_deepseek, call_gemini]:
        try:
            return provider(prompt, system_prompt)
        except Exception:
            continue

    raise Exception("All providers failed. Try again later.")
