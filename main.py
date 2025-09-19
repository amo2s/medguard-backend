from fastapi import FastAPI
from pydantic import BaseModel
import os

# === Load environment variables ===
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=env_path)

# === Middleware ===
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Import LLM services ===
from services.llm_clients import ask_any

# === Load system prompt from system_prompt.txt ===
PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "system_prompt.txt")

if not os.path.isfile(PROMPT_FILE):
    print("❌ ERROR: system_prompt.txt not found.")
    SYSTEM_PROMPT = "You are MedGuard AI."
else:
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read().strip()

    if not SYSTEM_PROMPT:
        print("❌ ERROR: system_prompt.txt is empty.")
        SYSTEM_PROMPT = "You are MedGuard AI."


# === Request model ===
class PromptRequest(BaseModel):
    prompt: str
    user_id: str | None = None


# === Endpoint ===
@app.post("/ask")
def ask_ai(request: PromptRequest):
    try:
        response_text = ask_any(request.prompt, SYSTEM_PROMPT)
        return {"response": response_text}
    except Exception as e:
        return {"error": str(e)}


print("✅ MedGuard AI server ready. System prompt & environment loaded successfully.")
