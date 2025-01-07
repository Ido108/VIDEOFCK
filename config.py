# config.py

import os
from dotenv import load_dotenv

load_dotenv()

# Retrieve API keys from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

def verify_api_keys():
    """Verify that all necessary API keys are set."""
    keys_present = True
    if not ELEVENLABS_API_KEY:
        print("Warning: ELEVENLABS_API_KEY not found. ElevenLabs TTS will not be available. TO SET KEYS, OPEN .env FILE")
    return keys_present