"""Configuration for API keys and model names."""

import os

from dotenv import load_dotenv


load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saarika:v2")
SARVAM_TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v2")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")
