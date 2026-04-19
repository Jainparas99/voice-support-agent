# Voice Support Agent

Speech-to-speech customer support prototype with:

- STT through Sarvam or Whisper
- LLM responses through Claude
- TTS through Sarvam Bulbul
- Gradio UI
- WER evaluation samples

# Multilingual Voice Support Agent

End-to-end voice-based customer support bot handling English, Hindi, and Hinglish (code-switched speech) — built for Indian enterprise use cases.

## Architecture

Speech Input → Sarvam Saarika STT → Claude Support Agent → Sarvam Bulbul TTS → Voice Output

## Why Sarvam?

Standard ASR (Whisper, Google STT) degrades on Indian-accented English and code-switched Hinglish.
Sarvam Saarika is trained natively on Indian languages — resulting in lower WER on Indian speech.

## Benchmark Results

| Input Type               | Sarvam WER | Whisper WER | Improvement |
| ------------------------ | ---------- | ----------- | ----------- |
| Indian-accented English  | X%         | Y%          | +Z%         |
| Hinglish (code-switched) | X%         | Y%          | +Z%         |
| Standard English         | X%         | Y%          | baseline    |

## Demo

[Hugging Face Spaces link]

## Stack

- STT: Sarvam.ai Saarika v2
- LLM: Anthropic Claude API
- TTS: Sarvam.ai Bulbul v1
- UI: Gradio

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with the API keys you need:

```bash
SARVAM_API_KEY=...
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
```

## Run

CLI:

```bash
python main.py
```

Gradio app:

```bash
python app.py
```

Evaluation:

```bash
python evaluation/evaluate.py
```
