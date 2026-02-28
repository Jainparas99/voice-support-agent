# Voice Support Agent

Speech-to-speech customer support prototype with:

- STT through Sarvam or Whisper
- LLM responses through Claude
- TTS through Sarvam Bulbul
- Gradio UI
- WER evaluation samples

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
