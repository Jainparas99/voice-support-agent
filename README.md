# Multilingual Voice Support Agent

End-to-end voice-based customer support bot for Indian enterprise use cases. It accepts spoken English, Hindi, and Hinglish, transcribes the audio with Sarvam, generates a concise support response with Claude, and speaks the answer back with Bulbul TTS.

## Architecture

Speech Input -> Sarvam Saaras STT -> Claude Support Agent -> Sarvam Bulbul TTS -> Voice Output

## Why Sarvam

Standard ASR models often degrade on Indian-accented English and code-switched Hinglish. Sarvam's speech stack is trained natively for Indian languages and accents, which makes it a better fit for multilingual support flows in India.

## Stack

- STT: Sarvam Saaras v3
- LLM: Anthropic Claude Sonnet 4
- TTS: Sarvam Bulbul v3
- UI: Gradio
- Evaluation: jiwer WER and CER comparison against Whisper

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```bash
SARVAM_API_KEY=...
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
```

Optional overrides:

```bash
SARVAM_STT_MODEL=saaras:v3
SARVAM_STT_MODE=transcribe
SARVAM_TTS_MODEL=bulbul:v3
CLAUDE_MODEL=claude-sonnet-4-20250514
```

## Run

CLI microphone mode:

```bash
python main.py
```

CLI file mode:

```bash
python main.py --mode file --audio-path path/to/input.wav --no-playback
```

Gradio UI:

```bash
python app.py
```

## Record Your 3 Test Audio Samples

You need 3 short WAV files inside `evaluation/test_samples/`:

1. `english_sample.wav` -> "Hello, I need help resetting my account password"
2. `hinglish_sample.wav` -> "Bhai mujhe apna subscription cancel karna hai, kaise karu?"
3. `hindi_accented_english.wav` -> "I am not able to login to my account since yesterday"

Fastest way to record them:

```bash
python evaluation/record_test_samples.py
```

The script records each file and refreshes `evaluation/ground_truth.json` with the matching phrases.

## Benchmark Evaluation

Run:

```bash
python evaluation/evaluate.py
```

This compares Sarvam and Whisper on:

- WER
- CER
- transcript output by sample type
- relative improvement of Sarvam vs Whisper

Results are saved to `evaluation/results.json`.

## Benchmark Results

Fill this table after you record real voice samples and run the evaluator:

| Input Type | Sarvam WER | Whisper WER | Improvement |
| --- | --- | --- | --- |
| Indian-accented English | X% | Y% | +Z% |
| Hinglish (code-switched) | X% | Y% | +Z% |
| Standard English | X% | Y% | baseline |

## Demo

Add your Hugging Face Spaces link here after deployment.

## Deploy To Hugging Face Spaces

Install the Hub CLI and log in:

```bash
pip install huggingface_hub
huggingface-cli login
```

Create a new public Gradio Space, then add it as a remote:

```bash
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/voice-support-agent
git push space main
```

At the top of the Spaces README, add this YAML front matter:

```yaml
---
title: Multilingual Voice Support Agent
emoji: 🎙️
colorFrom: orange
colorTo: teal
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
---
```

## Notes

- If Anthropic credits are unavailable, the app falls back to safe spoken support messages.
- Bulbul v3 uses a smaller valid speaker set than some older examples online. The code already uses v3-compatible defaults.
- Replace the placeholder benchmark WAV files before treating the metrics as real.
