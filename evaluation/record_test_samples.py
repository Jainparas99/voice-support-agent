"""Record the three benchmark WAV files used by evaluation/evaluate.py."""

import json
from pathlib import Path

from pipeline.audio import record_audio


ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT / "test_samples"
GROUND_TRUTH_PATH = ROOT / "ground_truth.json"

SAMPLE_PROMPTS = [
    {
        "file": "english_sample.wav",
        "ground_truth": "Hello, I need help resetting my account password",
        "type": "english",
    },
    {
        "file": "hinglish_sample.wav",
        "ground_truth": "Bhai mujhe apna subscription cancel karna hai, kaise karu?",
        "type": "hinglish",
    },
    {
        "file": "hindi_accented_english.wav",
        "ground_truth": "I am not able to login to my account since yesterday",
        "type": "hindi_accented_english",
    },
]


def main() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    print("Recording benchmark samples for WER evaluation.\n")

    for sample in SAMPLE_PROMPTS:
        print(f"Next file: {sample['file']}")
        print(f"Say this clearly: {sample['ground_truth']}")
        input("Press Enter when you are ready to record 6 seconds...")
        output_path = SAMPLES_DIR / sample["file"]
        record_audio(duration_seconds=6, output_path=str(output_path))
        print(f"Saved: {output_path}\n")

    GROUND_TRUTH_PATH.write_text(json.dumps(SAMPLE_PROMPTS, ensure_ascii=False, indent=2))
    print(f"Updated ground truth file: {GROUND_TRUTH_PATH}")


if __name__ == "__main__":
    main()
