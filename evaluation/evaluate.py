"""Compute WER for benchmark audio samples."""

import json
from pathlib import Path

from jiwer import wer

from pipeline.stt import transcribe_audio


ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT / "test_samples"
GROUND_TRUTH_PATH = ROOT / "ground_truth.json"


def evaluate(provider: str = "sarvam") -> dict[str, float]:
    """Return WER by sample filename."""
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text())
    scores: dict[str, float] = {}

    for filename, expected in ground_truth.items():
        audio_path = SAMPLES_DIR / filename
        predicted = transcribe_audio(str(audio_path), provider=provider)
        scores[filename] = wer(expected, predicted)

    return scores


if __name__ == "__main__":
    for sample, score in evaluate().items():
        print(f"{sample}: {score:.3f}")
