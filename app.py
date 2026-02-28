"""Gradio UI entry point for the voice support agent."""

import gradio as gr

from main import run


def process_audio(audio_path: str) -> str:
    """Process an uploaded audio file and return the response audio path."""
    if not audio_path:
        raise gr.Error("Please upload or record an audio file.")
    return str(run(audio_path))


demo = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath", label="Customer audio"),
    outputs=gr.Audio(type="filepath", label="Agent response"),
    title="Voice Support Agent",
    description="Speech-to-speech customer support prototype.",
)


if __name__ == "__main__":
    demo.launch()
