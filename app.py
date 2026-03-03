"""Gradio UI entry point for the voice support agent."""

import os
from pathlib import Path
import tempfile

from dotenv import load_dotenv
import gradio as gr

from pipeline.llm import get_support_response_with_metadata
from pipeline.stt import transcribe_sarvam
from pipeline.tts import SUPPORTED_TTS_LANGUAGE_CODES, detect_tts_language, synthesize_speech


load_dotenv()


def process_voice_input(
    audio_file_path: str | None,
    history_state: list[dict[str, str]] | None,
) -> tuple[str, str, str | None, list[dict[str, str]]]:
    """
    Main Gradio handler: audio file in, transcript/response/audio path out.
    """
    if not audio_file_path:
        return "No audio received.", "", None, list(history_state or [])

    stt_result = transcribe_sarvam(audio_file_path)
    transcript = str(stt_result.get("transcript", "")).strip()
    detected_lang = str(stt_result.get("language_code") or "unknown")

    if not transcript:
        display_transcript = f"[{detected_lang}] No speech detected."
        return display_transcript, "I could not hear any speech clearly. Please try again.", None, list(
            history_state or []
        )

    llm_result = get_support_response_with_metadata(
        transcript,
        conversation_history=history_state or [],
    )
    response_text = str(llm_result["text"])
    updated_history = list(llm_result["conversation_history"])

    tts_lang = _tts_language_from_stt(detected_lang, response_text)
    output_path = synthesize_speech(
        response_text,
        output_path=_temp_output_path(),
        target_language_code=tts_lang,
    )

    display_transcript = f"[{detected_lang}] {transcript}"
    return display_transcript, response_text, str(output_path), updated_history


def clear_conversation() -> tuple[list[dict[str, str]], str, str, None]:
    """Reset Gradio state and visible outputs."""
    return [], "", "", None


def create_ui() -> gr.Blocks:
    """Create the Gradio UI for the multilingual voice support demo."""
    with gr.Blocks(title="Multilingual Voice Support Agent") as demo:
        gr.Markdown("## Multilingual Voice Support Agent")
        gr.Markdown("Speak in English, Hindi, or Hinglish. The agent will transcribe, answer, and reply with speech.")

        history_state = gr.State([])

        with gr.Row():
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Your voice input",
            )

        with gr.Row():
            submit_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear Conversation")

        with gr.Row():
            transcript_box = gr.Textbox(
                label="What you said (detected language)",
                interactive=False,
            )
            response_box = gr.Textbox(
                label="Agent response",
                interactive=False,
            )

        audio_output = gr.Audio(
            label="Agent voice response",
            type="filepath",
            autoplay=True,
        )

        submit_btn.click(
            fn=process_voice_input,
            inputs=[audio_input, history_state],
            outputs=[transcript_box, response_box, audio_output, history_state],
        )

        clear_btn.click(
            fn=clear_conversation,
            outputs=[history_state, transcript_box, response_box, audio_output],
        )

    return demo


def _tts_language_from_stt(language_code: str, response_text: str) -> str:
    """Map STT language detection to a supported TTS language code."""
    if language_code in SUPPORTED_TTS_LANGUAGE_CODES:
        return language_code
    inferred = detect_tts_language(response_text, fallback="hi-IN")
    if inferred in SUPPORTED_TTS_LANGUAGE_CODES:
        return inferred
    return "hi-IN"


def _temp_output_path() -> str:
    """Create a unique WAV path for Gradio audio output."""
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="voice_agent_")
    os.close(fd)
    Path(path).unlink(missing_ok=True)
    return path


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(share=True)
