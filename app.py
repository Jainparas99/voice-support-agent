"""Gradio UI entry point for the voice support agent."""

from html import escape

from dotenv import load_dotenv
import gradio as gr

from main import process_turn
from pipeline.stt import SUPPORTED_LANGUAGE_CODES
from pipeline.tts import SUPPORTED_BULBUL_SPEAKERS, SUPPORTED_TTS_LANGUAGE_CODES


load_dotenv()


APP_CSS = """
:root {
  --panel: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(245,240,231,0.98));
  --accent: #d95d39;
  --accent-2: #1f6f78;
  --ink: #1e252b;
  --muted: #5e676f;
  --line: rgba(30, 37, 43, 0.12);
  --surface: #f7f1e7;
}

.gradio-container {
  background:
    radial-gradient(circle at top left, rgba(217, 93, 57, 0.16), transparent 28%),
    radial-gradient(circle at top right, rgba(31, 111, 120, 0.18), transparent 22%),
    linear-gradient(180deg, #f7f1e7 0%, #efe4d0 100%);
  color: var(--ink);
}

.hero-shell {
  padding: 28px 30px;
  border: 1px solid var(--line);
  border-radius: 28px;
  background: linear-gradient(135deg, rgba(255,255,255,0.94), rgba(244, 228, 206, 0.9));
  box-shadow: 0 18px 50px rgba(104, 72, 43, 0.12);
}

.eyebrow {
  display: inline-block;
  margin-bottom: 12px;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(31, 111, 120, 0.12);
  color: var(--accent-2);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.hero-title {
  margin: 0;
  font-size: 42px;
  line-height: 1.02;
  letter-spacing: -0.03em;
}

.hero-subtitle {
  max-width: 760px;
  margin: 14px 0 0;
  color: var(--muted);
  font-size: 17px;
  line-height: 1.6;
}

.info-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
  margin-top: 24px;
}

.info-card {
  padding: 14px 16px;
  border-radius: 18px;
  background: rgba(255,255,255,0.78);
  border: 1px solid rgba(30, 37, 43, 0.08);
}

.info-label {
  margin-bottom: 4px;
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.info-value {
  font-size: 15px;
  font-weight: 600;
}

.panel-shell {
  border: 1px solid var(--line);
  border-radius: 26px;
  background: var(--panel);
  box-shadow: 0 14px 36px rgba(64, 46, 24, 0.08);
}

.panel-shell .gr-form,
.panel-shell .gr-box {
  border: none !important;
  background: transparent !important;
}

.status-card {
  padding: 16px 18px;
  border-radius: 20px;
  border: 1px solid rgba(31, 111, 120, 0.16);
  background: linear-gradient(135deg, rgba(31,111,120,0.08), rgba(217,93,57,0.08));
}

.status-title {
  margin: 0 0 8px;
  font-size: 15px;
  font-weight: 700;
}

.status-line {
  margin: 4px 0;
  color: var(--muted);
  font-size: 14px;
}

.kit-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 14px;
}

.kit-table th,
.kit-table td {
  padding: 12px 14px;
  border-bottom: 1px solid rgba(30, 37, 43, 0.08);
  text-align: left;
  vertical-align: top;
}

.kit-table th {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
}

@media (max-width: 900px) {
  .hero-title {
    font-size: 34px;
  }

  .info-grid {
    grid-template-columns: 1fr;
  }
}
"""


BENCHMARK_TABLE_HTML = """
<table class="kit-table">
  <thead>
    <tr>
      <th>File</th>
      <th>Say This</th>
      <th>Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>english_sample.wav</code></td>
      <td>Hello, I need help resetting my account password</td>
      <td>English</td>
    </tr>
    <tr>
      <td><code>hinglish_sample.wav</code></td>
      <td>Bhai mujhe apna subscription cancel karna hai, kaise karu?</td>
      <td>Hinglish</td>
    </tr>
    <tr>
      <td><code>hindi_accented_english.wav</code></td>
      <td>I am not able to login to my account since yesterday</td>
      <td>Indian-accented English</td>
    </tr>
  </tbody>
</table>
"""


HF_SPACES_FRONT_MATTER = """---
title: Multilingual Voice Support Agent
emoji: 🎙️
colorFrom: orange
colorTo: teal
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
---
"""


def process_voice_input(
    audio_file_path: str | None,
    history_state: list[dict[str, str]] | None,
    tts_language_override: str,
    speaker_override: str,
    stt_language_code: str,
) -> tuple[str, str, str | None, list[dict[str, str]], list[tuple[str, str]], str, dict]:
    """Main Gradio handler for one audio turn."""
    history = list(history_state or [])
    if not audio_file_path:
        return (
            "No audio received.",
            "",
            None,
            history,
            _history_to_chatbot(history),
            _render_status(None, "Upload or record audio to start the demo."),
            {},
        )

    try:
        result = process_turn(
            audio_path=audio_file_path,
            conversation_history=history,
            playback=False,
            cleanup_input_audio=False,
            stt_language_code=stt_language_code,
            tts_language=None if tts_language_override == "auto" else tts_language_override,
            speaker=None if speaker_override == "auto" else speaker_override,
        )
        updated_history = list(result["conversation_history"])
        transcript = str(result["transcript"])
        detected_language = str(result["detected_language"])
        response_text = str(result["response_text"])
        output_path = str(result["output_path"])
        timings = dict(result["timings"])
        status = _render_status(result, "Turn completed successfully.")
        return (
            f"[{detected_language}] {transcript}",
            response_text,
            output_path,
            updated_history,
            _history_to_chatbot(updated_history),
            status,
            timings,
        )
    except Exception as exc:
        return (
            "Transcription failed.",
            "",
            None,
            history,
            _history_to_chatbot(history),
            _render_status(None, f"Error: {exc}"),
            {},
        )


def clear_conversation() -> tuple[list[dict[str, str]], list[tuple[str, str]], str, str, None, str, dict]:
    """Reset Gradio state and visible outputs."""
    return [], [], "", "", None, _render_status(None, "Conversation cleared."), {}


def create_ui() -> gr.Blocks:
    """Create the Gradio UI for the multilingual voice support demo."""
    with gr.Blocks(title="Multilingual Voice Support Agent", css=APP_CSS) as demo:
        gr.HTML(
            """
            <section class="hero-shell">
              <div class="eyebrow">Indian Voice Support Demo</div>
              <h1 class="hero-title">Multilingual Voice Support Agent</h1>
              <p class="hero-subtitle">
                Speak in English, Hindi, or Hinglish. The app transcribes with Sarvam Saaras,
                responds like a support agent with Claude, and talks back with Bulbul voice output.
              </p>
              <div class="info-grid">
                <div class="info-card">
                  <div class="info-label">Speech To Text</div>
                  <div class="info-value">Sarvam Saaras v3 for Indian English and code-mixed speech</div>
                </div>
                <div class="info-card">
                  <div class="info-label">Support Brain</div>
                  <div class="info-value">Claude with short spoken responses and escalation fallback</div>
                </div>
                <div class="info-card">
                  <div class="info-label">Text To Speech</div>
                  <div class="info-value">Bulbul v3 voices with language-aware playback</div>
                </div>
              </div>
            </section>
            """
        )

        history_state = gr.State([])

        with gr.Tabs():
            with gr.Tab("Live Demo"):
                with gr.Row():
                    with gr.Column(scale=4, elem_classes=["panel-shell"]):
                        gr.Markdown(
                            "### Input\nRecord directly from your mic or upload a WAV file. "
                            "Use the settings below only when you want to override the auto flow."
                        )
                        audio_input = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            label="Customer voice input",
                        )
                        with gr.Accordion("Voice Settings", open=False):
                            tts_language = gr.Dropdown(
                                choices=["auto", *sorted(SUPPORTED_TTS_LANGUAGE_CODES)],
                                value="auto",
                                label="TTS language override",
                            )
                            speaker = gr.Dropdown(
                                choices=["auto", *sorted(SUPPORTED_BULBUL_SPEAKERS)],
                                value="auto",
                                label="Speaker override",
                            )
                            stt_language = gr.Dropdown(
                                choices=sorted(SUPPORTED_LANGUAGE_CODES),
                                value="unknown",
                                label="STT language hint",
                            )
                        with gr.Row():
                            submit_btn = gr.Button("Run Voice Agent", variant="primary")
                            clear_btn = gr.Button("Clear Conversation")

                        status_box = gr.HTML(_render_status(None, "Ready for your first test."))

                    with gr.Column(scale=5, elem_classes=["panel-shell"]):
                        gr.Markdown("### Conversation")
                        chat_box = gr.Chatbot(
                            label="Conversation history",
                            height=360,
                            show_copy_button=True,
                        )
                        with gr.Row():
                            transcript_box = gr.Textbox(
                                label="Detected transcript",
                                interactive=False,
                                lines=3,
                            )
                            response_box = gr.Textbox(
                                label="Agent response",
                                interactive=False,
                                lines=3,
                            )
                        with gr.Row():
                            audio_output = gr.Audio(
                                label="Voice response",
                                type="filepath",
                                autoplay=True,
                            )
                            metrics_box = gr.JSON(label="Pipeline timings")

                submit_btn.click(
                    fn=process_voice_input,
                    inputs=[audio_input, history_state, tts_language, speaker, stt_language],
                    outputs=[
                        transcript_box,
                        response_box,
                        audio_output,
                        history_state,
                        chat_box,
                        status_box,
                        metrics_box,
                    ],
                )

                clear_btn.click(
                    fn=clear_conversation,
                    outputs=[
                        history_state,
                        chat_box,
                        transcript_box,
                        response_box,
                        audio_output,
                        status_box,
                        metrics_box,
                    ],
                )

            with gr.Tab("Benchmark Kit"):
                with gr.Row():
                    with gr.Column(scale=5, elem_classes=["panel-shell"]):
                        gr.Markdown(
                            "### Record Your Three Test Audio Samples\n"
                            "Use the helper script below or record manually into `evaluation/test_samples/`. "
                            "Keep each recording short and clear, around 5 to 10 seconds."
                        )
                        gr.HTML(BENCHMARK_TABLE_HTML)
                        gr.Code(
                            "python evaluation/record_test_samples.py",
                            language="bash",
                            label="Recording helper",
                        )
                        gr.Markdown(
                            "After recording, verify that `evaluation/ground_truth.json` matches exactly what you said. "
                            "Then run the evaluator to compare Sarvam and Whisper."
                        )
                        gr.Code(
                            "python evaluation/evaluate.py",
                            language="bash",
                            label="Run evaluation",
                        )
                    with gr.Column(scale=4, elem_classes=["panel-shell"]):
                        gr.Markdown(
                            "### What To Expect\n"
                            "- Sarvam should perform better on Hinglish and Indian-accented English.\n"
                            "- `evaluation/results.json` will contain the benchmark output.\n"
                            "- Replace the placeholder WAV files before treating the metrics as real."
                        )

            with gr.Tab("Deploy"):
                with gr.Row():
                    with gr.Column(scale=5, elem_classes=["panel-shell"]):
                        gr.Markdown(
                            "### Deploy To Hugging Face Spaces\n"
                            "Install the Hub CLI, create a Gradio Space, add the Space remote, and push your repo."
                        )
                        gr.Code(
                            "pip install huggingface_hub\n"
                            "huggingface-cli login\n"
                            "git remote add space https://huggingface.co/spaces/YOUR_USERNAME/voice-support-agent\n"
                            "git push space main",
                            language="bash",
                            label="HF Spaces commands",
                        )
                        gr.Markdown("Add this YAML front matter at the top of your Spaces README:")
                        gr.Code(
                            HF_SPACES_FRONT_MATTER,
                            language="yaml",
                            label="README front matter",
                        )
                    with gr.Column(scale=4, elem_classes=["panel-shell"]):
                        gr.Markdown(
                            "### Notes\n"
                            "- Use `sdk: gradio` and `app_file: app.py`.\n"
                            "- Add your API keys as Space secrets.\n"
                            "- If Claude credits are exhausted, the app will fall back to safe spoken responses."
                        )

    return demo


def _history_to_chatbot(history: list[dict[str, str]]) -> list[tuple[str, str]]:
    """Convert conversation history into chatbot message pairs."""
    chat_pairs: list[tuple[str, str]] = []
    pending_user: str | None = None

    for item in history:
        role = item.get("role")
        content = item.get("content", "")
        if role == "user":
            pending_user = content
        elif role == "assistant":
            chat_pairs.append((pending_user or "", content))
            pending_user = None

    return chat_pairs


def _render_status(result: dict[str, object] | None, message: str) -> str:
    """Render a compact HTML status card."""
    if not result:
        return (
            "<div class='status-card'>"
            "<p class='status-title'>Status</p>"
            f"<p class='status-line'>{escape(message)}</p>"
            "</div>"
        )

    timings = dict(result.get("timings", {}))
    return (
        "<div class='status-card'>"
        "<p class='status-title'>Pipeline Status</p>"
        f"<p class='status-line'><strong>Transcript language:</strong> {escape(str(result['detected_language']))}</p>"
        f"<p class='status-line'><strong>TTS language:</strong> {escape(str(result['tts_language']))}</p>"
        f"<p class='status-line'><strong>Output file:</strong> {escape(str(result['output_path']))}</p>"
        f"<p class='status-line'><strong>Timings:</strong> STT {timings.get('stt_seconds', 0)}s, "
        f"LLM {timings.get('llm_seconds', 0)}s, TTS {timings.get('tts_seconds', 0)}s</p>"
        f"<p class='status-line'>{escape(message)}</p>"
        "</div>"
    )


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(share=True)
