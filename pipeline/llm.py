"""Claude API module for support-agent responses."""

from typing import Optional

import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL


SYSTEM_PROMPT = """
You are a helpful customer support agent for a B2B SaaS company.
You answer user questions clearly and concisely in 2-3 sentences max.
If the user speaks in Hindi or Hinglish, respond in the same language mix they used.
If you don't know the answer, say so honestly and offer to escalate.
Do not use markdown formatting - your response will be spoken aloud.
""".strip()

ConversationHistory = list[dict[str, str]]


def generate_response(transcript: str) -> str:
    """Generate a support-agent response from a customer transcript."""
    response_text, _ = get_support_response(transcript)
    return response_text


def get_support_response(
    user_message: str,
    conversation_history: Optional[ConversationHistory] = None,
    max_tokens: int = 300,
) -> tuple[str, ConversationHistory]:
    """
    Get Claude's support-agent response for a user message.

    Returns the assistant response and updated conversation history.
    """
    if not user_message.strip():
        raise ValueError("user_message cannot be empty.")

    history = list(conversation_history or [])
    history.append({"role": "user", "content": user_message.strip()})

    response = _get_client().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=history,
    )

    assistant_message = _extract_text(response)
    history.append({"role": "assistant", "content": assistant_message})
    return assistant_message, history


def _get_client() -> anthropic.Anthropic:
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is missing. Add it to your .env file.")
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _extract_text(response: anthropic.types.Message) -> str:
    text_parts = [
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text" and getattr(block, "text", None)
    ]
    return "\n".join(text_parts).strip()
