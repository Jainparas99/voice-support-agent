"""Claude API module for support-agent responses."""

import re
import time
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
SupportResponse = dict[str, object]
MAX_HISTORY_MESSAGES = 10
MAX_SPOKEN_CHARS = 600
DEFAULT_MAX_RETRIES = 2
ESCALATION_KEYWORDS = {
    "angry",
    "cancel",
    "cancellation",
    "chargeback",
    "complaint",
    "compliance",
    "court",
    "data breach",
    "delete my data",
    "escalate",
    "fraud",
    "gdpr",
    "lawsuit",
    "legal",
    "manager",
    "not working",
    "refund",
    "security incident",
    "sue",
    "unauthorized",
}


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
    result = get_support_response_with_metadata(
        user_message=user_message,
        conversation_history=conversation_history,
        max_tokens=max_tokens,
    )
    return str(result["text"]), result["conversation_history"]  # type: ignore[return-value]


def get_support_response_with_metadata(
    user_message: str,
    conversation_history: Optional[ConversationHistory] = None,
    max_tokens: int = 300,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> SupportResponse:
    """Return a support response plus metadata useful for UI, logs, and TTS."""
    if not user_message.strip():
        raise ValueError("user_message cannot be empty.")

    sanitized_user_message = sanitize_for_speech(user_message)
    needs_escalation = detect_escalation_need(sanitized_user_message)
    history = trim_conversation_history(conversation_history or [])
    history.append({"role": "user", "content": sanitized_user_message})

    try:
        response = _create_message_with_retries(
            messages=history,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
        assistant_message = sanitize_for_speech(_extract_text(response))
        raw_response = response.model_dump() if hasattr(response, "model_dump") else None
        error = None
    except Exception as exc:
        assistant_message = fallback_response(needs_escalation=needs_escalation)
        raw_response = None
        error = str(exc)

    assistant_message = _limit_spoken_length(assistant_message)
    history.append({"role": "assistant", "content": assistant_message})
    history = trim_conversation_history(history)

    return {
        "text": assistant_message,
        "conversation_history": history,
        "needs_escalation": needs_escalation,
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "fallback_used": error is not None,
        "error": error,
        "raw_response": raw_response,
    }


def sanitize_for_speech(text: str) -> str:
    """Remove formatting that sounds bad when spoken by TTS."""
    cleaned = text.strip()
    cleaned = re.sub(r"```.*?```", " ", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"https?://\S+|www\.\S+", "link", cleaned)
    cleaned = re.sub(r"^[\s>*#-]+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"[*_~]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def trim_conversation_history(
    conversation_history: ConversationHistory,
    max_messages: int = MAX_HISTORY_MESSAGES,
) -> ConversationHistory:
    """Keep the most recent conversation messages within Claude context budget."""
    if max_messages <= 0:
        raise ValueError("max_messages must be greater than 0.")
    return list(conversation_history[-max_messages:])


def detect_escalation_need(user_message: str) -> bool:
    """Detect if a message should be escalated to a human support agent."""
    normalized = user_message.lower()
    return any(keyword in normalized for keyword in ESCALATION_KEYWORDS)


def fallback_response(needs_escalation: bool = False) -> str:
    """Return a safe spoken fallback if Claude is unavailable."""
    if needs_escalation:
        return (
            "I am sorry, I could not complete that automatically. "
            "This looks important, so I will escalate it to our support team."
        )
    return (
        "I am sorry, I am having trouble generating a response right now. "
        "Please try again, or I can connect you with support."
    )


def _get_client() -> anthropic.Anthropic:
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is missing. Add it to your .env file.")
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _create_message_with_retries(
    messages: ConversationHistory,
    max_tokens: int,
    max_retries: int,
) -> anthropic.types.Message:
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return _get_client().messages.create(
                model=CLAUDE_MODEL,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
        except (
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.InternalServerError,
            anthropic.RateLimitError,
        ) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(2**attempt)

    raise RuntimeError(f"Claude request failed: {last_error}") from last_error


def _extract_text(response: anthropic.types.Message) -> str:
    text_parts = [
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text" and getattr(block, "text", None)
    ]
    return "\n".join(text_parts).strip()


def _limit_spoken_length(text: str, max_chars: int = MAX_SPOKEN_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    shortened = text[:max_chars].rsplit(" ", 1)[0].strip()
    return f"{shortened}."
