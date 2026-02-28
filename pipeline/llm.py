"""Claude API module for support-agent responses."""


def generate_response(transcript: str) -> str:
    """Generate a support-agent response from a customer transcript."""
    if not transcript.strip():
        raise ValueError("Transcript cannot be empty.")
    raise NotImplementedError("Connect Claude API here.")
