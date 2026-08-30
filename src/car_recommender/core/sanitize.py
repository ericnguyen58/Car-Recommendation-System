"""Input sanitization for user-supplied text passed to the LLM advisor."""

import re

_TAG_RE = re.compile(r"<[^>]+>")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_text(text: str, max_len: int) -> str:
    text = _TAG_RE.sub("", text)  # strip HTML tags
    text = _CONTROL_RE.sub("", text)  # strip control chars
    return text.strip()[:max_len]
