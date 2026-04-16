"""Shared content-sanitization helpers for low-trust text injected into prompts.

Both SAGE recalled memories and RAG-retrieved documentation chunks originate
from sources that an adversarial codebase (or repo) could poison. Any text
that ends up inside an LLM prompt without being directly controlled by the
caller is treated as untrusted and run through this sanitizer before use.
"""

import re

# Per-entry cap when sanitizing a single block of untrusted text. Documentation
# chunks from RAG tend to be larger than SAGE memory entries, so callers that
# want a tighter cap should pass max_length explicitly.
DEFAULT_MAX_LENGTH = 500


def sanitize_untrusted_text(text: str, max_length: int = DEFAULT_MAX_LENGTH) -> str:
    """Sanitize low-trust text for safe inclusion in an LLM prompt.

    Applies the following transforms:
    - Strips control characters (except newline and tab)
    - Removes role-switch sequences (e.g. "system:", "assistant:", "user:")
    - Removes XML-style role tags
    - Removes natural-language imperative injection attempts
      (e.g. "ignore previous instructions", "disregard the above")
    - Truncates the result to ``max_length`` characters

    Args:
        text: Untrusted text to sanitize.
        max_length: Maximum length of the returned string. Defaults to 500.

    Returns:
        Sanitized text, safe to embed in an LLM prompt within a
        delimited low-trust block.
    """
    if not text:
        return ""

    # Strip control characters except newline and tab
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Remove common role-switch / prompt injection sequences at line starts
    text = re.sub(
        r'(?i)^(system|assistant|user|human|ai)\s*:', '', text, flags=re.MULTILINE
    )
    # Remove XML-style role tags
    text = re.sub(
        r'(?i)</?(?:system|assistant|user|human|instruction)[^>]*>', '', text
    )
    # Remove natural-language imperative injection attempts
    text = re.sub(
        r'(?i)^(ignore|disregard|forget|override|instead|do not follow|skip|bypass)\b[^\n]{0,200}',
        '', text, flags=re.MULTILINE,
    )
    # Truncate to prevent token bloat from a single poisoned entry
    if len(text) > max_length:
        text = text[:max_length] + "..."
    return text.strip()
