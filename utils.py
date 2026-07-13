"""
utils.py
--------
Utility functions for the PDF Summarizer Skill:
- Text chunking (for large documents)
- Word count
- Timing decorator
- Result formatting
"""

import time
import re
from typing import List


def chunk_text(text: str, max_tokens: int = 3000, overlap: int = 200) -> List[str]:
    """
    Split large text into overlapping chunks suitable for LLM processing.

    Args:
        text: The full document text.
        max_tokens: Approximate max tokens per chunk (1 token ≈ 4 chars).
        overlap: Character overlap between consecutive chunks for context continuity.

    Returns:
        List of text chunks.
    """
    max_chars = max_tokens * 4
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + max_chars, text_length)

        # Try to break at a sentence boundary
        if end < text_length:
            # Look backwards for a period/newline
            boundary = text.rfind('. ', start, end)
            if boundary == -1:
                boundary = text.rfind('\n', start, end)
            if boundary != -1 and boundary > start:
                end = boundary + 1

        chunks.append(text[start:end].strip())
        start = end - overlap if end - overlap > start else end

    return [c for c in chunks if c]


def count_words(text: str) -> int:
    """Return the word count of a string."""
    return len(re.findall(r'\b\w+\b', text))


def format_summary_output(result: dict, mode: str) -> str:
    """
    Format a summary result dict into a human-readable string.

    Args:
        result: The summary result dictionary.
        mode: The summary mode used.

    Returns:
        Formatted string for console output.
    """
    sep = "=" * 60
    lines = [
        sep,
        f"  PDF SUMMARY REPORT",
        sep,
        f"  File        : {result.get('file', 'N/A')}",
        f"  Title       : {result.get('title', 'N/A')}",
        f"  Author      : {result.get('author', 'N/A')}",
        f"  Pages       : {result.get('pages_processed', '?')} / {result.get('pages', '?')}",
        f"  Word Count  : {result.get('word_count', 0):,}",
        f"  Mode        : {mode.upper()}",
        f"  OCR Used    : {'Yes' if result.get('used_ocr') else 'No'}",
        f"  Time Taken  : {result.get('processing_time_seconds', 0):.2f}s",
        sep,
        "",
        "  SUMMARY:",
        "",
        result.get("summary", "No summary available."),
    ]

    if result.get("key_points"):
        lines += ["", "  KEY POINTS:"]
        for i, point in enumerate(result["key_points"], 1):
            lines.append(f"  {i}. {point}")

    if result.get("entities"):
        lines += ["", "  NOTABLE ENTITIES:"]
        for ent_type, values in result["entities"].items():
            lines.append(f"  {ent_type}: {', '.join(values)}")

    if result.get("conclusion"):
        lines += ["", "  CONCLUSION:"]
        lines.append(f"  {result['conclusion']}")

    lines.append(sep)
    return "\n".join(lines)


def timer(func):
    """Decorator that adds processing_time_seconds to the returned dict."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = round(time.time() - start, 3)
        if isinstance(result, dict):
            result["processing_time_seconds"] = elapsed
        return result
    return wrapper
