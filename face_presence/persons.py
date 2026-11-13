"""Validation helpers for person labels."""

from __future__ import annotations

import re

MAX_LABEL_LENGTH = 32
_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._'-]{0,31}$")
_RESERVED_LABEL_PATTERN = re.compile(r"^Person_\d+$", re.IGNORECASE)


def validate_label(value: str) -> str:
    """Normalize and validate a label used by the overlay and CSV exports.

    The overlay uses OpenCV's Hershey font, which does not render accents
    reliably, so labels are restricted to printable ASCII characters.
    """
    label = value.strip()
    if not label:
        raise ValueError("Label cannot be empty.")
    if label.isdigit():
        raise ValueError("Label cannot contain only digits.")
    if _RESERVED_LABEL_PATTERN.fullmatch(label):
        raise ValueError(
            "Labels matching Person_NNN are reserved for automatic enrollment."
        )
    if len(label) > MAX_LABEL_LENGTH:
        raise ValueError(f"Label cannot be longer than {MAX_LABEL_LENGTH} characters.")
    if not _LABEL_PATTERN.fullmatch(label):
        raise ValueError(
            "Use ASCII letters, digits, spaces, dots, underscores, apostrophes "
            "or hyphens; the first character must be a letter or digit."
        )
    return " ".join(label.split())
