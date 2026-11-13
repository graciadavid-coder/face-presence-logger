"""Small formatting helpers shared by the UI and exports."""

from __future__ import annotations


def format_duration(seconds: float) -> str:
    """Format a duration as MM:SS, or HH:MM:SS for sessions over an hour."""
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
