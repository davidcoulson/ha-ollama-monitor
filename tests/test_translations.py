"""Check the shipped translations."""

from __future__ import annotations

import json
from pathlib import Path

COMPONENT = Path(__file__).parent.parent / "custom_components" / "ollama_monitor"


def test_english_translation_matches_strings() -> None:
    """translations/en.json is a copy of strings.json without key references."""
    strings = (COMPONENT / "strings.json").read_text()
    assert strings == (COMPONENT / "translations" / "en.json").read_text()
    assert "[%key:" not in strings
    json.loads(strings)
