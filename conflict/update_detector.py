import re


def detect_update_signals(text):
    """
    Detect words and phrases that may indicate
    an explicit update or temporal change.
    """

    text = text.lower().strip()

    explicit_update_phrases = [
        "now",
        "from now on",
        "no longer",
        "anymore",
        "changed",
        "changed my",
        "instead of",
        "moved to",
        "switched to",
        "currently prefer",
        "currently use",
    ]

    temporal_phrases = [
        "currently",
        "now",
        "today",
        "recently",
        "previously",
        "before",
        "last year",
        "last month",
        "in the past",
        "earlier",
    ]

    explicit_updates = [
        phrase
        for phrase in explicit_update_phrases
        if phrase in text
    ]

    temporal_signals = [
        phrase
        for phrase in temporal_phrases
        if phrase in text
    ]

    return {
        "explicit_update": len(explicit_updates) > 0,
        "temporal_change": len(temporal_signals) > 0,
        "explicit_update_phrases": explicit_updates,
        "temporal_phrases": temporal_signals,
    }