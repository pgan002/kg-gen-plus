"""Generic lexical normalization for labels used throughout KGGen."""

from __future__ import annotations

import re
import unicodedata


def split_camel_case(value: str) -> str:
    """Insert word boundaries into camelCase, PascalCase, and acronym names."""
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    return re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", value)


def normalized_label_key(value: str) -> str:
    """Return a deterministic lexical comparison key for a label."""
    value = unicodedata.normalize("NFKC", value)
    value = split_camel_case(value)
    value = re.sub(r"[_-]+", " ", value)
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip().casefold()
