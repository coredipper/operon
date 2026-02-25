"""
Denaturation Layers: Wire-Level Anti-Prion Defense
===================================================

Paper §5.3: Intermediate paraphrasing/sanitization between agents to
disrupt prompt injection cascading ("prion disease").

In biology, denaturation unfolds a protein's tertiary structure,
destroying its functional conformation.  Applied to data flowing
between agents, denaturation strips the *syntactic* structure that
injection payloads rely on while preserving semantic content.

Each filter implements the DenatureFilter protocol and can be
attached to a Wire to transform data in transit.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable


@runtime_checkable
class DenatureFilter(Protocol):
    """Protocol for wire-level data transformation filters."""

    @property
    def name(self) -> str: ...

    def denature(self, value: str) -> str: ...


@dataclass(frozen=True)
class SummarizeFilter:
    """Truncation + prefix.

    Strips injection by compressing to a bounded summary.
    The prefix makes it clear the content has been processed,
    preventing the downstream agent from treating it as raw input.
    """

    name: str = "summarize"
    max_length: int = 500
    prefix: str = "[Summary] "

    def denature(self, value: str) -> str:
        cleaned = " ".join(value.split())
        if len(cleaned) > self.max_length:
            cleaned = cleaned[: self.max_length] + "..."
        return self.prefix + cleaned


@dataclass(frozen=True)
class StripMarkupFilter:
    """Removes code blocks, ChatML tokens, [INST] tags, XML role tags,
    role delimiters, and <|...|> patterns.

    These syntactic structures are the vectors prompt injections use
    to impersonate system/user roles or inject executable blocks.
    """

    name: str = "strip_markup"

    # Patterns ordered from most specific to most general
    _PATTERNS: tuple[re.Pattern[str], ...] = field(
        default=(
            re.compile(r"```[\s\S]*?```"),           # fenced code blocks
            re.compile(r"`[^`]+`"),                    # inline code
            re.compile(r"<\|[^|]*\|>"),                # <|...|> tokens (ChatML)
            re.compile(r"\[INST\].*?\[/INST\]", re.S), # [INST] blocks
            re.compile(r"\[/?INST\]"),                  # standalone [INST] tags
            re.compile(
                r"<(system|user|assistant|human|ai)[^>]*>.*?</\1>",
                re.S | re.I,
            ),                                          # XML role tags
            re.compile(
                r"<(system|user|assistant|human|ai)[^>]*/?>",
                re.I,
            ),                                          # self-closing role tags
            re.compile(r"^(system|user|assistant|human|ai)\s*:", re.I | re.M),
        ),
        init=False,
        repr=False,
    )

    def denature(self, value: str) -> str:
        result = value
        for pattern in self._PATTERNS:
            result = pattern.sub("", result)
        # Collapse extra whitespace left behind
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()


@dataclass(frozen=True)
class NormalizeFilter:
    """Lowercase, strip control chars, Unicode NFKC normalization.

    NFKC collapses homoglyphs and compatibility characters that
    attackers use to bypass keyword filters.
    """

    name: str = "normalize"
    lowercase: bool = True
    strip_control_chars: bool = True
    unicode_form: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFKC"

    def denature(self, value: str) -> str:
        result = unicodedata.normalize(self.unicode_form, value)
        if self.strip_control_chars:
            # Remove control chars (C0/C1) but preserve newlines and tabs
            result = "".join(
                ch for ch in result
                if ch in ("\n", "\t", "\r") or not unicodedata.category(ch).startswith("C")
            )
        if self.lowercase:
            result = result.lower()
        return result


@dataclass(frozen=True)
class ChainFilter:
    """Compose multiple filters in sequence.

    Filters are applied left-to-right: the output of each filter
    becomes the input of the next.
    """

    name: str = "chain"
    filters: tuple[DenatureFilter, ...] = ()

    def denature(self, value: str) -> str:
        result = value
        for f in self.filters:
            result = f.denature(result)
        return result
