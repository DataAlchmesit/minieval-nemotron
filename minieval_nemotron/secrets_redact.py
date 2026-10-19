"""Secrets guard for the cascade: nothing credential-like ever reaches a hosted model.

minieval-pro 1.2.1 (the pinned PyPI version) has no secrets guard, so the cascade
carries its own. It runs before routing, on every candidate:

  - a fact that itself contains a secret is REJECTED (rule R1) and never sent;
  - a source that contains a secret is redacted before any model call, and
    quotes are checked against the redacted text.

The patterns are deliberately conservative: each one needs a recognisable key
format or an explicit "password is ..." style statement. Generic long strings
are not flagged, to avoid redacting ordinary text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

REDACTED = "[REDACTED]"


def _luhn_ok(digits: str) -> bool:
    nums = [int(c) for c in digits][::-1]
    total = 0
    for i, n in enumerate(nums):
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


# (name, pattern, group to redact). Group 0 = whole match.
_PATTERNS: list[tuple[str, re.Pattern, int]] = [
    ("private key block",
     re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), 0),
    ("OpenAI-style API key", re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}"), 0),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), 0),
    ("GitHub token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b|\bgithub_pat_[A-Za-z0-9_]{20,}\b"), 0),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}\b"), 0),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), 0),
    ("JSON web token", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b"), 0),
    ("bearer token", re.compile(r"(?i)\bbearer\s+([A-Za-z0-9_\-\.=]{16,})"), 1),
    ("password", re.compile(
        r"(?i)\b(?:password|passwd|pwd|passcode|passphrase)\b\s*(?:is|was|:|=)\s*[\"']?([^\s\"',;]+)"), 1),
    ("PIN", re.compile(r"(?i)\bpin\b\s*(?:is|was|:|=)\s*(\d{4,8})\b"), 1),
    ("secret value", re.compile(
        r"(?i)\b(?:api[ _-]?key|access[ _-]?token|auth[ _-]?token|secret[ _-]?key|client[ _-]?secret)\b"
        r"\s*(?:is|was|:|=)\s*[\"']?([^\s\"',;]{8,})"), 1),
]

_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")


@dataclass(frozen=True)
class SecretScan:
    has_secret: bool
    kinds: tuple[str, ...]


def scan(text: str) -> SecretScan:
    if not text:
        return SecretScan(False, ())
    kinds = [name for name, pat, _ in _PATTERNS if pat.search(text)]
    for m in _CARD.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            kinds.append("payment card number")
            break
    return SecretScan(bool(kinds), tuple(dict.fromkeys(kinds)))


def redact(text: str) -> str:
    if not text:
        return text
    out = text
    for _, pat, group in _PATTERNS:
        if group == 0:
            out = pat.sub(REDACTED, out)
        else:
            out = pat.sub(lambda m, g=group: m.group(0).replace(m.group(g), REDACTED), out)

    def _card(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        return REDACTED if 13 <= len(digits) <= 19 and _luhn_ok(digits) else m.group(0)

    return _CARD.sub(_card, out)
