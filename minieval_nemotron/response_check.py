"""The response check: decides whether a Nemotron reply is allowed to count.

A reply counts only if it is well-formed JSON in the agreed shape and its quote
is copied word-for-word from the source. Moving a fact toward STORE needs more:
the quote must sit inside the sentence the local guards narrowed to, and that
sentence must have passed the local attribution guard.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from .policy import CascadePolicy
from .prompts import ENUMS, VERDICT_SCHEMA
from .types import Candidate, LocalResult, Verdict


@dataclass
class ParsedReply:
    subject: str
    temporal: str
    verdict: str
    quote: str
    suggested_fact: str


@dataclass
class CheckResult:
    usable: bool                  # False -> retry once, then REVIEW
    verdict: Optional[Verdict]    # final verdict if usable and settled
    escalate: bool                # True -> ask the escalation model
    reason: str
    parsed: Optional[ParsedReply] = None


def _norm(text: str) -> str:
    # Whitespace-only normalisation: the quote must still be the same words
    # in the same order, character for character.
    return re.sub(r"\s+", " ", text).strip()


def quote_in(quote: str, text: Optional[str]) -> bool:
    if not quote or not text:
        return False
    return _norm(quote) in _norm(text)


def parse_reply(raw: str) -> Optional[ParsedReply]:
    if not raw:
        return None
    if "</think>" in raw:  # reasoning models may think out loud first
        raw = raw.split("</think>", 1)[1]
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("{"):]
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    for key in VERDICT_SCHEMA["schema"]["required"]:
        if key not in data or not isinstance(data[key], str):
            return None
    for key, allowed in ENUMS.items():
        if data[key] not in allowed:
            return None
    return ParsedReply(**{k: data[k] for k in VERDICT_SCHEMA["schema"]["required"]})


def check_reply(
    raw: str, candidate: Candidate, local: LocalResult, policy: CascadePolicy
) -> CheckResult:
    reply = parse_reply(raw)
    if reply is None:
        return CheckResult(False, None, False, "Reply was not valid JSON in the agreed shape.")

    # Third party: never store as a fact about the user. Keep the suggestion
    # so the review queue can offer a one-click correction.
    if reply.subject == "third_party":
        return CheckResult(True, "REVIEW", False,
                           "Model says the fact is about someone else.", reply)

    if reply.verdict == "unclear" or reply.subject == "unclear":
        return CheckResult(True, None, True, "Model was unsure; escalating.", reply)

    # From here the verdict is supported or contradicted, so a quote is required.
    if not quote_in(reply.quote, candidate.source):
        return CheckResult(False, None, False,
                           "Quote is missing or not found word-for-word in the source.", reply)

    if reply.verdict == "contradicted":
        return CheckResult(True, "REJECT", False,
                           "Model found evidence against the fact, with a verified quote.", reply)

    # verdict == "supported": the strictest path, because it can store a fact.
    if local.narrowed_sentence and not quote_in(reply.quote, local.narrowed_sentence):
        return CheckResult(True, "REVIEW", False,
                           "Quote is in the source but outside the sentence this fact came from.", reply)
    if local.attribution_flag and not policy.allow_attribution_override:
        return CheckResult(True, "REVIEW", False,
                           "Local attribution guard flagged this sentence; a model cannot override it.", reply)
    return CheckResult(True, "STORE", False,
                       "Model found supporting evidence with a verified quote.", reply)
