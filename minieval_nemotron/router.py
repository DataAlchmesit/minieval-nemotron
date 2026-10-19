"""The router: decides whether a candidate needs Nemotron at all.

It is a pure function (no network, no randomness), so every routing decision
can be tested offline and reproduced later.

Rules, checked in this order:
  R1   safety flag                        -> REJECT, final, no model
  R2   injection phrasing                 -> REVIEW, no model
  R11  web/pasted fact about the user     -> REVIEW, no model
  R3   attribution flag                   -> judge
  R4   any local REJECT                   -> judge
  R5   temporal wording                   -> judge
  R6   uncertain band (local REVIEW)      -> judge
  R7   contradiction with low relatedness -> judge
  R12  hedged or future wording           -> judge
  R9   nothing flagged, local STORE       -> fast STORE, no model
"""

from __future__ import annotations

import re

from .policy import CascadePolicy
from .types import Candidate, LocalResult, Route


def _matches(patterns: tuple[str, ...], text: str) -> list[str]:
    return [p for p in patterns if re.search(p, text, flags=re.IGNORECASE)]


def find_injection(candidate: Candidate, policy: CascadePolicy) -> list[str]:
    patterns = (
        policy.injection_user
        if candidate.origin == "user_message"
        else policy.injection_strict
    )
    return _matches(patterns, candidate.source)


def find_temporal(candidate: Candidate, local: LocalResult, policy: CascadePolicy) -> list[str]:
    text = local.narrowed_sentence or candidate.source
    return _matches(policy.temporal_cues, text)


def find_hedge(candidate: Candidate, local: LocalResult, policy: CascadePolicy) -> list[str]:
    text = local.narrowed_sentence or candidate.source
    return _matches(policy.hedge_cues, text)


def route(candidate: Candidate, local: LocalResult, policy: CascadePolicy) -> Route:
    # R1: safety is final. A model is never asked to overrule it.
    if local.safety_flag:
        return Route("FAST", "REJECT", ["R1"], "Local safety check flagged the content.")

    # R2: injection attempts never reach a model.
    if find_injection(candidate, policy):
        return Route(
            "FAST", "REVIEW", ["R2"],
            f"Source ({candidate.origin}) contains instruction-like text aimed at the system.",
        )

    # R11: your identity comes from you, not from a web page or pasted text.
    if candidate.origin != "user_message" and candidate.about_user:
        return Route(
            "FAST", "REVIEW", ["R11"],
            f"A fact about the user came from {candidate.origin} content; a person must confirm it.",
        )

    rules: list[str] = []
    if local.attribution_flag:
        rules.append("R3")
    if local.verdict == "REJECT":
        rules.append("R4")
    if find_temporal(candidate, local, policy):
        rules.append("R5")
    if local.verdict == "REVIEW":
        rules.append("R6")
    if (
        local.relatedness is not None
        and local.contradiction is not None
        and local.relatedness < policy.min_relatedness
        and local.contradiction >= 0.5
    ):
        rules.append("R7")
    if find_hedge(candidate, local, policy):
        rules.append("R12")

    if rules:
        return Route("JUDGE", None, rules, "Escalated to Nemotron: " + ", ".join(rules))

    # R9: clean, confident local STORE. No model call needed.
    return Route("FAST", "STORE", ["R9"], "Local checks passed with no flags.")
