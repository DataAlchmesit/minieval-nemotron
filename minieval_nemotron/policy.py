"""The cascade policy: every setting that can change a decision, in one place.

The policy is immutable and has a SHA-256 fingerprint. Model IDs, the prompt
version and the temperature are part of it, so a decision made by an LLM is
still reproducible: the audit log records exactly which rules were in force.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from .prompts import PROMPT_VERSION

# Wording that signals a change over time. Matching any of these sends a fact
# to the judge, whatever the local confidence (fixes Delhi -> Bangalore).
TEMPORAL_CUES: tuple[str, ...] = (
    r"\bmoved (from|to|out|back)\b",
    r"\bused to\b",
    r"\bno longer\b",
    r"\banymore\b",
    r"\bpreviously\b",
    r"\bformerly\b",
    r"\bformer\b",
    r"\bswitched (from|to)\b",
    r"\bquit\b",
    r"\bleft (my|the|her|his)\b",
    r"\bnow\b",
    r"\bsince\b",
    r"\blast (week|month|year)\b",
    r"\brecently\b",
    r"\bthese days\b",
)

# Wording that makes a statement uncertain or about the future. Found with the
# real gate: "I think I'll probably switch to Python next year" scored 0.97
# faithful and was stored as "the user uses Python". The NLI model does not
# notice hedges or plans, so these always go to the judge.
HEDGE_CUES: tuple[str, ...] = (
    r"\bi think\b",
    r"\bprobably\b",
    r"\bmaybe\b",
    r"\bmight\b",
    r"\bperhaps\b",
    r"\bnot sure\b",
    r"\b(planning|plan|hoping|hope|want|going|thinking about|considering) (to|of)\b",
    r"\bconsidering\b",
    r"\bwill\b",
    r"\bi'll\b",
    r"\bnext (week|month|year|summer|winter|spring|fall)\b",
    r"\bsomeday\b",
    r"\bif\b",
)

# Content from outside the user (web pages, emails, tool output, pasted text):
# any imperative aimed at an AI or system is treated as an injection attempt.
INJECTION_STRICT: tuple[str, ...] = (
    r"\bignore (all |any |the |your |previous |prior |earlier )*(instructions|rules|prompts?)\b",
    r"\bdisregard (all |any |the |your |previous |prior )*(instructions|rules)\b",
    r"\bsystem prompt\b",
    r"\b(new|updated) instructions\b",
    r"\boverride\b",
    r"\byou (must|should|need to) (store|remember|save|approve|add|record)\b",
    r"\b(approve|store|save|remember|record) (this|it|the following)\b",
    r"\bas an ai\b",
    r"\b(assistant|ai|model|llm)\s*[:,]\s*(please )?(store|remember|save|approve)\b",
)

# The user's own messages: imperatives are normal ("remember that I'm
# vegetarian"), so only phrases aimed at the system itself count.
INJECTION_USER: tuple[str, ...] = (
    r"\bignore (all |any |the |your |previous |prior |earlier )*(instructions|rules|prompts?)\b",
    r"\bdisregard (all |any |the |your |previous |prior )*(instructions|rules)\b",
    r"\bsystem prompt\b",
    r"\b(approve|store|save) (this|it) regardless\b",
)


@dataclass(frozen=True)
class CascadePolicy:
    name: str = "minieval-nemotron-default"
    version: str = "0.1.0"

    # Models (Token Factory IDs). Confirm with `day1_smoke_test.py --list`.
    judge_model: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
    escalation_model: str = "nvidia/Nemotron-3-Ultra-550b-a55b"
    prompt_version: str = PROMPT_VERSION
    temperature: float = 0.0
    judge_reasoning: bool = False  # decided from Day 1 results

    # Routing
    min_relatedness: float = 0.58  # same default as minieval-pro
    temporal_cues: tuple[str, ...] = TEMPORAL_CUES
    hedge_cues: tuple[str, ...] = HEDGE_CUES
    injection_strict: tuple[str, ...] = INJECTION_STRICT
    injection_user: tuple[str, ...] = INJECTION_USER

    # Response check
    retries_on_bad_reply: int = 1
    # If the local attribution guard flagged the sentence, a model may NOT
    # upgrade the fact to STORE. Keeps injection and misattribution out.
    allow_attribution_override: bool = False
    # If R12 (hedged/future wording) fired, a model may NOT upgrade the fact to
    # STORE either: plans and guesses go to a person.
    hedge_never_auto_store: bool = True

    extra: dict = field(default_factory=dict)

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()