"""Plain data types passed between the router, the judge and the response check."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

Origin = Literal["user_message", "external", "pasted"]
Verdict = Literal["STORE", "REVIEW", "REJECT"]


@dataclass(frozen=True)
class Candidate:
    """A fact the extractor wants to store, plus where it came from."""

    fact: str
    source: str
    origin: Origin = "user_message"
    # Set by the extractor: is this fact about the user (identity, preferences,
    # life events) rather than about the world? Web pages may never set facts
    # about the user without a human check.
    about_user: bool = True
    source_url: Optional[str] = None


@dataclass(frozen=True)
class LocalResult:
    """What minieval-pro's local checks said. Built from its GateDecision.

    Only `verdict` is required. Missing scores are treated as "unknown",
    which never makes a decision less cautious.
    """

    verdict: Verdict
    entailment: Optional[float] = None
    contradiction: Optional[float] = None
    relatedness: Optional[float] = None
    attribution_flag: bool = False
    safety_flag: bool = False
    narrowed_sentence: Optional[str] = None
    # Extra context carried into the audit log.
    faithfulness: Optional[float] = None
    label: Optional[str] = None
    reason: Optional[str] = None
    policy_fingerprint: Optional[str] = None
    # The source text that may be sent to a hosted model. minieval-pro redacts
    # credential-like content (passwords, API keys) from its decision; the
    # cascade sends this redacted copy to Nemotron, never the original.
    model_source: Optional[str] = None


@dataclass
class Route:
    """The router's answer: where this candidate goes next, and why."""

    destination: Literal["FAST", "JUDGE"]
    fast_verdict: Optional[Verdict] = None
    rules: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class Decision:
    """The final, auditable outcome for one candidate."""

    verdict: Verdict
    reason: str
    local_verdict: Verdict
    route: list[str]
    rules: list[str]
    policy_fingerprint: str
    origin: Origin
    quote: Optional[str] = None
    suggested_fact: Optional[str] = None
    model_calls: list[dict] = field(default_factory=list)
    local_reason: Optional[str] = None
    local_policy_fingerprint: Optional[str] = None
