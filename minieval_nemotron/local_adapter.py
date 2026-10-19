"""Connects minieval-pro's MemoryGate to the cascade.

    gate.check(source, fact) -> GateDecision  ->  LocalResult  ->  decide()

Field mapping (from minieval_pro/gate.py):

    GateDecision.verdict          -> LocalResult.verdict
    GateDecision.entailment       -> entailment
    GateDecision.contradiction    -> contradiction
    GateDecision.relatedness      -> relatedness (only set on the contradiction path)
    GateDecision.evidence_span    -> narrowed_sentence (None = single-sentence source,
                                     so the whole source IS the sentence)
    GateDecision.source           -> model_source (redacted by minieval-pro only
                                     in versions that have a secrets guard; the
                                     cascade redacts it again regardless, see
                                     secrets_redact.py)
    reason "Possible misattribution..."  -> attribution_flag
    reason "Not stored: ..." + REJECT    -> safety_flag (secrets guard: final)

GateDecision has no explicit attribution or secrets field, so the two flags are
read from `reason`. Those strings come from gate.py and are covered by tests;
after the hackathon, adding explicit boolean fields to GateDecision would make
this sturdier.
"""

from __future__ import annotations

from typing import Any, Optional

from .cascade import CallModel, decide
from .policy import CascadePolicy
from .types import Candidate, Decision, LocalResult

ATTRIBUTION_PREFIX = "Possible misattribution"
SECRETS_PREFIX = "Not stored:"


def from_gate_decision(d: Any) -> LocalResult:
    """Translate a minieval-pro GateDecision into the cascade's LocalResult."""
    reason = getattr(d, "reason", "") or ""
    verdict = d.verdict
    source = getattr(d, "source", None)
    return LocalResult(
        verdict=verdict,
        entailment=getattr(d, "entailment", None),
        contradiction=getattr(d, "contradiction", None),
        relatedness=getattr(d, "relatedness", None),
        attribution_flag=reason.startswith(ATTRIBUTION_PREFIX),
        safety_flag=(verdict == "REJECT" and reason.startswith(SECRETS_PREFIX)),
        narrowed_sentence=getattr(d, "evidence_span", None) or source,
        faithfulness=getattr(d, "faithfulness", None),
        label=getattr(d, "label", None),
        reason=reason,
        policy_fingerprint=getattr(d, "policy_fingerprint", None),
        model_source=source,
    )


def make_gate(quiet: bool = True):
    """Load minieval-pro's MemoryGate (downloads the local models on first use)."""
    from minieval_pro.gate import MemoryGate
    return MemoryGate(quiet=quiet)


def check_with_cascade(
    candidate: Candidate,
    gate: Any,
    call_model: CallModel,
    policy: Optional[CascadePolicy] = None,
) -> Decision:
    """Full path for one fact: local gate first, then the cascade."""
    local = from_gate_decision(gate.check(source=candidate.source, fact=candidate.fact))
    return decide(candidate, local, call_model, policy)
