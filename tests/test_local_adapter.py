"""Tests for the minieval-pro -> cascade adapter.

FakeGateDecision mirrors the real GateDecision fields from minieval_pro/gate.py,
including the exact reason strings the gate produces. No models are loaded.
"""

import json
from dataclasses import dataclass
from typing import Optional

from minieval_nemotron import Candidate
from minieval_nemotron.local_adapter import check_with_cascade, from_gate_decision


@dataclass
class FakeGateDecision:
    verdict: str
    fact: str
    source: str
    faithfulness: float
    label: str
    reason: str
    policy_name: str = "default"
    policy_version: str = "1.1"
    policy_fingerprint: str = "abc123def456"
    timestamp: str = "2026-10-19T00:00:00+00:00"
    entailment: Optional[float] = None
    contradiction: Optional[float] = None
    neutral: Optional[float] = None
    relatedness: Optional[float] = None
    evidence_span: Optional[str] = None


class FakeGate:
    def __init__(self, decision):
        self.decision = decision

    def check(self, source, fact):
        return self.decision


class FakeModel:
    def __init__(self, *replies):
        self.replies, self.sent = list(replies), []

    def __call__(self, model_id, system, user):
        self.sent.append(user)
        return self.replies.pop(0)


def reply(verdict="supported", subject="user", quote=""):
    return json.dumps({"subject": subject, "temporal": "changed", "verdict": verdict,
                       "quote": quote, "suggested_fact": ""})


DELHI = "I moved from Delhi to Bangalore last month."


def test_delhi_real_gate_shape_end_to_end():
    gd = FakeGateDecision("REJECT", "The user lives in Bangalore.", DELHI, 0.12,
                          "contradicts", "Fact contradicts its source.",
                          contradiction=0.774, relatedness=0.71)
    model = FakeModel(reply(quote="moved from Delhi to Bangalore"))
    d = check_with_cascade(Candidate(gd.fact, DELHI), FakeGate(gd), model)
    assert d.verdict == "STORE" and d.local_verdict == "REJECT"
    assert d.local_policy_fingerprint == "abc123def456"
    assert d.local_reason == "Fact contradicts its source."


def test_single_sentence_source_uses_whole_source_as_narrowed():
    gd = FakeGateDecision("STORE", "f", "One sentence only.", 0.9, "faithful",
                          "Fact is supported by its source.")
    assert from_gate_decision(gd).narrowed_sentence == "One sentence only."


def test_multi_sentence_uses_evidence_span():
    gd = FakeGateDecision("STORE", "f", "A. B.", 0.9, "faithful",
                          "Fact is supported by its source.", evidence_span="B.")
    assert from_gate_decision(gd).narrowed_sentence == "B."


def test_attribution_reason_sets_flag():
    gd = FakeGateDecision("REVIEW", "The user is a lawyer.", "My brother is a lawyer.",
                          0.99, "faithful",
                          "Possible misattribution. 'My brother' is a third party.")
    local = from_gate_decision(gd)
    assert local.attribution_flag and not local.safety_flag


def test_secret_in_fact_is_final_reject_and_never_sent():
    gd = FakeGateDecision("REJECT", "The user's password is [REDACTED].",
                          "My password is [REDACTED].", 0.99, "faithful",
                          "Not stored: fact contains a password.")
    model = FakeModel()
    d = check_with_cascade(Candidate("The user's password is hunter2.",
                                     "My password is hunter2."), FakeGate(gd), model)
    assert d.verdict == "REJECT" and d.rules == ["R1"] and model.sent == []


def test_secret_in_source_only_is_redacted_before_model_call():
    original = "I moved to Pune last month. My API key is sk-live-123456."
    redacted = "I moved to Pune last month. My API key is [REDACTED]."
    gd = FakeGateDecision("REJECT", "The user lives in Pune.", redacted, 0.2,
                          "contradicts", "Fact contradicts its source.",
                          contradiction=0.7, relatedness=0.7,
                          evidence_span="I moved to Pune last month.")
    model = FakeModel(reply(quote="I moved to Pune last month"))
    d = check_with_cascade(Candidate("The user lives in Pune.", original), FakeGate(gd), model)
    assert "sk-live-123456" not in model.sent[0]
    assert "[REDACTED]" in model.sent[0]
    assert d.verdict == "STORE"


def test_plain_reject_is_not_a_safety_flag():
    gd = FakeGateDecision("REJECT", "f", "s", 0.1, "contradicts",
                          "Fact contradicts its source.")
    assert not from_gate_decision(gd).safety_flag
