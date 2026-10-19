"""The cascade's own secrets guard, tested against minieval-pro 1.2.1 behaviour.

minieval-pro 1.2.1 does not redact anything, so FakeGate here returns the
source untouched. The cascade must still keep secrets away from the model.
"""

import json

import pytest

from minieval_nemotron import Candidate, LocalResult, decide
from minieval_nemotron.secrets_redact import REDACTED, redact, scan


class SpyModel:
    def __init__(self, reply):
        self.reply, self.sent = reply, []

    def __call__(self, model_id, system, user):
        self.sent.append(user)
        return self.reply


@pytest.mark.parametrize("text,kind", [
    ("My OpenAI key is sk-proj-abcdEFGH1234ijklMNOP", "OpenAI-style API key"),
    ("AWS key AKIAIOSFODNN7EXAMPLE in the config", "AWS access key"),
    ("token ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ012345", "GitHub token"),
    ("my password is hunter2!", "password"),
    ("Password: Tr0ub4dor&3", "password"),
    ("my pin is 4821", "PIN"),
    ("the api key = 9f8e7d6c5b4a3210", "secret value"),
    ("card 4111 1111 1111 1111 expires soon", "payment card number"),
    ("Authorization: Bearer abcdefghijklmnop1234", "bearer token"),
])
def test_detects_and_redacts(text, kind):
    assert kind in scan(text).kinds
    assert REDACTED in redact(text)


@pytest.mark.parametrize("text", [
    "I use a password manager called 1Password.",
    "The pin is on the map.",
    "I moved from Delhi to Bangalore last month.",
    "My phone number ends in 1234.",
    "Order 1234 5678 9012 3456 shipped",  # fails the card checksum
])
def test_ordinary_text_is_left_alone(text):
    assert not scan(text).has_secret
    assert redact(text) == text


def test_redaction_keeps_the_label():
    assert redact("my password is hunter2") == f"my password is {REDACTED}"


def test_fact_with_secret_is_rejected_and_never_sent():
    model = SpyModel("{}")
    d = decide(Candidate("The user's password is hunter2.", "My password is hunter2."),
               LocalResult("STORE", entailment=0.99), model)
    assert d.verdict == "REJECT" and d.rules == ["R1"] and model.sent == []


def test_secret_in_source_never_reaches_model_with_unredacting_gate():
    source = "I moved to Pune last month. My API key is sk-live-abcdEFGH12345678."
    reply = json.dumps({"subject": "user", "temporal": "changed", "verdict": "supported",
                        "quote": "I moved to Pune last month", "suggested_fact": ""})
    model = SpyModel(reply)
    # Like minieval-pro 1.2.1: model_source is the raw, unredacted source.
    local = LocalResult("REJECT", narrowed_sentence="I moved to Pune last month.",
                        model_source=source)
    d = decide(Candidate("The user lives in Pune.", source), local, model)
    assert "sk-live" not in model.sent[0]
    assert REDACTED in model.sent[0]
    assert d.verdict == "STORE"
