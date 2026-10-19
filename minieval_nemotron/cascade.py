"""decide(): runs one candidate through the whole cascade.

    local checks -> router -> (fast verdict | Nemotron judge -> response check
    -> maybe escalation model) -> Decision

The model is passed in as a function, `call_model(model_id, system, user) -> str`.
Tests pass a fake one; the app passes the real Token Factory client. Any
exception from it fails closed: the result is REVIEW, never STORE.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Callable, Optional

from .policy import CascadePolicy
from .prompts import SYSTEM_PROMPT, user_message
from .response_check import CheckResult, check_reply
from .router import route
from .secrets_redact import redact, scan
from .types import Candidate, Decision, LocalResult

CallModel = Callable[[str, str, str], str]


def _ask(
    call_model: CallModel,
    model_id: str,
    candidate: Candidate,
    local: LocalResult,
    policy: CascadePolicy,
    log: list[dict],
) -> Optional[CheckResult]:
    """Ask one model, retrying a bad reply. Returns None if the call itself failed."""
    attempts = 1 + policy.retries_on_bad_reply
    result: Optional[CheckResult] = None
    for attempt in range(1, attempts + 1):
        t0 = time.perf_counter()
        try:
            raw = call_model(model_id, SYSTEM_PROMPT,
                             user_message(candidate.fact, candidate.source, candidate.origin))
        except Exception as exc:  # timeout, rate limit, network: fail closed
            log.append({"model": model_id, "attempt": attempt,
                        "error": f"{type(exc).__name__}: {exc}",
                        "latency_s": round(time.perf_counter() - t0, 3)})
            return None
        result = check_reply(raw, candidate, local, policy)
        log.append({"model": model_id, "attempt": attempt, "raw": raw,
                    "usable": result.usable, "check": result.reason,
                    "latency_s": round(time.perf_counter() - t0, 3)})
        if result.usable:
            return result
    return result  # still unusable after retries


def decide(
    candidate: Candidate,
    local: LocalResult,
    call_model: CallModel,
    policy: Optional[CascadePolicy] = None,
) -> Decision:
    policy = policy or CascadePolicy()
    fp = policy.fingerprint()

    # Secrets guard (R1). Runs whatever minieval-pro version is installed.
    fact_scan = scan(candidate.fact)
    if fact_scan.has_secret:
        return Decision(
            verdict="REJECT", route=["local"], rules=["R1"], policy_fingerprint=fp,
            origin=candidate.origin, local_verdict=local.verdict,
            reason="Not stored: the fact contains a " + ", ".join(fact_scan.kinds) + ".",
            local_reason=local.reason, local_policy_fingerprint=local.policy_fingerprint,
        )

    # Only ever send a redacted source to a hosted model, and check quotes
    # against that same redacted text.
    base_source = local.model_source if local.model_source is not None else candidate.source
    safe_source = redact(base_source)
    if safe_source != candidate.source:
        candidate = replace(candidate, source=safe_source)
    if local.narrowed_sentence:
        local = replace(local, narrowed_sentence=redact(local.narrowed_sentence))

    r = route(candidate, local, policy)

    base = dict(local_verdict=local.verdict, rules=r.rules,
                policy_fingerprint=fp, origin=candidate.origin,
                local_reason=local.reason,
                local_policy_fingerprint=local.policy_fingerprint)

    if r.destination == "FAST":
        return Decision(verdict=r.fast_verdict, reason=r.reason, route=["local"], **base)

    calls: list[dict] = []
    path = ["local", "judge"]
    result = _ask(call_model, policy.judge_model, candidate, local, policy, calls)

    if result is not None and result.usable and result.escalate:
        path.append("escalation")
        result = _ask(call_model, policy.escalation_model, candidate, local, policy, calls)
        if result is not None and result.usable and result.escalate:
            return Decision(verdict="REVIEW", route=path, model_calls=calls,
                            reason="Both models were unsure; a person should decide.",
                            quote=result.parsed.quote or None, **base)

    if result is None:
        return Decision(verdict="REVIEW", route=path, model_calls=calls,
                        reason="Model call failed; failing closed to REVIEW "
                               f"(local verdict was {local.verdict}).", **base)
    if not result.usable:
        return Decision(verdict="REVIEW", route=path, model_calls=calls,
                        reason=f"No usable model reply after retry: {result.reason}", **base)

    parsed = result.parsed
    return Decision(
        verdict=result.verdict,
        reason=result.reason,
        route=path,
        model_calls=calls,
        quote=(parsed.quote or None) if parsed else None,
        suggested_fact=(parsed.suggested_fact or None) if parsed else None,
        **base,
    )
