"""Evaluate MiniEval + Nemotron on the regression corpora contributed to Mem0
(mem0ai/mem0#7283): 14 attribution cases + 4 temporal-shift cases, written
before this cascade existed.

Mapping from the corpus's tri-state vocabulary:
    accepted              -> STORE
    uncertain             -> REVIEW
    rejected_with_reason  -> REJECT

Two scores are reported:
    strict  final verdict exactly matches the mapping
    safe    no false fact stored (non-accepted case never STOREd) and no true fact
            lost (accepted case never REJECTed); REVIEW is the safe middle ground

    python scripts/corpus_eval.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from corpora.attribution_corpus import CORPUS  # noqa: E402
from corpora.temporal_shift_corpus import TEMPORAL_CORPUS  # noqa: E402
from minieval_nemotron import Candidate, CascadePolicy  # noqa: E402

MAP = {"accepted": "STORE", "uncertain": "REVIEW", "rejected_with_reason": "REJECT"}


def load_cases():
    cases = []
    for c in list(CORPUS) + list(TEMPORAL_CORPUS):
        group = getattr(c, "category", "temporal_shift")
        cases.append((c.id, group, Candidate(c.candidate, c.source_text), c.expected))
    return cases


def is_safe(expected: str, verdict: str) -> bool:
    if expected == "accepted":
        return verdict != "REJECT"
    return verdict != "STORE"


def evaluate(gate, call_model, policy, cases, usage_log):
    from live_eval import run  # reuse the live evaluation loop
    rows = run(gate, call_model, policy,
               [(cid, cand, {MAP[exp]}) for cid, _, cand, exp in cases], usage_log)
    for r, (_, group, _, exp) in zip(rows, cases):
        r["group"] = group
        r["corpus_expected"] = exp
        r["safe"] = is_safe(exp, r["final_verdict"])
        r["local_safe"] = is_safe(exp, r["local_verdict"])
    return rows


def report(rows, policy):
    n = len(rows)
    print(f"\n{'case':<32}{'expect':<8}{'local':<8}{'final':<8}{'strict':<8}{'safe':<6}route")
    print("-" * 100)
    for r in rows:
        print(f"{r['case']:<32}{MAP[r['corpus_expected']]:<8}{r['local_verdict']:<8}"
              f"{r['final_verdict']:<8}{'yes' if r['final_correct'] else 'NO':<8}"
              f"{'yes' if r['safe'] else 'NO':<6}{r['route']}")
    print("-" * 100)
    print(f"Strict  - local alone: {sum(r['local_correct'] for r in rows)}/{n}   "
          f"MiniEval + Nemotron: {sum(r['final_correct'] for r in rows)}/{n}")
    print(f"Safe    - local alone: {sum(r['local_safe'] for r in rows)}/{n}   "
          f"MiniEval + Nemotron: {sum(r['safe'] for r in rows)}/{n}")
    print(f"Sent to Nemotron: {sum(1 for r in rows if r['model_calls'])}/{n}, "
          f"{sum(r['tokens'] for r in rows)} tokens")
    for r in rows:
        if not r["final_correct"]:
            print(f"\nNOT AS EXPECTED: {r['case']} -> {r['final_verdict']} "
                  f"(corpus says {r['corpus_expected']})")
            print(f"  reason: {r['reason']}")
            if r["raw_model_replies"]:
                print(f"  model said: {r['raw_model_replies'][-1]}")
    print(f"\nJudge: {policy.judge_model} | policy {policy.fingerprint()[:12]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="corpus_eval_results.json")
    args = ap.parse_args()

    from minieval_nemotron.local_adapter import make_gate
    from minieval_nemotron.nemotron_client import make_call_model

    policy = CascadePolicy()
    usage_log: list[dict] = []
    print("Loading local models...")
    gate = make_gate(quiet=True)
    call_model = make_call_model(policy, on_usage=usage_log.append)
    rows = evaluate(gate, call_model, policy, load_cases(), usage_log)
    report(rows, policy)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "policy": asdict(policy),
                   "policy_fingerprint": policy.fingerprint(), "results": rows},
                  f, indent=2, default=str)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
