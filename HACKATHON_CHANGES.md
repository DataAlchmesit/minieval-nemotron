# What changed during the Nebius x NVIDIA hackathon

MiniEval existed before the submission period (Oct 2026). This log records
what was built or significantly changed during it. One entry per working day.

## Baseline before the hackathon

- Local-only gate: DeBERTa NLI faithfulness, MiniLM relatedness guard,
  regex attribution guard, toxic-bert safety check.
- `adjudicate()` for overwrite conflicts (scores each side independently).
- Append-only JSONL audit log with a SHA-256 policy fingerprint.
- 18-case self-authored smoke test; 13-case attribution corpus and a
  temporal-shift corpus contributed to Mem0.
- Published as `minieval-pro` on PyPI. No LLM, no Nebius, no NVIDIA models, no UI.

## Daily log

### Day 1 — Oct 10
- Confirmed Token Factory API: OpenAI-compatible, base URL
  `https://api.tokenfactory.nebius.com/v1/`, structured output via
  `response_format: {"type": "json_schema"}`.
- Wrote `day1_smoke_test.py`: lists the Nemotron models visible to the
  account, runs the combined verification prompt on each tier, scores
  `subject` / `temporal` / `verdict` separately, and checks every quote is
  verbatim in the source.
- Results: _(paste the summary table from `day1_results.json` here)_
- Nemotron model IDs confirmed: _(fill in)_
- Reasoning on/off: _(which mode returned clean JSON, and latency difference)_

### Day 2 — Oct 11

- Built the verification cascade: router (R1–R12), evidence-span response check,
  one retry, fail-closed to REVIEW, Ultra escalation on "unclear". 33 offline tests.
- Connected minieval-pro's real MemoryGate via an adapter. Credential-like text is
  redacted before any model call, so secrets never reach Token Factory.
- Ran the real gate on 8 cases: Delhi→Bangalore confirmed as a local REJECT (goes to
  Nemotron); an injection email was locally STORED at 0.995 faithfulness and stopped
  only by rule R2.
- Found a real gap: "I'll probably switch to Python next year" was stored at 0.97.
  Added R12 (hedged/future wording) to escalate it.
- Docs: architecture diagrams added; clarified that toxic-bert is not used by the gate.