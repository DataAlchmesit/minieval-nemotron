# MiniEval × Nemotron — Architecture

**MiniEval is a memory firewall for personal AI.** Every fact an assistant wants
to remember passes through it first. Cheap local checks settle the clear cases;
Nemotron on Nebius Token Factory judges the hard ones; a person sees only what
Nemotron can't settle. A model's answer only counts if the quote it cites is
really in the source.

## 1. The product: where MiniEval sits

```mermaid
flowchart TD
    U["User chat<br/>trusted: the user speaking"] --> X
    W["Web + tool results<br/>Tavily search, untrusted"] -.-> X
    X["Fact extractor<br/>Nemotron Nano, tags origin"] --> G

    subgraph G["MiniEval gate: checks origin, evidence and conflicts"]
        L["Local checks<br/>clear cases, cheap"] --> J["Nemotron judge<br/>uncertain + rejects"]
        J --> H["Human review<br/>unsettled cases only"]
    end

    G --> M["Memory store<br/>fact, source, score, origin"]
    M -. "existing fact: conflict check" .-> G
    M --> A["Assistant reply<br/>answers only from stored facts"]

    classDef nemotron fill:#EEEDFE,stroke:#534AB7,color:#26215C
    classDef minieval fill:#E1F5EE,stroke:#0F6E56,color:#04342C
    classDef plain fill:#F1EFE8,stroke:#888780,color:#2C2C2A
    class X,J,A nemotron
    class L,H minieval
    class U,W,M plain
```

Dashed arrow = untrusted input. Purple = Nemotron on Token Factory. Teal = MiniEval.

## 2. Inside the gate: the verification cascade

```mermaid
flowchart TD
    C["Candidate fact + source<br/>tagged with origin"] --> T0
    E["Existing memory + source<br/>only for overwrites"] --> T0
    T0["Tier 0: local checks<br/>NLI, attribution, relatedness, safety"] --> R{"Router<br/>rules R1 to R11"}

    R -- "R1, R2, R11, R9" --> F["Fast verdict<br/>no model call"]
    R -- "R3 to R7" --> J["Combined call<br/>Nemotron Nano or Super"]
    R -- "R8 overwrite conflict" --> U["Nemotron Ultra"]
    J -- "unclear" --> U

    J --> RC["Response check<br/>valid JSON, quote in sentence, 1 retry"]
    U --> RC
    RC -- "passes" --> D["Decision + audit log<br/>route, model, prompt version, fingerprint"]
    RC -. "no valid quote, timeout, bad JSON" .-> RV["REVIEW, fail-closed<br/>local verdict kept in log"]
    F --> D

    classDef nemotron fill:#EEEDFE,stroke:#534AB7,color:#26215C
    classDef minieval fill:#E1F5EE,stroke:#0F6E56,color:#04342C
    classDef plain fill:#F1EFE8,stroke:#888780,color:#2C2C2A
    class J,U nemotron
    class T0,F,D minieval
    class C,E,R,RC,RV plain
```

## 3. Router rules

Checked in order. A fact skips the models only if a fast rule settles it.

| Rule | Trigger | Goes to |
|---|---|---|
| R1 | Local safety flag | REJECT, final |
| R2 | Injection phrasing (strict for web/pasted text, narrow list for the user's own messages) | REVIEW, no model |
| R11 | A fact **about the user** that came from web or pasted content | REVIEW, no model |
| R3 | Attribution flag ("my brother is a lawyer") | Combined call |
| R4 | Any local REJECT, however confident | Combined call |
| R5 | Temporal wording ("moved from", "used to", "no longer", ...) | Combined call |
| R6 | Uncertain band (local REVIEW) | Combined call |
| R7 | Contradiction on a low-relatedness pair | Combined call |
| R8 | Overwrite conflict in `adjudicate()`, or the combined call says "unclear" | Score comparison, then Ultra |
| R9 | Nothing flagged, local STORE | Fast STORE |
| R10 | Timeout, rate limit, bad JSON, no valid quote after 1 retry | REVIEW |

## 4. Design rules

1. **Evidence or nothing.** A model's verdict counts only if its quote appears word-for-word in the source.
2. **Asymmetric authority.** A model can always make a verdict more cautious. It can move a fact toward STORE only if the quote sits inside the sentence the fact came from **and** that sentence passed the local attribution guard.
3. **Fail closed.** Any failure ends in REVIEW, never STORE.
4. **Origin matters.** Your identity comes from you, not from a web page. Web facts can never overwrite what the user said.
5. **Reproducible.** Temperature 0. Model IDs, prompt version and thresholds are inside the policy's SHA-256 fingerprint, which is logged with every decision.
6. **Source is data, not instructions.** The model has no tools and can return only a label and a quote, so text inside a source can't make it act.

## 5. Where each part lives in the code

| Part | File | Status |
|---|---|---|
| Policy + fingerprint | `minieval_nemotron/policy.py` | Built |
| Prompt + JSON shape | `minieval_nemotron/prompts.py` | Built |
| Router (R1–R7, R9, R11) | `minieval_nemotron/router.py` | Built |
| Response check | `minieval_nemotron/response_check.py` | Built |
| `decide()`, retry, fail-closed, Ultra on "unclear" | `minieval_nemotron/cascade.py` | Built |
| Token Factory connection | `minieval_nemotron/nemotron_client.py` | Built, not yet run live |
| Local checks → `LocalResult` adapter | — | Next |
| `adjudicate()` + R8 | — | Planned |
| Fact extractor (Nano) | — | Planned |
| Tavily web input | — | Planned |
| App: chat, memory panel, review queue | — | Planned |

Update the Status column as each part lands.