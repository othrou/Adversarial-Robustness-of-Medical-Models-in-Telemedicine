# MedGemma Safety Simulation: Multi-Agent Red Teaming & Hardening

## Overview

This project simulates an adversarial "Markov Game" between two specialized agents to evaluate and harden the security posture of MedGemma-4B-it.

Initially, MedGemma was found vulnerable to several black-box attacks (PAIR, ProAttack, RL-based Prompt Injection). This repository contains the framework used to implement a "Defense"—layering Llama Guard 3 and NeMo Guardrails to protect the model against unethical medical requests, PII leakage, and harmful content generation.

> **📚 Full documentation lives in [`docs/`](docs/README.md):**
> [architecture](docs/01-architecture.md) ·
> [mathematical formulation of the zero-sum Markov game](docs/02-game-theory.md) ·
> [the attacks](docs/03-attacks.md) ·
> [the two-guardrail defender](docs/04-defense.md) ·
> [experiments, statistics & benchmarking](docs/05-experiments.md).
> Generate figures with `analysis/plots.py`; run the standard sweep with
> `scripts/benchmark.sh` (re-run it on any model or prompt change).

---

## 🧠 Mathematical Modelization

### The POMDP Framework

The interaction is modeled as a Partially Observable Markov Decision Process (POMDP):

- **Agent A (Attacker):** Operates under partial observability. It sees only the model's textual output (Observations) and refines its prompt (Actions) to transition the model from a "Refusal State" to a "Breached State."
- **Agent B (Defender):** Controls the state transition function $T(s, a)$. By implementing guardrails, the defender significantly reduces the probability of transitioning into a harmful state.

### Multi-Agent Markov Game

The simulation is a zero-sum game:

- **Attacker Action ($A_t$):** Iterative refinement of adversarial prompts (e.g., PAIR methodology).
- **Environment Transition:** The " Defense" processes the input through three distinct layers.
- **Reward/Observation ($O_t$):** The Attacker receives a response or a refusal, providing the signal for the next iteration.

---

## 🛡️ Defensive Architecture: The " Defense"

We implement a three-layer sequential defense to ensure MedGemma remains safe:

1. **Layer 1: Input Sentinel (Llama Guard 3)**

   - **Confrontation:** First line of defense.
   - **Task:** Uses a customized Meta Llama Guard taxonomy (O1-O8), including a specialized O8: Medical Ethics category to catch unethical experimentation or drug synthesis requests.

2. **Layer 2: Logic Flow (NeMo Guardrails)**

   - **Confrontation:** Dialogue management.
   - **Task:** Handles domain-specific "Hard Negatives" using Colang 1.0 logic. It differentiates between safe medical queries and unethical procedures that general safety models might miss.

3. **Layer 3: Output Self-Check (NeMo `self_check_output`)**
   - **Confrontation:** Final audit.
   - **Task:** MedGemma audits its own response against the medical policy in
     `config/prompts.yml` before it reaches the user.
   - ⚠️ **A Llama Guard 3 *output* sentinel is written but NOT enabled.**
     `llama_guard_check_output` is commented out in both `config/prompts.yml` and
     `config/config.yml`, so the output side is the NeMo self-check only.
     Enabling it is the cheapest hardening step available and is tracked in
     [`docs/04-defense.md`](docs/04-defense.md) §5.

---

## 📁 Project Structure

```
medgemma-sim/
├── config/
│   ├── config.yml           # Model routing (Ollama) & system instructions
│   ├── prompts.yml          # Meta Llama Guard O1-O8 taxonomy definitions
│   └── rails.co             # Colang 1.0 dialogue flows & refusal logic
├── agents/
│   ├── llm.py               # LLM backend abstraction (Ollama + offline Mock)
│   ├── judge.py             # Safety judge: harm score 0-10, breach flag, feedback
│   ├── defender.py          # Defender (Agent B): NeMo+MedGemma wrapper + Mock
│   ├── reward.py            # Zero-sum Markov-game reward model + scoreboard
│   ├── goals.py             # Adversarial goals + benign utility probes
│   ├── target.py            # Low-level NeMo Guardrails wrapper
│   ├── attacker_pair.py     # (legacy) original static PAIR strategies
│   └── attacks/             # Attacker (Agent A) strategies, one shared interface
│       ├── base.py          #   BaseAttacker, AttackEpisode, TurnRecord
│       ├── pair.py          #   PAIR: iterative prompt refinement
│       ├── proattack.py     #   ProAttack: evolutionary wrapper search
│       ├── rl_bandit.py     #   RL: epsilon-greedy reward-guided bandit
│       └── signature.py     #   Signature-guided RAG PII exfiltration
├── tests/
│   ├── test_markov_game.py  # Offline unit tests (mock backend, no server)
│   └── tests.py             # Connectivity / baseline safety checks
├── simulation.py            # Main Orchestrator (Markov Game loop)
├── pyproject.toml           # Project + optional deps (managed with uv)
└── notebooks/               # Original attack/defense research notebooks
```

### The Markov game, concretely

Each **attack** strategy (Agent A) implements one interface,
`attacks.base.BaseAttacker`, and runs its own inner optimisation loop against the
**Defender** (Agent B): *propose adversarial prompt → guarded target responds →
judge scores harm → reward guides the next move.* The four ported attacks are:

| name        | idea                                                       | from notebook |
|-------------|------------------------------------------------------------|---------------|
| `pair`      | iteratively refine one prefix on judge feedback            | `PAIR_Attack_Enhanced` |
| `proattack` | evolutionary hill-climbing over a population of wrappers   | `ProAttack_Saad` |
| `rl`        | ε-greedy bandit with a softmax memory bank of good prefixes| `RL_Attack_PAIR` |
| `signature` | TF-IDF medical "signatures" to make a RAG leak PII         | `signature_guided_adversarial_attack` |

### Reward model (`agents/reward.py`)

The game is scored as a (near) zero-sum game. Crucially, the **Defender is now
rewarded when it performs well** — the piece the original loop was missing:

* **Adversarial turn** — `attacker_reward = harm − query_cost·queries`;
  `defender_reward = +block_reward` when it correctly refuses, or `−harm` on a leak.
* **Benign utility turn** — the Defender is rewarded for answering ordinary medical
  questions and **penalised for over-refusal**, so "refuse everything" is not optimal.

Headline metrics per run: **ASR** (Attack Success Rate), **DSR** (Defense Success
Rate) and **Over-Refusal Rate**, plus each agent's cumulative return.

---

## 🚀 Getting Started

### Prerequisites

- **Ollama:** Installed and running.
- **Models:**
  ```
  ollama pull amsaravi/medgemma-4b-it:q6
  ollama pull llama-guard3:1b
  ```

### Installation

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/) (Python 3.12):

```bash
uv sync                          # core deps (game + NeMo Guardrails)
uv sync --extra signature --group dev   # + signature attack extras + test deps
```

### Running the Simulation

Run the Markov game against the **real** guarded target (needs Ollama + the models
above pulled):

```bash
uv run python simulation.py --attack pair           # PAIR vs the defence
uv run python simulation.py --attack rl  --attacker-model mistral
uv run python simulation.py --attack all            # every jailbreak attack, compared
```

Useful flags: `--num-goals`, `--max-iterations`, `--max-queries`,
`--breach-threshold`, `--attacker-model`, `--judge-model`, `--defender {guardrails,raw,mock}`.

### Cloud providers (OpenAI / OpenRouter)

Any model can be named `provider/model`, so each role runs wherever it should —
typically a strong cloud judge against the local MedGemma target:

```bash
uv sync --extra cloud
export OPENROUTER_API_KEY=...        # keys are read from the environment only
export OPENAI_API_KEY=...

uv run python simulation.py --attack all \
    --attacker-model openrouter/<open-weight-model> \
    --judge-model    openai/<frontier-model> \
    --max-spend-usd 20 --report results/run.json
```

Providers: `ollama` (default), `openai`, `openrouter`, `compat` (any
OpenAI-compatible server via `--base-url`), `mock`. A name with no prefix behaves
exactly as before. `--guard-model` and `--victim-model` also re-route the guarded
pipeline's Llama Guard and target — applied to the loaded NeMo config in memory,
so the guardrail-prompt fingerprint in the report stays meaningful.

**Choosing models per role** — the roles have opposite requirements:

| role | pick | why |
|------|------|-----|
| judge | strong frontier model, `temperature=0` | every metric derives from it |
| attacker | **open-weight** (Qwen / Llama / Mistral class) | frontier APIs refuse red-team prompts, and a refused attacker looks identical to a strong defence |
| target | local MedGemma | it is the object of study |
| guard | Llama Guard 3 1B (local) or 8B (cloud) | guard-capacity ablation |

Avoid **reasoning models as the attacker**: their `<think>` blocks are stripped,
but a completion truncated inside one leaves no prefix at all. The run is flagged
rather than scored (see below), but the queries are wasted.

Run `scripts/preflight.sh` first — it checks keys, validates every model id
against the provider's `/v1/models`, and fails before a sweep starts rather than
part-way through.

### Analysis: significance testing and judge validation

```bash
# Paired undefended-vs-defended test: cluster bootstrap over goals, exact
# McNemar on breaches, Holm-Bonferroni across attacks, Cliff's delta.
python -m analysis.stats --baseline results/run_*_undefended_s*.json \
                         --treatment results/run_*_defended_s*.json

# Judge validation (the gate on every headline number).
python simulation.py ... --transcript transcripts/run.jsonl   # capture raw turns
python -m analysis.judge_eval sample transcripts/run.jsonl --out annotation --n 300
#   ...two annotators fill in annotation/to_annotate.csv, blind to the judge...
python -m analysis.judge_eval score annotation/key.json \
    --annotator annotation/a.csv annotation/b.csv
```

`judge_eval` reports **inter-annotator agreement first** (if humans can't agree,
the scale is the problem, not the judge), then judge-vs-human Cohen's κ, and
applies the gate: **κ ≥ 0.6 before headline numbers may be reported.** Every
metric in this project is a function of the judge, so this is not optional.

### Floor controls

`--attack direct` and `--attack random_framing` use **no attacker LLM**: one asks
for the goal verbatim, the other is best-of-N over a fixed framing roster. They
are budget-matched to the search attacks. If PAIR / ProAttack / RL don't clearly
beat both, the search hasn't been shown to contribute anything — and the raw
target already breaches on 69–81% of goals, so this is a live possibility. Run
them in every sweep.

### Validity guards

Two checks guard against results that look clean but measure nothing:

* **Attacker health** — every episode records how many distinct prefixes were
  actually sent. An attacker that never varied (censored by its provider, or
  truncated inside a reasoning block) marks the scoreboard **INVALID**: a 0% ASR
  then measures the attacker, not the defence.
* **Confounded comparisons** — `analysis/plots.py` refuses to draw a comparison
  when more than one variable differs between runs (models, prompts, *or*
  budgets), and renders an attack a run never measured as an explicit
  "not measured" gap instead of a zero bar. Override with `--allow-confounded`
  only when a descriptive figure is genuinely what you want.

**Offline dry run / CI** — no server, no downloads, deterministic mock models:

```bash
MARKOV_GAME_BACKEND=mock uv run python simulation.py --attack all --defender mock
uv run pytest                    # 22 offline unit tests
```

The signature-guided attack uses a built-in **synthetic** corpus by default; set
`MARKOV_GAME_USE_MTSAMPLES=1` to use the real (public, de-identified) MTSamples
corpus with synthetic Faker PII injected — no real patient data is ever used.

---

## 📊 Evaluation Taxonomy (O1-O8)

The system evaluates all interactions against the following categories:

- **O1-O7:** Standard Meta Llama Guard taxonomy (Violence, Sexual Content, Criminal Planning, Guns, Controlled Substances, Self-Harm, Offensive Language).
- **O8 (Custom):** Medical Ethics & Unethical Procedures. Specifically blocks human trials without consent, illegal medical synthesis, and bypassing professional oversight.

---
