# MedGemma Safety Simulation: Multi-Agent Red Teaming & Hardening

## Overview

This project simulates an adversarial "Markov Game" between two specialized agents to evaluate and harden the security posture of MedGemma-4B-it.

Initially, MedGemma was found vulnerable to several black-box attacks (PAIR, ProAttack, RL-based Prompt Injection). This repository contains the framework used to implement a "Defense"—layering Llama Guard 3 and NeMo Guardrails to protect the model against unethical medical requests, PII leakage, and harmful content generation.

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

3. **Layer 3: Output Sentinel (Llama Guard 3)**
   - **Confrontation:** Final audit.
   - **Task:** Scans MedGemma's generated response for PII leakage or harmful medical advice before it reaches the user.

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
`--breach-threshold`, `--attacker-model`, `--judge-model`, `--defender {guardrails,mock}`.

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
