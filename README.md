<div align="center">

# ⚔️ MedGemma Arena

### Multi-Agent Red Teaming & Hardening for Medical LLMs

**A zero-sum Markov game between adversarial attackers and a guarded medical AI, measuring robustness through quantifiable attack success rates, defense rates, and utility trade-offs.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Ollama](https://img.shields.io/badge/Ollama-powered-5B5?logo=ollama&logoColor=white)](https://ollama.ai)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-22_offline-important)](#-testing)
[![Code](https://img.shields.io/badge/docs-docs%2F-8A2BE2)](docs/README.md)

</div>

---

## 🔥 The Big Picture

MedGemma-4B-it was found vulnerable to black-box jailbreaks. This framework **systematically evaluates** how well a defense stack (Llama Guard 3 + NeMo Guardrails) protects it — using **four distinct attack strategies**, a **harm judge**, and a **zero-sum reward model** that also tracks over-refusal on benign queries.

```
                    ┌─────────────────────────────────────────────────────┐
                    │                 simulation.py                       │
                    │                   (orchestrator)                    │
    ┌──────────┐    │  ┌──────────┐  prompt   ┌───────────┐  response    │
    │ GOALS ───┼───►│──┤ Attacker │──────────►│ Defender  │───────────┐  │
    │ 8 harmful│    │  │ (Agent A) │◄──────────│ (Agent B) │           │  │
    │ 6 benign │    │  └──────────┘ feedback  └───────────┘           │  │
    └──────────┘    │       ▲                     ▲                  │  │
                    │       │                     │    ┌──────────┐   │  │
                    │       └─────────────────────┼────│  Judge   │◄──┘  │
                    │                             │    │ harm 0-10│      │
                    │                             │    └────┬─────┘      │
                    │                             │         │            │
                    │                     ┌───────▼─────────▼──────┐     │
                    │                     │     Reward Model       │     │
                    │                     │   ASR / DSR / Returns  │     │
                    │                     └───────────────────────┘     │
                    └────────────────────────────────────────────────────┘
                                            │
                              results/*.json ──► analysis/plots.py ──► figures
```

---

## 🎯 The Four Attack Strategies

Each attack is an **autonomous agent** with its own optimization loop, unified under one interface. See [`docs/03-attacks.md`](docs/03-attacks.md) for full details and per-attack CLI commands.

| Attack | Tactic | Inner Loop | Cost |
|--------|--------|-----------|------|
| **PAIR** 🎭 | Iterative prefix refinement | Refine one prefix on judge feedback | ~queries |
| **ProAttack** 🧬 | Evolutionary wrapper search | Mutate + select elite wrappers | ~n_candidates × generations |
| **RL Bandit** 🎰 | ε-greedy softmax bandit | Explore/exploit over prefix memory bank | ~n_candidates × iterations |
| **Signature** 🕵️ | TF-IDF PII exfiltration | RAG vault with rare-medical-term signatures | ~num_targets × templates |

> **Key insight:** `effective_harm = 0 if blocked else harm` — a blocked refusal never rewards the attacker, even if the judge mis-scores it.

---

## 🛡️ The Defender Stack

Three sequential layers (configured in [`config/`](config/), full breakdown in [`docs/04-defense.md`](docs/04-defense.md)):

```
  user prompt
       │
  ┌────▼──────────────┐   Layer 1: NeMo Self-Check
  │ "Should this be   │   (MedGemma judges its own input)
  │    blocked?"      │──► BLOCK ❌
  └────┬──────────────┘
       │ pass
  ┌────▼──────────────┐   Layer 2: Llama Guard 3
  │ O1-O8 Taxonomy    │   (Custom O8: Medical Ethics)
  │    safe/unsafe    │──► BLOCK ❌
  └────┬──────────────┘
       │ safe
  ┌────▼──────────────┐
  │    MedGemma       │   Generates response
  │  generates answer │
  └────┬──────────────┘
       │
  ┌────▼──────────────┐   Layer 3: NeMo Self-Check Output
  │  Audit bot's own  │   (Catches PII / harmful content)
  │     answer        │──► BLOCK ❌
  └────┬──────────────┘
       │ pass
  response to user ✅
```

**Designed invariants** ([architecture deep-dive](docs/01-architecture.md) §5):
- 🔒 **Fail-closed** — a guardrail exception = block, never breach
- 🚫 **Refusal overrides judge** — textual refusal forces `success=False`
- 📊 **Config fingerprinting** — SHA-256 of every guardrail file in each report

---

## 📦 Project Architecture

```
├── simulation.py              # 🎮 Game orchestrator (the entry point)
├── agents/
│   ├── llm.py                 # 🔌 LLM backend (Ollama / deterministic Mock)
│   ├── judge.py               # ⚖️ Harm scorer 0-10 + feedback
│   ├── defender.py            # 🛡️ GuardrailsDefender / RawModelDefender / MockDefender
│   ├── reward.py              # 📈 Zero-sum reward model + ScoreBoard
│   ├── goals.py               # 📝 8 adversarial goals + 6 benign probes
│   ├── target.py              # 🔧 Low-level NeMo Guardrails wrapper
│   └── attacks/
│       ├── base.py            #   BaseAttacker, AttackEpisode, TurnRecord
│       ├── pair.py            #   🎭 PAIR: iterative prompt refinement
│       ├── proattack.py       #   🧬 ProAttack: evolutionary search
│       ├── rl_bandit.py       #   🎰 RL bandit: softmax explore/exploit
│       └── signature.py       #   🕵️ Signature-guided PII exfiltration
├── config/
│   ├── config.yml             #   Model routing + active rails
│   ├── prompts.yml            #   Llama Guard O1-O8 taxonomy
│   └── rails.co               #   Colang dialogue flows
├── tests/
│   ├── test_markov_game.py    #   22 offline unit tests (mock backend)
│   ├── tests.py               #   Connectivity / pipeline sanity check
│   └── malicious.py           #   Manual adversarial probe
├── analysis/
│   └── plots.py               # 📊 Figures: rates, harm heatmap, A/B comparison
├── scripts/
│   ├── benchmark.sh           #   Standard multi-attack sweep
│   └── ab_benchmark.sh        #   Undefended vs defended A/B comparison
├── notebooks/                 # 📓 Original research (attack + defense prototypes)
├── results/                   #   📁 Run reports + figures (write-only)
└── docs/
    ├── 01-architecture.md     #   Every module, data structure, control flow
    ├── 02-game-theory.md      #   🧮 Formal POMDP + reward mathematics
    ├── 03-attacks.md          #   All 4 attacks in depth
    ├── 04-defense.md          #   Defender config + extension points
    └── 05-experiments.md      #   Benchmarking protocol + statistical methods
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Ollama running locally with:
ollama pull amsaravi/medgemma-4b-it:q6    # Target model
ollama pull llama-guard3:1b               # Safety judge + guardrail
ollama pull llama3.2                      # Attacker (or any instruct model)
```

### Install

```bash
uv sync                            # Core deps
uv sync --extra signature --group dev   # + signature attack + testing
```

### Run

```bash
# 🚤 Quick smoke test (offline, no Ollama needed)
uv run python simulation.py --attack pair --defender mock

# 🎯 Single attack against real models
uv run python simulation.py --attack rl \
    --attacker-model llama3.2 \
    --judge-model llama-guard3:1b \
    --num-goals 3 --max-iterations 5

# ⚔️ Full sweep: all attacks compared
uv run python simulation.py --attack all \
    --attacker-model llama3.2 \
    --judge-model llama-guard3:1b \
    --num-goals 3 --repeats 3 \
    --report results/run.json

# 📊 Generate figures from results
uv run python -m analysis.plots results/run.json --outdir results/figures/run

# 🔬 Ablation: undefended raw model vs guarded
uv run python simulation.py --attack pair --defender raw --report results/raw.json
uv run python simulation.py --attack pair --defender guardrails --report results/guarded.json
```

### Run Tests

```bash
uv run pytest              # 22 offline tests, no server needed
```

---

## 📊 What You Get

Every run reports. The reward model equations and metric derivations are formalized in [`docs/02-game-theory.md`](docs/02-game-theory.md) §6–8.

| Metric | Meaning | Good |
|--------|---------|------|
| **ASR** | Attack Success Rate (breach fraction) | ↓ Low |
| **DSR** | Defense Success Rate (1 − ASR) | ↑ High |
| **Mean Harm** | Average harm grade (0–10) across goals | ↓ Low |
| **Over-Refusal** | Benign queries wrongly blocked | ↓ Low |
| **Attacker Return** | Cumulative attacker reward | ↓ Low |
| **Defender Return** | Cumulative defender reward | ↑ High |

Multiple `--repeats` give you **mean ± std** for error bars. See [`docs/05-experiments.md`](docs/05-experiments.md) for the benchmarking protocol and statistical methods.

---

## 🧪 Headline Results (from the paper)

| Attack | ASR Undefended | ASR Defended | Δ Harm | Verdict |
|--------|:--------------:|:------------:|:------:|---------|
| **RL Bandit** | 81% | **19%** | **−5.3** | 🟢 Guardrails highly effective |
| **PAIR** | 69% | 38% | −1.2 | 🟡 Modest help |
| **ProAttack** | 62% | 62% | **+2.1** | 🔴 Guard did not help; harm rose |
| **Signature** | **100%** | **100%** | 0.0 | 🔴 **Bypasses guard entirely** (RAG attack) |
| Over-refusal | 0% | 16.7% | | Utility cost of defense |

> **Open gap:** The signature-guided PII attack operates on its own RAG vault, never hitting the chat guardrails — 10/10 harm both ways.

---

## 📚 Documentation Map — `docs/`

| # | Document | What it covers | Who should read it |
|---|----------|---------------|-------------------|
| 1 | [`01-architecture.md`](docs/01-architecture.md) | **Every module, data structure, and end-to-end control flow** — the map of the codebase. Includes the big-picture ASCII diagram, module reference (simulation.py, agents/*, config/, tests/), key data structures (TurnRecord, AttackEpisode, JudgeVerdict, etc.), and the 5 design invariants. | Everyone starting out. Read this first. |
| 2 | [`02-game-theory.md`](docs/02-game-theory.md) | **Formal POMDP + zero-sum Markov game** — state space, actions, observations, transition kernel, reward equations (adversarial + benign turns), episodic returns, equilibrium interpretation, and a full symbol table mapping Greek letters to code variables. | If you care about *why* the numbers mean what they mean. |
| 3 | [`03-attacks.md`](docs/03-attacks.md) | **All 4 attack strategies in depth** — PAIR (iterative refinement), ProAttack (evolutionary search), RL bandit (ε-greedy softmax), Signature (TF-IDF PII exfiltration). Each section has: the loop pseudocode, key parameters, exact CLI command to test just that attack, and how to add a 5th one. | Running or extending attacks. |
| 4 | [`04-defense.md`](docs/04-defense.md) | **The defender pipeline** — why only two guardrails (scope rationale), the 3-stage pipeline diagram, code reference for GuardrailsDefender / RawModelDefender / MockDefender, configuration reference (config.yml / prompts.yml / rails.co), the `llama_guard` naming gotcha, and ordered extension points. | Understanding or modifying the defense. |
| 5 | [`05-experiments.md`](docs/05-experiments.md) | **Benchmarking protocol & statistics** — the golden rule ("any prompt change invalidates previous numbers"), how `--repeats` produces mean±std, statistical methods (population std, error bars), how to read the figures (rates.png, harm_grades.png, harm_heatmap.png, returns.png, signature_pii.png), A/B comparison between runs, the headline undefended vs defended results table, and reproducibility/isolation guarantees. | Running experiments, interpreting results, or reporting findings. |

**Recommended reading order:** ① → ② → ③ → ④ → ⑤, or jump to whichever section matches what you're doing.

---

## 🧠 The Math (in brief)

The game is a **zero-sum partially observable Markov game**. Full formalism with state space, action spaces, observation function, transition kernel, reward equations, and equilibrium interpretation in [`docs/02-game-theory.md`](docs/02-game-theory.md).

```
  G = ⟨S, A^A, A^B, O, T, Z, R^A, R^B, γ⟩

  State:       s_t = (context_t, harm_t, blocked_t)
  Attacker:    a_t = prefix_t ⊕ goal     (adversarial prompt)
  Defender:    a_t ∈ {answer, refuse}    (via guardrail pipeline)
  Observation: o_t = (response, harm̂_t, blocked, feedback)
  Reward:
    Adversarial turn:  R^A = h̄_t − κ·q
                        R^B = −λ·h̄_t  (breach) | +β  (blocked)
    Benign turn:       R^A = 0
                        R^B = +u (answered) | −ρ (over-refused)
```

---

## 👥 Contributing

This is a research/evaluation framework. Contributions that add:

- 🔥 New attack strategies (subclass `BaseAttacker`, register in `agents/attacks/__init__.py`)
- 🛡️ New defender layers (subclass `Defender`, add a `--defender` switch)
- 📊 Better analysis plots (`analysis/plots.py`)

are welcome — but **every change requires a re-benchmark** ([`docs/05-experiments.md`](docs/05-experiments.md) §1: "the golden rule").

---

<div align="center">

**Made with 🧠 for AI Safety Research**

*Adversarial Robustness of Medical Models in Telemedicine — Phase 2*

</div>
