# Documentation — MedGemma Safety Markov Game

Full documentation for the adversarial-robustness framework in this repository: a
two-player, zero-sum **Markov game** between an *Attacker* agent (a suite of
jailbreak / data-exfiltration strategies) and a *Defender* agent (MedGemma-4B-it
behind exactly two guardrails: **Llama Guard 3** + **NVIDIA NeMo Guardrails**).

> **Scope of this work.** AI-safety and cybersecurity *evaluation*. Every attack
> here exists to measure and then harden the guarded model; all patient data is
> synthetic. See [`04-defense.md`](04-defense.md) for the deliberate decision to
> keep the defender to just the two guardrails above.

## Read in this order

| # | Document | What it covers |
|---|----------|----------------|
| 1 | [`01-architecture.md`](01-architecture.md) | Every module, data structure, and the end-to-end control flow. The map of the codebase. |
| 2 | [`02-game-theory.md`](02-game-theory.md) | The **mathematical formulation** of the zero-sum Markov game: states, actions, observations, transition, reward, value, equilibrium, and how the code's metrics map onto it. |
| 3 | [`03-attacks.md`](03-attacks.md) | All four attacks the attacker agent can run, in depth, with the exact command to test **each** one and expected outputs. |
| 4 | [`04-defense.md`](04-defense.md) | The two-guardrail defender, its config, why it is kept minimal, and its extension points. |
| 5 | [`05-experiments.md`](05-experiments.md) | Benchmarking protocol, **statistical figures**, and when to re-benchmark (any model or prompt change). |

## 30-second orientation

```
simulation.py                 # orchestrator: runs the game, scores it, writes reports
agents/
  attacks/                    # Attacker (Agent A): pair, proattack, rl, signature
  defender.py                 # Defender (Agent B): Llama Guard 3 + NeMo Guardrails
  judge.py                    # harm scorer (0-10) -> the reward signal
  reward.py                   # zero-sum reward model + scoreboard (ASR / DSR / returns)
  goals.py                    # adversarial goals + benign utility probes
  llm.py                      # Ollama backend + deterministic offline mock
config/                       # NeMo rails + Llama-Guard taxonomy (O1-O8)
analysis/plots.py             # statistical figures from run reports
scripts/benchmark.sh          # standard sweep + figures (re-run on any model/prompt change)
results/                      # write-only run reports + figures
tests/                        # 22 offline unit tests (mock backend, no server)
notebooks/                    # original research (attack + defense); source of the ports
```

## Fastest path to a result (no GPU, no server)

```bash
MARKOV_GAME_BACKEND=mock python simulation.py --attack all --defender mock \
    --repeats 3 --report results/_demo.json
python -m analysis.plots results/_demo.json --outdir results/figures/_demo
```

The mock backend is deterministic, so this is a wiring / CI check, not a
measurement. Real numbers need Ollama — see [`05-experiments.md`](05-experiments.md).
