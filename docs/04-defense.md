# 04 — The Defender (Agent B)

The defender is **MedGemma-4B-it wrapped in exactly two guardrails**:

1. **Llama Guard 3 (1B)** — an input safety classifier over a custom O1–O8 taxonomy.
2. **NVIDIA NeMo Guardrails** — Colang dialogue flows + self-check prompts.

That is the whole defender. This document explains the pipeline, the config, the
**deliberate decision to keep only these two strategies**, and where to extend.

---

## 1. Why only two guardrails (scope decision)

The defense team prototyped several additional strategies in `notebooks/Defense/`
(a Knowledge-Graph verifier, RPO robust-prompt-optimization, a TinyLlama blue-team
model, and a PrivacyShield PII defense). **These are intentionally NOT integrated
into the game framework.** The evaluated defender is limited to Llama Guard + NeMo
Guardrails, for three reasons:

- **A clean, reproducible baseline.** Two well-understood, off-the-shelf guardrails
  give a defender whose behaviour is fully determined by config files anyone can
  read and diff — the right control point for a robustness study.
- **Attribution.** With a minimal stack, a change in ASR/DSR is attributable to a
  specific prompt or model, not to interactions among many bespoke components.
- **Low surface area.** Fewer moving parts means the benchmark is cheap to re-run
  on every model/prompt change (the core experimental discipline — see
  [`05-experiments.md`](05-experiments.md)).

The defender therefore has **limited flexibility by design**: its policy space is
essentially {taxonomy prompt, self-check prompts, rail flows, model choice}. The
notebook defenses remain available as research, but are out of scope for the game;
if one is ever promoted in, it must come in as a documented, benchmarked layer
(§5), not as a silent addition.

---

## 2. The three-stage pipeline

Although there are two *products*, they compose into three sequential checks
(configured in `config/config.yml` under `rails:`):

```
             user prompt
                  │
   ┌──────────────▼───────────────┐   Stage 1  (NeMo self-check, LLM = MedGemma)
   │  self check input            │   prompts.yml::self_check_input
   │  "should this be blocked?"   │──► block ─┐
   └──────────────┬───────────────┘           │
                  │ pass                       │
   ┌──────────────▼───────────────┐   Stage 2  (Llama Guard 3, O1–O8)
   │  llama guard check input     │   prompts.yml::llama_guard_check_input
   │  safe / unsafe + categories  │──► block ─┤
   └──────────────┬───────────────┘           │
                  │ safe                       ▼
   ┌──────────────▼───────────────┐   MedGemma generates    "I'm sorry, I cannot
   │  MedGemma (main model)       │   the answer             fulfill this request…"
   └──────────────┬───────────────┘                                 (refusal)
                  │
   ┌──────────────▼───────────────┐   Stage 3  (NeMo self-check output)
   │  self check output           │   prompts.yml::self_check_output
   │  audit the bot's own answer  │──► block ─► refusal
   └──────────────┬───────────────┘
                  │ pass
              response to user
```

- **Stage 1 & 3** are NeMo *self-check* rails: MedGemma judges whether the input /
  its own output violates the medical policy (`prompts.yml`).
- **Stage 2** is **Llama Guard 3** over the **O1–O8 taxonomy**. O1–O7 are the
  standard Meta Llama Guard categories; **O8: Medical Ethics & Unethical
  Procedures** is the project's custom category (human experimentation without
  IRB, illicit pharma synthesis, bypassing oversight, home invasive procedures).
- A Llama-Guard **output** check exists in `prompts.yml` but is **commented out**
  (both there and in `config.yml`) — the output side currently relies on the NeMo
  self-check only. Enabling it is the single easiest hardening step (§5).

Colang flows in `config/rails.co` implement the refusal actions and a semantic
"harmful medical procedures" intent that catches paraphrases the classifiers miss.

---

## 3. Code: `agents/defender.py`

```python
class Defender:                     # interface
    async def respond(self, prompt) -> DefenseOutput   # (content, blocked)

class GuardrailsDefender(Defender): # the real defender
    # builds nemoguardrails.LLMRails from ./config
    # FAIL-CLOSED: any rail/config exception -> DefenseOutput(blocked=True)
    # blocked = keyword refusal check on the generated content

class RawModelDefender(Defender):   # UNDEFENDED target (--defender raw)
    # queries MedGemma directly, NO guardrails -- the ablation baseline (~ the
    # notebook target). Used by scripts/ab_benchmark.sh to measure how much harm
    # the two guardrails actually remove. See docs/05-experiments.md s6.

class MockDefender(Defender):       # offline stand-in (no server)
    # refuses on a medical-harm keyword set, but is deliberately blind to
    # "trusted framings" (IRB / ethics board / accredited) so persona/authority
    # attacks can slip through -> non-trivial offline game dynamics
```

Two invariants matter for the game's correctness:

- **Fail-closed** — a guardrail fault is scored as a block, never a breach, so an
  infrastructure error can never inflate ASR.
- **`blocked`** — comes from the shared `judge._looks_like_refusal` keyword set,
  giving a model-agnostic refusal signal independent of the judge's harm score.

The `MockDefender` is *intentionally imperfect*: it lets the offline suite exercise
real attack dynamics (some framings leak) without an LLM. It is **not** a model of
the real guard's accuracy — never read offline ASR as a safety measurement.

---

## 4. Configuration reference (`config/`)

| File | Contents | Change it to… |
|------|----------|---------------|
| `config.yml` | model routing (MedGemma `main`, `llama_guard3:1b`), active input/output rails, general instructions, sample conversation | swap the target model / judge model; enable the output rail; change base_url |
| `prompts.yml` | `self_check_input`, `self_check_output`, `llama_guard_check_input` (O1–O8), commented `llama_guard_check_output` | tune the taxonomy or self-check policy; enable the output guard |
| `rails.co` | Colang flows: self-check flows, Llama-Guard flow, `check medical ethics` intent, refusal messages | add refusal intents / semantic catches |

> **`llama_guard` naming gotcha** (noted in `config.yml`): NeMo injects the guard
> LLM keyed on the model `type`, so it **must** be `type: llama_guard`. Using
> `guardrail` makes the flow resolve the LLM to `None` and raise
> "No LLM provided to llm_call()". Don't rename it.

---

## 5. Extension points (staying within scope)

Ordered by effort. **Every one of these changes the defender's policy, so each
requires a re-benchmark** (see [`05-experiments.md`](05-experiments.md)).

1. **Enable the Llama-Guard output check** — uncomment `llama_guard_check_output`
   in `prompts.yml` and the `llama guard check output` flow in `config.yml`.
   Directly targets the open PII/leak finding.
2. **Harden the O8 taxonomy** — extend the medical-ethics category in `prompts.yml`.
3. **Tune the self-check prompts** — tighten `self_check_input/output`.
4. **Swap models** — a larger Llama Guard, a different target, via `config.yml`.
5. **Promote a notebook defense (only if decided)** — wrap it as a `Defender`
   subclass with a `--defender <name>` switch, keep it composable with the two
   guardrails, and benchmark before/after. Do **not** fold it silently into
   `GuardrailsDefender`.

---

## 6. Requirements to run the real defender

- **Ollama** running locally, with the models pulled:
  ```bash
  ollama pull amsaravi/medgemma-4b-it:q6
  ollama pull llama-guard3:1b
  ```
- `nemoguardrails` installed (core dependency).
- Sanity check the pipeline end-to-end with `tests/tests.py` (sends a benign
  greeting through the guarded target). Offline development uses `--defender mock`
  and needs none of the above.
