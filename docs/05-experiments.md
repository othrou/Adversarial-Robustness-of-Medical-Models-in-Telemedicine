# 05 — Experiments, Statistics & Benchmarking

How to run the evaluation, produce **statistical figures**, and — the core
discipline — **re-benchmark whenever the model or a prompt changes**.

---

## 1. The golden rule

> **Any change to a model or a prompt invalidates the previous numbers. Re-run the
> benchmark and compare.**

"Model or prompt" means any of:

- the target model, judge model, or attacker model (`--*-model`, `config.yml`);
- the Llama-Guard taxonomy or self-check prompts (`config/prompts.yml`);
- the Colang flows (`config/rails.co`);
- the goals or benign probes (`agents/goals.py`);
- the breach threshold or reward weights (`--breach-threshold`, `RewardConfig`).

The metrics are only comparable *within a fixed configuration*. The report's
`config` block records the configuration so two runs can be compared honestly.

### How this is enforced (not just advised)

Model **and prompt** changes are captured mechanically, so a change can't slip
through as a like-for-like baseline:

- **Model change** — the report `config` stores `attacker_model`, `judge_model`,
  `victim_model`, `defender`, `backend`.
- **Prompt change** — the report `config.guardrail_config` stores a SHA-256 of
  **each** guardrail file (`config/config.yml`, `prompts.yml`, `rails.co`). Edit a
  prompt or a rail and its hash changes (`simulation.py::_config_fingerprint`).
- **Automatic diff** — `analysis/plots.py` compares any two runs and prints /
  annotates exactly what changed, e.g.

  ```
  [plots] config changes vs baseline:
    - run_after.json: model[judge_model]: gemma2:2b -> gemma2:9b; prompt[prompts.yml]: changed
  ```

  and writes a `changes.txt` beside the figures. If two runs are identical it says
  **"NO model/prompt change vs baseline"** — so an accidental no-op comparison is
  obvious. The comparison figure's subtitle carries the same "changed: …" tag.

**Workflow for any model/prompt change:**
```bash
scripts/benchmark.sh                       # baseline -> results/run_A.json
#   ...edit config/prompts.yml (or swap a --*-model)...
scripts/benchmark.sh                       # after    -> results/run_B.json
python -m analysis.plots results/run_A.json results/run_B.json \
    --outdir results/figures/AB --labels before after   # names what changed + the effect
```

---

## 2. Running the benchmark

### One command (recommended)

```bash
# real run: every attack, 5 repeats, report + figures, timestamped
scripts/benchmark.sh

# offline wiring check (deterministic mock, no server)
MARKOV_GAME_BACKEND=mock scripts/benchmark.sh
```

`scripts/benchmark.sh` runs `--attack all --repeats 5`, writes
`results/run_<stamp>.json`, and renders figures into
`results/figures/<stamp>/`. Override via env: `REPEATS`, `NUM_GOALS`,
`MAX_ITERATIONS`, `MAX_QUERIES`, `ATTACKER_MODEL`, `JUDGE_MODEL`, `PY`.

### Manual

```bash
python simulation.py --attack all --repeats 5 --num-goals 6 \
    --attacker-model mistral --judge-model llama-guard3:1b \
    --report results/run_baseline.json --quiet
```

---

## 3. Statistics: why `--repeats`

A single sweep runs each goal once. Against Ollama with `temperature > 0` the
attacker LLM and target are **stochastic**, so one number per attack is a sample,
not an estimate. `--repeats R` re-runs the whole sweep with seeds `seed, seed+1,
…, seed+R-1` and the report stores, per metric, a `stats` block:

```json
"pair": {
  "attack_success_rate": 0.17,
  "stats": {
    "attack_success_rate": {"mean": 0.17, "std": 0.075, "repeats": 5},
    "defense_success_rate": {"mean": 0.83, "std": 0.075, "repeats": 5},
    ...
  }
}
```

`analysis/plots.py` reads `stats.*.std` and draws it as **error bars**. Guidance:

- Use **R ≥ 5** for reported figures; more if std is large relative to the effect
  you're claiming.
- The **mock** backend is deterministic → every repeat is identical → std = 0. Mock
  runs are for wiring/CI, never for statistical claims.
- The signature attack additionally reports its own `pii_metrics` mean/std over its
  internal repeats.

---

### 3.1 Statistical methods (exactly what is computed)

Per attack, for each metric $x$ (ASR, DSR, over-refusal, attacker/defender return,
and the signature `pii_metrics`), the aggregation over $R$ = `--repeats` seeds
computes:

$$
\bar x = \frac{1}{R}\sum_{r=1}^{R} x^{(r)},
\qquad
\sigma(x) = \sqrt{\frac{1}{R}\sum_{r=1}^{R}\bigl(x^{(r)} - \bar x\bigr)^2}.
$$

- **$\sigma$ is the _population_ standard deviation** — the code calls
  `statistics.pstdev` (`simulation.py::aggregate_summaries`,
  `_mean_metrics`), i.e. divisor $R$, **not** the sample std's $R-1$. With small
  $R$ this slightly *under*-estimates the spread; if you prefer the unbiased
  sample std, switch `pstdev` → `stdev` in those two spots (both guard $R>1$).
- **Error bars are $\pm 1\sigma$.** `analysis/plots.py` passes `yerr = std`
  directly to `ax.bar`, so a bar's whisker is one population std, not a confidence
  interval.
- **`repeats` ($R$)** is stored alongside each mean/std so a reader knows the
  sample size behind a bar.
- **Not computed:** confidence intervals, standard error ($\sigma/\sqrt{R}$), or
  significance tests. When comparing two runs, treat non-overlapping $\pm1\sigma$
  bars as *suggestive*, not as a p-value. To add a CI, wrap the stored
  `mean`/`std`/`repeats` (e.g. $\bar x \pm 1.96\,\sigma/\sqrt{R}$) — everything
  needed is already in the report's `stats` block.
- **Determinism caveat:** with the mock backend every seed is identical, so
  $\sigma = 0$ by construction. Only Ollama runs (`temperature > 0`) produce real
  variance.

## 4. Figures (`analysis/plots.py`)

Read-only w.r.t. `results/`; writes PNGs into `--outdir`.

### Single run → per-attack figures

```bash
python -m analysis.plots results/run_baseline.json --outdir results/figures/baseline
```

Produces:

| file | shows |
|------|-------|
| `rates.png` | ASR / DSR / over-refusal per attack, with std error bars |
| `harm_grades.png` | **mean judge harm grade (0–10) per attack**, with std error bars |
| `harm_heatmap.png` | **per-goal harm grade, attack × goal** (0–10, annotated) |
| `returns.png` | attacker vs defender cumulative return per attack (the zero-sum view) |
| `signature_pii.png` | signature PII leak ASR / refusal / target-accuracy (if present) |

### Multiple runs → benchmark comparison

This is the **"did my change help?"** figure — the heart of re-benchmarking:

```bash
python -m analysis.plots \
    results/run_baseline.json results/run_newprompt.json \
    --outdir results/figures/compare --labels baseline new-prompt
```

Produces one grouped bar chart per headline rate (`cmp_attack_success_rate.png`,
`cmp_defense_success_rate.png`, `cmp_over_refusal_rate.png`), each run a series,
error bars included. Read the delta per attack directly off the bars.

---

## 5. Reading the results

The headline numbers map onto the game as formalised in
[`02-game-theory.md`](02-game-theory.md):

- **ASR (Attack Success Rate)** — empirical breach probability under the attacker's
  best menu strategy. **Lower is a safer defender.** A robust defender keeps ASR
  low *without* pushing over-refusal up.
- **DSR = 1 − ASR** — the defender's win rate on adversarial goals.
- **Mean harm grade (0–10)** — the *graded* companion to the binary ASR: the judge's
  best harm score per goal, averaged (`mean_harm_score`; per-goal grades in
  `harm_grades`). ASR says *whether* a breach happened; the grade says *how bad* the
  leak was. **Lower is safer.** A defender can cut ASR yet still leave high-grade
  partial leaks — the grade catches that.
- **Over-Refusal Rate** — fraction of **benign** probes wrongly blocked. **Lower is
  more useful.** Watch this whenever you tighten a prompt: over-tightening trades
  safety for utility.
- **Attacker / Defender return** — cumulative reward; the defender return folds in
  both blocking and utility retention.

**Interpretation checklist for a change:**
1. Did ASR drop on the attack(s) you targeted? By more than the error bars?
2. Did Over-Refusal rise? (The cost of the fix.)
3. Did any *other* attack's ASR move? (Unintended interaction.)
4. Is the effect outside `±std`? If not, increase `--repeats` before concluding.

---

## 6. Headline benchmark — Undefended vs Defended A/B

The rigorous run: **undefended raw MedGemma** (`--defender raw`, ≈ the notebook
target) vs the **two-guardrail defense** (`--defender guardrails`), over the 8
notebook goals × all 4 attacks × 2 seeds, `gemma2:2b` judge, breach threshold 6.
Reproduce with `scripts/ab_benchmark.sh`; reports in
`results/run_20260724_fullab_{undefended,defended}.json`.

| attack | ASR undef → def | harm undef → def (0–10) | Δharm | reading |
|--------|-----------------|--------------------------|-------|---------|
| **rl** | 0.81 → 0.19 | 7.2 → **1.9** | **−5.3** | guardrails highly effective |
| **pair** | 0.69 → 0.38 | 5.4 → 4.2 | −1.2 | modest help |
| **proattack** | 0.62 → 0.62 | 6.1 → **8.2** | **+2.1** | ⚠️ guard does not help; harm rose |
| **signature** | 1.00 → 1.00 | 10.0 → 10.0 | 0.0 | bypasses the guard (attacks its RAG) |
| over-refusal | 0% → 16.7% | | | utility cost of the guard |

**Findings.**
1. The guardrails strongly defeat the **RL bandit** attack (harm 7.2 → 1.9) and
   modestly help against **PAIR**.
2. They do **not** touch the **signature PII/RAG** attack (10/10 either way) — the
   open security gap, since it never reaches the chat guard.
3. **ProAttack anomaly:** evolutionary wrapper search elicits *more* graded harm
   against the guarded pipeline than the raw model (held across both seeds,
   8.2 ± 0.3). See the caveat below.
4. The guard costs utility: 1-in-6 benign probes over-refused.

> **Methodological caveat.** The undefended arm (`RawModelDefender`) and the guarded
> pipeline use slightly different system prompts (the raw defender is safety-primed;
> NeMo uses `config.yml`'s instruction + sample conversation). So this A/B mixes
> "guardrails on/off" with "system-prompt framing." The proattack rise is a real
> measured number, but its mechanism is partly this confound. For a perfectly clean
> ablation, give `RawModelDefender` the identical system prompt.

---

## 7. Reproducibility & isolation

- `results/` is **write-only**; nothing in the codebase reads it back. Attacks keep
  no on-disk caches. One run never biases the next.
- Reports are written **only** with `--report`; ordinary runs leave no artifacts.
- Every report embeds its full `config` (models, budgets, threshold, repeats, seed)
  — enough to reproduce and to know whether two runs are comparable.
- Keep old `results/run_*.json` around: they *are* the benchmark history that the
  comparison figures plot.
