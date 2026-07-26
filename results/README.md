# Experiment Results

Write-only output directory for Markov-game evaluation runs.

Populate it with the orchestrator's `--report` flag, e.g.:

```bash
uv run python simulation.py --attack all --report results/run_$(date +%Y%m%d).json
```

**Isolation guarantee:** nothing in the codebase ever *reads* from this folder —
it only receives run summaries. So results here never feed back into or bias a
later run. The attack strategies also keep no on-disk caches (unlike the original
notebooks), for the same reason.

Each JSON contains the run `config` (models, budgets, thresholds) and per-attack
`results` (ASR, DSR, over-refusal rate, attacker/defender return). Runs with
`--repeats > 1` also carry a per-metric `stats` block (mean/std over seeds).

## Figures

Render statistical figures from any report(s) with the read-only plotter:

```bash
# per-attack figures for one run -> results/figures/<name>/
python -m analysis.plots results/eval_all_20260722.json --outdir results/figures/eval_all_20260722

# compare two runs (e.g. before/after a prompt change)
python -m analysis.plots results/run_A.json results/run_B.json \
    --outdir results/figures/compare --labels before after
```

Figures live under `results/figures/`. See `docs/05-experiments.md` for the full
benchmarking + statistics protocol (and when to re-benchmark).
