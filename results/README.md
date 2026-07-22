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
`results` (ASR, DSR, over-refusal rate, attacker/defender return).
