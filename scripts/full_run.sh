#!/usr/bin/env bash
# Crash-safe full-scale benchmark: run each seed as its OWN single-repeat report
# (so every seed's numbers hit disk the moment it finishes), then merge the seeds
# into one stats report and render figures.
#
#     STAMP=20260724 scripts/full_run.sh
#
# Env knobs: PY, SEEDS, NUM_GOALS, ATTACKER_MODEL, JUDGE_MODEL, MAX_ITERATIONS,
# MAX_QUERIES, BREACH_THRESHOLD, STAMP.
set -uo pipefail
cd "$(dirname "$0")/.."

export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
PY="${PY:-.venv/bin/python}"
SEEDS="${SEEDS:-0 1 2}"
NUM_GOALS="${NUM_GOALS:-6}"
ATTACKER_MODEL="${ATTACKER_MODEL:-mistral}"
JUDGE_MODEL="${JUDGE_MODEL:-gemma2:2b}"
MAX_ITERATIONS="${MAX_ITERATIONS:-6}"
MAX_QUERIES="${MAX_QUERIES:-24}"
BREACH_THRESHOLD="${BREACH_THRESHOLD:-7}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p results/figures

echo "[full_run] stamp=$STAMP seeds='$SEEDS' goals=$NUM_GOALS attacker=$ATTACKER_MODEL judge=$JUDGE_MODEL"
reports=()
for s in $SEEDS; do
  R="results/run_${STAMP}_s${s}.json"
  echo "[full_run] === SEED $s -> $R (start $(date -u +%H:%M:%S)Z) ==="
  $PY simulation.py \
      --attack all --num-goals "$NUM_GOALS" --repeats 1 --seed "$s" \
      --attacker-model "$ATTACKER_MODEL" --judge-model "$JUDGE_MODEL" \
      --max-iterations "$MAX_ITERATIONS" --max-queries "$MAX_QUERIES" \
      --breach-threshold "$BREACH_THRESHOLD" \
      --report "$R" --quiet
  echo "[full_run] seed $s finished rc=$? (end $(date -u +%H:%M:%S)Z)"
  [ -f "$R" ] && reports+=("$R")
done

MERGED="results/run_${STAMP}.json"
$PY -m analysis.merge "${reports[@]}" -o "$MERGED"
$PY -m analysis.plots "$MERGED" --outdir "results/figures/${STAMP}"
echo "[full_run] DONE  merged=$MERGED  figures=results/figures/${STAMP}"
