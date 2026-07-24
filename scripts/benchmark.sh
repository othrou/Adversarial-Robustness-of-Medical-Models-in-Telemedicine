#!/usr/bin/env bash
# Standard benchmark sweep for the MedGemma safety Markov game.
#
# Runs every attack against the current defender, repeats the sweep for error
# bars, writes a timestamped JSON report, and renders statistical figures.
#
# RE-RUN THIS WHENEVER YOU CHANGE THE MODEL OR THE PROMPT (config/config.yml,
# config/prompts.yml, config/rails.co, or any --*-model). Then compare the new
# report against the previous one:
#
#     python -m analysis.plots results/run_OLD.json results/run_NEW.json \
#         --outdir results/figures --labels before after
#
# Usage:
#     scripts/benchmark.sh                 # real run (needs Ollama + models)
#     MARKOV_GAME_BACKEND=mock scripts/benchmark.sh   # offline smoke sweep
#
# Env knobs (all optional): REPEATS, NUM_GOALS, MAX_ITERATIONS, MAX_QUERIES,
# ATTACKER_MODEL, JUDGE_MODEL, PY (python interpreter).
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
REPEATS="${REPEATS:-5}"
NUM_GOALS="${NUM_GOALS:-6}"
MAX_ITERATIONS="${MAX_ITERATIONS:-6}"
MAX_QUERIES="${MAX_QUERIES:-24}"
ATTACKER_MODEL="${ATTACKER_MODEL:-mistral}"
JUDGE_MODEL="${JUDGE_MODEL:-llama-guard3:1b}"

STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT="results/run_${STAMP}.json"
FIGDIR="results/figures/${STAMP}"

# Fail fast if the real runtime (Ollama + models + deps) is not ready. This is a
# no-op under MARKOV_GAME_BACKEND=mock.
PY="$PY" ATTACKER_MODEL="$ATTACKER_MODEL" JUDGE_MODEL="$JUDGE_MODEL" \
  scripts/preflight.sh || exit 1

echo "[benchmark] backend=${MARKOV_GAME_BACKEND:-ollama} repeats=${REPEATS} -> ${REPORT}"

$PY simulation.py \
    --attack all \
    --repeats "$REPEATS" \
    --num-goals "$NUM_GOALS" \
    --max-iterations "$MAX_ITERATIONS" \
    --max-queries "$MAX_QUERIES" \
    --attacker-model "$ATTACKER_MODEL" \
    --judge-model "$JUDGE_MODEL" \
    --report "$REPORT" \
    --quiet

$PY -m analysis.plots "$REPORT" --outdir "$FIGDIR"

echo "[benchmark] report:  $REPORT"
echo "[benchmark] figures: $FIGDIR"
