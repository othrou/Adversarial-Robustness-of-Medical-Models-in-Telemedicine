#!/usr/bin/env bash
# Preflight readiness check for a REAL (Ollama-backed) full-scale run.
#
# Verifies the runtime the mock path does not need: the Ollama server, the
# required models, and the Python deps for the real defender. Exits non-zero with
# a precise reason if anything is missing, so `scripts/benchmark.sh` can fail fast
# instead of dying mid-sweep.
#
#     scripts/preflight.sh          # check defaults
#     PY=.venv311/bin/python scripts/preflight.sh
#
# Skipped automatically when MARKOV_GAME_BACKEND=mock (nothing to check offline).
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
BASE_URL="${BASE_URL:-http://127.0.0.1:11434}"
TARGET_MODEL="${TARGET_MODEL:-amsaravi/medgemma-4b-it:q6}"
GUARD_MODEL="${GUARD_MODEL:-llama-guard3:1b}"
ATTACKER_MODEL="${ATTACKER_MODEL:-mistral}"
JUDGE_MODEL="${JUDGE_MODEL:-llama-guard3:1b}"

if [ "${MARKOV_GAME_BACKEND:-ollama}" = "mock" ]; then
  echo "[preflight] MARKOV_GAME_BACKEND=mock -> offline, nothing to check. OK."
  exit 0
fi

fail=0
note() { printf '  %-4s %s\n' "$1" "$2"; }

echo "[preflight] real-run readiness ($BASE_URL)"

# 1. Python deps for the real defender + backend.
$PY -c "import nemoguardrails" 2>/dev/null \
  && note "OK" "python: nemoguardrails importable" \
  || { note "FAIL" "python: nemoguardrails missing (uv sync)"; fail=1; }
$PY -c "import ollama" 2>/dev/null \
  && note "OK" "python: ollama client importable" \
  || { note "FAIL" "python: ollama client missing (uv sync)"; fail=1; }

# 2. Ollama server reachable.
tags="$(curl -s --max-time 4 "$BASE_URL/api/tags" 2>/dev/null)"
if [ -z "$tags" ]; then
  note "FAIL" "ollama server unreachable at $BASE_URL (start: 'ollama serve')"
  echo "[preflight] NOT READY (server down) — cannot check models."
  exit 1
fi
note "OK" "ollama server reachable"

# 3. Required models present.
for m in "$TARGET_MODEL" "$GUARD_MODEL" "$ATTACKER_MODEL" "$JUDGE_MODEL"; do
  if printf '%s' "$tags" | grep -q "${m%%:*}"; then
    note "OK" "model present: $m"
  else
    note "FAIL" "model missing: $m  (ollama pull $m)"
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  echo "[preflight] NOT READY — resolve the FAIL lines above."
  exit 1
fi
echo "[preflight] READY for a full-scale run."
