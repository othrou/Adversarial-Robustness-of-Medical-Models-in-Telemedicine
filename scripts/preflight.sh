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

# Cloud roles are validated before anything else: an invalid model id or a
# missing key should fail here, not three hours into a sweep.
#   $1 = role label, $2 = 'provider/model' spec
check_cloud_role() {
  local role="$1" spec="$2" provider model url key_var key
  provider="${spec%%/*}"
  model="${spec#*/}"

  case "$provider" in
    openai)     url="https://api.openai.com/v1";      key_var="OPENAI_API_KEY" ;;
    openrouter) url="https://openrouter.ai/api/v1";   key_var="OPENROUTER_API_KEY" ;;
    compat)     url="${CLOUD_BASE_URL:-}";            key_var="LLM_API_KEY" ;;
    *)          return 0 ;;   # local model: handled by the Ollama checks below
  esac

  key="${!key_var:-}"
  if [ -z "$key" ]; then
    note "FAIL" "$role: \$$key_var is not set (keys come from the environment)"
    fail=1; return
  fi
  note "OK" "$role: \$$key_var present"

  if [ -z "$url" ]; then
    note "FAIL" "$role: provider 'compat' needs CLOUD_BASE_URL"
    fail=1; return
  fi

  # Model ids drift between provider releases -- verify before spending anything.
  local ids
  ids="$(curl -s --max-time 10 -H "Authorization: Bearer $key" "$url/models" 2>/dev/null)"
  if [ -z "$ids" ]; then
    note "WARN" "$role: could not list models at $url (skipping id check)"
    return
  fi
  if printf '%s' "$ids" | grep -q "\"$model\""; then
    note "OK" "$role: model available: $model"
  else
    note "FAIL" "$role: model id not offered by $provider: $model"
    printf '%s' "$ids" | grep -o '"id"[[:space:]]*:[[:space:]]*"[^"]*"' \
      | sed 's/.*"\([^"]*\)"$/       near: \1/' \
      | grep -i "$(printf '%s' "$model" | cut -c1-6)" | head -5
    fail=1
  fi
}

echo "[preflight] real-run readiness ($BASE_URL)"

# 0. Cloud roles (no-op when every role is local).
for pair in "attacker:$ATTACKER_MODEL" "judge:$JUDGE_MODEL" \
            "target:$TARGET_MODEL" "guard:${GUARD_MODEL_URI:-$GUARD_MODEL}"; do
  check_cloud_role "${pair%%:*}" "${pair#*:}"
done

if printf '%s %s %s %s' "$ATTACKER_MODEL" "$JUDGE_MODEL" "$TARGET_MODEL" \
     "$GUARD_MODEL" | grep -qE '(^| )(openai|openrouter|compat)/'; then
  $PY -c "import openai" 2>/dev/null \
    && note "OK" "python: openai client importable" \
    || { note "FAIL" "python: openai missing (uv sync --extra cloud)"; fail=1; }
fi

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
