#!/usr/bin/env bash
#
# Build a French faithfulness judge-calibration dataset, end to end — resumably.
#
#   Stage 1 (generate) — text-only conversation traces (no audio, no voice IDs),
#                        one agent-model tier per intended target rating:
#                            target 3  <- strong model (Opus)
#                            target 2  <- mid model    (GPT-5.4)
#                            target 1  <- weak model   (GPT-4.1-mini)
#   Stage 2 (judge)    — re-run the faithfulness judge N times per trace and
#                        assemble an app-compatible dataset (Tara's structure).
#
# RESUMABLE: OUTROOT is stable (not timestamped). A record that already has a
# valid faithfulness score is skipped, so re-running after an interruption picks
# up only what's missing. One record's failure does NOT abort the run — failures
# are collected and reported so you can just re-run to retry them.
#
# Prereqs:
#   - Branch off today's main (French-ready run_text_only.py, language fix applied).
#   - .env has EVA_LANGUAGE=fr, a "gpt-5.1" deployment (user simulator), and the
#     agent + judge models below in EVA_MODEL_LIST.
#   - Do NOT set JUDGE_MODEL (keeps the judge at its Opus-4-6 class default).
#
# Usage:
#   bash scripts/build_french_judge_dataset.sh            # generate + judge
#   bash scripts/build_french_judge_dataset.sh generate   # only generate traces
#   bash scripts/build_french_judge_dataset.sh judge       # only (re-)judge existing traces
#   OUTROOT=debug_output/my_run bash scripts/build_french_judge_dataset.sh   # custom dir
#
# To force-regenerate one record, delete its dir under OUTROOT and re-run.
#
# NOTE: no `set -e` — we handle per-record failures explicitly so one flaky
# record (timeout, rate limit) can't kill a long run.
set -uo pipefail

cd "$(dirname "$0")/.."

export EVA_LANGUAGE=fr
unset JUDGE_MODEL || true

PY=".venv/bin/python"
N_JUDGES=3
METRIC=faithfulness
OUT="${OUT:-eva_faithfulness_test_set_fr.jsonl}"
OUTROOT="${OUTROOT:-debug_output/french_faithfulness}"   # stable → resumable
STAGE="${1:-all}"                                         # all | generate | judge

# Each entry: "DOMAIN | AGENT_MODEL | TARGET | space-separated RECORD_IDS"
# ~20 traces total. Adjust freely; these IDs exist in data/<domain>_scenarios/.
# Same (domain, target) can appear in multiple rows (different models) — traces
# land in the same run dir (domain_t<target>) and just add more candidates for
# that rating bucket, as long as record IDs don't repeat across rows.
RUNS=(
  "airline    | us.anthropic.claude-opus-4-8       | 3 | 1.1.2 1.1.3"
  "airline    | gpt-5.4                            | 2 | 1.1.4 1.1.5"
  "airline    | gpt-4.1-mini                       | 1 | 1.2.1 1.2.2"
  "airline    | anthropic.claude-haiku-4-5-20251001-v1:0 | 1 | 1.2.3 1.3.1"
  "airline    | gemini-3-flash-preview             | 2 | 1.3.2 2.1.1"
  "airline    | gpt-4.1-mini                       | 1 | 2.1.2 2.1.6"
  "airline    | gpt-4.1-mini                       | 1 | 2.2.2 2.2.4"
  "airline    | ministral-3.3b                     | 1 | 2.3.2 2.3.4"
  "airline    | us.anthropic.claude-opus-4-6-v1     | 2 | 2.4.1 2.4.2"
  "airline    | us.anthropic.claude-opus-4-6-v1     | 3 | 3.1.3 3.1.5"
  "airline    | ministral-3.3b                     | 1 | 4.1.1 4.1.2"
  "airline    | gemini-3-flash-preview             | 2 | 4.1.3"
  "itsm       | us.anthropic.claude-opus-4-8       | 3 | 1 10"
  "itsm       | gpt-5.4                            | 2 | 100 101"
  "itsm       | gpt-4.1-mini                       | 1 | 102 103"
  "itsm       | anthropic.claude-haiku-4-5-20251001-v1:0 | 1 | 11 12"
  "itsm       | gemini-3-flash-preview             | 2 | 13 14"
  "itsm       | gpt-4.1-mini                       | 1 | 15 16"
  "itsm       | gpt-4.1-mini                       | 1 | 17 18"
  "itsm       | ministral-3.3b                     | 1 | 19 20"
  "itsm       | us.anthropic.claude-opus-4-6-v1     | 2 | 21 22"
  "itsm       | us.anthropic.claude-opus-4-6-v1     | 3 | 23 24"
  "itsm       | ministral-3.3b                     | 1 | 2 4"
  "itsm       | gemini-3-flash-preview             | 2 | 5"
  "medical_hr | us.anthropic.claude-opus-4-8       | 3 | 1.1 1.2"
  "medical_hr | gpt-5.4                            | 2 | 10.1 10.2"
  "medical_hr | gpt-4.1-mini                       | 1 | 11.1 11.2"
  "medical_hr | anthropic.claude-haiku-4-5-20251001-v1:0 | 1 | 2.1 2.2"
  "medical_hr | gemini-3-flash-preview             | 2 | 3.1 3.2"
  "medical_hr | gpt-4.1-mini                       | 1 | 4.1 4.2"
  "medical_hr | gpt-4.1-mini                       | 1 | 5.1 5.2"
  "medical_hr | ministral-3.3b                     | 1 | 6.1 6.2"
  "medical_hr | us.anthropic.claude-opus-4-6-v1     | 2 | 7.1 7.2"
  "medical_hr | us.anthropic.claude-opus-4-6-v1     | 3 | 8.1 8.2"
  "medical_hr | ministral-3.3b                     | 1 | 9.1 9.2"
  "medical_hr | gemini-3-flash-preview             | 2 | 12.1"
)

trim() { echo "$1" | xargs; }

# record_done RUN_DIR RECORD_ID — true if a metrics.json exists for this record
# with a non-errored faithfulness rating already computed.
record_done() {
  local run_dir="$1" rid="$2" mj
  for mj in "$run_dir"/*/"$rid"/metrics.json; do
    [ -f "$mj" ] || continue
    if "$PY" - "$mj" <<'PY'
import json, sys
try:
    m = json.load(open(sys.argv[1]))
    f = m.get("metrics", {}).get("faithfulness", {})
    ok = f.get("error") is None and (f.get("details") or {}).get("rating") is not None
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
PY
    then return 0; fi
  done
  return 1
}

run_dir_for() { echo "$OUTROOT/$1_t$2"; }   # domain, target

# ---------------------------------------------------------------------------
# Stage 1: generate
# ---------------------------------------------------------------------------
generate() {
  echo "=== Stage 1: generating French traces (text-only) → $OUTROOT ==="
  local GEN_FAIL=()
  for row in "${RUNS[@]}"; do
    IFS='|' read -r domain model target ids <<< "$row"
    domain=$(trim "$domain"); model=$(trim "$model"); target=$(trim "$target"); ids=$(trim "$ids")
    local run_dir; run_dir=$(run_dir_for "$domain" "$target")
    for rid in $ids; do
      if record_done "$run_dir" "$rid"; then
        echo "  skip (done): $domain/$rid  t=$target"
        continue
      fi
      echo "  gen: $domain/$rid  model=$model  target=$target"
      EVA_DOMAIN="$domain" "$PY" scripts/run_text_only.py \
        --llm-model "$model" --domain "$domain" \
        --metrics "$METRIC" --record-id "$rid" --output-dir "$run_dir" \
        || echo "    (run_text_only exited non-zero for $domain/$rid)"
      if ! record_done "$run_dir" "$rid"; then
        echo "    FAILED to produce a valid score: $domain/$rid"
        GEN_FAIL+=("$domain/$rid t=$target")
      fi
    done
  done
  if [ ${#GEN_FAIL[@]} -gt 0 ]; then
    echo ""
    echo "Stage 1 INCOMPLETE for ${#GEN_FAIL[@]} record(s): ${GEN_FAIL[*]}"
    echo "Re-run this script to retry just those (completed records are skipped)."
    return 1
  fi
  echo "Stage 1 complete."
  return 0
}

# ---------------------------------------------------------------------------
# Stage 2: judge + assemble
# ---------------------------------------------------------------------------
judge() {
  echo "=== Stage 2: re-running the $METRIC judge ${N_JUDGES}x per trace ==="
  local RUN_ARGS=()
  for row in "${RUNS[@]}"; do
    IFS='|' read -r domain model target ids <<< "$row"
    domain=$(trim "$domain"); target=$(trim "$target")
    local run_dir; run_dir=$(run_dir_for "$domain" "$target")
    for gen in "$run_dir"/*/; do
      [ -d "$gen" ] || continue
      RUN_ARGS+=(--run "${gen%/}:${domain}:${target}")
    done
  done
  if [ ${#RUN_ARGS[@]} -eq 0 ]; then
    echo "No generated traces found under $OUTROOT — run the 'generate' stage first."
    return 1
  fi
  "$PY" scripts/rerun_judge.py \
    --metric "$METRIC" --n "$N_JUDGES" --out "$OUT" --make-labels \
    "${RUN_ARGS[@]}"
}

# ---------------------------------------------------------------------------
case "$STAGE" in
  generate) generate ;;
  judge)    judge ;;
  all)
    if generate; then
      judge
    else
      echo "Skipping Stage 2 because Stage 1 is incomplete. Fix/retry, then re-run (or run 'judge' once all traces exist)."
      exit 1
    fi
    ;;
  *) echo "Unknown stage '$STAGE' (use: all | generate | judge)"; exit 2 ;;
esac

echo ""
echo "Dataset: $OUT"
echo "Point the labeling app at it (add a French '$METRIC' entry to apps/metrics_labeler.py METRICS)."
