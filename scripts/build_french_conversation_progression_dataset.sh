#!/usr/bin/env bash
#
# Build a French conversation-progression judge-calibration dataset, end to end — resumably.
#
# Conversation progression is driven by swapping the agent's "Conversational
# Behavior" prompt instructions while keeping the agent model FIXED. Each variant
# steers the agent toward a specific failure dimension:
#     target 1 (clear issue)   <- tool_caller | forgetful | bad_questions  variants
#     target 2 (minor issue)   <- redundant variant (milder failure mode)
#     target 3 (no issue)      <- default variant (unmodified prompt)
#
# Variants:
#   combined_severe → all four failure dimensions at once (tool spam, info loss,
#                     redundancy, bad questions) — targets rating 1
#   forgetful       → information_loss (re-asks for already-provided info) — targets 1
#   tool_caller     → unnecessary_tool_calls (retries, redundant lookups) — targets 1
#   combined_mild   → light redundancy + one extra verification call — targets rating 2
#   default         → unmodified prompt — targets rating 3
#
# The variants live in configs/prompts_variants/{combined_severe,forgetful,
# tool_caller,combined_mild}/simulation.yaml — full copies of configs/prompts/simulation.yaml
# with only the agent's "## Conversational Behavior" section replaced. Selected
# per-run via EVA_PROMPTS_DIR.
#
# RESUMABLE: OUTROOT is stable. A record with an existing non-errored
# conversation_progression score is skipped. One record's failure does not abort
# the run.
#
# Usage:
#   bash scripts/build_french_conversation_progression_dataset.sh            # generate + judge
#   bash scripts/build_french_conversation_progression_dataset.sh generate   # traces only
#   bash scripts/build_french_conversation_progression_dataset.sh judge      # (re-)judge existing traces
#
# To force-regenerate one record, delete its dir under OUTROOT and re-run.
set -uo pipefail

cd "$(dirname "$0")/.."

export EVA_LANGUAGE=fr
unset JUDGE_MODEL || true

PY=".venv/bin/python"
N_JUDGES=3
METRIC=conversation_progression
OUT="${OUT:-eva_conversation_progression_test_set_fr.jsonl}"
OUTROOT="${OUTROOT:-debug_output/french_conversation_progression}"
STAGE="${1:-all}"                                        # all | generate | judge
AGENT_MODEL="${AGENT_MODEL:-gpt-5.4}"                   # fixed across all variants

# Each entry: "DOMAIN | VARIANT | TARGET | space-separated RECORD_IDS"
# VARIANT maps to configs/prompts_variants/<variant>/ ; "default" = unmodified prompt.
#
# target 1: combined_severe + forgetful + tool_caller (strong failure modes)
# target 2: combined_mild (light redundancy + verification)
# target 3: default (no failure steering)
RUNS=(
  # --- airline  (18 total: 8×t1, 5×t2, 5×t3) ---
  "airline    | combined_severe | 1 | 1.1.2 1.1.3 2.1.2 2.1.6"
  "airline    | forgetful       | 1 | 1.1.4 2.2.2"
  "airline    | tool_caller     | 1 | 1.1.5 2.2.4"
  "airline    | combined_mild   | 2 | 1.2.1 1.2.2 1.2.3 2.2.5 2.3.2"
  "airline    | default         | 3 | 1.3.1 1.3.2 2.1.1 2.3.4 2.4.1"
  # --- itsm  (19 total: 7×t1, 7×t2, 5×t3) ---
  "itsm       | combined_severe | 1 | 1 15 16"
  "itsm       | forgetful       | 1 | 10 17"
  "itsm       | tool_caller     | 1 | 100 18"
  "itsm       | combined_mild   | 2 | 101 102 103 11 19 20 21"
  "itsm       | default         | 3 | 12 13 14 22 23"
  # --- medical_hr  (20 total: 7×t1, 6×t2, 7×t3) ---
  "medical_hr | combined_severe | 1 | 1.1 4.1 4.2"
  "medical_hr | forgetful       | 1 | 1.2 5.1"
  "medical_hr | tool_caller     | 1 | 10.1 5.2"
  "medical_hr | combined_mild   | 2 | 10.2 11.1 11.2 6.1 6.2 7.1"
  "medical_hr | default         | 3 | 2.1 2.2 3.1 3.2 7.2 8.1 8.2"
)

trim() { echo "$1" | xargs; }

prompts_dir_for() {  # variant -> EVA_PROMPTS_DIR value (empty = unset = default)
  case "$1" in
    default|"") echo "" ;;
    *) echo "configs/prompts_variants/$1" ;;
  esac
}

# record_done RUN_DIR RECORD_ID — true if metrics.json exists with a non-errored
# conversation_progression score.
record_done() {
  local run_dir="$1" rid="$2" mj
  for mj in "$run_dir"/*/"$rid"/metrics.json; do
    [ -f "$mj" ] || continue
    if "$PY" - "$mj" <<'PY'
import json, sys
try:
    m = json.load(open(sys.argv[1]))
    c = m.get("metrics", {}).get("conversation_progression", {})
    ok = c.get("error") is None and c.get("score") is not None
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
PY
    then return 0; fi
  done
  return 1
}

run_dir_for() { echo "$OUTROOT/$1_t$2"; }   # domain, target

generate() {
  echo "=== Stage 1: generating French conversation_progression traces (text-only) → $OUTROOT ==="
  echo "    agent model fixed at: $AGENT_MODEL"
  local GEN_FAIL=()
  for row in "${RUNS[@]}"; do
    IFS='|' read -r domain variant target ids <<< "$row"
    domain=$(trim "$domain"); variant=$(trim "$variant"); target=$(trim "$target"); ids=$(trim "$ids")
    local run_dir; run_dir=$(run_dir_for "$domain" "$target")
    local prompts_dir; prompts_dir=$(prompts_dir_for "$variant")
    for rid in $ids; do
      if record_done "$run_dir" "$rid"; then
        echo "  skip (done): $domain/$rid  variant=$variant"
        continue
      fi
      echo "  gen: $domain/$rid  variant=$variant  prompts_dir=${prompts_dir:-<default>}"
      EVA_DOMAIN="$domain" EVA_PROMPTS_DIR="$prompts_dir" "$PY" scripts/run_text_only.py \
        --llm-model "$AGENT_MODEL" --domain "$domain" \
        --metrics "$METRIC" --record-id "$rid" --output-dir "$run_dir" \
        || echo "    (run_text_only exited non-zero for $domain/$rid)"
      if ! record_done "$run_dir" "$rid"; then
        echo "    FAILED to produce a valid score: $domain/$rid"
        GEN_FAIL+=("$domain/$rid variant=$variant")
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

judge() {
  echo "=== Stage 2: re-running the $METRIC judge ${N_JUDGES}x per trace ==="
  local RUN_ARGS=()
  for row in "${RUNS[@]}"; do
    IFS='|' read -r domain variant target ids <<< "$row"
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
