#!/usr/bin/env bash
#
# Build a French conciseness judge-calibration dataset, end to end — resumably.
#
# Unlike faithfulness (driven by swapping the AGENT MODEL), conciseness is driven
# by swapping the agent's "Response Style" prompt instructions while keeping the
# agent model FIXED — mirroring Tara's approach (wordy / moderate / terse prompt
# variants to hit each target rating bucket):
#     target 1 (not concise)   <- wordy    variant
#     target 2 (adequate)      <- moderate variant  (loosened from production;
#                                  3-5 sentences, proactive detail, multi-option
#                                  descriptions — scores ~2.5 vs production's ~2.7)
#     target 3 (highly concise)<- terse    variant  (revised: 1-sentence cap,
#                                  no unsolicited detail, single-best-option only)
#
# NOTE: the production/default prompt already scores ~2.7 (near target 3) so it
# is NOT used for any target bucket — moderate sits between wordy and terse.
#
# The variants live in configs/prompts_variants/{wordy,moderate,terse}/simulation.yaml —
# full copies of configs/prompts/simulation.yaml with only the agent's
# "### Response Style" bullets replaced. Selected per-run via EVA_PROMPTS_DIR,
# which AgenticSystem (via a prompts_path override) reads INSTEAD of the shared
# configs/prompts/ directory. This means running this script never mutates the
# shared prompt file — safe to run concurrently with other generation (e.g. the
# faithfulness pipeline), which also reads configs/prompts/simulation.yaml on
# every record and would otherwise be corrupted by an in-place edit.
#
# RESUMABLE: OUTROOT is stable. A record with an existing non-errored conciseness
# score is skipped. One record's failure does not abort the run.
#
# Prereqs:
#   - Branch off today's main (French-ready run_text_only.py + language fix +
#     prompts_path override applied to AgenticSystem/run_text_only.py).
#   - .env has EVA_LANGUAGE=fr and a "gpt-5.1" deployment (user simulator).
#
# Usage:
#   bash scripts/build_french_conciseness_dataset.sh            # generate + judge
#   bash scripts/build_french_conciseness_dataset.sh generate   # traces only
#   bash scripts/build_french_conciseness_dataset.sh judge      # (re-)judge existing traces
#
# To force-regenerate one record, delete its dir under OUTROOT and re-run.
set -uo pipefail

cd "$(dirname "$0")/.."

export EVA_LANGUAGE=fr
unset JUDGE_MODEL || true

PY=".venv/bin/python"
N_JUDGES=3
METRIC=conciseness
OUT="${OUT:-eva_conciseness_test_set_fr.jsonl}"
OUTROOT="${OUTROOT:-debug_output/french_conciseness}"        # stable → resumable (t1, t2)
OUTROOT_T3="${OUTROOT_T3:-debug_output/french_conciseness_t3}"  # separate dir for t3 (production prompt)
                                                                 # keeps old terse traces untouched and avoids
                                                                 # record_done false-positives from the prior terse runs
STAGE="${1:-all}"                                        # all | generate | judge
AGENT_MODEL="${AGENT_MODEL:-gpt-5.4}"                     # fixed across all variants

# Each entry: "DOMAIN | VARIANT | TARGET | space-separated RECORD_IDS"
# VARIANT is one of: wordy | moderate | terse | default
# (maps to configs/prompts_variants/<variant>/; "default" = unmodified production prompt)
# Target 1 = wordy, Target 2 = moderate, Target 3 = terse
# Aiming for ~10 records per target bucket.
RUNS=(
  # --- target 1 (not concise) — wordy variant ---
  "airline    | wordy    | 1 | 1.1.2 1.1.3 2.1.2 2.2.2 2.4.1 2.4.2"
  "itsm       | wordy    | 1 | 1 10 11 12 15 16"
  "medical_hr | wordy    | 1 | 1.1 1.2 2.1 2.2 4.1 4.2"

  # --- target 2 (adequate) — moderate variant ---
  "airline    | moderate | 2 | 1.1.4 1.1.5 2.1.1 2.2.5 3.1.3 3.1.5"
  "itsm       | moderate | 2 | 100 101 13 14 17 18"
  "medical_hr | moderate | 2 | 10.1 10.2 3.1 3.2 5.1 5.2"

  # --- target 3 (highly concise) — production/default prompt ---
  # The production prompt already scores ~2.7-2.9, making it the natural target-3 tier.
  "airline    | default  | 3 | 1.2.1 1.2.2 2.3.2 2.3.4 4.1.1 4.1.2"
  "itsm       | default  | 3 | 102 103 25 26 19 20"
  "medical_hr | default  | 3 | 11.1 11.2 12.1 12.2 6.1 6.2"
)

trim() { echo "$1" | xargs; }

prompts_dir_for() {  # variant -> EVA_PROMPTS_DIR value (empty = unset = default)
  case "$1" in
    default|"") echo "" ;;
    *) echo "configs/prompts_variants/$1" ;;
  esac
}

# record_done RUN_DIR RECORD_ID — true if metrics.json exists with a non-errored
# conciseness score that rated at least one turn.
record_done() {
  local run_dir="$1" rid="$2" mj
  for mj in "$run_dir"/*/"$rid"/metrics.json; do
    [ -f "$mj" ] || continue
    if "$PY" - "$mj" <<'PY'
import json, sys
try:
    m = json.load(open(sys.argv[1]))
    c = m.get("metrics", {}).get("conciseness", {})
    ratings = (c.get("details") or {}).get("per_turn_ratings") or {}
    ok = c.get("error") is None and any(v is not None for v in ratings.values())
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
PY
    then return 0; fi
  done
  return 1
}

run_dir_for() {   # domain, target -> output dir
  if [ "$2" = "3" ]; then
    echo "$OUTROOT_T3/$1_t$2"
  else
    echo "$OUTROOT/$1_t$2"
  fi
}

generate() {
  echo "=== Stage 1: generating French conciseness traces (text-only) ==="
  echo "    t1/t2 → $OUTROOT  |  t3 → $OUTROOT_T3"
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
    echo "No generated traces found under $OUTROOT / $OUTROOT_T3 — run the 'generate' stage first."
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
