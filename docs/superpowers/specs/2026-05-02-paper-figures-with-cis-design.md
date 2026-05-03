# Paper figures with confidence intervals — design

**Branch:** `scratch/tara/stats`
**Status:** design (pre-implementation)
**Owner:** Tara Bogavelli

## Goal

Auto-generate the EVA paper's clean-condition tables and plots from the existing
stats pipeline on this branch, with bootstrap confidence intervals integrated
throughout. Replace the per-domain main tables with domain-collapsed pooled-CI
tables, add `pass@k` and `pass^k` aggregate columns, draw CI error bars on the
accuracy-vs-experience scatter plots, and emit a Pareto-frontier JSON that
includes CIs.

Out of scope: perturbation tables and figures. Those will be done in a later
pass and live in their own modules.

## Inputs

The new code consumes outputs of the existing pipeline only. No re-walking of
runs, no edits to the upstream stages.

- `output_processed/eva-bench-stats/CIs/stats/results_pooled.csv`
  (one row per `model_label × metric`, columns: `point_estimate`,
  `ci_lower`, `ci_upper`, `n=pooled`)
- `output_processed/eva-bench-stats/CIs/stats/results_per_domain.csv`
  (referenced for diagnostics but not required for the headline tables/plots)
- `local/eva-bench-stats/CIs_config.yaml` for the model list, aliases, and
  architecture types

`pass@k` and `pass^k` are added later: when the user re-pulls trial data with
explicit pass@k / pass^k metrics, those metric names are added to the
`metrics:` list in `CIs_config.yaml` and flow through `data_CIs.py` →
`stats_CIs.py` → `results_pooled.csv` unchanged. The new code looks them up by
configured name and renders `--` for any system/metric pair that isn't present
yet.

## New files

All under `analysis/eva-bench-stats/`:

- `run_paper.py` — single entry point. Reads pooled CIs, dispatches to the
  three writers below, prints a summary. Run as
  `uv run python analysis/eva-bench-stats/run_paper.py`.
- `paper_tables.py` — emits `accuracy_table.tex` and `experience_table.tex`.
- `paper_plots.py` — emits the accuracy-vs-experience scatter PDFs (pass@1,
  pass@k, pass^k) with CI error bars.
- `paper_frontier.py` — emits `pareto_frontier.json` with CI fields.
- `paper_config.py` — small module holding metric → display-name maps,
  palette definitions, and the model/architecture ordering used by all three
  writers. Sourced from `CIs_config.yaml` plus inline display constants.

Outputs go to `output_processed/eva-bench-stats/CIs/paper/`:

```
paper/
  accuracy_table.tex
  experience_table.tex
  accuracy_vs_experience_pass_at_1.pdf
  accuracy_vs_experience_pass_at_k.pdf
  accuracy_vs_experience_pass_power_k.pdf
  pareto_frontier.json
```

## Tables

Two LaTeX tables, generated from `results_pooled.csv` only.

**Accuracy table columns** (after collapse):

| Group | Columns |
| --- | --- |
| Identity | Arch., System |
| EVA-A (purple, 3 sub-cols) | pass@1, pass@k, pass^k |
| Per-metric (teal) | Task Completion, Faithfulness, Agent Speech Fidelity |

**Experience table columns** (parallel):

| Group | Columns |
| --- | --- |
| Identity | Arch., System |
| EVA-X (purple, 3 sub-cols) | pass@1, pass@k, pass^k |
| Per-metric (pink) | Turn-Taking, Conciseness, Conv. Progression |

Each cell renders `point ±halfwidth` where `halfwidth = max(point − ci_lower,
ci_upper − point)`. This is the pooled CI from `results_pooled.csv`. Cells
without a value (e.g., `pass@k` before the re-pull) render `--`.

Cell shading is unchanged from the current scheme: scaled per metric column
from observed min to max of the *point estimate*, using the existing
`acc1..7`, `tel1..7`, `pnk1..7` palettes. Numbers and the `±` half-width share
the cell's foreground color (white on dark, black on light), matching today's
look.

Architecture grouping (`Cascade` / `Hybrid` / `S2S`) uses `\multirow` exactly
as in the existing tables. Within an architecture, systems are sorted by
display name. Both tables share the same row order.

The header row uses three-level grouping:

```
                | EVA-A pass-rates              | Accuracy sub-metrics
                | pass@1 | pass@k | pass^k      | TC | FA | ASF
Arch | System  |  ITSM/HR/CSM collapsed         |  ITSM/HR/CSM collapsed
```

The single domain column is implicit — caption text says "pooled across the
three EVA domains, equal-weighted, with bootstrap percentile CI half-widths."

## Plots

Three scatter PDFs, each one point per system, error bars on each point:

- `accuracy_vs_experience_pass_at_1.pdf` — x = EVA-A pass@1, y = EVA-X pass@1
- `accuracy_vs_experience_pass_at_k.pdf` — pass@k axes
- `accuracy_vs_experience_pass_power_k.pdf` — pass^k axes

Point coordinates and error-bar lengths come from `results_pooled.csv` for
the relevant pass metric:

- horizontal bar: pooled CI for the EVA-A metric on this row
- vertical bar:   pooled CI for the EVA-X metric on this row

Bars are drawn with `matplotlib.pyplot.errorbar` using asymmetric
`(point − ci_lower, ci_upper − point)` so the bars accurately reflect the
percentile CI's asymmetry.

Marker shape and color follow architecture (Cascade circle, Hybrid diamond,
S2S square — matching the existing PDFs). Pareto frontier line is drawn over
the points using the existing `_pareto_frontier` rule (upper-right frontier,
both axes higher-is-better) computed on the *point estimates*.

Plots without complete data (e.g., pass@k before the re-pull) skip writing
that PDF and log a warning.

## Pareto frontier JSON

`pareto_frontier.json` shape:

```json
{
  "pass_at_1": [
    {
      "system": "<model_label>",
      "system_type": "cascade|hybrid|s2s",
      "eva_a": {"point": 0.428, "ci_low": 0.392, "ci_high": 0.461},
      "eva_x": {"point": 0.591, "ci_low": 0.553, "ci_high": 0.628}
    },
    ...
  ],
  "pass_at_k":    [ ... same shape, may be [] if metric absent ... ],
  "pass_power_k": [ ... ]
}
```

Frontier membership is computed on point estimates (consistent with the
plotted dashed line). Entries are sorted by `eva_a.point` ascending. CI
fields use the raw percentile bounds, not the half-width — half-width is a
table-display detail only.

## Configuration

A small `paper:` section is added to `CIs_config.yaml` (or kept inline in
`paper_config.py` if simpler — decided at implementation time):

```yaml
paper:
  metrics:
    accuracy_aggregate:  [EVA-A_pass, EVA-A_pass_at_k, EVA-A_pass_power_k]
    accuracy_sub:        [task_completion, faithfulness, agent_speech_fidelity]
    experience_aggregate: [EVA-X_pass, EVA-X_pass_at_k, EVA-X_pass_power_k]
    experience_sub:      [turn_taking, conciseness, conversation_progression]
  scatter_metrics:
    pass_at_1:    {x: EVA-A_pass,         y: EVA-X_pass}
    pass_at_k:    {x: EVA-A_pass_at_k,    y: EVA-X_pass_at_k}
    pass_power_k: {x: EVA-A_pass_power_k, y: EVA-X_pass_power_k}
```

`pass_at_k` / `pass_power_k` metric names are placeholders — final names get
locked in when the re-pull happens. The code resolves them from config; the
strings appear in exactly one place.

## Error handling

- Missing `results_pooled.csv` → fail fast with a message pointing at
  `stats_CIs.py`.
- Metric configured in `paper:` but absent from pooled CSV → emit `--` in
  tables, skip the plot for that variant, return `[]` in the JSON for that
  variant. Print one warning per missing metric, not per cell.
- A single system × metric pair missing while the metric exists for other
  systems → `--` in that cell, system absent from the relevant scatter and
  frontier list. Warning printed.
- No silent fallbacks (e.g., per-domain CIs substituted for missing pooled
  CIs). Errors are explicit.

## Testing

- Unit-test the cell formatter (`format_cell(point, lo, hi)` →
  `"0.428 ±0.035"`, `"--"` for None).
- Unit-test the scatter error-bar conversion (`(point, lo, hi) →
  (point − lo, hi − point)`, both non-negative, asymmetric allowed).
- Unit-test the frontier filter (point-estimate input only, expected
  upper-right frontier).
- A small integration test: feed a hand-built `results_pooled.csv` through
  `run_paper.py` in a tmp dir; assert the four `.tex/.pdf/.json` files exist
  and contain expected substrings/keys.
- No visual regression tests — paper iteration is manual.

## Decisions log

- Code home: new modules in `analysis/eva-bench-stats/`, not a port of
  `aggregate_eva_results.py`.
- Cell display: `point ±halfwidth`, single line.
- Heatmap basis: point estimate.
- pass@k / pass^k: explicit metrics from upcoming re-pull, not recomputed
  from trial-level binary pass.
- Hybrid model with previously missing HR/CSM is fully present in the latest
  pull → no missing-domain handling required.
- Error bars on scatter: pooled CIs from `results_pooled.csv`.
- Frontier JSON: all three variants, with point + CI bounds.
- Output dir: `output_processed/eva-bench-stats/CIs/paper/`.
- Runner: new `run_paper.py`, separate from `run_stats.py`.
- Aggregate columns: three sub-columns under one EVA-A / EVA-X header.
