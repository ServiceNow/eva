# eva-bench-stats

Statistical analyses for EVA-Bench.

## Purpose

Produces variance analyses, perturbation tests, and confidence intervals from EVA-Bench
evaluation runs. Results are explored interactively in a Streamlit app and exported as
a single HTML file.

---

## Quick start (new machine)

### 1. Install dependencies

```bash
uv sync
```

### 2. Get config files

Config files live in `local/eva-bench-stats/` (gitignored — get them from a teammate):

| File | Used by |
|------|---------|
| `local/eva-bench-stats/variance_config.yaml` | Variance analysis |
| `local/eva-bench-stats/perturbations_config.yaml` | Perturbation analysis |

See [Config format](#config-format) below for the full schema.

### 3. Ensure run archives are present

**Variance analysis** reads from `output/judge_variance_analysis/<run_id>/`:

```
output/judge_variance_analysis/
  <run_id>/
    iter_1/
      records/
        <record_id>/
          trial_1/
            metrics.json
          trial_2/
            metrics.json
          trial_3/
            metrics.json
    iter_2/
      ...
    iter_3/
      ...
```

The `run_id` values come from your `variance_config.yaml`. Variance archives must follow
the iteration-archive layout above; standard single-run output directories are not supported.

**Perturbation analysis** reads trial-score CSVs — see your `perturbations_config.yaml`
for the expected path.

### 4. Run the pipeline

The easiest way is through the Streamlit app's **Run pipeline** sidebar (see step 5). To
run from the command line instead:

```bash
# Step 1: process raw runs → CSVs
uv run python analysis/eva-bench-stats/run_data.py

# Step 2: compute statistics
uv run python analysis/eva-bench-stats/run_stats.py
```

Both scripts process all configured analyses (perturbations + variance). You must run
`run_data.py` before `run_stats.py`.

### 5. Launch the app

```bash
uv run streamlit run analysis/eva-bench-stats/app.py
```

The **Run pipeline** expander in the sidebar lets you trigger `run_data.py` and
`run_stats.py` without leaving the browser. It auto-expands when data is missing.

---

## Output layout

All processed outputs go under `output_processed/eva-bench-stats/` (gitignored):

```
output_processed/eva-bench-stats/
  variance/
    data/                        ← written by run_data.py
      scores.csv                 (per-iteration scores for all runs × metrics)
      judge_var.csv              (per-(record,trial) judge std dev)
      trial_var.csv              (per-record trial std dev)
      judge_summary.csv          (per-(run,metric) judge variance summary)
      trial_summary.csv          (per-(run,metric) trial variance summary)
      composite_stability.csv    (per-iteration EVA composite values)
      borderline_scenarios.csv   (pass/fail flips from judge stochasticity)
    stats/                       ← written by run_stats.py
      icc_per_model.csv
      icc_pooled_centered.csv
      icc_pooled_twoway.csv
      q0_judge_pooled.csv
      q0_judge_per_model.csv
      q0_trial_pooled.csv
      q0_trial_per_model.csv
      q1a.csv
      q1b.csv
      q2_kw.csv
      q2_pairwise.csv
      q3_kw.csv
      q3_pairwise.csv
      within_type_*.csv          (within-type KW/pairwise results, one per run type × test)
      lmm/                       ← written by stats_variance_lmm.py (run_stats.py --skip-lmm to omit)
        lmm_variance_components.csv         (pooled: σ², proportions, CIs per metric × component)
        lmm_fixed_effects.csv               (pooled: domain + model coefficients + CIs per metric)
        lmm_convergence.csv                 (all fits: converged, log-likelihood, notes)
        lmm_per_model_variance_components.csv (per-model: σ², proportions, CIs per metric × model)
        lmm_per_model_fixed_effects.csv     (per-model: domain coefficients + CIs per metric × model)
  perturbations/
    results_pooled.csv
    results_per_domain.csv
    completeness_report.csv
```

---

## Config format

### `variance_config.yaml`

```yaml
output_dir: output_processed/eva-bench-stats/variance
alpha: 0.05
n_bootstrap: 1000

metrics:
  - faithfulness
  - agent_speech_fidelity
  - conversation_progression
  - conciseness

runs:
  "display label for this run":
    run_id: "2026-03-09_06-30-04.153584"
    type: cascade          # "cascade" or "s2s"
    stt: elevenlabs (scribe_v2)
    llm: sonnet-4-6
    tts: elevenlabs (turbo)

  "another run (s2s example)":
    run_id: "2026-04-15_22-21-54.234893_gemini-3.1-flash-live-preview"
    type: s2s
    s2s: gemini-3.1-flash-live-preview
    voice: Leda
```

The display label (the top-level key under `runs`) is how the run appears in all charts
and tables. `run_id` must match the directory name under `output/judge_variance_analysis/`.

### `perturbations_config.yaml`

See the file itself for the full schema — it includes model definitions, condition names,
domain names, and paths to trial-score CSVs.

---

## Files

| File | Purpose |
|------|---------|
| `load_data.py` | Read raw `output/<run_id>/` and iteration-archive run directories into DataFrames |
| `plots_utils.py` | Shared plot helpers (`empty_fig`, `fmt_p`, `download_button`) used across all analysis modules |
| `run_data.py` | Entry point: run all data preprocessing (perturbations + variance) |
| `run_stats.py` | Entry point: run all statistical analyses (perturbations + variance) |
| `data_perturbations.py` | Process perturbation run dirs → `output_processed/eva-bench-stats/perturbations/` |
| `stats_perturbations.py` | Perturbation statistical tests (paired permutation, Holm-Bonferroni, bootstrap CIs) |
| `plots_perturbations.py` | Perturbation figures and display tables |
| `data_variance.py` | Process variance iteration archives → `output_processed/eva-bench-stats/variance/data/` |
| `stats_variance.py` | Variance stats: judge/trial variance, ICC, Q0/Q1/Q2/Q3 tests |
| `stats_variance_lmm.py` | LMM variance decomposition: REML mixed effects models, variance component CIs, fixed effect coefficients |
| `plots_variance.py` | Variance figures and display tables |
| `data_CIs.py` | Process clean trial scores → scenario-level means for CI analysis |
| `stats_CIs.py` | Bootstrap stability check, per-domain, and equal-weighted pooled CIs |
| `plots_CIs.py` | CI forest plots and paper-ready summary table |
| `data_frontier.py` | Process runs for frontier analysis (placeholder) |
| `stats_frontier.py` | Frontier stats (placeholder) |
| `plots_frontier.py` | Frontier figures and tables (placeholder) |
| `app.py` | Multi-page Streamlit app |
| `generate_report.py` | HTML export |

---

## Statistical tests

### Variance analysis (`stats_variance.py`)

The variance study measures three sources of variance in EVA metric scores. Each run must
have been evaluated N times by the same judge on the same conversations (iteration archives),
with M simulation trials per scenario.

**Q0 — Is variance significantly greater than zero?**
One-sample Wilcoxon signed-rank test against 0 (H₁: median > 0), run separately for judge
variance and trial variance, pooled across models and per model × metric. A significant result
means variance is reliably non-zero for that metric.

**Q1a — Is judge variance significantly different from trial variance?**
Paired Wilcoxon signed-rank test per model × metric (Bonferroni-corrected). Paired on
record_id to remove scenario-difficulty confound.

**Q1b — Does the judge-vs-trial gap vary by model?**
Kruskal-Wallis on per-record deltas (judge_std − trial_std) across models.

**Q2 — Does judge variance differ across models?**
Kruskal-Wallis H test on per-(record,trial) judge std devs. If significant: pairwise
Mann-Whitney U (Bonferroni-corrected) with rank-biserial correlation as effect size.
Run both pooled and within each model type (cascade / S2S) separately.

**Q3 — Does trial variance differ across models?**
Same approach as Q2, applied to per-record trial std devs.

**ICC — What fraction of variance is explained by scenario identity?**
Three estimates: pooled centered (within-model), pooled two-way random effects with
interaction F-test, and per-model one-way ANOVA.

---

### Perturbation robustness (`stats_perturbations.py`)

**Question:** Does a model's performance change significantly when the user speaks with an
accent, in background noise, or both?

**Unit of analysis:** Scenario-level mean delta — the difference between a model's mean
score under a perturbation condition and its mean score on the clean baseline for the
same scenario. 30 paired deltas per (model, condition, domain).

#### Sign-flip permutation test

- **H₀:** Mean perturbation effect (delta) = 0
- **H₁:** Mean perturbation effect ≠ 0 (two-sided)
- **Test statistic:** Mean of the 30 scenario-level deltas
- **Procedure:** For each of 10,000 permutations, independently flip each delta's sign with
  p = 0.5; p-value = fraction of permuted means where |permuted mean| ≥ |observed mean|
- **Why this test:** Score distributions are bounded and skewed; the sign-flip permutation
  test is the natural paired test for this setting

#### Bootstrap confidence intervals

- **Method:** Percentile bootstrap, resampling scenarios with replacement
- **Resamples:** 1,000
- **CI:** 2.5th–97.5th percentile → 95% CI on mean delta

#### Multiple testing correction: Holm-Bonferroni

- **Method:** Holm-Bonferroni step-down FWER control at α = 0.05
- **Correction families:** Defined per (model, metric)
  - *Pooled* — 3 tests per family (one per condition)
  - *Per-domain* — 9 tests per family (3 conditions × 3 domains)

---

#### Pairwise comparisons — secondary family (`run_pairwise_analysis`, line 194)

- **H₀:** Mean delta for condition A equals mean delta for condition B (one H₀ per pair)
- **Pairs tested:** accent vs background noise, accent vs both, background noise vs both
- **Correction family:** Holm-Bonferroni across the 3 pairwise tests per (model, metric).
  Applied separately from the primary vs-baseline family.
- **Why separate families:** The primary question ("does perturbation affect performance?")
  and the secondary question ("which conditions differ from each other?") are distinct
  inferential goals. Pooling would dilute power for each independently.

---

#### Compact letter display (`compute_cld`, line 328)

- **Input:** 3×3 boolean significance matrix from secondary-family reject flags
- **Algorithm:** Insert-absorption maximal-clique: find all maximal subsets of conditions
  where no pair is significantly different; assign letters a, b, c ... to subsets; each
  condition receives all letters of subsets it belongs to.
- **Interpretation:** Conditions sharing a letter are not significantly different from each
  other. Conditions with no shared letter are significantly different.

---

#### Additivity test (`run_additivity_analysis`, line 276)

- **H₀:** The combined (both) perturbation effect equals the sum of the individual effects
- **Statistic:** Mean of `delta_both − (delta_accent + delta_background_noise)` across scenarios
- **Correction:** None — one test per (model, metric, domain). Raw p-value reported directly.
- **Interpretation:** Positive residual = synergistic (combined effect exceeds sum of parts);
  negative residual = sub-additive (combined effect less than sum of parts).

---

### Variance decomposition — mixed effects (`stats_variance_lmm.py`)

Partitions score variance into scenario, trial, judge (judge-graded metrics only), and
residual components using a linear mixed effects model fitted with REML.

**Pipeline:**
```
_detect_judge_metrics          → classify metrics as judge-graded or deterministic
fit_lmm_pooled                 → one REML model per metric (all 4 models together)
  extract_variance_components  → σ², proportions, Wald CIs, fixed effects, convergence
fit_lmm_per_model              → one REML model per (metric, model_id)
  extract_variance_components  → same extraction
results_to_dataframes          → assemble 5 output CSVs from result dicts
main                           → load scores.csv, run pipeline, write lmm/ CSVs
```

Three model variants are used depending on metric type and analysis level:

**Pooled, judge-graded** (all models together, Judge vc_formula included):
```
score ~ C(model_id, Sum) + C(domain, Sum)
      + (1 | scenario_uid)
      + Trial:  0 + C(trial_uid)   [vc_formula]
      + Judge:  0 + C(judge_uid)   [vc_formula]
```
Residual = "Judge + interactions": judge_uid is shared across models, so the judge random
effect cannot isolate per-model judge stochasticity; it absorbs instead model×scenario
interactions. Judge and interactions are confounded in the pooled residual.

**Pooled, deterministic** — same but without the Judge vc_formula term.

**Per-model, judge-graded** (one fit per model, Judge vc_formula *omitted*):
```
score ~ C(domain, Sum)
      + (1 | scenario_uid)
      + Trial:  0 + C(trial_uid)   [vc_formula]
```
Residual = "Judge stochasticity": judge_uid has 1 obs/level in a single-model fit, making
judge variance unidentifiable. Dropping Judge makes Trial identifiable (3 obs/level) and
the residual is pure judge stochasticity.

**Per-model, deterministic** — iterations are collapsed (identical across iterations);
two-level model fitted: `score ~ C(domain, Sum) + (1 | scenario_uid)`.
Residual = σ²_trial; only Scenario and Trial components are reported.

Sum-to-zero (effects) coding means coefficients are deviations from the grand mean.
Variance component CIs are approximate Wald (likelihood-based) intervals from the REML
Hessian (clipped to zero). See the app tab for interpretation guidance.

Skip LMM when it's not needed: `uv run python analysis/eva-bench-stats/run_stats.py --skip-lmm`

**Future:** A model × scenario random interaction term (requiring pymer4 / lme4 via rpy2)
is planned as a follow-on tab "Variance decomp (interaction)". See `local/superpowers/specs/`
for the future spec.
