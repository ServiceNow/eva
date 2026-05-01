# eva-bench-stats

Statistical analyses for EVA-Bench.

## Purpose

Produces confidence intervals, variance analyses, frontier analyses, and perturbation
tests from EVA-Bench evaluation runs. Results are explored interactively in a Streamlit
app and exported as a single HTML file.

## Files

| File | Purpose |
|------|---------|
| `load_data.py` | Shared: read raw `output/<run_id>/` run directories into DataFrames |
| `data_perturbations.py` | Process perturbation run dirs → `output_processed/eva-bench-stats/perturbations/` |
| `stats_perturbations.py` | Perturbation statistical tests (paired permutation, Holm-Bonferroni, bootstrap CIs) |
| `plots_perturbations.py` | Perturbation figures and display tables |
| `data_CIs.py` | Process runs for CI analysis (placeholder) |
| `stats_CIs.py` | Cluster bootstrap CI computation (placeholder) |
| `plots_CIs.py` | CI figures and tables (placeholder) |
| `data_variance.py` | Process runs for variance analysis (placeholder) |
| `stats_variance.py` | Variance stats: distributions, ICC, judge/trial variance, LME (placeholder) |
| `plots_variance.py` | Variance figures and tables (placeholder) |
| `data_frontier.py` | Process runs for frontier analysis (placeholder) |
| `stats_frontier.py` | Frontier stats: quantile regression, SFA/DEA (placeholder) |
| `plots_frontier.py` | Frontier figures and tables (placeholder) |
| `app.py` | Multi-page Streamlit app — run with `uv run streamlit run analysis/eva-bench-stats/app.py` |
| `generate_report.py` | HTML export — run with `uv run python analysis/eva-bench-stats/generate_report.py` |

## Statistical tests

### Perturbation robustness (`stats_perturbations.py`)

**Question:** Does a model's performance change significantly when the user speaks with an
accent, in background noise, or both?

**Unit of analysis:** Scenario-level mean delta — the difference between a model's mean
score under a perturbation condition and its mean score on the clean baseline for the
same scenario. 30 paired deltas per (model, condition, domain).

---

#### Sign-flip permutation test (`permutation_test`, lines 44–75)

- **H₀:** Mean perturbation effect (delta) = 0
- **H₁:** Mean perturbation effect ≠ 0 (two-sided)
- **Test statistic:** Mean of the 30 scenario-level deltas
- **Procedure:** For each of 10,000 permutations, independently flip each delta's sign with
  p = 0.5; p-value = fraction of permuted means where |permuted mean| ≥ |observed mean|
- **Assumptions:** None beyond exchangeability under H₀ — the sign of any delta is equally
  likely to be positive or negative, which holds under the null
- **Why this test:** Score distributions are bounded and skewed; parametric tests (t-test)
  are inappropriate. The sign-flip permutation test is the natural paired test for this setting

---

#### Bootstrap confidence intervals (`bootstrap_ci`, lines 78–105)

- **Method:** Percentile bootstrap, resampling scenarios with replacement
- **Resamples:** 1,000
- **CI:** 2.5th–97.5th percentile of bootstrap means → 95% CI on mean delta
- **Interpretation:** Sampling variability of the mean effect across the scenario sample.
  The CI excludes 0 if and only if the effect is of consistent sign, but statistical
  significance is determined by the corrected permutation p-value, not the CI alone

---

#### Multiple testing correction: Holm-Bonferroni (`run_analysis`, lines 108–202)

- **Method:** Holm-Bonferroni step-down FWER control at α = 0.05
  (`statsmodels.stats.multitest.multipletests`, `method='holm'`)
- **Correction families:** Defined per (model, metric). Two analysis modes:
  - *Pooled* — 3 tests per family (one per condition: Accent, Background noise, Both);
    call `run_analysis(df, config)` (default `correction_groupby`)
  - *Per-domain* — 9 tests per family (3 conditions × 3 domains);
    call `run_analysis(df, config, correction_groupby=["model_label", "metric"])`
- **Why Holm over Bonferroni:** Uniformly more powerful while maintaining the same FWER
  guarantee; appropriate when tests are not independent across conditions

---

#### Seed strategy

Per-cell seed = global seed + `hash(f"{group_meta}:{cond}:{domain}") % (2**31)`.
Each cell gets a distinct, deterministic seed so permutation draws are not correlated
across cells (`run_analysis`, line 171).

## Running

```bash
# Interactive app
uv run streamlit run analysis/eva-bench-stats/app.py

# HTML export
uv run python analysis/eva-bench-stats/generate_report.py
```
