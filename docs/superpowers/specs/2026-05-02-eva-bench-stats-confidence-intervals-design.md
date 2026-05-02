# Confidence Intervals tab for eva-bench-stats — Design Spec

**Date:** 2026-05-02
**Owner:** tara.bogavelli@servicenow.com
**Status:** Draft
**Source directions:** `lindsay_files/confidence_interval_directions.md`

---

## 1. Problem

Compute 95% bootstrapped confidence intervals on model-level metric scores for all configured models, all metrics, and pass@1 — both per-domain and pooled across domains with equal domain weighting. Surface results in the existing `eva-bench-stats` Streamlit app's **CIs** tab (currently a placeholder), and produce a paper-ready summary for EVA-A / EVA-X composites.

---

## 2. Scope and inputs

- **Domains:** `airline` (50 scenarios), `itsm` (80 scenarios), `medical_hr` (83 scenarios).
- **Trials per scenario:** k = 5 for clean (non-perturbation) runs.
- **Metrics:**
  - 6 individual: `task_completion`, `faithfulness`, `agent_speech_fidelity`, `conversation_progression`, `turn_taking`, `conciseness`.
  - 2 composite means: `EVA-A_mean`, `EVA-X_mean`.
  - 2 pass@1: `EVA-A_pass`, `EVA-X_pass` (per-trial binary → scenario-level pass proportion).
- **Source data:** clean (non-perturbation) trial-level scores, extracted by a new sibling of `local/eva-bench-stats/pull_perturbation_data.py`.

---

## 3. Architecture

### 3.1 New / changed files

| File | Status | Role |
|---|---|---|
| `local/eva-bench-stats/pull_clean_data.py` | NEW | Remote-side puller. Walks `/mnt/voice_agent`, keeps clean-only runs, dedupes by latest per (alias, domain), writes `trial_scores.csv` + `run_manifest.csv` + `coverage_summary.csv` + zip. Emits all clean runs; manifest flags rows below `--success-threshold`. Same long-form schema as `pull_perturbation_data.py`. |
| `local/eva-bench-stats/CIs_config.yaml` | NEW | Config (see §4). |
| `analysis/eva-bench-stats/stats_utils.py` | NEW | Hosts shared `bootstrap_resample(values, n_boot, seed) -> ndarray` (full bootstrap distribution of means). `bootstrap_ci(...)` becomes a thin wrapper that takes percentiles. |
| `analysis/eva-bench-stats/stats_perturbations.py` | EDIT | Re-import `bootstrap_ci` from `stats_utils` so the perturbation pipeline keeps working unchanged. |
| `analysis/eva-bench-stats/data_CIs.py` | REPLACE | Loads `trial_scores.csv` (auto-pick newest `eva_clean_data_*`), filters to clean + configured models, validates completeness, computes scenario-level means, writes `scenario_means.csv` + `completeness_report.csv`. |
| `analysis/eva-bench-stats/stats_CIs.py` | REPLACE | Stability check + per-domain bootstrap + equal-weighted pooled bootstrap. Writes `results_per_domain.csv`, `results_pooled.csv`, `stability_log.csv`. |
| `analysis/eva-bench-stats/plots_CIs.py` | REPLACE | Forest plot per metric (model × {3 domains, pooled}); paper-ready summary table for `EVA-A_mean` / `EVA-X_mean`. |
| `analysis/eva-bench-stats/run_data.py` | EDIT | Add `_CIs_data()` call. |
| `analysis/eva-bench-stats/run_stats.py` | EDIT | Add `_CIs_stats()` call. |
| `analysis/eva-bench-stats/app.py` | EDIT | Replace `CIs_page()` placeholder with metric-tab layout + Stability / Paper-summary / Completeness expanders. |

### 3.2 Output layout

```
output_processed/eva-bench-stats/CIs/
  data/
    scenario_means.csv          (model_label, system_alias, domain, scenario_id, metric, scenario_mean)
    completeness_report.csv     (model_label, alias, domain, n_scenarios, n_expected, complete, model_complete, issues)
  stats/
    results_per_domain.csv      (model_label, metric, domain, n, point_estimate, ci_lower, ci_upper)
    results_pooled.csv          (model_label, metric, domain="pooled", n="pooled", point_estimate, ci_lower, ci_upper)
    stability_log.csv           (metric, n_boot_ref, n_boot_test, max_abs_ci_diff, within_tolerance)
```

---

## 4. Config — `local/eva-bench-stats/CIs_config.yaml`

```yaml
trial_scores_dir: output/eva-bench-stats     # auto-pick latest "eva_clean_data_*" subfolder
# trial_scores_path: output/eva-bench-stats/.../trial_scores.csv   # alternative explicit path
output_dir: output_processed/eva-bench-stats/CIs
random_seed: 42
n_bootstrap: 1000        # may be raised to 2000 from stability check
alpha: 0.05
stability_threshold: 0.002
stability_check_model: "<display label>"   # optional; defaults to first model in models:

expected_domains: [airline, itsm, medical_hr]
expected_scenarios:
  airline: 50
  itsm: 80
  medical_hr: 83
expected_k: 5

metrics:
  - task_completion
  - faithfulness
  - agent_speech_fidelity
  - conversation_progression
  - turn_taking
  - conciseness
  - EVA-A_mean
  - EVA-X_mean
  - EVA-A_pass     # pass@1 for EVA-A
  - EVA-X_pass     # pass@1 for EVA-X

models:
  "<display label>":
    alias: "<system_alias from trial_scores.csv>"
    type: cascade   # or s2s / hybrid
```

---

## 5. Data pipeline

```
[remote] /mnt/voice_agent/<run_dirs>/
    └── pull_clean_data.py
        → eva_clean_data_YYYYMMDD_HHMMSS/{trial_scores.csv, run_manifest.csv, coverage_summary.csv, .zip}

[local] output/eva-bench-stats/eva_clean_data_*/trial_scores.csv
    └── data_CIs.py
        ↳ filter to (configured aliases) ∩ (perturbation_category == "clean")
        ↳ check_model_completeness:
              per (model, domain): n_scenarios == expected_scenarios[domain],
              every scenario has expected_k trials, every (scenario, trial) has all metrics.
              → log warning + skip on mismatch
        ↳ compute_scenario_means: groupby (model, domain, scenario_id, metric), mean over trials.
              For EVA-A_pass / EVA-X_pass (binary), the trial mean is the scenario-level pass proportion (= pass@1).
        → scenario_means.csv, completeness_report.csv

[local] scenario_means.csv
    └── stats_CIs.py
        ↳ Step A — stability check: one representative model, all metrics; bootstrap @ 1000 and @ 2000;
              max |Δ CI bound| > stability_threshold → use 2000, else 1000. Log to stability_log.csv.
        ↳ Step B — per-domain bootstrap: for each (model × metric × domain):
              x = scenario_means array, length n_d ∈ {50, 80, 83}.
              point_estimate = mean(x).
              draw indices (n_boot, n_d) with replacement → boot_means shape (n_boot,).
              ci_lower, ci_upper = percentile(boot_means, [2.5, 97.5]).
              keep boot_means in memory keyed by (model, metric, domain).
        ↳ Step C — pooled bootstrap (equal domain weights): for each (model × metric):
              pool_dist = mean over the 3 domains' boot_means arrays (elementwise).
              pooled_point = mean of the 3 domain point estimates (NOT the bootstrap mean).
              pooled_ci = percentile(pool_dist, [2.5, 97.5]).
        → results_per_domain.csv, results_pooled.csv, stability_log.csv

[local] *.csv
    └── app.py CIs_page + plots_CIs.py
        ↳ forest plot per metric (model × {airline, itsm, medical_hr, pooled})
        ↳ paper-ready table: EVA-A_mean & EVA-X_mean as "point [lo, hi]"
        ↳ stability + completeness expanders
```

### 5.1 Critical invariants

- **Domain pooling is equal-weighted**, not n-weighted. The pooled point estimate averages the 3 domain means; the pooled CI averages the 3 domain bootstrap distributions elementwise.
- **Bootstrap unit = scenario** with replacement, not trial. Trial averaging happens once, upstream, in `compute_scenario_means`.
- **Composites are re-aggregated implicitly.** The puller emits `EVA-A_mean` / `EVA-A_pass` / `EVA-X_mean` / `EVA-X_pass` as their own metrics from each trial's `aggregate_metrics`. The bootstrap therefore resamples scenario-level composites directly — equivalent to recomputing composites within each resample.
- **Reproducibility:** `random_seed` from config drives both stability check and main run. Per-cell seeds derived as in `stats_perturbations` (`seed + hash(...) % 2**31`).

---

## 6. Statistical functions (in `stats_utils.py` and `stats_CIs.py`)

```python
# stats_utils.py
def bootstrap_resample(values: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Return shape (n_boot,) array of bootstrap-resample means."""

def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI from bootstrap_resample."""

# stats_CIs.py
def stability_check(scenario_means_df, model_label, metrics, seed, n_low=1000, n_high=2000) -> pd.DataFrame:
    """For one model, run bootstrap CIs at n_low and n_high for every metric × domain.
    Return per-metric max |CI-boundary diff| and within-tolerance flag."""

def compute_domain_ci(values: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05) \
        -> tuple[float, float, float, np.ndarray]:
    """Return (point_estimate, ci_lower, ci_upper, boot_dist) for one model × metric × domain."""

def compute_pooled_ci(domain_points: list[float], domain_dists: list[np.ndarray], alpha: float = 0.05) \
        -> tuple[float, float, float]:
    """Equal-weighted pooled CI: mean of domain point estimates,
    percentiles of the elementwise mean of domain bootstrap distributions."""

def run_analysis(scenario_means_df, config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full pipeline: stability check → per-domain CIs → pooled CIs.
    Returns (per_domain_df, pooled_df, stability_log_df)."""
```

---

## 7. App UI — `CIs_page()`

- **Header + brief intro** describing percentile bootstrap, equal-domain pooling, n_boot from stability check.
- **Tabs:** one per metric, ordered: pass@1 (EVA-A_pass, EVA-X_pass), individual metrics, composite means.
- **Per tab:**
  - **Forest plot**: y = model; x = score; point + horizontal CI bar; 4 facet columns (airline / itsm / medical_hr / pooled). Color by `system_type` (cascade / s2s / hybrid), consistent with variance page styling.
  - **Table**: model × {domain, point, ci_lower, ci_upper, n}, with `download_button`.
- **Always-visible expanders:**
  - **Stability** — `stability_log.csv` rendered as a small table.
  - **Paper-ready summary** — `point [lo, hi]` table for EVA-A_mean & EVA-X_mean, model × {airline, itsm, medical_hr, pooled}, with `download_button`.
  - **Completeness** — `completeness_report.csv`.
- **Empty state**: if `output_processed/eva-bench-stats/CIs/stats/` is missing, show the same "Run pipeline" prompt used by the perturbations page.

---

## 8. Error handling & edge cases

- **Missing model in trial_scores.csv** → warning, skip, recorded in `completeness_report.csv`.
- **Wrong scenario count for a domain** → warning, skip that (model, domain). If pooled would have < 3 domains, pooled is skipped for that model and recorded.
- **Partial trials** (< expected_k for a scenario) → warning, skip that scenario for that model. If this drops the domain below expected_scenarios, falls into the previous case.
- **NaN metric values** — already filtered upstream in the puller; if a scenario ends with zero valid trials for a metric, skip the scenario for that metric only.
- **Auto-pick latest data folder** mirrors `data_perturbations`.
- **No models eligible** → write empty CSVs, log clearly, exit 0.
- **Stability check failure** (representative model missing) → fall back to default `n_boot`, log warning, do not block.

---

## 9. Testing

- **Unit:** `bootstrap_resample`, `compute_domain_ci`, `compute_pooled_ci` validated against synthetic inputs with fixed seeds (deterministic outputs).
- **Regression:** run the perturbation pipeline on existing `scenario_deltas.csv` after the `bootstrap_ci` extraction; verify byte-identical `results_pooled.csv` / `results_per_domain.csv`.
- **Smoke:** end-to-end `run_data.py` + `run_stats.py` on whatever clean data exists; verify CSV shapes match §3.2 and that pooled CIs are produced for every (model, metric) where all 3 domains are complete.
- **Manual UI check:** open the Streamlit app, confirm the CIs page renders forest plots, the stability expander shows expected n_boot, the paper-ready summary renders correctly, and download buttons produce the expected CSVs.

---

## 10. Out of scope

- Statistical significance testing across models (no pairwise tests, no multiple-comparison correction). The CIs are descriptive only.
- HTML report export — `generate_report.py` integration is left for a follow-up if needed.
- Migrating perturbation analysis to use `stats_utils.bootstrap_resample`'s richer interface; only the existing `bootstrap_ci` shim is preserved for backward compatibility.
