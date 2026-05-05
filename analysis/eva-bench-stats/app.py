# analysis/eva-bench-stats/app.py
"""EVA-Bench Statistics — multi-page Streamlit app.

Run from project root:
    uv run streamlit run analysis/eva-bench-stats/app.py
"""

import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


import streamlit as st
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "local" / "eva-bench-stats"
PROCESSED_DIR = PROJECT_ROOT / "output_processed" / "eva-bench-stats"

st.set_page_config(page_title="EVA-Bench Statistics", layout="wide")


def _load_config(area: str) -> dict | None:
    path = CONFIG_DIR / f"{area}_config.yaml"
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


_VARIANCE_DATA_DIR = PROCESSED_DIR / "variance" / "data"
_VARIANCE_STATS_DIR = PROCESSED_DIR / "variance" / "stats"
_RUN_DATA_SCRIPT = PROJECT_ROOT / "analysis" / "eva-bench-stats" / "run_data.py"
_RUN_STATS_SCRIPT = PROJECT_ROOT / "analysis" / "eva-bench-stats" / "run_stats.py"
_VARIANCE_DATA_FILES = [
    "scores.csv",
    "judge_var.csv",
    "trial_var.csv",
    "judge_summary.csv",
    "trial_summary.csv",
    "composite_stability.csv",
    "borderline_scenarios.csv",
]


def _variance_data_ready() -> bool:
    import pandas as pd

    for f in _VARIANCE_DATA_FILES:
        p = _VARIANCE_DATA_DIR / f
        if not p.exists():
            return False
        try:
            if pd.read_csv(p, nrows=1).empty:
                return False
        except Exception:
            return False
    return True


def _run_script(script_path: Path) -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    output = result.stdout + ("\n" + result.stderr if result.stderr else "")
    return result.returncode == 0, output


def overview_page():
    st.header("EVA-Bench Statistics")
    st.markdown("""
Statistical analyses for EVA-Bench.

| Page | Status |
|------|--------|
| Perturbations | Implemented |
| CIs | Placeholder |
| Variance | Placeholder |
| Frontier | Placeholder |
""")


def perturbations_page():
    import pandas as pd
    from plots_perturbations import (
        _DOMAIN_DISPLAY,
        data_coverage_table,
        perturbation_additivity_table,
        perturbation_delta_plot,
        perturbation_overview_plot,
        perturbation_pairwise_pvalue_table,
        perturbation_pairwise_results_table,
        perturbation_pvalue_table,
        perturbation_results_table,
        perturbation_summary_table,
    )

    st.header("Perturbation Tests")
    config = _load_config("perturbations")
    if config is None:
        st.warning(f"Config not found: `{CONFIG_DIR / 'perturbations_config.yaml'}`. Create it to get started.")
        return

    # ── Data coverage ─────────────────────────────────────────────────
    coverage_path = PROCESSED_DIR / "perturbations" / "completeness_report.csv"
    if coverage_path.exists():
        completeness_df = pd.read_csv(coverage_path)
        with st.expander("Data coverage", expanded=False):
            st.caption(
                "n_complete_domains / n_expected_domains per condition. Included = all conditions × all domains complete."
            )
            cov_tbl = data_coverage_table(completeness_df)
            if not cov_tbl.empty:
                st.dataframe(cov_tbl, width="stretch")

    pooled_path = PROCESSED_DIR / "perturbations" / "results_pooled.csv"
    per_domain_path = PROCESSED_DIR / "perturbations" / "results_per_domain.csv"
    if not pooled_path.exists() or not per_domain_path.exists():
        st.info("Statistical results not found.\n\nRun `uv run python analysis/eva-bench-stats/run_stats.py` first.")
        return

    results_pooled = pd.read_csv(pooled_path)
    results_per_domain = pd.read_csv(per_domain_path)
    results_pooled["reject"] = results_pooled["reject"].astype(bool)
    results_per_domain["reject"] = results_per_domain["reject"].astype(bool)

    pairwise_pooled = pairwise_per_domain = None
    additivity_pooled = additivity_per_domain = None
    cld_pooled_df = cld_per_domain_df = None

    _pw_pooled_path = PROCESSED_DIR / "perturbations" / "results_pairwise_pooled.csv"
    _pw_domain_path = PROCESSED_DIR / "perturbations" / "results_pairwise_per_domain.csv"
    _add_pooled_path = PROCESSED_DIR / "perturbations" / "results_additivity_pooled.csv"
    _add_domain_path = PROCESSED_DIR / "perturbations" / "results_additivity_per_domain.csv"
    _cld_pooled_path = PROCESSED_DIR / "perturbations" / "cld_pooled.csv"
    _cld_domain_path = PROCESSED_DIR / "perturbations" / "cld_per_domain.csv"

    _pairwise_available = _pw_pooled_path.exists() and _pw_domain_path.exists()
    _additivity_available = _add_pooled_path.exists() and _add_domain_path.exists()
    _cld_available = _cld_pooled_path.exists() and _cld_domain_path.exists()

    if _pairwise_available:
        pairwise_pooled = pd.read_csv(_pw_pooled_path)
        pairwise_per_domain = pd.read_csv(_pw_domain_path)
        pairwise_pooled["reject"] = pairwise_pooled["reject"].astype(bool)
        pairwise_per_domain["reject"] = pairwise_per_domain["reject"].astype(bool)

    if _additivity_available:
        additivity_pooled = pd.read_csv(_add_pooled_path)
        additivity_per_domain = pd.read_csv(_add_domain_path)
        additivity_pooled["reject"] = additivity_pooled["reject"].astype(bool)
        additivity_per_domain["reject"] = additivity_per_domain["reject"].astype(bool)

    if _cld_available:
        cld_pooled_df = pd.read_csv(_cld_pooled_path)
        cld_per_domain_df = pd.read_csv(_cld_domain_path)

    metrics: list[str] = config.get("metrics", [])
    alpha: float = config.get("alpha", 0.05)

    domains = sorted(results_per_domain["domain"].unique())

    # Sort models: cascade first, then hybrid, then s2s, unknown types last, alphabetical within each group.
    # Ordering is fully driven by the config — set `type:` to cascade / hybrid / s2s when adding models.
    _TYPE_ORDER = {"cascade": 0, "hybrid": 1, "s2s": 2}
    _TYPE_LABEL = {"cascade": "Cascade", "hybrid": "Hybrid", "s2s": "S2S"}
    _model_type = {label: cfg.get("type", "") for label, cfg in config["models"].items()}
    models = sorted(
        results_pooled["model_label"].unique(),
        key=lambda m: (_TYPE_ORDER.get(_model_type.get(m, ""), 99), m),
    )
    # Build boundaries (cumulative counts of present types) and per-region labels.
    _present_types = [t for t in ("cascade", "hybrid", "s2s") if any(_model_type.get(m) == t for m in models)]
    _counts = [sum(1 for m in models if _model_type.get(m) == t) for t in _present_types]
    if len(_present_types) > 1:
        cumulative = []
        running = 0
        for c in _counts[:-1]:
            running += c
            cumulative.append(running)
        group_boundary = cumulative
        group_labels = tuple(_TYPE_LABEL[t] for t in _present_types)
    else:
        group_boundary = None
        group_labels = None

    _METRIC_TAB_LABELS = {
        "EVA-A_pass": "EVA-A (pass@1)",
        "EVA-A_pass@3": "EVA-A (pass@3)",
        "EVA-A_pass^3": "EVA-A (pass^3)",
        "EVA-X_pass": "EVA-X (pass@1)",
        "EVA-X_pass@3": "EVA-X (pass@3)",
        "EVA-X_pass^3": "EVA-X (pass^3)",
    }

    def _tlabel(m: str) -> str:
        return _METRIC_TAB_LABELS.get(m, m)

    # Desired tab order: pass@1s + individual metrics first, other EVA aggregates last
    _MAIN_METRIC_ORDER = [
        "EVA-A_pass",
        "EVA-X_pass",
        "task_completion",
        "agent_speech_fidelity",
        "faithfulness",
        "turn_taking",
        "conciseness",
        "conversation_progression",
    ]
    _OTHER_METRIC_ORDER = [
        "EVA-A_mean",
        "EVA-X_mean",
        "EVA-overall_mean",
        "EVA-A_pass@3",
        "EVA-A_pass^3",
        "EVA-X_pass@3",
        "EVA-X_pass^3",
    ]
    available = set(metrics)
    main_metrics = [m for m in _MAIN_METRIC_ORDER if m in available]
    other_metrics = [m for m in _OTHER_METRIC_ORDER if m in available]
    remaining = [m for m in metrics if m not in set(_MAIN_METRIC_ORDER + _OTHER_METRIC_ORDER)]

    n_main = len(main_metrics)
    n_remaining = len(remaining)
    stats_tab_idx = 2 + n_main

    tab_labels = (
        ["Summary", "Per-model plots"]
        + [_tlabel(m) for m in main_metrics]
        + ["Statistics methods"]
        + [_tlabel(m) for m in remaining]
        + [_tlabel(m) for m in other_metrics]
    )
    tabs = st.tabs(tab_labels)

    def _render_metric_tab(metric: str) -> None:
        sig_note = f"\\* = p < {alpha} after Holm-Bonferroni correction"
        metric_results = pd.concat(
            [
                results_per_domain[results_per_domain["metric"] == metric],
                results_pooled[results_pooled["metric"] == metric],
            ]
        )
        if (
            metric_results.empty
            or metric_results[["ci_lower", "ci_upper", "observed_mean_delta"]].dropna(how="all").empty
        ):
            st.info(f"No results for metric '{metric}'.")
            return
        y_min = min(metric_results["ci_lower"].min(), metric_results["observed_mean_delta"].min())
        y_max = max(metric_results["ci_upper"].max(), metric_results["observed_mean_delta"].max())
        if pd.isna(y_min) or pd.isna(y_max):
            st.info(f"No results for metric '{metric}'.")
            return
        y_range = (math.floor(y_min * 10) / 10 - 0.05, math.ceil(y_max * 10) / 10 + 0.05)

        _cld_pooled_metric = cld_pooled_df[cld_pooled_df["metric"] == metric] if cld_pooled_df is not None else None

        # ── Pooled ────────────────────────────────────────────────────
        st.subheader("Pooled (all domains, 90 scenarios)")
        st.caption(f"{sig_note} across 3 conditions per model.")
        st.plotly_chart(
            perturbation_delta_plot(
                results_pooled,
                metric,
                title="Pooled",
                y_range=y_range,
                model_order=models,
                group_boundary=group_boundary,
                group_labels=group_labels,
                cld_df=_cld_pooled_metric,
            ),
            width="stretch",
        )
        with st.expander("Mean deltas and 95% CIs (pooled)", expanded=False):
            tbl = perturbation_results_table(results_pooled, metric=metric, model_order=models)
            if not tbl.empty:
                st.dataframe(tbl, width="stretch")

        st.divider()

        # ── Per domain ────────────────────────────────────────────────
        st.subheader("Per domain (30 scenarios each)")
        st.caption(f"{sig_note} across 9 tests (3 conditions × 3 domains) per model.")
        for domain in domains:
            domain_results = results_per_domain[results_per_domain["domain"] == domain]
            _cld_domain_metric = (
                cld_per_domain_df[(cld_per_domain_df["metric"] == metric) & (cld_per_domain_df["domain"] == domain)]
                if cld_per_domain_df is not None
                else None
            )
            st.plotly_chart(
                perturbation_delta_plot(
                    domain_results,
                    metric,
                    title=_DOMAIN_DISPLAY.get(domain, domain),
                    y_range=y_range,
                    model_order=models,
                    group_boundary=group_boundary,
                    group_labels=group_labels,
                    cld_df=_cld_domain_metric,
                ),
                width="stretch",
            )
        with st.expander("Mean deltas and 95% CIs (per domain)", expanded=False):
            for domain in domains:
                st.markdown(f"**{_DOMAIN_DISPLAY.get(domain, domain)}**")
                domain_results = results_per_domain[results_per_domain["domain"] == domain]
                tbl = perturbation_results_table(domain_results, metric=metric, model_order=models)
                if not tbl.empty:
                    st.dataframe(tbl, width="stretch")

        st.divider()

        # ── P-values ──────────────────────────────────────────────────
        st.subheader("Significance (corrected p-values)")
        st.caption("Holm-Bonferroni corrected. ✓ = reject H₀ at α = " + str(alpha) + ".")
        tbl_pv = perturbation_pvalue_table(results_pooled, metric=metric, model_order=models)
        if not tbl_pv.empty:
            st.markdown("**Pooled**")
            st.dataframe(tbl_pv, width="stretch")
        with st.expander("Per-domain p-values", expanded=False):
            for domain in domains:
                st.markdown(f"**{_DOMAIN_DISPLAY.get(domain, domain)}**")
                domain_results = results_per_domain[results_per_domain["domain"] == domain]
                tbl = perturbation_pvalue_table(domain_results, metric=metric, model_order=models)
                if not tbl.empty:
                    st.dataframe(tbl, width="stretch")

        # ── Pairwise comparisons (secondary family) ───────────────────────────
        st.divider()
        st.subheader("Pairwise comparisons (secondary family)")
        st.caption(
            "Holm-Bonferroni corrected across 3 pairwise comparisons per model. "
            "CLD letters on the delta plots above identify groups not significantly different from each other."
        )
        if not _pairwise_available:
            st.info("Run `uv run python analysis/eva-bench-stats/run_stats.py` to generate pairwise results.")
        else:
            pw_metric = pairwise_pooled[pairwise_pooled["metric"] == metric]
            tbl_pw = perturbation_pairwise_pvalue_table(pw_metric, metric=metric, model_order=models)
            if not tbl_pw.empty:
                st.markdown("**Pooled**")
                st.dataframe(tbl_pw, width="stretch")
            with st.expander("Mean deltas and 95% CIs — pairwise (pooled)", expanded=False):
                tbl_pw_res = perturbation_pairwise_results_table(pw_metric, metric=metric, model_order=models)
                if not tbl_pw_res.empty:
                    st.dataframe(tbl_pw_res, width="stretch")
            with st.expander("Per-domain pairwise p-values", expanded=False):
                for domain in domains:
                    st.markdown(f"**{_DOMAIN_DISPLAY.get(domain, domain)}**")
                    domain_pw = pairwise_per_domain[
                        (pairwise_per_domain["metric"] == metric) & (pairwise_per_domain["domain"] == domain)
                    ]
                    tbl = perturbation_pairwise_pvalue_table(domain_pw, metric=metric, model_order=models)
                    if not tbl.empty:
                        st.dataframe(tbl, width="stretch")

        # ── Additivity test ────────────────────────────────────────────────────
        st.divider()
        st.subheader("Additivity test (both vs expected sum of accent + background noise)")
        st.caption("Raw p-value, uncorrected — one test per model per metric.")
        if not _additivity_available:
            st.info("Run `uv run python analysis/eva-bench-stats/run_stats.py` to generate additivity results.")
        else:
            add_metric = additivity_pooled[additivity_pooled["metric"] == metric]
            tbl_add = perturbation_additivity_table(add_metric, metric=metric, model_order=models)
            if not tbl_add.empty:
                st.markdown("**Pooled**")
                st.dataframe(tbl_add, width="stretch")
            with st.expander("Per-domain additivity results", expanded=False):
                for domain in domains:
                    st.markdown(f"**{_DOMAIN_DISPLAY.get(domain, domain)}**")
                    domain_add = additivity_per_domain[
                        (additivity_per_domain["metric"] == metric) & (additivity_per_domain["domain"] == domain)
                    ]
                    tbl = perturbation_additivity_table(domain_add, metric=metric, model_order=models)
                    if not tbl.empty:
                        st.dataframe(tbl, width="stretch")

    # ── Summary tab ────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Effect of perturbations — pooled (all domains, 90 scenarios)")
        st.caption(f"\\* = p < {alpha} after Holm-Bonferroni correction across 3 conditions per model.")

        for metric in main_metrics:
            metric_results = pd.concat(
                [
                    results_per_domain[results_per_domain["metric"] == metric],
                    results_pooled[results_pooled["metric"] == metric],
                ]
            )
            y_min = min(metric_results["ci_lower"].min(), metric_results["observed_mean_delta"].min())
            y_max = max(metric_results["ci_upper"].max(), metric_results["observed_mean_delta"].max())
            y_range = (math.floor(y_min * 10) / 10 - 0.05, math.ceil(y_max * 10) / 10 + 0.05)
            st.plotly_chart(
                perturbation_delta_plot(
                    results_pooled,
                    metric,
                    title=_tlabel(metric),
                    y_range=y_range,
                    model_order=models,
                    group_boundary=group_boundary,
                    group_labels=group_labels,
                    cld_df=(cld_pooled_df[cld_pooled_df["metric"] == metric] if cld_pooled_df is not None else None),
                ),
                width="stretch",
            )

        st.divider()
        st.subheader("Significant models per metric (pooled)")
        summary_results = results_pooled[results_pooled["metric"].isin(main_metrics)]
        summary_tbl = perturbation_summary_table(summary_results)
        if not summary_tbl.empty:
            st.dataframe(summary_tbl, width="stretch")

    # ── Per-model plots tab ────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Effect of perturbations across metrics (pooled)")
        st.caption("Pooled results (90 scenarios across all domains). One plot per model.")

        overview_df = results_pooled[results_pooled["metric"].isin(main_metrics)]
        overview_y_min = min(overview_df["observed_mean_delta"].min(), overview_df["ci_lower"].min())
        overview_y_max = max(overview_df["observed_mean_delta"].max(), overview_df["ci_upper"].max())
        overview_y_range = (math.floor(overview_y_min * 10) / 10 - 0.05, math.ceil(overview_y_max * 10) / 10 + 0.05)

        layout = st.radio(
            "Group bars by",
            options=[
                "Metric (x-axis) — compare conditions within each metric",
                "Condition (x-axis) — compare metrics within each condition",
            ],
            horizontal=True,
            label_visibility="collapsed",
        )
        group_by = "metric" if layout.startswith("Metric") else "condition"

        for model in models:
            fig = perturbation_overview_plot(
                overview_df,
                model,
                group_by=group_by,
                y_range=overview_y_range,
                metric_order=main_metrics,
            )
            st.plotly_chart(fig, width="stretch")

    # ── Main metric tabs ───────────────────────────────────────────────
    for tab, metric in zip(tabs[2 : 2 + n_main], main_metrics):
        with tab:
            _render_metric_tab(metric)

    # ── Statistics methods tab ─────────────────────────────────────────
    with tabs[stats_tab_idx]:
        st.header("Statistical methods")
        st.write("""
Perturbation robustness is assessed by comparing each system's score on perturbed speech
(accented, noisy, or both) to its own clean-speech baseline, using paired scenario-level
deltas and a sign-flip permutation test with bootstrap confidence intervals.
""")

        with st.expander("Data preparation", expanded=False):
            st.markdown(f"""
**Unit of analysis: scenario-level mean delta.**

1. For each scenario, compute the mean score across trials separately for the clean
   baseline (k ≈ 5 trials) and each perturbation condition (k = 3 trials).
2. Match clean and perturbed means by `scenario_id` (inner join).
   Because the clean run includes more scenarios than the perturbation run,
   only the {config.get("expected_scenarios", 30)} scenarios present in both sets are retained.
3. Delta = perturbed mean − clean mean.
   Negative delta = score drops under perturbation.
4. Each (model, condition, domain) combination yields {config.get("expected_scenarios", 30)} paired deltas —
   one per scenario.

**Completeness criteria for inclusion:**
A model is included only if all three conditions (Accent, Background noise, Accent +
Background noise) are fully present in all three domains with exactly
{config.get("expected_scenarios", 30)} scenarios and {config.get("expected_pert_trials", 3)} trials each.
Models failing this check are reported in the *Data coverage* expander and excluded.

*Implemented in:* `data_perturbations.py` — `compute_scenario_means`, `compute_deltas`,
`check_model_completeness`.
""")

        with st.expander("Sign-flip permutation test", expanded=False):
            st.markdown(f"""
**Test:** Two-sided paired sign-flip permutation test.
**H₀:** The mean perturbation effect (delta) is zero.
**H₁:** The mean perturbation effect is non-zero (two-sided).

**Why sign-flip permutation?**
Under H₀, flipping the sign of any delta is equally likely — there is no systematic
direction. No distributional assumption is required. The permutation distribution of
the mean is exact and makes no assumption about normality, which matters because
score distributions are often bounded and skewed.

**Calculation steps:**
1. Collect the {config.get("expected_scenarios", 30)} scenario-level deltas for a given
   (model, condition, domain).
2. Compute the observed mean delta.
3. Generate {config.get("n_permutations", 10000):,} permutations: for each, independently
   flip each delta's sign with probability 0.5 using a fixed RNG seed.
4. Compute the mean of each permuted set.
5. p-value = fraction of permuted means where |permuted mean| ≥ |observed mean|.

*Implemented in:* `stats_perturbations.py` — `permutation_test` (lines 44–75).
""")

        with st.expander("Bootstrap confidence intervals", expanded=False):
            st.markdown(f"""
**Method:** Percentile bootstrap, resampling across scenarios.

**Why bootstrap?**
Scenario-level deltas are not i.i.d. normal (bounded scores, scenario difficulty
variation). The bootstrap makes no distributional assumption and provides an interval
on the sampling variability of the mean delta.

**Calculation steps:**
1. Resample the {config.get("expected_scenarios", 30)} deltas with replacement,
   {config.get("n_bootstrap", 1000):,} times, using a fixed per-cell RNG seed.
2. Compute the mean of each resample.
3. CI = [2.5th percentile, 97.5th percentile] of the {config.get("n_bootstrap", 1000):,} bootstrap means.

**Interpretation:**
95% CI on the mean perturbation effect across the scenario sample. A CI that
excludes 0 is consistent with a real effect, but statistical significance is
determined by the permutation test (corrected), not the CI.

*Implemented in:* `stats_perturbations.py` — `bootstrap_ci` (lines 78–105).
""")

        with st.expander("Multiple testing correction: Holm-Bonferroni", expanded=False):
            st.markdown(f"""
**Method:** Holm-Bonferroni step-down procedure
(`statsmodels.stats.multitest.multipletests`, method=`'holm'`).
**Controls:** Family-wise error rate (FWER) at α = {alpha}.

**Why Holm-Bonferroni over Bonferroni?**
Bonferroni divides α by m (number of tests) for all hypotheses uniformly.
Holm-Bonferroni is uniformly more powerful: it step-down adjusts each p-value
in sequence, so tests with very small p-values benefit from weaker correction
while FWER is still controlled at α.

**Correction families:**

| View | Tests per family | Reasoning |
|------|-----------------|-----------|
| Pooled (all domains) | 3 (one per condition) | Domains are pooled; only conditions vary |
| Per domain | 9 (3 conditions × 3 domains) | Both conditions and domains vary |

Each family is defined per (model, metric). Correction is applied within the family;
models and metrics are independent hypotheses and are not pooled further.

*Implemented in:* `stats_perturbations.py` — `run_analysis` (lines 108–202).
""")

        with st.expander("Seed strategy", expanded=False):
            st.markdown(f"""
All random operations use a deterministic per-cell seed derived from the global
seed ({config.get("random_seed", 42)}) plus a hash of the cell's identity
(model, condition, domain). This makes every individual test reproducible without
all cells sharing the same seed — which would inadvertently correlate their
permutation draws.

```python
cell_seed = seed + hash(f"{{group_meta}}:{{cond}}:{{domain}}") % (2**31)
```

*Implemented in:* `stats_perturbations.py` — `run_analysis` (line 171).
""")

    # ── Remaining metric tabs (between Statistics methods and other EVA) ─
    remaining_start = stats_tab_idx + 1
    for tab, metric in zip(tabs[remaining_start : remaining_start + n_remaining], remaining):
        with tab:
            _render_metric_tab(metric)

    # ── Other EVA metric tabs ──────────────────────────────────────────
    other_start = remaining_start + n_remaining
    for tab, metric in zip(tabs[other_start:], other_metrics):
        with tab:
            _render_metric_tab(metric)


def CIs_page():
    import pandas as pd
    from plots_CIs import (
        forest_plot,
        metric_label,
        order_metrics,
        paper_summary_table,
    )

    def download_button(df: pd.DataFrame, filename: str) -> None:
        st.download_button("Download CSV", df.to_csv(index=False).encode(), filename, "text/csv")

    st.header("Confidence Intervals")
    st.caption(
        "95% percentile bootstrap CIs over scenario-level means. "
        "Pooled estimates use equal weighting across the 3 domains. "
        "Bootstrap resample count is selected by a 1k-vs-2k stability check (threshold 0.002 on the metric scale)."
    )

    config = _load_config("CIs")
    if config is None:
        st.warning(f"Config not found: `{CONFIG_DIR / 'CIs_config.yaml'}`. Create it to get started.")
        return

    output_dir = PROJECT_ROOT / config["output_dir"]
    stats_dir = output_dir / "stats"
    data_dir = output_dir / "data"

    per_domain_path = stats_dir / "results_per_domain.csv"
    pooled_path = stats_dir / "results_pooled.csv"
    stability_path = stats_dir / "stability_log.csv"
    completeness_path = data_dir / "completeness_report.csv"

    if not per_domain_path.exists() or not pooled_path.exists():
        st.warning(
            f"CI outputs not found in `{stats_dir}`. "
            f"Run the pipeline (Run pipeline expander in the sidebar) or "
            f"`uv run python analysis/eva-bench-stats/run_stats.py`."
        )
        return

    per_domain_df = pd.read_csv(per_domain_path)
    pooled_df = pd.read_csv(pooled_path)
    stability_df = pd.read_csv(stability_path) if stability_path.exists() else pd.DataFrame()
    completeness_df = pd.read_csv(completeness_path) if completeness_path.exists() else pd.DataFrame()

    # Color map by system_type
    type_colors = {"cascade": "#3B82F6", "s2s": "#CC61B0", "hybrid": "#A855F7"}
    model_to_color: dict[str, str] = {}
    for label, model_cfg in (config.get("models") or {}).items():
        model_to_color[label] = type_colors.get(model_cfg.get("type", "cascade"), "#3B82F6")

    with st.expander("Stability check", expanded=False):
        if stability_df.empty:
            st.info("No stability log available.")
        else:
            st.dataframe(stability_df, use_container_width=True)
            download_button(stability_df, "stability_log.csv")

    with st.expander("Paper-ready summary (EVA-A & EVA-X means)", expanded=False):
        summary = paper_summary_table(
            per_domain_df,
            pooled_df,
            metrics=("EVA-A_mean", "EVA-X_mean"),
            domains=tuple(config["expected_domains"]),
        )
        st.dataframe(summary, use_container_width=True)
        download_button(summary, "ci_summary_paper.csv")

    with st.expander("Completeness report", expanded=False):
        if completeness_df.empty:
            st.info("No completeness report available.")
        else:
            st.dataframe(completeness_df, use_container_width=True)
            download_button(completeness_df, "completeness_report.csv")

    metrics_present = sorted(set(per_domain_df["metric"]) | set(pooled_df["metric"]))
    metrics_ordered = order_metrics(metrics_present)
    if not metrics_ordered:
        st.info("No metrics to display.")
        return

    tabs = st.tabs([metric_label(m) for m in metrics_ordered])
    for tab, metric in zip(tabs, metrics_ordered):
        with tab:
            st.plotly_chart(
                forest_plot(
                    per_domain_df,
                    pooled_df,
                    metric=metric,
                    domains=config["expected_domains"],
                    color_map=model_to_color,
                ),
                use_container_width=True,
            )
            combined = pd.concat(
                [
                    per_domain_df[per_domain_df["metric"] == metric],
                    pooled_df[pooled_df["metric"] == metric],
                ],
                ignore_index=True,
            )[["model_label", "domain", "n", "point_estimate", "ci_lower", "ci_upper"]]
            st.dataframe(combined, use_container_width=True)
            download_button(combined, f"ci_{metric}.csv")


def variance_page():
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from plots_utils import fmt_p
    from plots_variance import (
        clean_composite_label,
        composite_sort_key,
        composite_stability_fig,
        deep_dive_scatter_fig,
        icc_bar_centered_fig,
        icc_bar_per_model_fig,
        icc_bar_twoway_fig,
        icc_heatmap_fig,
        llm_name,
        variance_bar_fig,
        variance_histogram,
    )

    st.header("Variance Analysis")
    config = _load_config("variance")
    if config is None:
        st.warning(f"Config not found: `{CONFIG_DIR / 'variance_config.yaml'}`. Create it to get started.")
        return

    runs: dict = config.get("runs", {})
    metrics: list[str] = config.get("metrics", [])

    # ── Color infrastructure ──────────────────────────────────────────────────
    _CASCADE_COLORS = ["#3B82F6", "#99C945", "#A855F7"]
    _S2S_COLORS = ["#CC61B0", "#F97316", "#EAB308"]
    RUN_COLOR_MAP: dict[str, str] = {}
    RUN_SYMBOL_MAP: dict[str, str] = {}
    RUN_GROUP_MAP: dict[str, str] = {}
    cascade_labels = [lbl for lbl, cfg in runs.items() if cfg.get("type") == "cascade"]
    s2s_labels = [lbl for lbl, cfg in runs.items() if cfg.get("type") != "cascade"]
    for i, lbl in enumerate(cascade_labels):
        RUN_COLOR_MAP[lbl] = _CASCADE_COLORS[i % len(_CASCADE_COLORS)]
        RUN_SYMBOL_MAP[lbl] = "circle"
        RUN_GROUP_MAP[lbl] = "cascade"
    for i, lbl in enumerate(s2s_labels):
        RUN_COLOR_MAP[lbl] = _S2S_COLORS[i % len(_S2S_COLORS)]
        RUN_SYMBOL_MAP[lbl] = "star"
        RUN_GROUP_MAP[lbl] = "s2s"

    # Model-id-keyed views (run_label minus the trailing ' — domain' suffix).
    # Used for plots/tables that consume the per-domain stats CSVs, where the
    # `model` column is a model_id rather than a (model × domain) compound.
    def _strip_domain(lbl: str) -> str:
        return lbl.split(" — ")[0].strip()

    MODEL_COLOR_MAP: dict[str, str] = {}
    cascade_models: list[str] = []
    s2s_models: list[str] = []
    for lbl in cascade_labels:
        mid = _strip_domain(lbl)
        if mid not in MODEL_COLOR_MAP:
            MODEL_COLOR_MAP[mid] = RUN_COLOR_MAP[lbl]
            cascade_models.append(mid)
    for lbl in s2s_labels:
        mid = _strip_domain(lbl)
        if mid not in MODEL_COLOR_MAP:
            MODEL_COLOR_MAP[mid] = RUN_COLOR_MAP[lbl]
            s2s_models.append(mid)
    MODEL_LABEL_ORDER: list[str] = cascade_models + s2s_models

    # ── Output paths ──────────────────────────────────────────────────────────
    data_dir = _VARIANCE_DATA_DIR
    stats_dir = _VARIANCE_STATS_DIR
    data_ready = _variance_data_ready()

    if not data_ready:
        st.warning("Processed data not found. Use **Run pipeline** in the sidebar to get started.")
        return

    # ── Load all CSVs ─────────────────────────────────────────────────────────
    scores_df = pd.read_csv(data_dir / "scores.csv")
    judge_var = pd.read_csv(data_dir / "judge_var.csv")
    trial_var = pd.read_csv(data_dir / "trial_var.csv")
    judge_summary = pd.read_csv(data_dir / "judge_summary.csv")
    trial_summary = pd.read_csv(data_dir / "trial_summary.csv")
    stability_df = pd.read_csv(data_dir / "composite_stability.csv")
    borderlines_df = pd.read_csv(data_dir / "borderline_scenarios.csv")

    def _read_stat(name: str) -> pd.DataFrame:
        p = stats_dir / name
        if not p.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(p)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    # Pooled-across-domains stats (Q0, Q1a, Q1b primary, ICC pooled_twoway)
    icc_tw = _read_stat("icc_pooled_twoway.csv")
    q0_judge_pooled = _read_stat("q0_judge_pooled.csv")
    q0_judge_per_model = _read_stat("q0_judge_per_model.csv")
    q0_trial_pooled = _read_stat("q0_trial_pooled.csv")
    q0_trial_per_model = _read_stat("q0_trial_per_model.csv")
    q1a_pooled = _read_stat("q1a.csv")
    q1b_pooled = _read_stat("q1b.csv")

    _TYPE_LABELS_ALL = {"cascade": "Cascade", "s2s": "S2S / audio-native"}
    within_type_results: dict[str, dict[str, pd.DataFrame]] = {}
    for rtype in ("cascade", "s2s"):
        wt: dict[str, pd.DataFrame] = {}
        for key in ("q2_kw", "q2_pairwise", "q3_kw", "q3_pairwise"):
            wt[key] = _read_stat(f"within_type_{rtype}_{key}.csv")
        if not wt["q2_kw"].empty or not wt["q3_kw"].empty:
            within_type_results[rtype] = wt

    # Shared axis style constant (matches plots_variance module)
    _axis_style = {"title_font": {"color": "#222"}, "tickfont": {"color": "#222"}}

    _raw_var_max = max(
        judge_var["std"].max() if not judge_var.empty else 0.0,
        trial_var["std"].max() if not trial_var.empty else 0.0,
    )
    global_var_ymax = _raw_var_max * 1.25

    _pass_thresholds: dict[str, float] = {}
    if not borderlines_df.empty:
        for _, row in borderlines_df.drop_duplicates("metric").iterrows():
            _pass_thresholds[row["metric"]] = float(row["threshold"])

    def download_button(df: pd.DataFrame, filename: str) -> None:
        st.download_button("Download CSV", df.to_csv(index=False).encode(), filename, "text/csv")

    # ── Domain selector ───────────────────────────────────────────────────────
    # Single-choice radio. Drives which within-domain CSV is loaded for
    # Q2/Q3/ICC views and which slice of the data CSVs is shown. Pooled stats
    # (Q0, Q1a, Q1b primary) are computed across all domains and don't move
    # with this selector.
    all_domains = sorted({cfg.get("domain") for cfg in runs.values() if cfg.get("domain")})
    selected_domain: str | None = None
    if all_domains:
        selected_domain = st.sidebar.radio(
            "Domain",
            options=all_domains,
            index=0,
            help=(
                "Selects which domain's within-domain stats and data slices are displayed. "
                "Pooled-across-domains stats (Q0, Q1a, Q1b primary) ignore this and always "
                "show the all-domains answer."
            ),
        )

        def _filter_domain(df: pd.DataFrame) -> pd.DataFrame:
            return df[df["domain"] == selected_domain] if "domain" in df.columns else df

        scores_df = _filter_domain(scores_df)
        judge_var = _filter_domain(judge_var)
        trial_var = _filter_domain(trial_var)
        judge_summary = _filter_domain(judge_summary)
        trial_summary = _filter_domain(trial_summary)
        stability_df = _filter_domain(stability_df)
        borderlines_df = _filter_domain(borderlines_df)

        # Within a single domain, the run_label is uniquely identified by its
        # model prefix. Strip the trailing ' — domain' so the run_label column
        # equals the model_id used in the per-domain stats CSVs.
        for _df in (judge_var, trial_var, judge_summary, trial_summary, stability_df, borderlines_df):
            if "run_label" in _df.columns and not _df.empty:
                _df["run_label"] = _df["run_label"].map(_strip_domain)

    # Load within-domain stats for the selected domain.
    _d_suffix = f"_{selected_domain}" if selected_domain else ""
    icc_pm = _read_stat(f"icc_per_model{_d_suffix}.csv")
    icc_pc = _read_stat(f"icc_pooled_centered{_d_suffix}.csv")
    q1a_within = _read_stat(f"q1a{_d_suffix}.csv") if selected_domain else pd.DataFrame()
    q1b_within = _read_stat(f"q1b{_d_suffix}.csv") if selected_domain else pd.DataFrame()
    q2_kw = _read_stat(f"q2_kw{_d_suffix}.csv")
    q2_pw = _read_stat(f"q2_pairwise{_d_suffix}.csv")
    q3_kw = _read_stat(f"q3_kw{_d_suffix}.csv")
    q3_pw = _read_stat(f"q3_pairwise{_d_suffix}.csv")
    # Q1a/Q1b pooled remain the primary view even when a domain is selected.
    q1a = q1a_pooled
    q1b = q1b_pooled
    _domain_label = f"Domain: **{selected_domain}**" if selected_domain else "All data"
    _pooled_label = "Pooled across all 3 domains"

    # ── 11 Tabs ───────────────────────────────────────────────────────────────
    tabs = st.tabs(
        [
            "Overview",
            "Variance overview",
            "Judge vs. trial variance",
            "Judge variance",
            "Trial variance",
            "EVA score stability",
            "Borderline scenarios",
            "Intraclass correlation",
            "Variance budget",
            "Per-metric deep dive",
            "Statistical tests",
        ]
    )

    # ── Tab 0: Overview ───────────────────────────────────────────────────────
    with tabs[0]:
        st.header("Overview")
        st.write("""
        This study measures three sources of variance in EVA metric scores:
        - **Judge variance** (stochasticity): the LLM judge producing different outputs when
          re-evaluating the *same* conversation (same audio, transcript, tool calls).
        - **Trial variance**: genuine differences in how conversations unfold across trials
          (different simulations of the same scenario).
        - **Scenario variance (ICC)**: how well scores differentiate between scenarios vs.
          within-scenario noise.

        Judge variance and trial variance are isolated together in one experiment (N iterations ×
        M trials per run). ICC is a separate but related calculation derived from the same data — it answers what fraction of the total variance is explained by scenario identity.
        """)

        meta_rows = []
        for lbl, cfg in runs.items():
            meta_rows.append({"run_label": lbl, **{k: v for k, v in cfg.items() if k != "run_id"}})
        st.subheader("Runs in this analysis")
        st.dataframe(pd.DataFrame(meta_rows), width="stretch")

        st.subheader("Metric reference")
        st.caption(
            "Scoring details as of 2026-04-27. Pass thresholds are read live from EVA_COMPOSITES "
            "(src/eva/metrics/aggregation.py) and update automatically. Verify metric descriptions "
            "and normalization against current prompt definitions if rubrics have changed."
        )
        _metric_rows = [
            (
                "faithfulness",
                "1, 2, or 3 per conversation",
                "5 faithfulness dimensions: fabricating tool parameters, misrepresenting tool results, "
                "violating policies, failing to disambiguate, hallucination",
                "1→0.0, 2→0.5, 3→1.0",
                "Mean of per-conversation normalized scores across all record × trial pairs",
            ),
            (
                "agent_speech_fidelity",
                "0 or 1 per turn",
                "Whether TTS output faithfully reproduced intended text, especially key entities (codes, amounts, names)",
                "fraction of turns scored 1 (0 to 1)",
                "Mean of per-conversation normalized scores across all record × trial pairs",
            ),
            (
                "conversation_progression",
                "1, 2, or 3 per conversation",
                "4 progression dimensions: unnecessary tool calls, information loss, "
                "redundant statements, question quality",
                "1→0.0, 2→0.5, 3→1.0",
                "Mean of per-conversation normalized scores across all record × trial pairs",
            ),
            (
                "conciseness",
                "1, 2, or 3 per turn, then averaged",
                "Whether assistant responses were voice-appropriately brief and easy to digest",
                "mean of (1→0.0, 2→0.5, 3→1.0) across all turns",
                "Mean of per-conversation normalized scores across all record × trial pairs",
            ),
        ]
        _th = "text-align:left;padding:7px 14px;border-bottom:2px solid rgba(128,128,128,0.35);white-space:nowrap;font-weight:600"
        _td_base = "padding:7px 14px;vertical-align:top;border-bottom:1px solid rgba(128,128,128,0.15)"
        _td_nowrap = f"{_td_base};white-space:nowrap"
        _td_wrap = f"{_td_base};min-width:220px"
        _rows_html = "".join(
            f"<tr>"
            f"<td style='{_td_nowrap}'>{r[0]}</td>"
            f"<td style='{_td_nowrap}'>{r[1]}</td>"
            f"<td style='{_td_wrap}'>{r[2]}</td>"
            f"<td style='{_td_nowrap}'>{r[3]}</td>"
            f"<td style='{_td_wrap}'>{r[4]}</td>"
            f"</tr>"
            for r in _metric_rows
        )
        st.html(f"""
        <table style="width:100%;border-collapse:collapse;font-size:14px;font-family:inherit">
          <thead><tr>
            <th style="{_th}">metric</th>
            <th style="{_th}">raw score values</th>
            <th style="{_th}">what it measures</th>
            <th style="{_th}">normalization</th>
            <th style="{_th}">model-level aggregation</th>
          </tr></thead>
          <tbody>{_rows_html}</tbody>
        </table>
        """)

        st.subheader("Sample sizes")
        st.caption(
            "Judge variance n = number of (record, trial) pairs per model "
            "(each pair contributes one std dev across the 3 judge iterations). "
            "Trial variance n = number of records per model "
            "(each record contributes one std dev across the 3 simulation trials)."
        )
        if not judge_summary.empty and not trial_summary.empty:
            _n_judge = (
                judge_summary.groupby("run_label")["n"]
                .max()
                .reset_index()
                .rename(columns={"run_label": "model", "n": "judge variance n (record×trial pairs)"})
            )
            _n_trial = (
                trial_summary.groupby("run_label")["n"]
                .max()
                .reset_index()
                .rename(columns={"run_label": "model", "n": "trial variance n (records)"})
            )
            st.dataframe(_n_judge.merge(_n_trial, on="model"), width="stretch")

    # ── Tab 1: Variance overview ──────────────────────────────────────────────
    with tabs[1]:
        st.header("Variance overview")
        if selected_domain:
            st.caption(f"Plots show {_domain_label}; statistical tests below are {_pooled_label}.")
        st.write("""
        High-level view of how much variance exists in each metric, and whether it is
        statistically distinguishable from zero. Judge variance (stochasticity) and trial
        variance (conversation-to-conversation) are shown on a shared scale for direct comparison.
        """)

        st.subheader("Judge vs. trial variance per metric (averaged across models)")
        st.caption(
            "Each bar is the mean std dev for that variance type and metric, averaged across "
            "all models. Error bars show the spread (std dev) across models."
        )

        _j_agg = (
            judge_summary.groupby("metric")["mean_std"]
            .agg(mean="mean", std="std")
            .reset_index()
            .assign(variance_type="Judge (stochasticity)")
        )
        _t_agg = (
            trial_summary.groupby("metric")["mean_std"]
            .agg(mean="mean", std="std")
            .reset_index()
            .assign(variance_type="Trial (conversation-to-conversation)")
        )
        _vov_df = pd.concat([_j_agg, _t_agg], ignore_index=True)
        _metric_order_vov = sorted(_vov_df["metric"].unique())

        _vov_fig = px.bar(
            _vov_df,
            x="metric",
            y="mean",
            color="variance_type",
            barmode="group",
            error_y="std",
            category_orders={"metric": _metric_order_vov},
            color_discrete_map={
                "Judge (stochasticity)": "#636EFA",
                "Trial (conversation-to-conversation)": "#EF553B",
            },
            labels={"mean": "Mean std dev", "metric": "Metric", "variance_type": "Variance type"},
        )
        _vov_fig.update_layout(
            yaxis_range=[0, global_var_ymax],
            legend_title_text="Variance type",
            legend={"orientation": "v", "yanchor": "middle", "y": 0.5, "xanchor": "left", "x": 1.02},
            margin={"r": 260},
        )
        _vov_fig.update_xaxes(**_axis_style)
        _vov_fig.update_yaxes(**_axis_style)
        st.plotly_chart(_vov_fig, width="stretch")

        _jt_table = pd.merge(
            judge_summary.groupby("metric")["mean_std"].mean().rename("mean judge std"),
            trial_summary.groupby("metric")["mean_std"].mean().rename("mean trial std"),
            left_index=True,
            right_index=True,
        ).reset_index()
        _jt_table["trial / judge"] = (_jt_table["mean trial std"] / _jt_table["mean judge std"]).round(1)
        _jt_table = _jt_table.sort_values("trial / judge", ascending=False).reset_index(drop=True)
        st.caption("Mean std dev averaged across models; ratio = trial ÷ judge.")
        st.dataframe(_jt_table.round({"mean judge std": 3, "mean trial std": 3}), hide_index=True, width="stretch")

        st.subheader("Is variance significantly greater than zero?")
        st.write(
            "One-sample Wilcoxon signed-rank test against 0 (one-sided, H₁: median > 0), "
            "pooled across all models per metric. "
            "A significant result means the variance is reliably non-zero — not just noise."
        )

        if not q0_judge_pooled.empty and not q0_trial_pooled.empty:
            _sig_map = {True: "✓ yes", False: "✗ no"}
            _q0_summary = pd.merge(
                q0_judge_pooled[["metric", "median_std", "p", "significant"]].rename(
                    columns={"median_std": "judge median std", "p": "judge p", "significant": "judge sig?"}
                ),
                q0_trial_pooled[["metric", "median_std", "p", "significant"]].rename(
                    columns={"median_std": "trial median std", "p": "trial p", "significant": "trial sig?"}
                ),
                on="metric",
            ).sort_values("metric")
            _q0_summary["judge p"] = _q0_summary["judge p"].map(fmt_p)
            _q0_summary["trial p"] = _q0_summary["trial p"].map(fmt_p)
            _q0_summary["judge sig?"] = _q0_summary["judge sig?"].map(_sig_map)
            _q0_summary["trial sig?"] = _q0_summary["trial sig?"].map(_sig_map)
            st.dataframe(
                _q0_summary.round({"judge median std": 4, "trial median std": 4}),
                hide_index=True,
                width="stretch",
            )

        with st.expander("Methodology"):
            st.markdown("""
**Why one-sample Wilcoxon against 0?**
Std devs are bounded at zero and right-skewed, so a t-test against 0 is inappropriate.
The Wilcoxon signed-rank test is a non-parametric alternative that ranks the absolute
values of the non-zero observations and tests whether they tend to be positive.

**Calculation steps:**
1. For each metric, collect all per-(record,trial) judge std devs (or per-record trial std devs)
   across all selected models.
2. Remove exact zeros (zero_method equivalent: only non-zero values are ranked).
3. Run `scipy.stats.wilcoxon(vals, alternative="greater")` — tests H₁: median > 0.
4. A significant result (p < 0.05) means variance is reliably above zero for that metric.

**Note:** This is not corrected for multiple comparisons across metrics, as each metric is
treated as a separate analysis question.
""")

    # ── Tab 2: Judge vs. trial variance ──────────────────────────────────────
    with tabs[2]:
        st.header("Judge vs. trial variance")
        if selected_domain:
            st.caption(
                f"Plots show {_domain_label}. Q1a/Q1b shown below are **{_pooled_label}** "
                "(per model, pooling its data across all 3 domains)."
            )
        st.write("""
        **What this measures:** Side-by-side comparison of judge variance (stochasticity from
        re-running the same judge) vs. trial variance (differences across simulation trials).

        **Why it matters:** Answers the core question — which source of variance dominates?
        If judge variance is much larger than trial variance, the benchmark results are noisy;
        if trial variance dominates, the noise comes from the simulation itself.
        """)

        combined = pd.merge(
            judge_summary[["run_id", "run_label", "metric", "mean_std", "std_of_std"]].rename(
                columns={"mean_std": "judge_std", "std_of_std": "judge_std_err"}
            ),
            trial_summary[["run_id", "run_label", "metric", "mean_std", "std_of_std"]].rename(
                columns={"mean_std": "trial_std", "std_of_std": "trial_std_err"}
            ),
            on=["run_id", "run_label", "metric"],
        )

        COLORS = {
            "Judge (stochasticity)": "#636EFA",
            "Trial (across scenarios)": "#EF553B",
        }

        st.subheader("Judge vs. trial variance across models")
        st.caption(
            "Error bars = std dev of per-(record,trial) std devs across the group. "
            "Asterisks indicate a significant difference between judge and trial variance "
            "for that model × metric (paired Wilcoxon signed-rank, Bonferroni-corrected): "
            "\\* p < 0.05, \\*\\* p < 0.01, \\*\\*\\* p < 0.001."
        )

        run_labels_cmp = list(combined["run_label"].unique())
        fig = make_subplots(rows=len(run_labels_cmp), cols=1, subplot_titles=run_labels_cmp, shared_xaxes=True)
        for row_idx, run_label in enumerate(run_labels_cmp, start=1):
            run_data = combined[combined["run_label"] == run_label]
            for source, color in COLORS.items():
                is_judge = source == "Judge (stochasticity)"
                fig.add_trace(
                    go.Bar(
                        name=source,
                        x=run_data["metric"],
                        y=run_data["judge_std"] if is_judge else run_data["trial_std"],
                        error_y={
                            "type": "data",
                            "array": (run_data["judge_std_err"] if is_judge else run_data["trial_std_err"]).fillna(0),
                        },
                        marker_color=color,
                        legendgroup=source,
                        showlegend=(row_idx == 1),
                    ),
                    row=row_idx,
                    col=1,
                )
            fig.update_yaxes(range=[0, global_var_ymax], row=row_idx, col=1)

        q1a_sig = q1a[q1a["significant"]] if not q1a.empty else pd.DataFrame()
        y_pad = global_var_ymax * 0.04
        for row_idx, run_label in enumerate(run_labels_cmp, start=1):
            run_data = combined[combined["run_label"] == run_label]
            xref_str = "x" if row_idx == 1 else f"x{row_idx}"
            yref_str = "y" if row_idx == 1 else f"y{row_idx}"
            # q1a's `model` is the model_id (run_label minus trailing ' — domain').
            model_id = run_label.split(" — ")[0].strip()
            model_sig = q1a_sig[q1a_sig["model"] == model_id] if not q1a_sig.empty else pd.DataFrame()
            for _, qrow in model_sig.iterrows():
                metric = qrow["metric"]
                mdata = run_data[run_data["metric"] == metric]
                if mdata.empty:
                    continue
                y_top = max(
                    mdata["judge_std"].iloc[0] + mdata["judge_std_err"].fillna(0).iloc[0],
                    mdata["trial_std"].iloc[0] + mdata["trial_std_err"].fillna(0).iloc[0],
                )
                p_bonf = qrow["p_bonferroni"]
                stars = "***" if p_bonf < 0.001 else "**" if p_bonf < 0.01 else "*"
                fig.add_annotation(
                    x=metric,
                    y=y_top + y_pad,
                    text=f"<b>{stars}</b>",
                    showarrow=False,
                    xref=xref_str,
                    yref=yref_str,
                    font={"size": 14, "color": "#444"},
                )

        fig.update_layout(
            barmode="group",
            height=200 * max(len(run_labels_cmp), 1),
            legend={"orientation": "v", "yanchor": "middle", "y": 0.5, "xanchor": "left", "x": 1.02},
            margin={"t": 40, "r": 160},
        )
        for ann in fig.layout.annotations:
            if ann.yref == "paper":
                ann.font.size = 12
        st.plotly_chart(fig, width="stretch")

        display_combined = combined.copy()
        st.dataframe(display_combined.round(4), width="stretch")
        download_button(display_combined, "variance_comparison.csv")

        judge_dom = combined[combined["judge_std"] > combined["trial_std"]]
        if not judge_dom.empty:
            st.warning(
                f"**Key finding:** Judge variance exceeds trial variance for "
                f"{len(judge_dom)} metric/run combinations: "
                f"{', '.join(judge_dom['metric'].unique())}."
            )
        else:
            st.info("**Key finding:** Trial variance exceeds judge variance for all metric/run combinations.")

        st.subheader("Statistical test: Is judge variance significantly different from trial variance?")
        st.caption(f"**Primary view: {_pooled_label}** — each model's data is pooled across all 3 task domains.")
        if q1a.empty:
            st.info("Not enough data to run statistical tests (need ≥ 5 paired records per model × metric).")
        else:
            st.markdown("**Q1a — Paired Wilcoxon signed-rank test (per model × metric, Bonferroni-corrected)**")
            q1a_disp = q1a[
                [
                    "metric",
                    "model",
                    "n_records",
                    "median_judge_std",
                    "median_trial_std",
                    "median_delta",
                    "p_bonferroni",
                    "significant",
                    "direction",
                ]
            ].copy()
            q1a_disp["model"] = q1a_disp["model"].apply(llm_name)
            q1a_disp["p_bonferroni"] = q1a_disp["p_bonferroni"].map(fmt_p)
            st.dataframe(
                q1a_disp.round({"median_judge_std": 4, "median_trial_std": 4, "median_delta": 4}), width="stretch"
            )

            if not q1b.empty:
                st.markdown("**Q1b — Does the gap vary by model? (Kruskal-Wallis on per-record deltas)**")
                q1b_disp = q1b[["metric", "H", "p", "significant"]].copy()
                q1b_disp["H"] = q1b_disp["H"].round(2)
                q1b_disp["p"] = q1b_disp["p"].map(fmt_p)
                st.dataframe(q1b_disp, width="stretch")

            if selected_domain and (not q1a_within.empty or not q1b_within.empty):
                with st.expander(f"Within-domain drill-down ({_domain_label})"):
                    st.caption(
                        "Same Q1a and Q1b tests, but restricted to the selected domain only. "
                        "Use this to check whether the pooled conclusion holds inside this domain."
                    )
                    if not q1a_within.empty:
                        st.markdown("**Q1a — within-domain**")
                        q1a_w_disp = q1a_within[
                            [
                                "metric",
                                "model",
                                "n_records",
                                "median_judge_std",
                                "median_trial_std",
                                "median_delta",
                                "p_bonferroni",
                                "significant",
                                "direction",
                            ]
                        ].copy()
                        q1a_w_disp["model"] = q1a_w_disp["model"].apply(llm_name)
                        q1a_w_disp["p_bonferroni"] = q1a_w_disp["p_bonferroni"].map(fmt_p)
                        st.dataframe(
                            q1a_w_disp.round({"median_judge_std": 4, "median_trial_std": 4, "median_delta": 4}),
                            width="stretch",
                        )
                    if not q1b_within.empty:
                        st.markdown("**Q1b — within-domain**")
                        q1b_w_disp = q1b_within[["metric", "H", "p", "significant"]].copy()
                        q1b_w_disp["H"] = q1b_w_disp["H"].round(2)
                        q1b_w_disp["p"] = q1b_w_disp["p"].map(fmt_p)
                        st.dataframe(q1b_w_disp, width="stretch")

            def _model_list(rows):
                return ", ".join(f"{llm_name(r['model'])} (p={fmt_p(r['p_bonferroni'])})" for r in rows)

            st.markdown("**Plain-English interpretation:**")
            for metric in sorted(metrics):
                q1a_sub = q1a[q1a["metric"] == metric] if not q1a.empty else pd.DataFrame()
                q1b_row = (
                    q1b[q1b["metric"] == metric].iloc[0]
                    if (not q1b.empty and (q1b["metric"] == metric).any())
                    else None
                )

                if q1a_sub.empty:
                    st.markdown(f"- **{metric}**: insufficient data for test.")
                    continue

                sig_less = [r for _, r in q1a_sub.iterrows() if r["significant"] and r["direction"] == "judge < trial"]
                sig_more = [r for _, r in q1a_sub.iterrows() if r["significant"] and r["direction"] == "judge > trial"]
                not_sig = [r for _, r in q1a_sub.iterrows() if not r["significant"]]

                sentences = []
                if sig_less:
                    sentences.append(
                        f"Judge variance is significantly less than trial variance for {_model_list(sig_less)}"
                    )
                if sig_more:
                    sentences.append(
                        f"Judge variance is significantly greater than trial variance for {_model_list(sig_more)}"
                    )
                if not_sig:
                    sentences.append(
                        f"Judge variance is not significantly different from trial variance for {_model_list(not_sig)}"
                    )

                q1b_txt = ""
                if q1b_row is not None:
                    if q1b_row["significant"]:
                        q1b_txt = (
                            f" The size of the gap varies significantly across models (K-W p={fmt_p(q1b_row['p'])})."
                        )
                    else:
                        q1b_txt = f" The gap is consistent across models (K-W p={fmt_p(q1b_row['p'])})."

                st.markdown(f"- **{metric}**: " + ". ".join(sentences) + "." + q1b_txt)

            with st.expander("Methodology"):
                st.markdown("""
**Why paired Wilcoxon signed-rank (Q1a)?**
We're comparing two variance estimates for the same set of records. Using the same records in
both groups removes the scenario-difficulty confound — a hard scenario would inflate both judge
and trial variance. Wilcoxon signed-rank is the non-parametric equivalent of a paired t-test;
it doesn't assume normality of the differences.

**Calculation steps (Q1a):**
1. For each (record, trial, metric, model): compute judge std dev across the 3 judge iterations.
2. Average those judge std devs over trials → one judge-variance estimate per (record, model, metric).
3. Trial variance per record is from `compute_trajectory_variance()` — std dev of the mean score
   across 3 trials (after averaging over iterations to remove judge noise).
4. Pair the two estimates by `record_id` and run a Wilcoxon signed-rank test
   (scipy.stats.wilcoxon, two-sided, zero_method="wilcox").
5. Bonferroni correction: multiply each p-value by the number of models being tested.

**Why Kruskal-Wallis on deltas (Q1b)?**
Q1b asks whether the *gap* between judge and trial variance is consistent across models or
model-dependent. We compute delta = mean_judge_std − traj_std for each record in each model,
then run Kruskal-Wallis across models. If significant, the judge-vs-trial relationship differs
by model — i.e., some models have noisier judges relative to their trial variance than others.
""")

        st.subheader("Is variance significantly greater than zero? (per model)")
        if q0_judge_per_model.empty or q0_trial_per_model.empty:
            st.info("Not enough data to run Q0 tests.")
        else:
            _sig_map_jt = {True: "✓ yes", False: "✗ no"}
            for run_label in MODEL_LABEL_ORDER:
                if run_label not in combined["run_label"].values:
                    continue
                st.markdown(f"**{llm_name(run_label)}** — {run_label}")
                _jm_sub = q0_judge_per_model[q0_judge_per_model["model"] == run_label]
                _tm_sub = q0_trial_per_model[q0_trial_per_model["model"] == run_label]
                if _jm_sub.empty or _tm_sub.empty:
                    st.info("No data for this model.")
                    continue
                _merged = pd.merge(
                    _jm_sub[["metric", "median_std", "p", "significant"]].rename(
                        columns={"median_std": "judge median std", "p": "judge p", "significant": "judge sig?"}
                    ),
                    _tm_sub[["metric", "median_std", "p", "significant"]].rename(
                        columns={"median_std": "trial median std", "p": "trial p", "significant": "trial sig?"}
                    ),
                    on="metric",
                ).sort_values("metric")
                _merged["judge p"] = _merged["judge p"].map(fmt_p)
                _merged["trial p"] = _merged["trial p"].map(fmt_p)
                _merged["judge sig?"] = _merged["judge sig?"].map(_sig_map_jt)
                _merged["trial sig?"] = _merged["trial sig?"].map(_sig_map_jt)
                st.dataframe(
                    _merged.round({"judge median std": 4, "trial median std": 4}),
                    hide_index=True,
                    width="stretch",
                )

    # ── Tab 3: Judge variance ─────────────────────────────────────────────────
    with tabs[3]:
        st.header("Judge variance (stochasticity)")
        if selected_domain:
            st.caption(
                f"All views on this tab are **within-domain** ({_domain_label}). "
                "Use the sidebar radio to switch domains."
            )
        st.write("""
        **What this measures:** For each (record, trial) pair, how much does the metric score vary
        across the judge re-runs on identical conversation data?

        **Why it matters:** If std dev here is high, the judge is unreliable — two runs of the
        same benchmark on the same data could yield meaningfully different scores.
        """)

        st.subheader("Judge std dev per metric")
        st.caption(
            "Each bar = mean std dev of normalized scores across all (record, trial) pairs "
            "for that run × metric; error bars = std dev of those std devs. "
            "Tick label markers show overall significance (Kruskal-Wallis: * p < 0.05, ** p < 0.01, *** p < 0.001). "
            "Bar labels are compact letter display (CLD): bars with different letters differ significantly "
            "(pairwise Mann-Whitney U, Bonferroni-corrected)."
        )
        st.plotly_chart(
            variance_bar_fig(
                judge_summary,
                q2_kw,
                q2_pw,
                "Mean std dev (judge)",
                y_max=global_var_ymax,
                color_map=MODEL_COLOR_MAP,
                label_order=MODEL_LABEL_ORDER,
            ),
            width="stretch",
        )

        fig_box = px.box(
            judge_var,
            x="metric",
            y="std",
            color="run_label",
            points="all",
            hover_data=["record_id", "trial"],
            labels={"std": "Std dev (judge)", "metric": "Metric", "run_label": "Model(s)"},
            title="Distribution (median, IQR, all points)",
            color_discrete_map=MODEL_COLOR_MAP,
            category_orders={"run_label": MODEL_LABEL_ORDER},
        )
        fig_box.update_traces(jitter=0.4, pointpos=0)
        fig_box.update_layout(yaxis_range=[0, global_var_ymax], legend_title_text="Model(s)")
        fig_box.update_xaxes(**_axis_style)
        fig_box.update_yaxes(**_axis_style)
        st.plotly_chart(fig_box, width="stretch")

        download_button(judge_summary, "judge_variance_overview.csv")

        st.plotly_chart(
            variance_histogram(
                judge_var,
                x_label="Std dev (judge)",
                title="Distribution of per-(record,trial) judge std dev",
                color_map=MODEL_COLOR_MAP,
                label_order=MODEL_LABEL_ORDER,
            ),
            width="stretch",
        )

        if not scores_df.empty:
            fig_iter = px.box(
                scores_df[scores_df["metric"].isin(metrics)],
                x="iteration",
                y="normalized_score",
                color="run_label",
                facet_col="metric",
                facet_col_wrap=3,
                labels={"normalized_score": "Score", "iteration": "Iteration", "run_label": "Model(s)"},
                title="Score distributions across iterations",
                facet_row_spacing=0.15,
                height=500,
                color_discrete_map=MODEL_COLOR_MAP,
                category_orders={"run_label": MODEL_LABEL_ORDER},
            )
            fig_iter.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            fig_iter.update_xaxes(**_axis_style)
            fig_iter.update_yaxes(**_axis_style)
            st.plotly_chart(fig_iter, width="stretch")

        st.subheader("Summary statistics")

        # Mean std dev per metric, averaged across all selected models
        _cross_model_mean = (
            judge_summary.groupby("metric")["mean_std"]
            .mean()
            .reset_index()
            .rename(columns={"mean_std": "mean_std (across models)"})
            .set_index("metric")
            .T.reindex(columns=sorted(judge_summary["metric"].unique()))
        )
        _cross_model_mean.index = ["mean judge std dev (all models)"]
        _cross_model_mean.index.name = None
        st.dataframe(_cross_model_mean.round(4), width="stretch")

        st.caption(
            "**Column guide for per-model table:** `mean_std` = mean std dev across (record, trial) pairs; "
            "`std_of_std` = spread of those std devs; "
            "`std_min` / `std_max` = observed min and max std dev; "
            "`mean_range` = mean of (max score − min score) across pairs — a simpler spread "
            "measure than std dev; "
            "`pct_changed` = fraction of pairs where the score differed across iterations; "
            "`n` = number of (record, trial) pairs."
        )
        st.dataframe(judge_summary.round(4), width="stretch")
        download_button(judge_summary, "judge_variance_summary.csv")

        st.subheader("Std dev range per model (max − min)")
        st.caption(
            "How much do per-(record, trial) std devs vary within each model × metric? "
            "High delta = some pairs are very consistent, others very noisy."
        )
        _jdelta = judge_summary.assign(std_delta=judge_summary["std_max"] - judge_summary["std_min"]).pivot(
            index="run_label", columns="metric", values="std_delta"
        )
        _jdelta.columns.name = None
        _jdelta.index.name = "model"
        st.dataframe(_jdelta.round(4), width="stretch")

        if not judge_summary.empty:
            worst = judge_summary.loc[judge_summary["mean_std"].idxmax()]
            st.info(
                f"**Key finding:** Highest judge std dev is {worst['mean_std']:.4f} for "
                f"**{worst['metric']}** (run: {worst['run_label']}). "
                f"{worst['pct_changed']:.1%} of (record, trial) pairs changed score across iterations."
            )

        st.subheader("Statistical test: Does judge variance differ across models?")
        if q2_kw.empty:
            st.info("Not enough data to run statistical tests (need ≥ 2 models with ≥ 2 observations).")
        else:
            sig_pairs_j: dict[str, str] = {}
            if not q2_pw.empty:
                for _metric, _grp in q2_pw[q2_pw["significant"]].groupby("metric"):
                    sig_pairs_j[_metric] = "; ".join(
                        f"{llm_name(r['model_1'])} vs {llm_name(r['model_2'])} (r={r['effect_r']:.2f})"
                        for _, r in _grp.iterrows()
                    )
            q2_disp = q2_kw[["metric", "H", "p", "significant", "n_models"]].copy()
            q2_disp["H"] = q2_disp["H"].round(2)
            q2_disp["p"] = q2_disp["p"].map(fmt_p)
            q2_disp["significant pairs"] = q2_disp["metric"].map(lambda m: sig_pairs_j.get(m, "—"))
            st.dataframe(q2_disp, width="stretch")

            for rtype, wt in within_type_results.items():
                _wt_kw = wt.get("q2_kw", pd.DataFrame())
                _wt_pw = wt.get("q2_pairwise", pd.DataFrame())
                if _wt_kw.empty:
                    continue
                st.markdown(f"**Within {_TYPE_LABELS_ALL.get(rtype, rtype)} models**")
                _wt_sig: dict[str, str] = {}
                if not _wt_pw.empty:
                    for _m, _g in _wt_pw[_wt_pw["significant"]].groupby("metric"):
                        _wt_sig[_m] = "; ".join(
                            f"{llm_name(r['model_1'])} vs {llm_name(r['model_2'])} (r={r['effect_r']:.2f})"
                            for _, r in _g.iterrows()
                        )
                _wt_disp = _wt_kw[["metric", "H", "p", "significant", "n_models"]].copy()
                _wt_disp["H"] = _wt_disp["H"].round(2)
                _wt_disp["p"] = _wt_disp["p"].map(fmt_p)
                _wt_disp["significant pairs"] = _wt_disp["metric"].map(lambda m: _wt_sig.get(m, "—"))  # noqa: B023
                st.dataframe(_wt_disp, width="stretch")

            st.markdown("**Plain-English interpretation:**")
            for _, row in q2_kw.iterrows():
                metric = row["metric"]
                pw_rows = q2_pw[q2_pw["metric"] == metric] if not q2_pw.empty else pd.DataFrame()
                sig_pw = pw_rows[pw_rows["significant"]] if not pw_rows.empty else pd.DataFrame()
                hstr = f"H={row['H']:.2f}, p={fmt_p(row['p'])}"
                if row["significant"] and not sig_pw.empty:
                    pairs_txt = ", ".join(
                        f"{llm_name(r['model_1'])} vs {llm_name(r['model_2'])}" for _, r in sig_pw.iterrows()
                    )
                    st.markdown(
                        f"- **{metric}**: Judge variance differs significantly across models "
                        f"({hstr}). Significant pairs (Bonferroni-corrected): {pairs_txt}."
                    )
                elif row["significant"]:
                    st.markdown(
                        f"- **{metric}**: Judge variance differs significantly across models "
                        f"({hstr}), but no individual pair survives Bonferroni correction."
                    )
                else:
                    st.markdown(f"- **{metric}**: No significant difference in judge variance across models ({hstr}).")

            if len(_TYPE_LABELS_ALL) > 1 and len(within_type_results) > 1:
                st.markdown(
                    "> **Note:** The table above pools cascade and S2S models together. "
                    "A significant result may simply reflect the type difference rather than "
                    "within-type variation. Results within each type are shown below."
                )

            with st.expander("Methodology"):
                st.markdown("""
**Why Kruskal-Wallis?**
Per-(record,trial) judge std devs are not normally distributed — they are right-skewed with
many zeros. Kruskal-Wallis is a non-parametric one-way ANOVA on ranks; it tests whether the
distributions of judge std devs differ across models without assuming normality.

**Calculation steps:**
1. For each metric, collect all per-(record,trial) judge std devs grouped by model.
   (These come from `compute_judge_variance()`: std dev of the 3 iteration scores for each pair.)
2. Run Kruskal-Wallis H test (scipy.stats.kruskal) across the groups.
3. If the overall test is significant (p < 0.05): run pairwise Mann-Whitney U tests
   for all pairs of models (scipy.stats.mannwhitneyu, two-sided).
4. Apply Bonferroni correction: multiply each p-value by the number of pairs
   (e.g., 3 pairs for 3 models).
5. Report rank-biserial correlation as effect size: r = 1 − 2U/(n₁·n₂).
   Values near ±1 = large effect; near 0 = negligible. Positive r means model 1 has
   higher std devs than model 2.
6. When both cascade and S2S models are selected, tests are also run within each type
   separately so that cross-type differences don't dominate the signal.
""")

    # ── Tab 4: Trial variance ─────────────────────────────────────────────────
    with tabs[4]:
        st.header("Trial variance (conversation-to-conversation)")
        if selected_domain:
            st.caption(
                f"All views on this tab are **within-domain** ({_domain_label}). "
                "Use the sidebar radio to switch domains."
            )
        st.write("""
        **What this measures:** For each scenario (record), how much does the metric score vary
        across the simulation trials (different conversations of the same scenario)?
        Judge noise is removed by averaging scores across iterations before computing std dev.

        **Why it matters:** High trial variance means the same scenario plays out very
        differently across trials — the system is sensitive to conversation randomness.
        """)

        st.subheader("Trial std dev per metric")
        st.caption(
            "Each bar = mean std dev of normalized scores across records for that run × metric; "
            "error bars = std dev of those std devs. "
            "Tick label markers show overall significance (Kruskal-Wallis: * p < 0.05, ** p < 0.01, *** p < 0.001). "
            "Bar labels are compact letter display (CLD)."
        )
        st.plotly_chart(
            variance_bar_fig(
                trial_summary,
                q3_kw,
                q3_pw,
                "Mean std dev (trial)",
                y_max=global_var_ymax,
                color_map=MODEL_COLOR_MAP,
                label_order=MODEL_LABEL_ORDER,
            ),
            width="stretch",
        )

        fig_box_t = px.box(
            trial_var,
            x="metric",
            y="std",
            color="run_label",
            points="all",
            hover_data=["record_id"],
            labels={"std": "Std dev (trial)", "metric": "Metric", "run_label": "Model(s)"},
            title="Distribution (median, IQR, all points)",
            color_discrete_map=MODEL_COLOR_MAP,
            category_orders={"run_label": MODEL_LABEL_ORDER},
        )
        fig_box_t.update_traces(jitter=0.4, pointpos=0)
        fig_box_t.update_layout(yaxis_range=[0, global_var_ymax], legend_title_text="Model(s)")
        fig_box_t.update_xaxes(**_axis_style)
        fig_box_t.update_yaxes(**_axis_style)
        st.plotly_chart(fig_box_t, width="stretch")

        download_button(trial_summary, "trial_variance_overview.csv")

        st.plotly_chart(
            variance_histogram(
                trial_var,
                x_label="Std dev (trial)",
                title="Distribution of per-scenario trial std dev",
                color_map=MODEL_COLOR_MAP,
                label_order=MODEL_LABEL_ORDER,
            ),
            width="stretch",
        )

        st.subheader("Summary statistics")
        st.caption(
            "**Column guide:** `mean_std` = mean std dev across records; "
            "`std_of_std` = spread of those std devs; "
            "`std_min` / `std_max` = observed min and max std dev; "
            "`mean_range` = mean of (max score − min score) across records — a simpler spread "
            "measure than std dev; "
            "`n` = number of records."
        )
        st.dataframe(trial_summary.round(4), width="stretch")
        download_button(trial_summary, "trial_variance_summary.csv")

        st.subheader("Std dev range per model (max − min)")
        st.caption("How much do per-record std devs vary within each model × metric?")
        _tdelta = trial_summary.assign(std_delta=trial_summary["std_max"] - trial_summary["std_min"]).pivot(
            index="run_label", columns="metric", values="std_delta"
        )
        _tdelta.columns.name = None
        _tdelta.index.name = "model"
        st.dataframe(_tdelta.round(4), width="stretch")

        if not trial_summary.empty:
            worst_t = trial_summary.loc[trial_summary["mean_std"].idxmax()]
            st.info(
                f"**Key finding:** Highest trial std dev is {worst_t['mean_std']:.4f} for "
                f"**{worst_t['metric']}** (run: {worst_t['run_label']})."
            )

        st.subheader("Statistical test: Does trial variance differ across models?")
        if q3_kw.empty:
            st.info("Not enough data to run statistical tests.")
        else:
            sig_pairs_t: dict[str, str] = {}
            if not q3_pw.empty:
                for _metric, _grp in q3_pw[q3_pw["significant"]].groupby("metric"):
                    sig_pairs_t[_metric] = "; ".join(
                        f"{llm_name(r['model_1'])} vs {llm_name(r['model_2'])} (r={r['effect_r']:.2f})"
                        for _, r in _grp.iterrows()
                    )
            q3_disp = q3_kw[["metric", "H", "p", "significant", "n_models"]].copy()
            q3_disp["H"] = q3_disp["H"].round(2)
            q3_disp["p"] = q3_disp["p"].map(fmt_p)
            q3_disp["significant pairs"] = q3_disp["metric"].map(lambda m: sig_pairs_t.get(m, "—"))
            st.dataframe(q3_disp, width="stretch")

            for rtype, wt in within_type_results.items():
                _wt_kw = wt.get("q3_kw", pd.DataFrame())
                _wt_pw = wt.get("q3_pairwise", pd.DataFrame())
                if _wt_kw.empty:
                    continue
                st.markdown(f"**Within {_TYPE_LABELS_ALL.get(rtype, rtype)} models**")
                _wt_sig2: dict[str, str] = {}
                if not _wt_pw.empty:
                    for _m, _g in _wt_pw[_wt_pw["significant"]].groupby("metric"):
                        _wt_sig2[_m] = "; ".join(
                            f"{llm_name(r['model_1'])} vs {llm_name(r['model_2'])} (r={r['effect_r']:.2f})"
                            for _, r in _g.iterrows()
                        )
                _wt_disp2 = _wt_kw[["metric", "H", "p", "significant", "n_models"]].copy()
                _wt_disp2["H"] = _wt_disp2["H"].round(2)
                _wt_disp2["p"] = _wt_disp2["p"].map(fmt_p)
                _wt_disp2["significant pairs"] = _wt_disp2["metric"].map(lambda m: _wt_sig2.get(m, "—"))  # noqa: B023
                st.dataframe(_wt_disp2, width="stretch")

            st.markdown("**Plain-English interpretation:**")
            for _, row in q3_kw.iterrows():
                metric = row["metric"]
                pw_rows = q3_pw[q3_pw["metric"] == metric] if not q3_pw.empty else pd.DataFrame()
                sig_pw = pw_rows[pw_rows["significant"]] if not pw_rows.empty else pd.DataFrame()
                hstr = f"H={row['H']:.2f}, p={fmt_p(row['p'])}"
                if row["significant"] and not sig_pw.empty:
                    pairs_txt = ", ".join(
                        f"{llm_name(r['model_1'])} vs {llm_name(r['model_2'])}" for _, r in sig_pw.iterrows()
                    )
                    st.markdown(
                        f"- **{metric}**: Trial variance differs significantly across models "
                        f"({hstr}). Significant pairs (Bonferroni-corrected): {pairs_txt}."
                    )
                elif row["significant"]:
                    st.markdown(
                        f"- **{metric}**: Trial variance differs significantly across models "
                        f"({hstr}), but no individual pair survives Bonferroni correction."
                    )
                else:
                    st.markdown(f"- **{metric}**: No significant difference in trial variance across models ({hstr}).")

            if len(_TYPE_LABELS_ALL) > 1 and len(within_type_results) > 1:
                st.markdown(
                    "> **Note:** The table above pools cascade and S2S models together. "
                    "A significant result may simply reflect the type difference rather than "
                    "within-type variation. Results within each type are shown below."
                )

            with st.expander("Methodology"):
                st.markdown("""
**Same approach as Q2 (judge variance), applied to trial variance.**

Trial variance is measured as the std dev of a scenario's score across 3 simulation trials,
after averaging over judge iterations to remove judge noise. The resulting per-record std dev
is the unit of analysis here — one value per (record, model, metric).

**Why the same test (Kruskal-Wallis + Mann-Whitney U)?**
The question is structurally identical to Q2: does this variance measure differ across models?
The data have the same properties — bounded at zero, right-skewed — so the same non-parametric
approach applies. The one practical difference is that Q3 groups contain N records per model
(not N×K (record,trial) pairs as in Q2), so the groups are smaller and the test is slightly
less powerful for the same true effect size.

**Calculation steps:**
1. For each metric: group per-record trial std devs by model (from compute_trajectory_variance()).
2. Run Kruskal-Wallis H test (scipy.stats.kruskal) across the groups.
3. If significant: run pairwise Mann-Whitney U for all model pairs, Bonferroni-corrected.
4. Report rank-biserial correlation r = 1 − 2U/(n₁·n₂) as effect size.
5. When both cascade and S2S models are selected, tests are also run within each type
   separately so that cross-type differences don't dominate the signal.
""")

    # ── Tab 5: EVA score stability ────────────────────────────────────────────
    with tabs[5]:
        st.header("EVA score stability")
        st.write("""
        **What this measures:** For each iteration, the headline EVA composite metrics are
        recomputed from scratch using that iteration's judge scores. If judge stochasticity
        is low, these numbers should be nearly identical across iterations.

        **Why it matters:** Even if individual metric std dev is small, it could systematically
        flip borderline scenarios, shifting the headline numbers that teams use to compare models.
        """)

        with st.expander("Composite metric definitions"):
            st.markdown("""
| Composite | Components | Pass condition |
|---|---|---|
| **EVA-A** | task_completion, faithfulness, agent_speech_fidelity | task_completion == 1.0 AND faithfulness >= 0.5 AND agent_speech_fidelity >= 0.95 |
| **EVA-X** | conversation_progression, turn_taking, conciseness | all >= 0.5 |
| **EVA-overall** | — | EVA-A pass AND EVA-X pass (derived) |
| **_mean variants** | same components as _pass | simple mean of normalized scores |

**Statistics computed per composite, per iteration:**
- **pass@1**: average fraction of trials that pass per scenario — expected pass rate for a single-trial benchmark
- **pass@k (k=3)**: fraction of scenarios where at least one trial passes — probability of seeing a passing result if you run once
- **pass^k (k=3)**: average (c/n)³ per scenario — theoretical probability all 3 draws pass (conservative lower bound)
- **mean**: mean of the composite value across all (record, trial) rows
            """)

        if stability_df.empty:
            st.warning("No composite stability data available.")
        else:
            composite_cols = [
                c
                for c in stability_df.columns
                if c not in ("run_id", "run_label", "iteration")
                and any(c.endswith(s) for s in ("_pass_at_1", "_pass_at_k", "_pass_power_k", "_mean"))
            ]
            eva_ax_cols = sorted(
                [c for c in composite_cols if not clean_composite_label(c).startswith("EVA-overall")],
                key=lambda c: composite_sort_key(clean_composite_label(c)),
            )

            st.plotly_chart(
                composite_stability_fig(stability_df, MODEL_COLOR_MAP, MODEL_LABEL_ORDER),
                width="stretch",
            )

            st.subheader("Delta (max − min) across iterations, per model")
            delta_df = stability_df.groupby("run_label")[eva_ax_cols].agg(lambda x: x.max() - x.min())
            delta_df.columns = [clean_composite_label(c) for c in delta_df.columns]
            delta_df = delta_df.reset_index()
            _num_cols = delta_df.select_dtypes("number").columns
            delta_df[_num_cols] = delta_df[_num_cols].apply(lambda s: s.map(lambda v: float(f"{v:.2g}")))
            st.dataframe(delta_df, width="stretch")

            st.subheader("Values per iteration")
            _vpi = stability_df[["run_id", "run_label", "iteration"] + eva_ax_cols].copy()
            _vpi = _vpi.rename(columns={c: clean_composite_label(c) for c in eva_ax_cols})
            st.dataframe(_vpi.round(4), width="stretch")
            download_button(stability_df, "composite_stability.csv")

            ranges = stability_df.groupby("run_label")[eva_ax_cols].agg(lambda x: x.max() - x.min())
            max_range = ranges.max().max()
            max_metric = clean_composite_label(ranges.max().idxmax())
            st.info(
                f"**Key finding:** Largest composite shift across iterations is {max_range:.4f} for **{max_metric}**."
            )

    # ── Tab 6: Borderline scenarios ───────────────────────────────────────────
    with tabs[6]:
        st.header("Borderline scenarios")
        st.write(
            "**What this measures:** (Record, trial) pairs where judge stochasticity flipped the score "
            "across an EVA composite pass/fail threshold — meaning one judge iteration scored the metric "
            "as passing and another as failing on identical conversation data.\n\n"
            "**Why it matters:** These are the scenarios where judge noise directly affects benchmark "
            "conclusions. Only metrics that feed into EVA-A or EVA-X pass conditions are included."
        )

        if borderlines_df.empty:
            st.info("No pass/fail threshold crossings found in the selected data.")
        else:
            flip_counts_bar = borderlines_df.groupby(["run_label", "metric"]).size().reset_index(name="flip_count")
            # Ensure all (run_label, metric) combinations appear even if count is 0
            _all_combos = pd.MultiIndex.from_product(
                [flip_counts_bar["run_label"].unique(), list(_pass_thresholds.keys())],
                names=["run_label", "metric"],
            )
            flip_counts_bar = (
                flip_counts_bar.set_index(["run_label", "metric"]).reindex(_all_combos, fill_value=0).reset_index()
            )
            _y_max_pairs = (
                int(
                    borderlines_df.groupby("run_label")[["record_id", "trial"]]
                    .apply(lambda g: g.drop_duplicates().shape[0])
                    .max()
                )
                if not borderlines_df.empty
                else 1
            )
            fig_bc = px.bar(
                flip_counts_bar,
                x="metric",
                y="flip_count",
                color="run_label",
                barmode="group",
                labels={"flip_count": "Pass/fail flips (record×trial pairs)", "metric": "Metric"},
                title="Judge-stochasticity pass/fail flips per metric",
                color_discrete_map=MODEL_COLOR_MAP,
                category_orders={"run_label": MODEL_LABEL_ORDER},
            )
            fig_bc.update_layout(yaxis_range=[0, _y_max_pairs], legend_title_text="Model(s)")
            fig_bc.update_xaxes(**_axis_style)
            fig_bc.update_yaxes(**_axis_style)
            st.plotly_chart(fig_bc, width="stretch")

            # ── Heatmap: one subplot per metric, x = model, y = record ──────────
            st.subheader("Scenario instability heatmap")
            st.write(
                "Each cell shows how many trials (out of 3) had a pass/fail flip for that "
                "scenario and model. A cell of 3 means the score crossed the threshold on "
                "every trial — maximum judge instability for that scenario."
            )

            threshold_metrics = list(_pass_thresholds.keys())
            active_run_labels = list(borderlines_df["run_label"].unique())
            short_labels = [llm_name(lbl) for lbl in active_run_labels]
            all_records = sorted(borderlines_df["record_id"].unique())

            import plotly.colors as pc

            _viridis_4_rev = pc.sample_colorscale("Viridis", [1, 2 / 3, 1 / 3, 0])
            _disc_viridis = [
                [0.00, _viridis_4_rev[0]],
                [0.25, _viridis_4_rev[0]],
                [0.25, _viridis_4_rev[1]],
                [0.50, _viridis_4_rev[1]],
                [0.50, _viridis_4_rev[2]],
                [0.75, _viridis_4_rev[2]],
                [0.75, _viridis_4_rev[3]],
                [1.00, _viridis_4_rev[3]],
            ]

            n_metrics = len(threshold_metrics)
            if n_metrics > 0 and active_run_labels:
                fig_heat = make_subplots(
                    rows=1,
                    cols=n_metrics,
                    subplot_titles=[m.replace("_", " ") for m in threshold_metrics],
                    shared_yaxes=True,
                    horizontal_spacing=0.04,
                )
                for col_idx, metric in enumerate(threshold_metrics, start=1):
                    metric_crossings = borderlines_df[borderlines_df["metric"] == metric]
                    # Count flips per (run_label, record_id)
                    flip_per = (
                        metric_crossings.groupby(["run_label", "record_id"]).size().reset_index(name="flip_count")
                    )
                    # Build pivot: y=record_id, x=model (short label)
                    pivot_rows = []
                    for run_lbl, short in zip(active_run_labels, short_labels):
                        sub = flip_per[flip_per["run_label"] == run_lbl].set_index("record_id")["flip_count"]
                        col_vals = [float(sub.get(rec, 0)) for rec in all_records]
                        pivot_rows.append((short, col_vals))
                    # z shape: (n_records, n_models) — rows=records, cols=models
                    z_vals = [[row[1][i] for row in pivot_rows] for i in range(len(all_records))]
                    x_labels = [row[0] for row in pivot_rows]

                    fig_heat.add_trace(
                        go.Heatmap(
                            z=z_vals,
                            x=x_labels,
                            y=all_records,
                            colorscale=_disc_viridis,
                            zmin=0,
                            zmax=3,
                            showscale=(col_idx == n_metrics),
                            colorbar={
                                "title": "# trials<br>flipped",
                                "tickvals": [0, 1, 2, 3],
                                "ticktext": ["0", "1", "2", "3"],
                                "len": 0.5,
                            },
                            hovertemplate=("record: %{y}<br>model: %{x}<br>trials flipped: %{z:.0f}<extra></extra>"),
                            customdata=[
                                [active_run_labels[j] for j in range(len(active_run_labels))] for _ in all_records
                            ],
                        ),
                        row=1,
                        col=col_idx,
                    )
                    fig_heat.update_xaxes(tickangle=30, row=1, col=col_idx)
                    fig_heat.update_yaxes(autorange="reversed", row=1, col=col_idx)

                _heat_height = max(600, len(all_records) * 22 + 160)
                fig_heat.update_layout(
                    height=_heat_height,
                    margin={"l": 140, "r": 120, "b": 140},
                )
                fig_heat.update_annotations(font_size=11)
                st.plotly_chart(fig_heat, width="stretch")

            download_button(borderlines_df, "borderline_scenarios.csv")

            # ── Scenario instability ranking ──────────────────────────────────
            st.subheader("Scenario instability ranking")
            st.write(
                "Total flip count per scenario across all metrics and models. "
                "Scenarios with flips across multiple models are universally sensitive; "
                "those concentrated in one model are model-specific."
            )
            _ranked = (
                borderlines_df.groupby("record_id")
                .size()
                .reset_index(name="total_flips")
                .sort_values("total_flips", ascending=True)
            )
            _max_flips = int(_ranked["total_flips"].max()) if not _ranked.empty else 1
            _flip_ceiling = len(_pass_thresholds) * len(active_run_labels) * 3

            fig_rank = px.bar(
                _ranked,
                x="total_flips",
                y="record_id",
                orientation="h",
                labels={
                    "total_flips": "Total pass/fail flips (all metrics × models × trials)",
                    "record_id": "Scenario",
                },
                color="total_flips",
                color_continuous_scale=_disc_viridis,
                range_color=[0, _flip_ceiling],
            )
            fig_rank.update_layout(
                yaxis={"autorange": "reversed"},
                xaxis_range=[0, _flip_ceiling],
                coloraxis_showscale=False,
                height=max(400, len(_ranked) * 18 + 100),
                margin={"l": 140, "r": 40, "b": 60},
            )
            fig_rank.update_xaxes(**_axis_style)
            fig_rank.update_yaxes(**_axis_style)
            st.plotly_chart(fig_rank, width="stretch")

            # Breakdown: how many models contributed flips per scenario
            _model_flip_counts = (
                borderlines_df.groupby(["record_id", "run_label"])
                .size()
                .reset_index(name="n")
                .groupby("record_id")["run_label"]
                .count()
                .reset_index(name="n_models_with_flips")
            )
            _ranked_detail = (
                _ranked.merge(_model_flip_counts, on="record_id", how="left")
                .sort_values("total_flips", ascending=False)
                .reset_index(drop=True)
            )
            _ranked_detail.index += 1
            _ranked_detail.index.name = "rank"
            st.dataframe(_ranked_detail, width="stretch")

    # ── Tab 7: Intraclass correlation ─────────────────────────────────────────
    with tabs[7]:
        st.header("Intraclass correlation (ICC)")
        if selected_domain:
            st.caption(
                f"Per-model and pooled-centered ICCs are **within-domain** ({_domain_label}). "
                "Two-way ICC (Option B) below also uses the selected domain."
            )
        st.write("""
        **What this measures:** ICC = σ²_scenario / σ²_total quantifies what fraction
        of score variance is attributable to *scenario identity* — i.e., how much of
        the spread in scores comes from some scenarios being consistently harder or
        easier, vs. noise from trial-to-trial conversation differences.

        **High ICC** → scores primarily reflect genuine scenario difficulty differences;
        the benchmark discriminates well across scenarios.
        **Low ICC** → scores are dominated by within-scenario noise; scenario identity
        explains little of the variance.

        Two pooled estimates and one per-model breakdown are shown.
        """)

        st.subheader("Pooled ICC")
        st.markdown("**Option A — Centered (within-model variance explained by scenario)**")
        st.caption(
            "Each model's mean score is subtracted before pooling. ICC answers: "
            "what fraction of within-model score variance is explained by scenario identity?"
        )
        st.plotly_chart(icc_bar_centered_fig(icc_pc), width="stretch")

        st.markdown("**Option B — Two-way random effects with interaction**")
        st.caption(
            "ICC_scenario = σ²_scenario / σ²_total. ICC_model = σ²_model / σ²_total. "
            "Both are fractions of total variance (scenario + model + interaction + residual). "
            "F-tests use MS_interaction as denominator (Cornfield-Tukey rule for random effects). "
            "Computed per task domain (record_ids do not overlap across domains)."
        )
        if not icc_tw.empty and "domain" in icc_tw.columns and selected_domain:
            icc_tw_view = icc_tw[icc_tw["domain"] == selected_domain]
            st.caption(f"Showing {_domain_label} (use the sidebar to change).")
        else:
            icc_tw_view = icc_tw
        st.plotly_chart(icc_bar_twoway_fig(icc_tw_view), width="stretch")

        if not icc_tw_view.empty:
            st.markdown("**Scenario × model interaction F-test**")
            st.caption(
                "A significant interaction means models do not rank scenarios consistently "
                "— some scenarios are disproportionately harder/easier for specific models."
            )
            _int_disp = icc_tw_view[
                ["metric", "f_interaction", "p_interaction", "sigma2_interaction", "sigma2_total"]
            ].copy()
            _int_disp["f_interaction"] = _int_disp["f_interaction"].round(2)
            _int_disp["p_interaction"] = _int_disp["p_interaction"].map(fmt_p)
            _int_disp["sigma2_interaction"] = _int_disp["sigma2_interaction"].round(4)
            _int_disp["% of total variance"] = (
                (_int_disp["sigma2_interaction"] / _int_disp["sigma2_total"].replace(0, float("nan"))) * 100
            ).round(1)
            _int_disp = _int_disp.drop(columns=["sigma2_total"])
            st.dataframe(_int_disp, hide_index=True, width="stretch")

        st.subheader("Per-model ICC")
        st.caption(
            "One-way ANOVA per (model, metric): ICC = σ²_scenario / (σ²_scenario + σ²_residual). "
            "How much of this model's score variance is explained by which scenario it is?"
        )
        # icc_pm has a `model_id` column from the new per-domain pipeline; the
        # legacy plotters expect run_label, so rename for the call site.
        icc_pm_view = icc_pm.rename(columns={"model_id": "run_label"}) if "model_id" in icc_pm.columns else icc_pm
        st.plotly_chart(icc_heatmap_fig(icc_pm_view, MODEL_LABEL_ORDER), width="stretch")
        st.plotly_chart(icc_bar_per_model_fig(icc_pm_view, MODEL_COLOR_MAP, MODEL_LABEL_ORDER), width="stretch")

        if not icc_pm_view.empty:
            with st.expander("Full per-model ICC table"):
                _pm_disp = icc_pm_view[
                    [
                        "run_label",
                        "metric",
                        "icc",
                        "ci_lower",
                        "ci_upper",
                        "sigma2_scenario",
                        "sigma2_residual",
                        "n_scenarios",
                        "n_trials",
                        "f_stat",
                        "p_value",
                    ]
                ].copy()
                _pm_disp["p_value"] = _pm_disp["p_value"].map(fmt_p)
                st.dataframe(_pm_disp.round(4), hide_index=True, width="stretch")
                download_button(_pm_disp, "icc_per_model.csv")

        if not icc_pc.empty:
            _icc_max = icc_pc.loc[icc_pc["icc"].idxmax()]
            _icc_min = icc_pc.loc[icc_pc["icc"].idxmin()]
            st.info(
                f"**Key finding:** Highest pooled ICC: **{_icc_max['metric']}** "
                f"({_icc_max['icc']:.2f}, 95% CI [{_icc_max['ci_lower']:.2f}–{_icc_max['ci_upper']:.2f}]). "
                f"Lowest: **{_icc_min['metric']}** "
                f"({_icc_min['icc']:.2f}, 95% CI [{_icc_min['ci_lower']:.2f}–{_icc_min['ci_upper']:.2f}])."
            )

    # ── Tab 8: Variance budget ────────────────────────────────────────────────
    with tabs[8]:
        st.header("Variance budget")
        st.write("""
        **What this measures:** A decomposition of total score variance into the sources
        that produced it. For a given (model, metric), how much of the score wobble comes
        from:
        - **Domain** — which task domain (itsm / medical_hr / airline) the scenario belongs to
        - **Scenario** — which scenario *within* a domain was graded
        - **Trial** — which of the 5 conversation simulations of that scenario
        - **Judge** — leftover stochasticity in the LLM judge re-grading the same trial

        Judge-graded metrics get all 4 pots; deterministic metrics (e.g., `task_completion`,
        `tool_call_validity`, `turn_taking`) get 3 pots — there is no judge stochasticity to
        measure since the same conversation always produces the same score. This generalises
        the ICC tab (which compares one source — scenario — against everything lumped
        together) into a full budget that sums to 100%.
        """)

        budget_df = _read_stat("variance_budget.csv")
        if budget_df.empty:
            st.warning(
                "`variance_budget.csv` is empty or missing. Run the statistical tests step "
                "in the sidebar to populate it."
            )
        else:
            if selected_domain:
                st.info(
                    "The sidebar Domain selector does not apply to this tab — the variance "
                    "budget computation pools data from all domains to estimate σ²_domain."
                )
            from plots_variance import variance_budget_absolute_bar, variance_budget_stacked_bar

            model_ids = sorted(budget_df["model_id"].dropna().unique().tolist())
            selected_model = st.selectbox("Model", model_ids, index=0, key="variance_budget_model")

            sub = budget_df[budget_df["model_id"] == selected_model].copy()
            non_converged = sub[~sub["converged"].fillna(False)]
            if not non_converged.empty:
                st.warning(
                    "Some metrics did not produce a usable variance budget for this model:\n\n"
                    + "\n".join(
                        f"- **{r['metric']}**: {r.get('fit_error') or 'fit did not converge'}"
                        for _, r in non_converged.iterrows()
                    )
                )

            converged = sub[sub["converged"].fillna(False)]
            if not converged.empty:
                st.subheader("Absolute variance (σ²)")
                st.caption(
                    "Bar height = total variance. Use this to gauge **how much** there is to "
                    "explain. A metric with very low total variance means scores are nearly "
                    "constant across the whole study — the % breakdown below is informative but "
                    "the magnitude is small, so the practical impact may be limited."
                )
                st.plotly_chart(variance_budget_absolute_bar(converged, selected_model), width="stretch")

                st.subheader("Variance breakdown (% of total)")
                st.caption(
                    "Same data, normalized so each bar reaches 100%. The σ value above each "
                    "bar is the standard deviation (sqrt of σ²_total): for a metric scored 0–1, "
                    "**σ = 0.05 means scores typically wobble ±5 percentage points** around the "
                    "mean. σ = 0.20 means ±20 pp, etc."
                )
                fig = variance_budget_stacked_bar(converged, selected_model)
                st.plotly_chart(fig, width="stretch")

                st.subheader("Detail")
                display_cols = [
                    "metric",
                    "metric_type",
                    "pct_domain",
                    "pct_scenario",
                    "pct_trial",
                    "pct_judge",
                    "sigma2_domain",
                    "sigma2_scenario",
                    "sigma2_trial",
                    "sigma2_judge",
                    "sigma2_total",
                    "n_obs",
                    "converged",
                ]
                display_cols = [c for c in display_cols if c in converged.columns]
                pct_cols = [c for c in display_cols if c.startswith("pct_")]
                sigma_cols = [c for c in display_cols if c.startswith("sigma2_")]
                styled = converged[display_cols].style.format(
                    {**dict.fromkeys(pct_cols, "{:.1%}"), **dict.fromkeys(sigma_cols, "{:.5f}")},
                    na_rep="—",
                )
                st.dataframe(styled, width="stretch")
                download_button(converged[display_cols], f"variance_budget_{selected_model}.csv")

    # ── Tab 9: Per-metric deep dive ───────────────────────────────────────────
    with tabs[9]:
        st.header("Per-metric deep dive")
        st.write("""
        **What this measures:** For each metric, how does judge variance relate to trial
        variance at the individual scenario level? Scenarios in the upper-right are high-variance
        from *both* sources; upper-left are judge-driven; lower-right are trial-driven.
        """)

        if metrics:
            selected_metric = st.selectbox("Metric", metrics, index=0)
            st.plotly_chart(
                deep_dive_scatter_fig(judge_var, trial_var, selected_metric, MODEL_COLOR_MAP, MODEL_LABEL_ORDER),
                width="stretch",
            )

            jv_m = judge_var[judge_var["metric"] == selected_metric][
                ["run_id", "run_label", "record_id", "trial", "std"]
            ].rename(columns={"std": "judge_std"})
            tv_m = trial_var[trial_var["metric"] == selected_metric][
                ["run_id", "run_label", "record_id", "std"]
            ].rename(columns={"std": "trial_std"})
            merged_dd = pd.merge(jv_m, tv_m, on=["run_id", "run_label", "record_id"])
            if not merged_dd.empty:
                st.dataframe(merged_dd.round(4), width="stretch")
                download_button(merged_dd, f"deep_dive_{selected_metric}.csv")
                mean_j = merged_dd["judge_std"].mean()
                mean_t = merged_dd["trial_std"].mean()
                dominant = "judge stochasticity" if mean_j > mean_t else "trial differences"
                st.info(
                    f"**Key finding:** For **{selected_metric}**, mean judge std dev = {mean_j:.4f}, "
                    f"mean trial std dev = {mean_t:.4f}. "
                    f"Variance is primarily driven by **{dominant}**."
                )

    # ── Tab 10: Statistical tests ─────────────────────────────────────────────
    with tabs[10]:
        st.header("Statistical tests")
        st.write("""
        Full results for all statistical tests. High-level summaries with plain-English
        interpretations appear inline on the **Judge vs. trial variance**, **Judge variance**,
        and **Trial variance** tabs.
        """)

        # ── Q0 ────────────────────────────────────────────────────────────────────
        with st.expander("Q0 — Is variance significantly greater than zero? (→ Variance overview tab)"):
            st.markdown("One-sample Wilcoxon signed-rank (H₁: median > 0). Non-zero std devs only.")

            st.markdown("**Judge variance — pooled across models**")
            if q0_judge_pooled.empty:
                st.info("No results.")
            else:
                _q0jp_disp = q0_judge_pooled.copy()
                _q0jp_disp["p"] = _q0jp_disp["p"].map(fmt_p)
                _q0jp_disp["W"] = _q0jp_disp["W"].round(1)
                st.dataframe(_q0jp_disp.round({"median_std": 4, "mean_std": 4}), hide_index=True, width="stretch")
                download_button(q0_judge_pooled, "stat_q0_judge_pooled.csv")

            st.markdown("**Judge variance — per model**")
            if q0_judge_per_model.empty:
                st.info("No results.")
            else:
                _q0jm_disp = q0_judge_per_model.copy()
                _q0jm_disp["p"] = _q0jm_disp["p"].map(fmt_p)
                _q0jm_disp["W"] = _q0jm_disp["W"].round(1)
                _q0jm_disp["model"] = _q0jm_disp["model"].apply(llm_name)
                st.dataframe(_q0jm_disp.round({"median_std": 4, "mean_std": 4}), hide_index=True, width="stretch")
                download_button(q0_judge_per_model, "stat_q0_judge_per_model.csv")

            st.markdown("**Trial variance — pooled across models**")
            if q0_trial_pooled.empty:
                st.info("No results.")
            else:
                _q0tp_disp = q0_trial_pooled.copy()
                _q0tp_disp["p"] = _q0tp_disp["p"].map(fmt_p)
                _q0tp_disp["W"] = _q0tp_disp["W"].round(1)
                st.dataframe(_q0tp_disp.round({"median_std": 4, "mean_std": 4}), hide_index=True, width="stretch")
                download_button(q0_trial_pooled, "stat_q0_trial_pooled.csv")

            st.markdown("**Trial variance — per model**")
            if q0_trial_per_model.empty:
                st.info("No results.")
            else:
                _q0tm_disp = q0_trial_per_model.copy()
                _q0tm_disp["p"] = _q0tm_disp["p"].map(fmt_p)
                _q0tm_disp["W"] = _q0tm_disp["W"].round(1)
                _q0tm_disp["model"] = _q0tm_disp["model"].apply(llm_name)
                st.dataframe(_q0tm_disp.round({"median_std": 4, "mean_std": 4}), hide_index=True, width="stretch")
                download_button(q0_trial_per_model, "stat_q0_trial_per_model.csv")

            with st.expander("Q0 full methodology"):
                st.markdown("""
**Test choice:** One-sample Wilcoxon signed-rank test (scipy.stats.wilcoxon, alternative="greater").

**Why non-parametric?** Std devs are bounded at zero and right-skewed; a one-sample t-test
against 0 would violate normality assumptions. The Wilcoxon test ranks the absolute values
of the non-zero observations and tests whether they are systematically positive.

**Calculation steps:**
1. For each metric (and optionally each model): collect the relevant std devs.
   - Judge: per-(record,trial) std devs across 3 judge iterations.
   - Trial: per-record std devs across 3 trials (judge-noise-removed).
2. Remove exact zeros before ranking (equivalent to zero_method="wilcox").
3. Run scipy.stats.wilcoxon(nonzero_vals, alternative="greater").
4. p < 0.05 → variance is significantly greater than zero for this metric/model.

**Pooled vs. per-model:** Pooled analysis combines all models' std devs, giving higher power
to detect small but consistent variance. Per-model analysis can reveal if a specific model's
judge is unusually deterministic (e.g., always returns the same score for a metric).
""")

        # ── Q1a ───────────────────────────────────────────────────────────────────
        with st.expander("Q1a — Paired Wilcoxon: judge vs. trial variance (→ Judge vs. trial variance tab)"):
            if q1a.empty:
                st.info("No results (need ≥ 5 paired records per model × metric).")
            else:
                _q1a_disp = q1a.copy()
                _q1a_disp["p_raw"] = _q1a_disp["p_raw"].map(fmt_p)
                _q1a_disp["p_bonferroni"] = _q1a_disp["p_bonferroni"].map(fmt_p)
                _q1a_disp["W"] = _q1a_disp["W"].round(1)
                _q1a_disp["model"] = _q1a_disp["model"].apply(llm_name)
                st.dataframe(
                    _q1a_disp.round({"median_judge_std": 4, "median_traj_std": 4, "median_delta": 4}),
                    width="stretch",
                )
                download_button(q1a, "stat_q1a_wilcoxon.csv")

            with st.expander("Q1a full methodology"):
                st.markdown("""
**Test choice:** Wilcoxon signed-rank test (non-parametric paired t-test).

**Why paired?** Both judge std dev and trial std dev are computed from the same set of records.
Pairing by record_id removes the scenario-difficulty confound — a hard scenario would inflate
both variance measures, making an unpaired comparison misleading.

**Why non-parametric?** The differences between paired judge/trial std devs are not normally
distributed (bounded at zero, right-skewed). The Wilcoxon test ranks the *absolute values* of
the differences and tests whether positive and negative ranks are symmetric around zero.

**Calculation steps:**
1. For each (record, trial, metric, model): judge std dev = std dev of scores across 3 iterations.
2. Average judge std devs over trials per record →
   one judge-variance estimate per (record, model, metric).
   (This matches the granularity of trial variance, which is computed per record.)
3. Trial variance per record = std dev of the iteration-averaged scores across 3 trials
   (from compute_trajectory_variance()).
4. Merge on record_id. For each model × metric pair:
   run scipy.stats.wilcoxon(judge_stds, traj_stds, alternative="two-sided", zero_method="wilcox").
   zero_method="wilcox" excludes tied pairs (where judge_std == traj_std) — conservative choice.
5. Bonferroni correction: multiply p_raw by the number of models tested per metric.
6. Direction is determined by the sign of the median delta (judge_std − traj_std).
""")

        # ── Q1b ───────────────────────────────────────────────────────────────────
        with st.expander(
            "Q1b — Kruskal-Wallis: does the judge-vs-trial gap vary across models? (→ Judge vs. trial variance tab)"
        ):
            if q1b.empty:
                st.info("No results.")
            else:
                _q1b_disp = q1b.copy()
                _q1b_disp["H"] = _q1b_disp["H"].round(3)
                _q1b_disp["p"] = _q1b_disp["p"].map(fmt_p)
                st.dataframe(_q1b_disp, width="stretch")
                download_button(q1b, "stat_q1b_delta_kruskal_wallis.csv")

            with st.expander("Q1b full methodology"):
                st.markdown("""
**What this tests:** Whether the *gap* between judge and trial variance differs across models.

**Why a separate test?** Q1a answers "is judge variance different from trial variance for each
model?" but doesn't say whether that relationship is model-dependent. Q1b tests the interaction:
do different models have different judge-vs-trial variance ratios?

**Calculation steps:**
1. For each (record, model, metric): compute delta = mean_judge_std − traj_std.
   (Using the same per-record aggregation as Q1a.)
2. Group the deltas by model.
3. Run Kruskal-Wallis across groups (same rationale as Q2: non-parametric, right-skewed data).
4. A significant result means the gap is not uniform across models — some models may have
   judge variance that dominates more (or less) than trial variance compared to others.

**No Bonferroni correction here:** Q1b is one test per metric (not a family of pairwise tests),
so no correction is needed within this test. The 0.05 threshold is applied per metric.
""")

        # ── Q2 ────────────────────────────────────────────────────────────────────
        with st.expander("Q2 — Does judge variance differ across models? (→ Judge variance tab)"):
            st.markdown("**Kruskal-Wallis (overall test, per metric)**")
            if q2_kw.empty:
                st.info("No results (need ≥ 2 models).")
            else:
                _q2_kw_disp = q2_kw.copy()
                _q2_kw_disp["H"] = _q2_kw_disp["H"].round(3)
                _q2_kw_disp["p"] = _q2_kw_disp["p"].map(fmt_p)
                st.dataframe(_q2_kw_disp, width="stretch")
                download_button(q2_kw, "stat_q2_kruskal_wallis.csv")

            st.markdown("**Pairwise Mann-Whitney U (Bonferroni-corrected)**")
            if q2_pw.empty:
                st.info("No pairwise results.")
            else:
                _q2_pw_disp = q2_pw.copy()
                _q2_pw_disp["p_raw"] = _q2_pw_disp["p_raw"].map(fmt_p)
                _q2_pw_disp["p_bonferroni"] = _q2_pw_disp["p_bonferroni"].map(fmt_p)
                _q2_pw_disp["U"] = _q2_pw_disp["U"].round(1)
                _q2_pw_disp["effect_r"] = _q2_pw_disp["effect_r"].round(3)
                st.dataframe(_q2_pw_disp, width="stretch")
                download_button(q2_pw, "stat_q2_pairwise.csv")

            if len(within_type_results) > 1:
                st.markdown(
                    "> **Note:** Tables above pool all models. Within-type results (cascade / S2S "
                    "separately) are shown below to isolate within-group variation."
                )
            for _rtype in ("cascade", "s2s"):
                _wt = within_type_results.get(_rtype, {})
                _wt_kw2 = _wt.get("q2_kw", pd.DataFrame())
                _wt_pw2 = _wt.get("q2_pairwise", pd.DataFrame())
                if _wt_kw2.empty:
                    continue
                st.markdown(f"**Within {_TYPE_LABELS_ALL.get(_rtype, _rtype)} — K-W**")
                _d = _wt_kw2.copy()
                _d["H"] = _d["H"].round(3)
                _d["p"] = _d["p"].map(fmt_p)
                st.dataframe(_d, width="stretch")
                if not _wt_pw2.empty:
                    st.markdown(f"**Within {_TYPE_LABELS_ALL.get(_rtype, _rtype)} — pairwise MWU**")
                    _dp = _wt_pw2.copy()
                    _dp["p_raw"] = _dp["p_raw"].map(fmt_p)
                    _dp["p_bonferroni"] = _dp["p_bonferroni"].map(fmt_p)
                    _dp["U"] = _dp["U"].round(1)
                    _dp["effect_r"] = _dp["effect_r"].round(3)
                    st.dataframe(_dp, width="stretch")

            with st.expander("Q2 full methodology"):
                st.markdown("""
**Test choice:** Kruskal-Wallis (non-parametric one-way ANOVA on ranks).

**Why non-parametric?** Per-(record,trial) judge std devs are right-skewed with many zeros —
they do not satisfy the normality assumption required for a one-way ANOVA.

**Why ranks?** Kruskal-Wallis converts all observations to ranks across all groups combined,
then tests whether the average rank differs across groups. This is robust to skew and outliers.

**Calculation steps:**
1. For each metric: group per-(record,trial) judge std devs by model.
2. Run scipy.stats.kruskal(*groups). The H statistic approximates a χ² distribution
   with df = n_models − 1 under the null that all groups have the same distribution.
3. If p < 0.05: run pairwise Mann-Whitney U (scipy.stats.mannwhitneyu, two-sided) for
   all pairs of models.
4. Bonferroni correction: multiply each p_raw by the number of pairs
   (3 pairs for 3 models, capped at 1.0).
5. Effect size: rank-biserial correlation r = 1 − 2U/(n₁·n₂).
   Ranges from −1 to +1. Positive means model_1 tends to have larger std devs.
   Convention: |r| < 0.1 negligible, 0.1–0.3 small, 0.3–0.5 medium, > 0.5 large.

**Family-wise error:** The 0.05 threshold is applied per metric independently.
No cross-metric correction is applied, consistent with treating each metric as a
separate analysis question.
""")

        # ── Q3 ────────────────────────────────────────────────────────────────────
        with st.expander("Q3 — Does trial variance differ across models? (→ Trial variance tab)"):
            st.markdown("**Kruskal-Wallis (overall test, per metric)**")
            if q3_kw.empty:
                st.info("No results (need ≥ 2 models).")
            else:
                _q3_kw_disp = q3_kw.copy()
                _q3_kw_disp["H"] = _q3_kw_disp["H"].round(3)
                _q3_kw_disp["p"] = _q3_kw_disp["p"].map(fmt_p)
                st.dataframe(_q3_kw_disp, width="stretch")
                download_button(q3_kw, "stat_q3_kruskal_wallis.csv")

            st.markdown("**Pairwise Mann-Whitney U (Bonferroni-corrected)**")
            if q3_pw.empty:
                st.info("No pairwise results.")
            else:
                _q3_pw_disp = q3_pw.copy()
                _q3_pw_disp["p_raw"] = _q3_pw_disp["p_raw"].map(fmt_p)
                _q3_pw_disp["p_bonferroni"] = _q3_pw_disp["p_bonferroni"].map(fmt_p)
                _q3_pw_disp["U"] = _q3_pw_disp["U"].round(1)
                _q3_pw_disp["effect_r"] = _q3_pw_disp["effect_r"].round(3)
                st.dataframe(_q3_pw_disp, width="stretch")
                download_button(q3_pw, "stat_q3_pairwise.csv")

            for _rtype in ("cascade", "s2s"):
                _wt = within_type_results.get(_rtype, {})
                _wt_kw3 = _wt.get("q3_kw", pd.DataFrame())
                _wt_pw3 = _wt.get("q3_pairwise", pd.DataFrame())
                if _wt_kw3.empty:
                    continue
                st.markdown(f"**Within {_TYPE_LABELS_ALL.get(_rtype, _rtype)} — K-W**")
                _d3 = _wt_kw3.copy()
                _d3["H"] = _d3["H"].round(3)
                _d3["p"] = _d3["p"].map(fmt_p)
                st.dataframe(_d3, width="stretch")
                if not _wt_pw3.empty:
                    st.markdown(f"**Within {_TYPE_LABELS_ALL.get(_rtype, _rtype)} — pairwise MWU**")
                    _dp3 = _wt_pw3.copy()
                    _dp3["p_raw"] = _dp3["p_raw"].map(fmt_p)
                    _dp3["p_bonferroni"] = _dp3["p_bonferroni"].map(fmt_p)
                    _dp3["U"] = _dp3["U"].round(1)
                    _dp3["effect_r"] = _dp3["effect_r"].round(3)
                    st.dataframe(_dp3, width="stretch")

            with st.expander("Q3 full methodology"):
                st.markdown("""
**Test choice:** Same as Q2 — Kruskal-Wallis + pairwise Mann-Whitney U with Bonferroni correction.

**Why the same test?** The question is structurally identical to Q2: does this variance measure
differ across models? Trial std devs have the same distributional properties as judge std devs
(bounded at zero, right-skewed), so the same non-parametric approach is appropriate.

**Key difference from Q2:** Trial variance is at the per-record level (one std dev per record,
computed across 3 trials after averaging over judge iterations). Q2's groups contained N×K
(record,trial) observations; Q3's groups contain only N records per model. This means Q3 has
smaller groups and is slightly less powerful for the same true effect size.

**Calculation steps:**
1. For each metric: group per-record trial std devs by model (from compute_trajectory_variance()).
2. Run scipy.stats.kruskal(*groups).
3. If significant: pairwise Mann-Whitney U, Bonferroni-corrected.
4. Effect size: rank-biserial correlation r = 1 − 2U/(n₁·n₂).
""")

        # ── Q4 ────────────────────────────────────────────────────────────────────
        with st.expander("Q4 — Intraclass correlation (→ Intraclass correlation tab)"):
            st.markdown("**Per-model ICC (one-way ANOVA)**")
            if icc_pm.empty:
                st.info("No results.")
            else:
                _q4_pm = icc_pm_view[
                    [
                        "run_label",
                        "metric",
                        "icc",
                        "ci_lower",
                        "ci_upper",
                        "f_stat",
                        "p_value",
                        "n_scenarios",
                        "n_trials",
                    ]
                ].copy()
                _q4_pm["p_value"] = _q4_pm["p_value"].map(fmt_p)
                st.dataframe(_q4_pm.round(4), hide_index=True, width="stretch")
                download_button(icc_pm, "stat_q4_icc_per_model.csv")

            st.markdown("**Pooled ICC — centered (Option A)**")
            if icc_pc.empty:
                st.info("No results.")
            else:
                st.dataframe(icc_pc.round(4), hide_index=True, width="stretch")
                download_button(icc_pc, "stat_q4_icc_pooled_centered.csv")

            st.markdown("**Pooled ICC — two-way random effects (Option B)**")
            if icc_tw.empty:
                st.info("No results.")
            else:
                _q4_tw = icc_tw.copy()
                for col in ["p_scenario", "p_model", "p_interaction"]:
                    _q4_tw[col] = _q4_tw[col].map(fmt_p)
                st.dataframe(_q4_tw.round(4), hide_index=True, width="stretch")
                download_button(icc_tw, "stat_q4_icc_pooled_twoway.csv")

            with st.expander("Q4 full methodology"):
                st.markdown("""
**What ICC measures here**

ICC = σ²_scenario / σ²_total. Scores are first averaged over judge iterations (removing
judge noise), giving one score per (model, scenario, trial). ICC then asks: of all the
variance in those scores, what fraction is attributable to which scenario it is?

**Per-model: one-way ANOVA ICC(1,1)**

For each (model, metric): one-way ANOVA with n = 50 scenario groups, k = 3 trials per group.

Variance components:
- σ²_scenario = max(0, (MS_between − MS_within) / k)
- σ²_residual = MS_within

ICC = σ²_scenario / (σ²_scenario + σ²_residual)

Confidence intervals: Shrout & Fleiss (1979) exact CI using the F distribution:
- ci_lower = max(0, (F_obs/F_upper − 1) / (F_obs/F_upper + k − 1))
- ci_upper = min(1, (F_obs/F_lower − 1) / (F_obs/F_lower + k − 1))

where F_lower, F_upper are the α/2 and 1−α/2 critical values of F(df_between, df_within).

**Pooled centered (Option A)**

Each model's mean score is subtracted before pooling. The one-way ANOVA is then run with
k = n_models × n_trials observations per scenario group. This answers: what fraction of
within-model score variance is explained by scenario identity, pooled across models?
The centering removes model-level mean differences so they do not inflate σ²_residual.

**Pooled two-way (Option B)**

Model: Y_ijk = μ + α_i (scenario) + β_j (model) + (αβ)_ij (interaction) + ε_ijk (residual/trial)

All four terms are treated as random effects. Variance components from expected mean squares
(Cornfield-Tukey rule): the correct denominator for testing scenario and model main effects
is MS_interaction, not MS_residual, because the interaction term inflates the expected value
of MS_scenario and MS_model.

Variance components:
- σ²_residual    = MS_residual
- σ²_interaction = max(0, (MS_interaction − MS_residual) / n_trials)
- σ²_scenario    = max(0, (MS_scenario − MS_interaction) / (n_trials × n_models))
- σ²_model       = max(0, (MS_model − MS_interaction)    / (n_trials × n_scenarios))
- σ²_total       = σ²_scenario + σ²_model + σ²_interaction + σ²_residual

ICC_scenario = σ²_scenario / σ²_total
ICC_model    = σ²_model    / σ²_total

F-tests (random-effects Cornfield-Tukey denominators):
- F_scenario    = MS_scenario    / MS_interaction  (df = df_s, df_sm)
- F_model       = MS_model       / MS_interaction  (df = df_m, df_sm)
- F_interaction = MS_interaction / MS_residual     (df = df_sm, df_e)

Degrees of freedom (n_s=50, n_m=6, n_t=3 for most metrics):
- df_scenario = 49, df_model = 5, df_interaction = 245, df_residual = 600

Confidence intervals: Satterthwaite approximation for σ²_scenario:
- L = MS_scenario − MS_interaction (linear combination)
- Effective df: df_L = L² / (MS_scenario²/df_scenario + MS_interaction²/df_interaction)
- 95% CI on σ²_scenario: [df_L × σ²_scenario / χ²(df_L, 0.975),  df_L × σ²_scenario / χ²(df_L, 0.025)]
- CI on ICC_scenario: divide bounds by σ²_total (σ²_total treated as fixed — standard approximation)

**Interpreting negative variance components**

Variance component estimates can be slightly negative due to sampling variability when the
true value is near zero. These are clipped to 0. A clipped estimate means the factor explains
essentially none of the variance; no strong conclusion can be drawn about the sign.

**Interpreting the scenario × model interaction**

A significant interaction F-test (F_interaction, df_interaction = 245, df_residual = 600)
means models do not rank scenarios consistently — some scenarios are disproportionately
harder or easier for specific models. A non-significant interaction supports the additivity
assumption and means the benchmark discriminates scenarios consistently across models.
""")


def frontier_page():
    import pandas as pd
    from plots_frontier import frontier_placeholder_fig, frontier_scatter_plot, qr_results_table

    st.header("Frontier Analysis")

    config = _load_config("frontier")
    if config is None:
        st.warning(
            f"Config not found: `{CONFIG_DIR / 'frontier_config.yaml'}`. Create it using the template in the spec."
        )
        return

    alpha: float = config.get("alpha", 0.05)
    scores_path = PROCESSED_DIR / "frontier" / "model_scores.csv"
    qr_path = PROCESSED_DIR / "frontier" / "qr_results.csv"

    if not scores_path.exists() or not qr_path.exists():
        st.info(
            "Results not found. Run:\n"
            "```\n"
            "uv run python analysis/eva-bench-stats/run_data.py\n"
            "uv run python analysis/eva-bench-stats/run_stats.py\n"
            "```"
        )
        st.plotly_chart(frontier_placeholder_fig(), use_container_width=True)
        return

    scores_df = pd.read_csv(scores_path)
    qr_results_df = pd.read_csv(qr_path)

    fig = frontier_scatter_plot(scores_df, qr_results_df, alpha=alpha)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Quantile regression results", expanded=False):
        tbl = qr_results_table(qr_results_df, alpha=alpha)
        if not tbl.empty:
            st.dataframe(tbl, use_container_width=True)
        st.caption(f"n={len(scores_df)} models. Results are exploratory — see methods below.")

    with st.expander("Statistical methods", expanded=False):
        st.markdown(f"""
**Quantile regression** models the upper boundary of the EVA-X pass@1 distribution
given EVA-A pass@1 (and vice versa). A negative slope at the q-th percentile means
models with higher EVA-A tend to have a lower ceiling on EVA-X — a frontier trade-off.

**Two directions** are run because the trade-off may be asymmetric. *EVA-X \\~ EVA-A*
asks whether EVA-A constrains the ceiling on EVA-X; *EVA-A \\~ EVA-X* asks the reverse.
The reverse-direction lines on the plot are shown inverted onto the same axes
(y = (x − intercept) / slope).

**Two quantiles (q=0.75 and q=0.90)** serve as a sensitivity check. At n=11, the
90th percentile is driven by ~1–2 observations near the boundary; the 75th percentile
(~3 observations) is more stable. Convergence of direction and significance across
both quantiles strengthens conclusions.

**Bootstrapped 95% CIs** on the slope are computed by resampling models with replacement
{config.get("n_bootstrap", 1000):,} times and refitting QR on each resample. Analytical
standard errors are not used because they assume n >> 11.

**⚠️ Standing caution:** n=11 models severely limits statistical power. All results should
be treated as hypothesis-generating and exploratory, not confirmatory. Statistical
significance at α = {alpha} does not imply a robust finding at this sample size.
""")


with st.sidebar:
    _data_ready = _variance_data_ready()
    with st.expander("Run pipeline", expanded=not _data_ready):
        if st.button("Process data"):
            with st.spinner("Running run_data.py…"):
                ok, out = _run_script(_RUN_DATA_SCRIPT)
            with st.expander("Output", expanded=not ok):
                st.text(out)
            if ok:
                st.rerun()
            else:
                st.error("run_data.py failed.")
        if st.button("Run statistical tests"):
            with st.spinner("Running run_stats.py…"):
                ok, out = _run_script(_RUN_STATS_SCRIPT)
            with st.expander("Output", expanded=not ok):
                st.text(out)
            if ok:
                st.rerun()
            else:
                st.error("run_stats.py failed.")

    st.markdown("### Export")
    if st.button("Export HTML Report"):
        from generate_report import build_report

        with st.spinner("Generating report..."):
            path = build_report()
        st.success(f"Saved to `{path.relative_to(PROJECT_ROOT)}`")


pg = st.navigation(
    [
        st.Page(overview_page, title="Overview"),
        st.Page(perturbations_page, title="Perturbations"),
        st.Page(CIs_page, title="CIs"),
        st.Page(variance_page, title="Variance"),
        st.Page(frontier_page, title="Frontier"),
    ]
)
pg.run()
