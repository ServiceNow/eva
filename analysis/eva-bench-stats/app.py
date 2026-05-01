# analysis/eva-bench-stats/app.py
"""EVA-Bench Statistics — multi-page Streamlit app.

Run from project root:
    uv run streamlit run analysis/eva-bench-stats/app.py
"""

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
        perturbation_delta_plot,
        perturbation_overview_plot,
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

    metrics: list[str] = config.get("metrics", [])
    alpha: float = config.get("alpha", 0.05)

    domains = sorted(results_per_domain["domain"].unique())

    # Sort models: cascade first, then s2s, alphabetical within each group
    _model_type = {label: cfg.get("type", "z") for label, cfg in config["models"].items()}
    models = sorted(
        results_pooled["model_label"].unique(),
        key=lambda m: (0 if _model_type.get(m) == "cascade" else 1, m),
    )

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
        y_min = min(metric_results["ci_lower"].min(), metric_results["observed_mean_delta"].min())
        y_max = max(metric_results["ci_upper"].max(), metric_results["observed_mean_delta"].max())
        pad = (y_max - y_min) * 0.20
        y_range = (y_min - pad, y_max + pad)

        # ── Pooled ────────────────────────────────────────────────────
        st.subheader("Pooled (all domains, 90 scenarios)")
        st.caption(f"{sig_note} across 3 conditions per model.")
        st.plotly_chart(
            perturbation_delta_plot(results_pooled, metric, title="Pooled", y_range=y_range, model_order=models),
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
            st.plotly_chart(
                perturbation_delta_plot(
                    domain_results,
                    metric,
                    title=_DOMAIN_DISPLAY.get(domain, domain),
                    y_range=y_range,
                    model_order=models,
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
            pad = (y_max - y_min) * 0.20
            y_range = (y_min - pad, y_max + pad)
            st.plotly_chart(
                perturbation_delta_plot(
                    results_pooled, metric, title=_tlabel(metric), y_range=y_range, model_order=models
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
        overview_pad = (overview_y_max - overview_y_min) * 0.15
        overview_y_range = (overview_y_min - overview_pad, overview_y_max + overview_pad)

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
    st.header("Confidence Intervals")
    st.info("Coming soon.")


def variance_page():
    st.header("Variance Analysis")
    st.info("Coming soon.")


def frontier_page():
    st.header("Frontier Analysis")
    st.info("Coming soon.")


with st.sidebar:
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
