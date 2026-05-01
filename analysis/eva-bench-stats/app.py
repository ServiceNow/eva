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

    # Sort models: cascade first, then s2s, unknown types last, alphabetical within each group.
    # Ordering is fully driven by the config — add `type: cascade` or `type: s2s` when adding models.
    _TYPE_ORDER = {"cascade": 0, "s2s": 1}
    _model_type = {label: cfg.get("type", "") for label, cfg in config["models"].items()}
    models = sorted(
        results_pooled["model_label"].unique(),
        key=lambda m: (_TYPE_ORDER.get(_model_type.get(m, ""), 2), m),
    )
    n_cascade = sum(1 for m in models if _model_type.get(m) == "cascade")
    group_boundary = n_cascade if 0 < n_cascade < len(models) else None
    group_labels = ("Cascade", "S2S") if group_boundary is not None else None

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
            perturbation_delta_plot(
                results_pooled,
                metric,
                title="Pooled",
                y_range=y_range,
                model_order=models,
                group_boundary=group_boundary,
                group_labels=group_labels,
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
            st.plotly_chart(
                perturbation_delta_plot(
                    domain_results,
                    metric,
                    title=_DOMAIN_DISPLAY.get(domain, domain),
                    y_range=y_range,
                    model_order=models,
                    group_boundary=group_boundary,
                    group_labels=group_labels,
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
                    results_pooled,
                    metric,
                    title=_tlabel(metric),
                    y_range=y_range,
                    model_order=models,
                    group_boundary=group_boundary,
                    group_labels=group_labels,
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
    import subprocess
    import sys

    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from plots_utils import fmt_p
    from plots_variance import (
        composite_stability_fig,
        deep_dive_scatter_fig,
        icc_bar_centered_fig,
        icc_bar_per_model_fig,
        icc_bar_twoway_fig,
        icc_heatmap_fig,
        llm_name,
        threshold_crossings_instability_fig,
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
    RUN_LABEL_ORDER: list[str] = cascade_labels + s2s_labels
    _TYPE_LABELS = {"cascade": "Cascade", "s2s": "S2S / audio-native"}

    # ── Output paths ──────────────────────────────────────────────────────────
    variance_dir = PROCESSED_DIR / "variance"
    data_dir = variance_dir / "data"
    stats_dir = variance_dir / "stats"
    run_data_script = PROJECT_ROOT / "analysis" / "eva-bench-stats" / "run_data.py"
    run_stats_script = PROJECT_ROOT / "analysis" / "eva-bench-stats" / "run_stats.py"

    def _run_script(script_path: Path) -> tuple[bool, str]:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        output = result.stdout + ("\n" + result.stderr if result.stderr else "")
        return result.returncode == 0, output

    # ── Run buttons ───────────────────────────────────────────────────────────
    data_ready = (data_dir / "scores.csv").exists() and (data_dir / "judge_var.csv").exists()
    stats_ready = (stats_dir / "q2_kw.csv").exists()

    if not data_ready or not stats_ready:
        st.warning("Processed data not found. Run the pipeline to get started.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Process data", disabled=data_ready):
                with st.spinner("Running run_data.py…"):
                    ok, out = _run_script(run_data_script)
                with st.expander("Output", expanded=not ok):
                    st.text(out)
                if ok:
                    st.rerun()
                else:
                    st.error("run_data.py failed. See output above.")
        with col2:
            if st.button("Run statistical tests", disabled=not data_ready or stats_ready):
                with st.spinner("Running run_stats.py…"):
                    ok, out = _run_script(run_stats_script)
                with st.expander("Output", expanded=not ok):
                    st.text(out)
                if ok:
                    st.rerun()
                else:
                    st.error("run_stats.py failed. See output above.")
        if not data_ready:
            return

    # ── Sidebar rebuild expander ──────────────────────────────────────────────
    with st.sidebar.expander("Rebuild data"):
        if st.button("Re-run data processing"):
            with st.spinner("Running run_data.py…"):
                ok, out = _run_script(run_data_script)
            with st.expander("Output"):
                st.text(out)
            if ok:
                st.rerun()
        if st.button("Re-run statistical tests"):
            with st.spinner("Running run_stats.py…"):
                ok, out = _run_script(run_stats_script)
            with st.expander("Output"):
                st.text(out)
            if ok:
                st.rerun()

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
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    icc_pm = _read_stat("icc_per_model.csv")
    icc_pc = _read_stat("icc_pooled_centered.csv")
    icc_tw = _read_stat("icc_pooled_twoway.csv")
    q0_judge_pooled = _read_stat("q0_judge_pooled.csv")
    q0_judge_per_model = _read_stat("q0_judge_per_model.csv")
    q0_trial_pooled = _read_stat("q0_trial_pooled.csv")
    q0_trial_per_model = _read_stat("q0_trial_per_model.csv")
    q1a = _read_stat("q1a.csv")
    q1b = _read_stat("q1b.csv")
    q2_kw = _read_stat("q2_kw.csv")
    q2_pw = _read_stat("q2_pairwise.csv")
    q3_kw = _read_stat("q3_kw.csv")
    q3_pw = _read_stat("q3_pairwise.csv")

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
    n_active_runs = len(runs)

    def download_button(df: pd.DataFrame, filename: str) -> None:
        st.download_button("Download CSV", df.to_csv(index=False).encode(), filename, "text/csv")

    # ── 10 Tabs ───────────────────────────────────────────────────────────────
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
        M trials per run). ICC is a separate but related calculation derived from the same data.
        """)

        meta_rows = []
        for lbl, cfg in runs.items():
            meta_rows.append({"run_label": lbl, **{k: v for k, v in cfg.items() if k != "run_id"}})
        st.subheader("Runs in this analysis")
        st.dataframe(pd.DataFrame(meta_rows), width="stretch")

        st.subheader("Metrics")
        st.dataframe(pd.DataFrame({"metric": metrics}), hide_index=True, width="stretch")

        st.subheader("Sample sizes")
        st.caption(
            "Judge variance n = number of (record, trial) pairs per model. "
            "Trial variance n = number of records per model."
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

    # ── Tab 2: Judge vs. trial variance ──────────────────────────────────────
    with tabs[2]:
        st.header("Judge vs. trial variance")
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
            model_sig = q1a_sig[q1a_sig["model"] == run_label] if not q1a_sig.empty else pd.DataFrame()
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

        st.subheader("Is variance significantly greater than zero? (per model)")
        if q0_judge_per_model.empty or q0_trial_per_model.empty:
            st.info("Not enough data to run Q0 tests.")
        else:
            _sig_map_jt = {True: "✓ yes", False: "✗ no"}
            for run_label in RUN_LABEL_ORDER:
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
                color_map=RUN_COLOR_MAP,
                label_order=RUN_LABEL_ORDER,
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
            color_discrete_map=RUN_COLOR_MAP,
            category_orders={"run_label": RUN_LABEL_ORDER},
        )
        fig_box.update_traces(jitter=0.4, pointpos=0)
        fig_box.update_layout(yaxis_range=[0, global_var_ymax], legend_title_text="Model(s)")
        fig_box.update_xaxes(**_axis_style)
        fig_box.update_yaxes(**_axis_style)
        st.plotly_chart(fig_box, width="stretch")

        download_button(judge_summary, "judge_variance_overview.csv")

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
                color_discrete_map=RUN_COLOR_MAP,
                category_orders={"run_label": RUN_LABEL_ORDER},
            )
            fig_iter.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            fig_iter.update_xaxes(**_axis_style)
            fig_iter.update_yaxes(**_axis_style)
            st.plotly_chart(fig_iter, width="stretch")

        st.plotly_chart(
            variance_histogram(
                judge_var,
                x_label="Std dev (judge)",
                title="Distribution of per-(record,trial) judge std dev",
                color_map=RUN_COLOR_MAP,
                label_order=RUN_LABEL_ORDER,
            ),
            width="stretch",
        )

        st.subheader("Summary statistics")
        st.caption(
            "**Column guide:** `mean_std` = mean std dev across (record, trial) pairs; "
            "`std_of_std` = spread; `pct_changed` = fraction of pairs where score differed; `n` = n pairs."
        )
        st.dataframe(judge_summary.round(4), width="stretch")
        download_button(judge_summary, "judge_variance_summary.csv")

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

    # ── Tab 4: Trial variance ─────────────────────────────────────────────────
    with tabs[4]:
        st.header("Trial variance (conversation-to-conversation)")
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
                color_map=RUN_COLOR_MAP,
                label_order=RUN_LABEL_ORDER,
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
            color_discrete_map=RUN_COLOR_MAP,
            category_orders={"run_label": RUN_LABEL_ORDER},
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
                color_map=RUN_COLOR_MAP,
                label_order=RUN_LABEL_ORDER,
            ),
            width="stretch",
        )

        st.subheader("Summary statistics")
        st.dataframe(trial_summary.round(4), width="stretch")
        download_button(trial_summary, "trial_variance_summary.csv")

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

    # ── Tab 5: EVA score stability ────────────────────────────────────────────
    with tabs[5]:
        st.header("EVA score stability")
        st.write("""
        How stable are composite EVA scores (pass@1, pass@k, pass^k, mean) across judge iterations?
        Each point is the composite value for one iteration; spread across iterations = judge noise on composites.
        """)
        st.plotly_chart(
            composite_stability_fig(stability_df, RUN_COLOR_MAP, RUN_LABEL_ORDER),
            width="stretch",
        )
        download_button(stability_df, "composite_stability.csv")

    # ── Tab 6: Borderline scenarios ───────────────────────────────────────────
    with tabs[6]:
        st.header("Borderline scenarios")
        st.write("""
        **What this shows:** Scenarios where judge stochasticity flipped a score across a pass/fail threshold.
        For each (run, metric, record, trial): if the min score across iterations is below the threshold
        AND the max is at or above it, that trial had a pass/fail flip due to judge noise.
        """)

        if borderlines_df.empty:
            st.info("No borderline scenarios found — judge was stable across all iterations.")
        else:
            st.subheader("Borderline scenario table")
            st.dataframe(borderlines_df.round(4), width="stretch")
            download_button(borderlines_df, "borderline_scenarios.csv")

            st.subheader("Scenario instability ranking")
            st.write(
                "Total flip count per scenario across all metrics and models. "
                "Scenarios with flips across multiple models are universally sensitive."
            )
            st.plotly_chart(
                threshold_crossings_instability_fig(borderlines_df, n_active_runs, _pass_thresholds, RUN_COLOR_MAP),
                width="stretch",
            )

    # ── Tab 7: Intraclass correlation ─────────────────────────────────────────
    with tabs[7]:
        st.header("Intraclass correlation (ICC)")
        st.write("""
        **What this measures:** ICC = σ²_scenario / σ²_total quantifies what fraction
        of score variance is attributable to *scenario identity* — i.e., how much of
        the spread in scores comes from some scenarios being consistently harder or
        easier, vs. noise from trial-to-trial conversation differences.

        **High ICC** → scores primarily reflect genuine scenario difficulty differences.
        **Low ICC** → scores are dominated by within-scenario noise.
        """)

        st.subheader("Pooled ICC — Option A (centered)")
        st.caption(
            "Each model's mean score is subtracted before pooling. ICC answers: "
            "what fraction of within-model score variance is explained by scenario identity?"
        )
        st.plotly_chart(icc_bar_centered_fig(icc_pc), width="stretch")

        st.subheader("Pooled ICC — Option B (two-way random effects with interaction)")
        st.caption("ICC_scenario = σ²_scenario / σ²_total. ICC_model = σ²_model / σ²_total.")
        st.plotly_chart(icc_bar_twoway_fig(icc_tw), width="stretch")

        if not icc_tw.empty:
            st.markdown("**Scenario × model interaction F-test**")
            _int_disp = icc_tw[
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
        st.caption("One-way ANOVA per (model, metric): ICC = σ²_scenario / (σ²_scenario + σ²_residual).")
        st.plotly_chart(icc_heatmap_fig(icc_pm, RUN_LABEL_ORDER), width="stretch")
        st.plotly_chart(icc_bar_per_model_fig(icc_pm, RUN_COLOR_MAP, RUN_LABEL_ORDER), width="stretch")

        if not icc_pc.empty:
            _icc_max = icc_pc.loc[icc_pc["icc"].idxmax()]
            _icc_min = icc_pc.loc[icc_pc["icc"].idxmin()]
            st.info(
                f"**Key finding:** Highest pooled ICC: **{_icc_max['metric']}** "
                f"({_icc_max['icc']:.2f}, 95% CI [{_icc_max['ci_lower']:.2f}–{_icc_max['ci_upper']:.2f}]). "
                f"Lowest: **{_icc_min['metric']}** "
                f"({_icc_min['icc']:.2f}, 95% CI [{_icc_min['ci_lower']:.2f}–{_icc_min['ci_upper']:.2f}])."
            )

    # ── Tab 8: Per-metric deep dive ───────────────────────────────────────────
    with tabs[8]:
        st.header("Per-metric deep dive")
        st.write("""
        **What this measures:** For each metric, how does judge variance relate to trial
        variance at the individual scenario level? Scenarios in the upper-right are high-variance
        from *both* sources; upper-left are judge-driven; lower-right are trial-driven.
        """)

        if metrics:
            selected_metric = st.selectbox("Metric", metrics, index=0)
            st.plotly_chart(
                deep_dive_scatter_fig(judge_var, trial_var, selected_metric, RUN_COLOR_MAP, RUN_LABEL_ORDER),
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

    # ── Tab 9: Statistical tests ──────────────────────────────────────────────
    with tabs[9]:
        st.header("Statistical tests")
        st.write("Full results for all statistical tests.")

        with st.expander("Q0 — Is variance significantly greater than zero?"):
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

            st.markdown("**Trial variance — pooled across models**")
            if q0_trial_pooled.empty:
                st.info("No results.")
            else:
                _q0tp_disp = q0_trial_pooled.copy()
                _q0tp_disp["p"] = _q0tp_disp["p"].map(fmt_p)
                _q0tp_disp["W"] = _q0tp_disp["W"].round(1)
                st.dataframe(_q0tp_disp.round({"median_std": 4, "mean_std": 4}), hide_index=True, width="stretch")
                download_button(q0_trial_pooled, "stat_q0_trial_pooled.csv")

        with st.expander("Q1a — Paired Wilcoxon: judge vs. trial per model × metric"):
            if q1a.empty:
                st.info("No results.")
            else:
                _q1a_disp = q1a.copy()
                _q1a_disp["p_bonferroni"] = _q1a_disp["p_bonferroni"].map(fmt_p)
                st.dataframe(_q1a_disp.round(4), hide_index=True, width="stretch")
                download_button(q1a, "stat_q1a.csv")

        with st.expander("Q1b — KW: does judge-vs-trial gap vary by model?"):
            if q1b.empty:
                st.info("No results.")
            else:
                _q1b_disp = q1b.copy()
                _q1b_disp["p"] = _q1b_disp["p"].map(fmt_p)
                st.dataframe(_q1b_disp.round(4), hide_index=True, width="stretch")
                download_button(q1b, "stat_q1b.csv")

        with st.expander("Q2 — Judge variance across models (KW + pairwise MWU)"):
            if q2_kw.empty:
                st.info("No results.")
            else:
                _q2_disp = q2_kw.copy()
                _q2_disp["p"] = _q2_disp["p"].map(fmt_p)
                st.dataframe(_q2_disp.round(4), hide_index=True, width="stretch")
                download_button(q2_kw, "stat_q2_kw.csv")
                if not q2_pw.empty:
                    _q2pw_disp = q2_pw.copy()
                    _q2pw_disp["p_raw"] = _q2pw_disp["p_raw"].map(fmt_p)
                    _q2pw_disp["p_bonferroni"] = _q2pw_disp["p_bonferroni"].map(fmt_p)
                    st.dataframe(_q2pw_disp.round(4), hide_index=True, width="stretch")
                    download_button(q2_pw, "stat_q2_pairwise.csv")

        with st.expander("Q3 — Trial variance across models (KW + pairwise MWU)"):
            if q3_kw.empty:
                st.info("No results.")
            else:
                _q3_disp = q3_kw.copy()
                _q3_disp["p"] = _q3_disp["p"].map(fmt_p)
                st.dataframe(_q3_disp.round(4), hide_index=True, width="stretch")
                download_button(q3_kw, "stat_q3_kw.csv")
                if not q3_pw.empty:
                    _q3pw_disp = q3_pw.copy()
                    _q3pw_disp["p_raw"] = _q3pw_disp["p_raw"].map(fmt_p)
                    _q3pw_disp["p_bonferroni"] = _q3pw_disp["p_bonferroni"].map(fmt_p)
                    st.dataframe(_q3pw_disp.round(4), hide_index=True, width="stretch")
                    download_button(q3_pw, "stat_q3_pairwise.csv")


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
