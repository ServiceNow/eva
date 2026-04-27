# apps/judge_variance_analysis/analysis.py
"""Variance analysis computations for the judge variance study."""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.stats import chi2
from scipy.stats import f as f_dist

JUDGE_METRICS = [
    "faithfulness",
    "agent_speech_fidelity",
    "conversation_progression",
    "turn_taking",
    "conciseness",
    "transcription_accuracy_key_entities",
]


def _oneway_icc(
    groups: list[np.ndarray],
    alpha: float = 0.05,
) -> dict:
    """One-way ANOVA ICC(1,1) from a list of equal-length observation arrays.

    Args:
        groups: List of 1-D arrays, one per group (scenario). All must have the
                same length k (number of observations per group, e.g. trials).
        alpha:  Two-sided confidence level (default 0.05 → 95% CI).

    Returns:
        Dict with keys: icc, ci_lower, ci_upper, sigma2_scenario, sigma2_residual,
        ms_between, ms_within, f_stat, p_value.
    """
    k = len(groups[0])  # observations per group (trials)
    n = len(groups)  # number of groups (scenarios)

    all_vals = np.concatenate(groups)
    grand_mean = all_vals.mean()
    group_means = np.array([g.mean() for g in groups])

    ss_between = k * np.sum((group_means - grand_mean) ** 2)
    ss_within = sum(np.sum((g - g.mean()) ** 2) for g in groups)

    df_between = n - 1
    df_within = n * (k - 1)

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    sigma2_scenario = max(0.0, (ms_between - ms_within) / k)
    sigma2_residual = ms_within
    denom = sigma2_scenario + sigma2_residual
    icc = sigma2_scenario / denom if denom > 0 else 0.0

    # F stat and p-value
    f_stat = ms_between / ms_within if ms_within > 0 else float("nan")
    p_value = float(1 - f_dist.cdf(f_stat, df_between, df_within)) if not np.isnan(f_stat) else float("nan")

    # Shrout & Fleiss (1979) ICC(1,1) confidence interval
    f_lower = f_dist.ppf(alpha / 2, df_between, df_within)
    f_upper = f_dist.ppf(1 - alpha / 2, df_between, df_within)

    def _ci_bound(f_crit: float) -> float:
        return (f_stat / f_crit - 1) / (f_stat / f_crit + k - 1)

    ci_lower = max(0.0, _ci_bound(f_upper))
    ci_upper = min(1.0, _ci_bound(f_lower))

    return {
        "icc": icc,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "sigma2_scenario": sigma2_scenario,
        "sigma2_residual": sigma2_residual,
        "ms_between": ms_between,
        "ms_within": ms_within,
        "f_stat": f_stat,
        "p_value": p_value,
    }


def _twoway_icc(Y: np.ndarray, alpha: float = 0.05) -> dict:
    """Two-way random-effects ANOVA ICC with interaction.

    Args:
        Y: 3-D array of shape (n_scenarios, n_models, n_trials).
           Balanced design required (equal n_trials per cell).
        alpha: Two-sided confidence level (default 0.05 → 95% CI).

    Returns:
        Dict with SS, MS, df, F, p, variance components, ICC_scenario, ICC_model,
        and Satterthwaite CIs for both.
    """
    n_s, n_m, n_t = Y.shape
    grand_mean = Y.mean()
    scenario_means = Y.mean(axis=(1, 2))  # shape (n_s,)
    model_means = Y.mean(axis=(0, 2))  # shape (n_m,)
    cell_means = Y.mean(axis=2)  # shape (n_s, n_m)

    ss_scenario = n_m * n_t * np.sum((scenario_means - grand_mean) ** 2)
    ss_model = n_s * n_t * np.sum((model_means - grand_mean) ** 2)
    ss_interaction = n_t * np.sum((cell_means - scenario_means[:, None] - model_means[None, :] + grand_mean) ** 2)
    ss_residual = np.sum((Y - cell_means[:, :, None]) ** 2)

    df_s = n_s - 1
    df_m = n_m - 1
    df_sm = df_s * df_m
    df_e = n_s * n_m * (n_t - 1)

    ms_s = ss_scenario / df_s
    ms_m = ss_model / df_m
    ms_sm = ss_interaction / df_sm
    ms_e = ss_residual / df_e

    # Variance components (random effects, Cornfield-Tukey denominators)
    sigma2_e = ms_e
    sigma2_sm = max(0.0, (ms_sm - ms_e) / n_t)
    sigma2_s = max(0.0, (ms_s - ms_sm) / (n_t * n_m))
    sigma2_m = max(0.0, (ms_m - ms_sm) / (n_t * n_s))
    sigma2_tot = sigma2_s + sigma2_m + sigma2_sm + sigma2_e

    icc_s = sigma2_s / sigma2_tot if sigma2_tot > 0 else 0.0
    icc_m = sigma2_m / sigma2_tot if sigma2_tot > 0 else 0.0

    # F-tests (Cornfield-Tukey: scenario and model tested against interaction)
    f_s = ms_s / ms_sm if ms_sm > 0 else float("nan")
    f_m = ms_m / ms_sm if ms_sm > 0 else float("nan")
    f_sm = ms_sm / ms_e if ms_e > 0 else float("nan")

    p_s = float(1 - f_dist.cdf(f_s, df_s, df_sm)) if not np.isnan(f_s) else float("nan")
    p_m = float(1 - f_dist.cdf(f_m, df_m, df_sm)) if not np.isnan(f_m) else float("nan")
    p_sm = float(1 - f_dist.cdf(f_sm, df_sm, df_e)) if not np.isnan(f_sm) else float("nan")

    # Satterthwaite CI for sigma2_s
    def _satterthwaite_ci(
        sigma2_hat: float, ms_num: float, ms_den: float, df_num: int, df_den: int, divisor: float
    ) -> tuple[float, float]:
        """Satterthwaite CI for a variance component estimated as (ms_num - ms_den)/divisor."""
        if sigma2_hat <= 0 or divisor <= 0:
            return 0.0, min(1.0, icc_s + 0.0)  # degenerate — return point estimate bounds
        L = ms_num - ms_den
        df_L = L**2 / (ms_num**2 / df_num + ms_den**2 / df_den)
        df_L = max(df_L, 1.0)
        ci_lo_var = df_L * sigma2_hat / chi2.ppf(1 - alpha / 2, df_L)
        ci_hi_var = df_L * sigma2_hat / chi2.ppf(alpha / 2, df_L)
        ci_lo_icc = max(0.0, min(1.0, ci_lo_var / sigma2_tot))
        ci_hi_icc = max(0.0, min(1.0, ci_hi_var / sigma2_tot))
        return ci_lo_icc, ci_hi_icc

    ci_lo_s, ci_hi_s = _satterthwaite_ci(sigma2_s, ms_s, ms_sm, df_s, df_sm, n_t * n_m)
    ci_lo_m, ci_hi_m = _satterthwaite_ci(sigma2_m, ms_m, ms_sm, df_m, df_sm, n_t * n_s)

    return {
        "icc_scenario": icc_s,
        "icc_model": icc_m,
        "ci_lower_scenario": ci_lo_s,
        "ci_upper_scenario": ci_hi_s,
        "ci_lower_model": ci_lo_m,
        "ci_upper_model": ci_hi_m,
        "sigma2_scenario": sigma2_s,
        "sigma2_model": sigma2_m,
        "sigma2_interaction": sigma2_sm,
        "sigma2_residual": sigma2_e,
        "sigma2_total": sigma2_tot,
        "ss_scenario": ss_scenario,
        "ss_model": ss_model,
        "ss_interaction": ss_interaction,
        "ss_residual": ss_residual,
        "ms_scenario": ms_s,
        "ms_model": ms_m,
        "ms_interaction": ms_sm,
        "ms_residual": ms_e,
        "f_scenario": f_s,
        "p_scenario": p_s,
        "f_model": f_m,
        "p_model": p_m,
        "f_interaction": f_sm,
        "p_interaction": p_sm,
        "n_scenarios": n_s,
        "n_models": n_m,
        "n_trials": n_t,
    }


def compute_icc(
    df: pd.DataFrame,
    metrics: list[str] = JUDGE_METRICS,
) -> dict[str, pd.DataFrame]:
    """Intraclass correlation at the scenario level for each metric.

    ICC = σ²_scenario / σ²_total quantifies what fraction of score variance
    is attributable to scenario identity (i.e. which scenario is being evaluated).

    Args:
        df: Flat scores DataFrame from load_data.load_scores().
        metrics: Metric names to include.

    Returns:
        Dict with keys:
          "per_model"       – one row per (run_id, metric)
          "pooled_centered" – one row per metric (Option A: model-mean-centered)
          "pooled_twoway"   – one row per metric (Option B: two-way random effects)
    """
    # ── Average over iterations to remove judge noise ─────────────────────────
    mean_by_trial = (
        df[df["metric"].isin(metrics)]
        .groupby(["run_id", "run_label", "metric", "record_id", "trial"])["normalized_score"]
        .mean()
        .reset_index()
        .rename(columns={"normalized_score": "mean_score"})
    )

    # ── per_model ─────────────────────────────────────────────────────────────
    pm_rows: list[dict] = []
    for (run_id, run_label, metric), grp in mean_by_trial.groupby(["run_id", "run_label", "metric"]):
        # Build list of arrays, one per scenario, containing trial scores
        scenario_groups = [sub["mean_score"].values for _, sub in grp.groupby("record_id") if len(sub) >= 2]
        if len(scenario_groups) < 2:
            continue
        # All groups must have same length for balanced ANOVA; use min length
        min_k = min(len(g) for g in scenario_groups)
        scenario_groups = [g[:min_k] for g in scenario_groups]

        r = _oneway_icc(scenario_groups)
        pm_rows.append(
            {
                "run_id": run_id,
                "run_label": run_label,
                "metric": metric,
                "icc": r["icc"],
                "ci_lower": r["ci_lower"],
                "ci_upper": r["ci_upper"],
                "sigma2_scenario": r["sigma2_scenario"],
                "sigma2_residual": r["sigma2_residual"],
                "ms_between": r["ms_between"],
                "ms_within": r["ms_within"],
                "n_scenarios": len(scenario_groups),
                "n_trials": min_k,
                "f_stat": r["f_stat"],
                "p_value": r["p_value"],
            }
        )

    per_model = pd.DataFrame(pm_rows)

    # ── pooled_centered (Option A) ────────────────────────────────────────────
    pc_rows: list[dict] = []
    for metric, grp in mean_by_trial.groupby("metric"):
        # Center each model's scores by its own mean
        model_means = grp.groupby("run_id")["mean_score"].transform("mean")
        centered = grp.assign(mean_score=grp["mean_score"] - model_means)

        scenario_groups = [sub["mean_score"].values for _, sub in centered.groupby("record_id") if len(sub) >= 2]
        if len(scenario_groups) < 2:
            continue
        min_k = min(len(g) for g in scenario_groups)
        scenario_groups = [g[:min_k] for g in scenario_groups]

        n_models = grp["run_id"].nunique()
        r = _oneway_icc(scenario_groups)
        pc_rows.append(
            {
                "metric": metric,
                "icc": r["icc"],
                "ci_lower": r["ci_lower"],
                "ci_upper": r["ci_upper"],
                "sigma2_scenario": r["sigma2_scenario"],
                "sigma2_residual": r["sigma2_residual"],
                "n_scenarios": len(scenario_groups),
                "n_models": n_models,
                "n_trials": min_k,
            }
        )

    pooled_centered = pd.DataFrame(pc_rows)

    # ── pooled_twoway (Option B: two-way random effects with interaction) ──────
    tw_rows: list[dict] = []
    for metric, grp in mean_by_trial.groupby("metric"):
        # Build balanced 3-D array Y[scenario, model, trial]
        scenarios = sorted(grp["record_id"].unique())
        models = sorted(grp["run_id"].unique())
        n_s, n_m = len(scenarios), len(models)
        n_t = grp.groupby(["record_id", "run_id"])["trial"].nunique().min()
        if n_s < 2 or n_m < 2 or n_t < 2:
            continue

        s_idx = {s: i for i, s in enumerate(scenarios)}
        m_idx = {m: i for i, m in enumerate(models)}

        Y = np.full((n_s, n_m, n_t), np.nan)
        for _, row in grp.iterrows():
            si = s_idx.get(row["record_id"])
            mi = m_idx.get(row["run_id"])
            ti = int(row["trial"])
            if si is not None and mi is not None and ti < n_t:
                Y[si, mi, ti] = row["mean_score"]

        if np.isnan(Y).any():
            continue  # skip if any cell is missing

        r = _twoway_icc(Y)
        tw_rows.append(
            {
                "metric": metric,
                **{k: v for k, v in r.items() if k not in ("ss_scenario", "ss_model", "ss_interaction", "ss_residual")},
            }
        )

    pooled_twoway = pd.DataFrame(tw_rows)

    return {
        "per_model": per_model,
        "pooled_centered": pooled_centered,
        "pooled_twoway": pooled_twoway,
    }


def compute_judge_variance(df: pd.DataFrame, metrics: list[str] = JUDGE_METRICS) -> pd.DataFrame:
    """Per (run_id, metric, record_id, trial): std dev, range, and whether score changed across iterations.

    Args:
        df: Flat scores dataframe from load_data.load_scores()
        metrics: List of metric names to include

    Returns:
        Dataframe with columns: run_id, run_label, metric, record_id, trial, std, range, score_changed
    """
    filtered = df[df["metric"].isin(metrics)]
    grouped = filtered.groupby(["run_id", "run_label", "metric", "record_id", "trial"])["normalized_score"]
    result = grouped.agg(
        std=lambda x: float(np.std(x, ddof=0)),  # population std (descriptive over these N iterations)
        range=lambda x: float(x.max() - x.min()),
    ).reset_index()
    # Build score_changed as a Python-bool object-dtype column.
    # Using .agg() produces a numpy bool_ dtype column; .iloc indexing then returns
    # numpy.bool_ which fails `is True`/`is False` identity checks. Build separately
    # with explicit Python bools in an object-dtype Series to avoid this.
    nunique = grouped.nunique().reset_index(name="nunique")
    result["score_changed"] = pd.array([bool(v > 1) for v in nunique["nunique"]], dtype=object)
    return result


def compute_trajectory_variance(df: pd.DataFrame, metrics: list[str] = JUDGE_METRICS) -> pd.DataFrame:
    """Per (run_id, metric, record_id): std dev and range across trials, with judge noise removed.

    Judge noise is removed by averaging over iterations before computing cross-trial std dev.

    Args:
        df: Flat scores dataframe from load_data.load_scores()
        metrics: List of metric names to include

    Returns:
        Dataframe with columns: run_id, run_label, metric, record_id, std, range
    """
    filtered = df[df["metric"].isin(metrics)]
    # Average across iterations to remove judge noise
    mean_by_trial = (
        filtered.groupby(["run_id", "run_label", "metric", "record_id", "trial"])["normalized_score"]
        .mean()
        .reset_index()
        .rename(columns={"normalized_score": "mean_score"})
    )
    # Std dev across trials
    result = (
        mean_by_trial.groupby(["run_id", "run_label", "metric", "record_id"])["mean_score"]
        .agg(
            std=lambda x: float(np.std(x, ddof=0)),  # population std (descriptive over these N trials)
            range=lambda x: float(x.max() - x.min()),
        )
        .reset_index()
    )
    return result


def compute_judge_variance_summary(judge_var: pd.DataFrame) -> pd.DataFrame:
    """Aggregate judge variance stats per (run_id, metric).

    Args:
        judge_var: Output of compute_judge_variance()

    Returns:
        Dataframe with columns: run_id, run_label, metric, mean_std, std_of_std,
        std_min, std_max, mean_range, pct_changed, n
    """
    return (
        judge_var.groupby(["run_id", "run_label", "metric"])
        .agg(
            mean_std=("std", "mean"),
            std_of_std=("std", "std"),
            std_min=("std", "min"),
            std_max=("std", "max"),
            mean_range=("range", "mean"),
            pct_changed=("score_changed", "mean"),
            n=("std", "count"),
        )
        .reset_index()
    )


def compute_trajectory_variance_summary(traj_var: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trajectory variance stats per (run_id, metric).

    Args:
        traj_var: Output of compute_trajectory_variance()

    Returns:
        Dataframe with columns: run_id, run_label, metric, mean_std, std_of_std,
        std_min, std_max, mean_range, n
    """
    return (
        traj_var.groupby(["run_id", "run_label", "metric"])
        .agg(
            mean_std=("std", "mean"),
            std_of_std=("std", "std"),
            std_min=("std", "min"),
            std_max=("std", "max"),
            mean_range=("range", "mean"),
            n=("std", "count"),
        )
        .reset_index()
    )


def compute_composite_stability(agg_df: pd.DataFrame) -> pd.DataFrame:
    """Per (run_id, iteration): compute run-level pass@1, pass@k, pass^k (theoretical), and mean composites.

    For each composite (EVA-overall_pass, EVA-A_pass, EVA-X_pass):
      - pass@1 = mean(c/n) across scenarios, where c = passing trials per scenario
      - pass@k = mean(1 if c>=1 else 0) across scenarios
      - pass^k = mean((c/n)^K) across scenarios, where K=3 (fixed number of trials)

    For each mean composite (EVA-A_mean, EVA-X_mean, EVA-overall_mean):
      - Mean across all (record, trial) rows

    Args:
        agg_df: Output of load_data.load_aggregate_scores()

    Returns:
        Dataframe with one row per (run_id, iteration) and composite columns
    """
    K = 3  # fixed number of trials (k in pass@k and pass^k)
    PASS_THRESHOLD = 1.0  # EVA composite pass columns store exactly 0.0 (fail) or 1.0 (pass)

    pass_composites = [c for c in ["EVA-overall_pass", "EVA-A_pass", "EVA-X_pass"] if c in agg_df.columns]
    mean_composites = [c for c in ["EVA-A_mean", "EVA-X_mean", "EVA-overall_mean"] if c in agg_df.columns]

    results = []
    for (run_id, run_label, iteration), grp in agg_df.groupby(["run_id", "run_label", "iteration"]):
        row: dict = {"run_id": run_id, "run_label": run_label, "iteration": iteration}

        for composite in pass_composites:
            scenario_stats = []
            for _record_id, scenario_grp in grp.groupby("record_id"):
                vals = scenario_grp[composite].dropna().values
                n = len(vals)
                if n == 0:
                    continue
                c = int(np.sum(vals >= PASS_THRESHOLD))
                scenario_stats.append(
                    {
                        "pass_at_1": c / n,
                        "pass_at_k": 1.0 if c >= 1 else 0.0,
                        "pass_power_k": (c / n) ** K,
                    }
                )
            if scenario_stats:
                s = pd.DataFrame(scenario_stats)
                col_base = composite.removesuffix("_pass")
                row[f"{col_base}_pass_at_1"] = float(s["pass_at_1"].mean())
                row[f"{col_base}_pass_at_k"] = float(s["pass_at_k"].mean())
                row[f"{col_base}_pass_power_k"] = float(s["pass_power_k"].mean())

        for composite in mean_composites:
            if composite in grp.columns:
                row[composite] = float(grp[composite].mean())

        results.append(row)

    return pd.DataFrame(results)


def find_borderline_scenarios(judge_var: pd.DataFrame) -> pd.DataFrame:
    """Return (record_id, trial, metric) rows where the judge score changed across iterations.

    Args:
        judge_var: Output of compute_judge_variance()

    Returns:
        Filtered dataframe of rows where score_changed is True
    """
    return judge_var[judge_var["score_changed"]].copy()


def compute_statistical_tests(
    judge_var: pd.DataFrame,
    traj_var: pd.DataFrame,
    metrics: list[str] = JUDGE_METRICS,
) -> dict[str, pd.DataFrame]:
    """Run statistical tests comparing judge and trial variance.

    Q1a: Is judge variance different from trial variance (per model × metric)?
        Aggregate judge stds to record level (mean over trials) to match
        trial variance granularity, then run paired Wilcoxon signed-rank.
        Bonferroni correction across number of models.

    Q1b: Does the judge-vs-trial gap vary across models (per metric)?
        Compute delta = mean_judge_std - traj_std per record.
        Kruskal-Wallis on deltas across models.

    Q2: Does judge variance differ significantly across models (per metric)?
        Kruskal-Wallis on per-(record,trial) judge stds across models.
        If significant: pairwise Mann-Whitney U with Bonferroni correction.
        Effect size: rank-biserial correlation r = 1 - 2U/(n1*n2).

    Q3: Does trial variance differ significantly across models (per metric)?
        Same approach as Q2 but applied to per-record trial stds.

    Args:
        judge_var: Output of compute_judge_variance()
        traj_var: Output of compute_trajectory_variance()
        metrics: List of metric names to test

    Returns:
        Dict with DataFrames keyed by "q1a", "q1b", "q2_kw", "q2_pairwise",
        "q3_kw", "q3_pairwise"
    """
    alpha = 0.05

    # ── Q2: Kruskal-Wallis + pairwise Mann-Whitney U across models ────────────
    q2_kw_rows: list[dict] = []
    q2_pw_rows: list[dict] = []

    for metric in metrics:
        sub = judge_var[judge_var["metric"] == metric]
        run_ids = sub["run_id"].unique()

        groups: dict[str, np.ndarray] = {}
        labels: dict[str, str] = {}
        for r in run_ids:
            mask = sub["run_id"] == r
            groups[r] = sub.loc[mask, "std"].dropna().values  # drop NaN (e.g. skipped metrics)
            labels[r] = sub.loc[mask, "run_label"].iloc[0]

        valid = {r: g for r, g in groups.items() if len(g) >= 2}
        if len(valid) < 2:
            continue
        # kruskal raises ValueError when all values across all groups are identical
        all_vals = np.concatenate(list(valid.values()))
        if len(np.unique(all_vals)) < 2:
            continue

        try:
            stat, p = scipy_stats.kruskal(*valid.values())
        except ValueError:
            continue
        q2_kw_rows.append(
            {
                "metric": metric,
                "H": float(stat),
                "p": float(p),
                "significant": bool(p < alpha),
                "n_models": len(valid),
            }
        )

        pairs = list(combinations(list(valid.keys()), 2))
        n_comparisons = len(pairs)
        for r1, r2 in pairs:
            g1, g2 = valid[r1], valid[r2]
            u_stat, p_raw = scipy_stats.mannwhitneyu(g1, g2, alternative="two-sided")
            p_bonf = min(float(p_raw) * n_comparisons, 1.0)
            n1, n2 = len(g1), len(g2)
            r_rb = float(1 - (2 * u_stat) / (n1 * n2))
            q2_pw_rows.append(
                {
                    "metric": metric,
                    "model_1": labels[r1],
                    "model_2": labels[r2],
                    "U": float(u_stat),
                    "p_raw": float(p_raw),
                    "p_bonferroni": p_bonf,
                    "significant": bool(p_bonf < alpha),
                    "effect_r": r_rb,
                    "n_1": int(n1),
                    "n_2": int(n2),
                }
            )

    # ── Q1a: Paired Wilcoxon (per model × metric) ─────────────────────────────
    # Aggregate judge std devs to record level (mean over trials)
    judge_by_record = (
        judge_var[judge_var["metric"].isin(metrics)]
        .groupby(["run_id", "run_label", "metric", "record_id"])["std"]
        .mean()
        .reset_index()
        .rename(columns={"std": "mean_judge_std"})
    )

    q1a_rows: list[dict] = []

    for metric in metrics:
        sub_j = judge_by_record[judge_by_record["metric"] == metric]
        sub_t = traj_var[traj_var["metric"] == metric][["run_id", "run_label", "record_id", "std"]].rename(
            columns={"std": "traj_std"}
        )
        run_ids = sub_j["run_id"].unique()
        bonf_factor = max(len(run_ids), 1)

        for run_id in run_ids:
            j_sub = sub_j[sub_j["run_id"] == run_id]
            t_sub = sub_t[sub_t["run_id"] == run_id]
            run_label = j_sub["run_label"].iloc[0]

            merged = pd.merge(
                j_sub[["record_id", "mean_judge_std"]],
                t_sub[["record_id", "traj_std"]],
                on="record_id",
            ).dropna()

            if len(merged) < 5:
                continue

            diffs = merged["mean_judge_std"] - merged["traj_std"]
            median_delta = float(diffs.median())
            direction = "judge > trial" if median_delta > 0 else ("judge < trial" if median_delta < 0 else "equal")

            try:
                with np.errstate(divide="ignore", invalid="ignore"):
                    stat, p_raw = scipy_stats.wilcoxon(
                        merged["mean_judge_std"].values,
                        merged["traj_std"].values,
                        alternative="two-sided",
                        zero_method="wilcox",
                    )
                p_bonf = min(float(p_raw) * bonf_factor, 1.0)
                q1a_rows.append(
                    {
                        "metric": metric,
                        "model": run_label,
                        "n_records": int(len(merged)),
                        "median_judge_std": float(merged["mean_judge_std"].median()),
                        "median_traj_std": float(merged["traj_std"].median()),
                        "median_delta": median_delta,
                        "W": float(stat),
                        "p_raw": float(p_raw),
                        "p_bonferroni": p_bonf,
                        "significant": bool(p_bonf < alpha),
                        "direction": direction,
                    }
                )
            except Exception:
                q1a_rows.append(
                    {
                        "metric": metric,
                        "model": run_label,
                        "n_records": int(len(merged)),
                        "median_judge_std": float(merged["mean_judge_std"].median()),
                        "median_traj_std": float(merged["traj_std"].median()),
                        "median_delta": median_delta,
                        "W": float("nan"),
                        "p_raw": float("nan"),
                        "p_bonferroni": float("nan"),
                        "significant": False,
                        "direction": direction,
                    }
                )

    # ── Q1b: K-W on per-record deltas across models ───────────────────────────
    merged_all = pd.merge(
        judge_by_record[judge_by_record["metric"].isin(metrics)],
        traj_var[traj_var["metric"].isin(metrics)][["run_id", "metric", "record_id", "std"]].rename(
            columns={"std": "traj_std"}
        ),
        on=["run_id", "metric", "record_id"],
    ).dropna(subset=["mean_judge_std", "traj_std"])
    merged_all["delta"] = merged_all["mean_judge_std"] - merged_all["traj_std"]

    q1b_rows: list[dict] = []
    for metric in metrics:
        sub = merged_all[merged_all["metric"] == metric]
        run_ids = sub["run_id"].unique()
        groups_delta = [sub[sub["run_id"] == r]["delta"].values for r in run_ids]
        groups_delta = [g for g in groups_delta if len(g) >= 2]
        if len(groups_delta) < 2:
            continue
        stat, p = scipy_stats.kruskal(*groups_delta)
        q1b_rows.append(
            {
                "metric": metric,
                "H": float(stat),
                "p": float(p),
                "significant": bool(p < alpha),
                "n_models": len(groups_delta),
            }
        )

    # ── Q3: Kruskal-Wallis + pairwise Mann-Whitney U on trial stds ───────────
    # Same structure as Q2 but on per-record trial stds (traj_var["std"]).
    # Trial variance is already at record level (one std dev per record), so
    # no aggregation step is needed — the groups are smaller than Q2's (N records
    # rather than N×K (record,trial) pairs) but the test is identical.
    q3_kw_rows: list[dict] = []
    q3_pw_rows: list[dict] = []

    for metric in metrics:
        sub = traj_var[traj_var["metric"] == metric]
        run_ids_t = sub["run_id"].unique()

        t_groups: dict[str, np.ndarray] = {}
        t_labels: dict[str, str] = {}
        for r in run_ids_t:
            mask = sub["run_id"] == r
            t_groups[r] = sub.loc[mask, "std"].dropna().values
            t_labels[r] = sub.loc[mask, "run_label"].iloc[0]

        t_valid = {r: g for r, g in t_groups.items() if len(g) >= 2}
        if len(t_valid) < 2:
            continue

        t_all_vals = np.concatenate(list(t_valid.values()))
        if len(np.unique(t_all_vals)) < 2:
            continue

        try:
            stat, p = scipy_stats.kruskal(*t_valid.values())
        except ValueError:
            continue
        q3_kw_rows.append(
            {
                "metric": metric,
                "H": float(stat),
                "p": float(p),
                "significant": bool(p < alpha),
                "n_models": len(t_valid),
            }
        )

        t_pairs = list(combinations(list(t_valid.keys()), 2))
        n_comparisons_t = len(t_pairs)
        for r1, r2 in t_pairs:
            g1, g2 = t_valid[r1], t_valid[r2]
            u_stat, p_raw = scipy_stats.mannwhitneyu(g1, g2, alternative="two-sided")
            p_bonf = min(float(p_raw) * n_comparisons_t, 1.0)
            n1, n2 = len(g1), len(g2)
            r_rb = float(1 - (2 * u_stat) / (n1 * n2))
            q3_pw_rows.append(
                {
                    "metric": metric,
                    "model_1": t_labels[r1],
                    "model_2": t_labels[r2],
                    "U": float(u_stat),
                    "p_raw": float(p_raw),
                    "p_bonferroni": p_bonf,
                    "significant": bool(p_bonf < alpha),
                    "effect_r": r_rb,
                    "n_1": int(n1),
                    "n_2": int(n2),
                }
            )

    return {
        "q1a": pd.DataFrame(q1a_rows),
        "q1b": pd.DataFrame(q1b_rows),
        "q2_kw": pd.DataFrame(q2_kw_rows),
        "q2_pairwise": pd.DataFrame(q2_pw_rows),
        "q3_kw": pd.DataFrame(q3_kw_rows),
        "q3_pairwise": pd.DataFrame(q3_pw_rows),
    }


def compute_within_type_tests(
    judge_var: pd.DataFrame,
    traj_var: pd.DataFrame,
    run_type_map: dict[str, str],
    metrics: list[str] = JUDGE_METRICS,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Run Q2/Q3 Kruskal-Wallis + pairwise tests separately within each model type.

    Prevents cross-type comparisons (cascade vs S2S) from dominating the K-W
    signal and masking within-type differences.

    Args:
        judge_var: Output of compute_judge_variance()
        traj_var: Output of compute_trajectory_variance()
        run_type_map: Dict mapping run_id → type string (e.g. "cascade" or "s2s")
        metrics: List of metric names to test

    Returns:
        Dict keyed by type string → same sub-dict structure as compute_statistical_tests
        (keys: "q2_kw", "q2_pairwise", "q3_kw", "q3_pairwise").
        Only types with ≥ 2 runs are included.
    """
    alpha = 0.05

    # Group run_ids by type
    type_to_run_ids: dict[str, list[str]] = {}
    for run_id, rtype in run_type_map.items():
        type_to_run_ids.setdefault(rtype, []).append(run_id)

    results: dict[str, dict[str, pd.DataFrame]] = {}

    for rtype, run_ids_for_type in type_to_run_ids.items():
        if len(run_ids_for_type) < 2:
            continue

        q2_kw_rows: list[dict] = []
        q2_pw_rows: list[dict] = []
        q3_kw_rows: list[dict] = []
        q3_pw_rows: list[dict] = []

        # ── Q2 within type ───────────────────────────────────────────────────
        for metric in metrics:
            sub = judge_var[(judge_var["metric"] == metric) & (judge_var["run_id"].isin(run_ids_for_type))]
            groups: dict[str, np.ndarray] = {}
            labels: dict[str, str] = {}
            for r in run_ids_for_type:
                mask = sub["run_id"] == r
                vals = sub.loc[mask, "std"].dropna().values
                if len(vals) >= 2:
                    groups[r] = vals
                    lbl = sub.loc[mask, "run_label"]
                    if not lbl.empty:
                        labels[r] = lbl.iloc[0]

            if len(groups) < 2:
                continue

            wt_all_vals = np.concatenate(list(groups.values()))
            if len(np.unique(wt_all_vals)) < 2:
                continue

            try:
                stat, p = scipy_stats.kruskal(*groups.values())
            except ValueError:
                continue
            q2_kw_rows.append(
                {
                    "metric": metric,
                    "H": float(stat),
                    "p": float(p),
                    "significant": bool(p < alpha),
                    "n_models": len(groups),
                }
            )

            pairs = list(combinations(list(groups.keys()), 2))
            n_comp = len(pairs)
            for r1, r2 in pairs:
                g1, g2 = groups[r1], groups[r2]
                u_stat, p_raw = scipy_stats.mannwhitneyu(g1, g2, alternative="two-sided")
                p_bonf = min(float(p_raw) * n_comp, 1.0)
                n1, n2 = len(g1), len(g2)
                q2_pw_rows.append(
                    {
                        "metric": metric,
                        "model_1": labels.get(r1, r1),
                        "model_2": labels.get(r2, r2),
                        "U": float(u_stat),
                        "p_raw": float(p_raw),
                        "p_bonferroni": p_bonf,
                        "significant": bool(p_bonf < alpha),
                        "effect_r": float(1 - (2 * u_stat) / (n1 * n2)),
                        "n_1": int(n1),
                        "n_2": int(n2),
                    }
                )

        # ── Q3 within type ───────────────────────────────────────────────────
        for metric in metrics:
            sub = traj_var[(traj_var["metric"] == metric) & (traj_var["run_id"].isin(run_ids_for_type))]
            t_groups: dict[str, np.ndarray] = {}
            t_labels: dict[str, str] = {}
            for r in run_ids_for_type:
                mask = sub["run_id"] == r
                vals = sub.loc[mask, "std"].dropna().values
                if len(vals) >= 2:
                    t_groups[r] = vals
                    lbl = sub.loc[mask, "run_label"]
                    if not lbl.empty:
                        t_labels[r] = lbl.iloc[0]

            if len(t_groups) < 2:
                continue

            wt_t_all_vals = np.concatenate(list(t_groups.values()))
            if len(np.unique(wt_t_all_vals)) < 2:
                continue

            try:
                stat, p = scipy_stats.kruskal(*t_groups.values())
            except ValueError:
                continue
            q3_kw_rows.append(
                {
                    "metric": metric,
                    "H": float(stat),
                    "p": float(p),
                    "significant": bool(p < alpha),
                    "n_models": len(t_groups),
                }
            )

            t_pairs = list(combinations(list(t_groups.keys()), 2))
            n_comp_t = len(t_pairs)
            for r1, r2 in t_pairs:
                g1, g2 = t_groups[r1], t_groups[r2]
                u_stat, p_raw = scipy_stats.mannwhitneyu(g1, g2, alternative="two-sided")
                p_bonf = min(float(p_raw) * n_comp_t, 1.0)
                n1, n2 = len(g1), len(g2)
                q3_pw_rows.append(
                    {
                        "metric": metric,
                        "model_1": t_labels.get(r1, r1),
                        "model_2": t_labels.get(r2, r2),
                        "U": float(u_stat),
                        "p_raw": float(p_raw),
                        "p_bonferroni": p_bonf,
                        "significant": bool(p_bonf < alpha),
                        "effect_r": float(1 - (2 * u_stat) / (n1 * n2)),
                        "n_1": int(n1),
                        "n_2": int(n2),
                    }
                )

        results[rtype] = {
            "q2_kw": pd.DataFrame(q2_kw_rows),
            "q2_pairwise": pd.DataFrame(q2_pw_rows),
            "q3_kw": pd.DataFrame(q3_kw_rows),
            "q3_pairwise": pd.DataFrame(q3_pw_rows),
        }

    return results


def compute_q0_tests(
    judge_var: pd.DataFrame,
    traj_var: pd.DataFrame,
    metrics: list[str] = JUDGE_METRICS,
) -> dict[str, pd.DataFrame]:
    """Q0: Is variance significantly greater than zero?

    One-sample Wilcoxon signed-rank test (alternative="greater") against 0.
    Std devs are bounded at zero and right-skewed, so a non-parametric test is used.
    A significant result means the median std dev is reliably above zero — i.e. there
    is genuine variance to study, not just measurement noise.

    Run pooled across models (per metric) and per (metric, model).

    Args:
        judge_var: Output of compute_judge_variance() — per (run, metric, record, trial) std devs
        traj_var: Output of compute_trajectory_variance() — per (run, metric, record) std devs
        metrics: Metrics to test

    Returns:
        Dict with DataFrames:
            "q0_judge_pooled"    — per metric, std devs pooled across all models
            "q0_judge_per_model" — per (metric, model)
            "q0_trial_pooled"    — per metric, std devs pooled across all models
            "q0_trial_per_model" — per (metric, model)
    """
    alpha = 0.05

    def _wilcoxon_vs_zero(vals: np.ndarray) -> tuple[float, float]:
        """One-sample Wilcoxon against 0, alternative='greater'. Returns (W, p)."""
        nonzero = vals[vals != 0]
        if len(nonzero) < 5:
            return float("nan"), float("nan")
        with np.errstate(divide="ignore", invalid="ignore"):
            stat, p = scipy_stats.wilcoxon(nonzero, alternative="greater")
        return float(stat), float(p)

    def _run_q0(var_df: pd.DataFrame, std_col: str = "std") -> tuple[list, list]:
        pooled_rows: list[dict] = []
        per_model_rows: list[dict] = []

        for metric in metrics:
            sub = var_df[var_df["metric"] == metric]

            # Pooled across all models
            all_vals = sub[std_col].values
            W, p = _wilcoxon_vs_zero(all_vals)
            pooled_rows.append(
                {
                    "metric": metric,
                    "n": int(len(all_vals)),
                    "median_std": float(np.median(all_vals)),
                    "mean_std": float(np.mean(all_vals)),
                    "W": W,
                    "p": p,
                    "significant": bool(not np.isnan(p) and p < alpha),
                }
            )

            # Per model
            for run_id in sub["run_id"].unique():
                run_sub = sub[sub["run_id"] == run_id]
                run_label = run_sub["run_label"].iloc[0]
                vals = run_sub[std_col].values
                W_m, p_m = _wilcoxon_vs_zero(vals)
                per_model_rows.append(
                    {
                        "metric": metric,
                        "model": run_label,
                        "n": int(len(vals)),
                        "median_std": float(np.median(vals)),
                        "mean_std": float(np.mean(vals)),
                        "W": W_m,
                        "p": p_m,
                        "significant": bool(not np.isnan(p_m) and p_m < alpha),
                    }
                )

        return pooled_rows, per_model_rows

    j_pooled, j_per_model = _run_q0(judge_var[judge_var["metric"].isin(metrics)])
    t_pooled, t_per_model = _run_q0(traj_var[traj_var["metric"].isin(metrics)])

    return {
        "q0_judge_pooled": pd.DataFrame(j_pooled),
        "q0_judge_per_model": pd.DataFrame(j_per_model),
        "q0_trial_pooled": pd.DataFrame(t_pooled),
        "q0_trial_per_model": pd.DataFrame(t_per_model),
    }
