# analysis/eva-bench-stats/stats_variance.py
# Config: local/eva-bench-stats/variance_config.yaml
#
# Reads from: output_processed/eva-bench-stats/variance/data/
# Writes to:  output_processed/eva-bench-stats/variance/stats/
"""Statistical tests for variance analysis.

Pure computation: reads DataFrames, returns DataFrames. No Streamlit.

Pipeline:
  compute_icc                — intraclass correlation (per_model, pooled_centered, pooled_twoway)
  compute_statistical_tests  — Q1a, Q1b, Q2, Q3 tests
  compute_q0_tests           — Q0 one-sample Wilcoxon (is variance > 0?)
  compute_within_type_tests  — Q2/Q3 separately per model type (cascade / s2s)
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats as scipy_stats
from scipy.stats import chi2
from scipy.stats import f as f_dist

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "local" / "eva-bench-stats" / "variance_config.yaml"


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
        sigma2_hat: float, icc_hat: float, ms_num: float, ms_den: float, df_num: int, df_den: int, divisor: float
    ) -> tuple[float, float]:
        """Satterthwaite CI for a variance component estimated as (ms_num - ms_den)/divisor."""
        if sigma2_hat <= 0 or divisor <= 0:
            return 0.0, min(1.0, icc_hat)  # degenerate — return point estimate bounds
        L = ms_num - ms_den
        df_L = L**2 / (ms_num**2 / df_num + ms_den**2 / df_den)
        df_L = max(df_L, 1.0)
        ci_lo_var = df_L * sigma2_hat / chi2.ppf(1 - alpha / 2, df_L)
        ci_hi_var = df_L * sigma2_hat / chi2.ppf(alpha / 2, df_L)
        ci_lo_icc = max(0.0, min(1.0, ci_lo_var / sigma2_tot))
        ci_hi_icc = max(0.0, min(1.0, ci_hi_var / sigma2_tot))
        return ci_lo_icc, ci_hi_icc

    ci_lo_s, ci_hi_s = _satterthwaite_ci(sigma2_s, icc_s, ms_s, ms_sm, df_s, df_sm, n_t * n_m)
    ci_lo_m, ci_hi_m = _satterthwaite_ci(sigma2_m, icc_m, ms_m, ms_sm, df_m, df_sm, n_t * n_s)

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
    metrics: list[str],
    domain: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Intraclass correlation at the scenario level for each metric.

    ICC = σ²_scenario / σ²_total quantifies what fraction of score variance
    is attributable to scenario identity (i.e. which scenario is being evaluated).

    `per_model` and `pooled_centered` are computed within the selected domain
    when `domain` is given, or across all data when `domain` is None.
    `pooled_twoway` is always computed per domain because record_ids don't
    overlap across domains.

    Args:
        df: Flat scores DataFrame from load_data.load_scores().
        metrics: Metric names to include.
        domain: Optional domain to restrict to.

    Returns:
        Dict with keys:
          "per_model"       – one row per (model_id, metric)
          "pooled_centered" – one row per metric (Option A: model-mean-centered)
          "pooled_twoway"   – one row per (domain, metric) (Option B)
    """
    # ── Average over iterations to remove judge noise ─────────────────────────
    has_domain_col = "domain" in df.columns
    df_filtered = df[df["metric"].isin(metrics)]
    if domain is not None and has_domain_col:
        df_filtered = df_filtered[df_filtered["domain"] == domain]
    group_cols = (
        ["run_id", "run_label"]
        + (["domain"] if has_domain_col else [])
        + [
            "metric",
            "record_id",
            "trial",
        ]
    )
    mean_by_trial = (
        df_filtered.groupby(group_cols, dropna=False)["normalized_score"]
        .mean()
        .reset_index()
        .rename(columns={"normalized_score": "mean_score"})
    )
    if not mean_by_trial.empty:
        mean_by_trial["model_id"] = mean_by_trial["run_label"].map(_model_id_from_label)
    else:
        mean_by_trial["model_id"] = pd.Series(dtype=object)

    # ── per_model ─────────────────────────────────────────────────────────────
    # Group by model_id (actual model identity). When a model has multiple
    # (model × domain) cells (i.e. domain is None and we're pooling), all of
    # them feed the one ICC fit for that model.
    pm_rows: list[dict] = []
    for (model_id, metric), grp in mean_by_trial.groupby(["model_id", "metric"]):
        # When pooling across domains, scenarios from different domains share
        # different record_id namespaces — give them globally-unique ids so
        # the per-scenario groupby doesn't accidentally merge cross-domain.
        if has_domain_col:
            grp = grp.assign(_scen=grp["domain"].astype(str) + "::" + grp["record_id"].astype(str))
            scen_col = "_scen"
        else:
            scen_col = "record_id"

        scenario_groups = [sub["mean_score"].values for _, sub in grp.groupby(scen_col) if len(sub) >= 2]
        if len(scenario_groups) < 2:
            continue
        min_k = min(len(g) for g in scenario_groups)
        scenario_groups = [g[:min_k] for g in scenario_groups]

        r = _oneway_icc(scenario_groups)
        pm_rows.append(
            {
                "model_id": model_id,
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
    # Center each model_id's data (across whatever cells survived the domain
    # filter) by its own mean, then run ICC over the centered pool.
    pc_rows: list[dict] = []
    for metric, grp in mean_by_trial.groupby("metric"):
        if has_domain_col:
            grp = grp.assign(_scen=grp["domain"].astype(str) + "::" + grp["record_id"].astype(str))
            scen_col = "_scen"
        else:
            scen_col = "record_id"
        model_means = grp.groupby("model_id")["mean_score"].transform("mean")
        centered = grp.assign(mean_score=grp["mean_score"] - model_means)

        scenario_groups = [sub["mean_score"].values for _, sub in centered.groupby(scen_col) if len(sub) >= 2]
        if len(scenario_groups) < 2:
            continue
        min_k = min(len(g) for g in scenario_groups)
        scenario_groups = [g[:min_k] for g in scenario_groups]

        n_models = grp["model_id"].nunique()
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
    # Stratified by domain when present: each domain has its own (scenario × model)
    # balanced array, since record_ids are domain-specific.
    tw_rows: list[dict] = []
    has_domain = "domain" in mean_by_trial.columns and mean_by_trial["domain"].notna().any()
    group_keys = ["domain", "metric"] if has_domain else ["metric"]
    for keys, grp in mean_by_trial.groupby(group_keys):
        if has_domain:
            domain, metric = keys
        else:
            domain, metric = None, keys

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
        row_out = {
            "domain": domain,
            "metric": metric,
            **{k: v for k, v in r.items() if k not in ("ss_scenario", "ss_model", "ss_interaction", "ss_residual")},
        }
        if not has_domain:
            row_out.pop("domain")
        tw_rows.append(row_out)

    pooled_twoway = pd.DataFrame(tw_rows)

    return {
        "per_model": per_model,
        "pooled_centered": pooled_centered,
        "pooled_twoway": pooled_twoway,
    }


def _model_id_from_label(run_label: str) -> str:
    """Strip the trailing ' — <domain>' from a per-domain run_label."""
    return run_label.split(" — ")[0].strip()


def _filter_and_relabel(df: pd.DataFrame, domain: str | None) -> pd.DataFrame:
    """Filter rows to a single domain (or pass through if None) and add `model_id`.

    `model_id` is the actual model identity — i.e. run_label minus the trailing
    domain suffix. When `domain` is None, rows from all domains are kept and
    per-model tests pool across domains. When `domain` is set, only that
    domain's rows survive.
    """
    out = df.copy()
    if domain is not None and "domain" in out.columns:
        out = out[out["domain"] == domain]
    if not out.empty:
        out["model_id"] = out["run_label"].map(_model_id_from_label)
    else:
        out["model_id"] = pd.Series(dtype=object)
    return out


def _filter_and_relabel(df: pd.DataFrame, domain: str | None) -> pd.DataFrame:
    """Filter rows to a single domain (or pass through if None) and add `model_id`.

    `model_id` is the actual model identity — i.e. run_label minus the trailing
    domain suffix. When `domain` is None, rows from all domains are kept and
    per-model tests pool across domains. When `domain` is set, only that
    domain's rows survive.
    """
    if df.empty:
        out = df.copy()
        out["model_id"] = pd.Series(dtype=object)
        return out
    out = df.copy()
    if domain is not None and "domain" in out.columns:
        out = out[out["domain"] == domain]
    out["model_id"] = out["run_label"].map(_model_id_from_label)
    return out


def compute_variance_budget(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Per (model × metric) variance decomposition via classical sum-of-squares.

    Partitions the total observed variance into nested levels:

        SS_total = SS_domain + SS_scenario + SS_trial + SS_judge   (judge-graded)
        SS_total = SS_domain + SS_scenario + SS_trial              (deterministic)

    where each SS_X measures the spread of group means at level X around their
    parent group's mean (Pythagorean identity for orthogonal nested factors).
    σ²_X = SS_X / N. Descriptive only — no inference, no confidence intervals.

    Why not REML: with only 3 domain levels, REML's variance estimator is
    unreliable and can report nonsensical values (see
    lindsay_files/2026-05-02-variance-budget-reml-vs-ss.md).

    Deterministic metrics — where scores are constant across iterations within
    each (record, trial) — get a 3-pot decomposition with sigma2_judge / pct_judge
    set to NaN. Iterations are deduped to one before computing so duplicates
    don't inflate N.
    """
    if "domain" not in df.columns:
        raise ValueError("compute_variance_budget requires a 'domain' column on df")

    work = df[df["metric"].isin(metrics)].copy()
    work["model_id"] = work["run_label"].map(_model_id_from_label)
    work["scenario_uid"] = work["domain"].astype(str) + "::" + work["record_id"].astype(str)
    work["trial_uid"] = work["scenario_uid"] + "::t" + work["trial"].astype(str)

    rows: list[dict] = []
    for (model_id, metric), grp in work.groupby(["model_id", "metric"]):
        per_trial_unique = grp.groupby(["record_id", "trial"])["normalized_score"].nunique()
        is_deterministic = bool((per_trial_unique <= 1).all()) if not per_trial_unique.empty else False
        metric_type = "deterministic" if is_deterministic else "judge_graded"

        if is_deterministic:
            min_iter = grp["iteration"].min()
            grp = grp[grp["iteration"] == min_iter]

        n_obs = len(grp)
        n_domains = grp["domain"].nunique()
        n_scenarios = grp["scenario_uid"].nunique()
        n_trials = grp["trial_uid"].nunique()

        base_row = {
            "model_id": model_id,
            "metric": metric,
            "metric_type": metric_type,
            "n_obs": n_obs,
            "n_domains": n_domains,
            "n_scenarios": n_scenarios,
            "n_trials": n_trials,
        }

        if n_domains < 2 or n_scenarios < 2 or n_trials < 2 or n_obs < 10:
            rows.append({**base_row, "converged": False, "fit_error": "insufficient data"})
            continue

        y = grp["normalized_score"].astype(float).to_numpy()
        grand_mean = float(y.mean())

        # Per-row level means. Using transform() so each row has its parent
        # group's mean — this lets SS terms be computed as elementwise diffs.
        domain_mean = grp.groupby("domain")["normalized_score"].transform("mean").to_numpy()
        scenario_mean = grp.groupby(["domain", "scenario_uid"])["normalized_score"].transform("mean").to_numpy()
        trial_mean = (
            grp.groupby(["domain", "scenario_uid", "trial_uid"])["normalized_score"].transform("mean").to_numpy()
        )

        ss_domain = float(((domain_mean - grand_mean) ** 2).sum())
        ss_scenario = float(((scenario_mean - domain_mean) ** 2).sum())
        if is_deterministic:
            # No iteration replication; trial is the residual.
            ss_trial = float(((y - scenario_mean) ** 2).sum())
            ss_judge = None
            ss_total = ss_domain + ss_scenario + ss_trial
        else:
            ss_trial = float(((trial_mean - scenario_mean) ** 2).sum())
            ss_judge = float(((y - trial_mean) ** 2).sum())
            ss_total = ss_domain + ss_scenario + ss_trial + ss_judge

        if ss_total <= 0:
            rows.append({**base_row, "converged": False, "fit_error": "zero total variance"})
            continue

        sigma2_domain = ss_domain / n_obs
        sigma2_scenario = ss_scenario / n_obs
        sigma2_trial = ss_trial / n_obs
        sigma2_total = ss_total / n_obs
        sigma2_judge = (ss_judge / n_obs) if ss_judge is not None else float("nan")

        rows.append(
            {
                **base_row,
                "sigma2_domain": sigma2_domain,
                "sigma2_scenario": sigma2_scenario,
                "sigma2_trial": sigma2_trial,
                "sigma2_judge": sigma2_judge,
                "sigma2_total": sigma2_total,
                "pct_domain": ss_domain / ss_total,
                "pct_scenario": ss_scenario / ss_total,
                "pct_trial": ss_trial / ss_total,
                "pct_judge": (ss_judge / ss_total) if ss_judge is not None else float("nan"),
                "converged": True,
                "fit_error": "",
            }
        )

    return pd.DataFrame(rows)


def compute_statistical_tests(
    judge_var: pd.DataFrame,
    trial_var: pd.DataFrame,
    metrics: list[str],
    domain: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Run statistical tests comparing judge and trial variance.

    When `domain` is None, rows from all domains are pooled and per-model tests
    treat each `model_id` (model name minus trailing domain suffix) as the unit
    of analysis. When `domain` is set, only that domain's rows are used. In
    both cases, comparisons across "models" use the actual model identity, not
    the (model × domain) compound run_label.

    Q1a: Is judge variance different from trial variance (per model × metric)?
        Aggregate judge stds to record level (mean over trials) to match
        trial variance granularity, then run paired Wilcoxon signed-rank.
        Bonferroni correction across number of models.

    Q1b: Does the judge-vs-trial gap vary across models (per metric)?
        Compute delta = mean_judge_std - trial_std per record.
        Kruskal-Wallis on deltas across models.

    Q2: Does judge variance differ significantly across models (per metric)?
        Kruskal-Wallis on per-(record,trial) judge stds across models.
        If significant: pairwise Mann-Whitney U with Bonferroni correction.
        Effect size: rank-biserial correlation r = 1 - 2U/(n1*n2).

    Q3: Does trial variance differ significantly across models (per metric)?
        Same approach as Q2 but applied to per-record trial stds.

    Args:
        judge_var: Output of compute_judge_variance()
        trial_var: Output of compute_trial_variance()
        metrics: List of metric names to test
        domain: If set, filter to this domain; if None, pool across all domains.

    Returns:
        Dict with DataFrames keyed by "q1a", "q1b", "q2_kw", "q2_pairwise",
        "q3_kw", "q3_pairwise"
    """
    alpha = 0.05

    judge_var = _filter_and_relabel(judge_var, domain)
    trial_var = _filter_and_relabel(trial_var, domain)

    # ── Q2: Kruskal-Wallis + pairwise Mann-Whitney U across models ────────────
    q2_kw_rows: list[dict] = []
    q2_pw_rows: list[dict] = []

    for metric in metrics:
        sub = judge_var[judge_var["metric"] == metric]
        model_ids = sub["model_id"].unique()

        groups: dict[str, np.ndarray] = {}
        labels: dict[str, str] = {}
        for r in model_ids:
            mask = sub["model_id"] == r
            groups[r] = sub.loc[mask, "std"].dropna().values  # drop NaN (e.g. skipped metrics)
            labels[r] = r

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
    # Aggregate judge std devs to record level (mean over trials).
    # When pooling across domains, we key the merge on (domain, record_id) so we
    # don't accidentally cross-domain-merge records that share the same id.
    j_by_record_keys = ["model_id", "metric", "record_id"]
    if "domain" in judge_var.columns:
        j_by_record_keys.append("domain")
    judge_by_record = (
        judge_var[judge_var["metric"].isin(metrics)]
        .groupby(j_by_record_keys, dropna=False)["std"]
        .mean()
        .reset_index()
        .rename(columns={"std": "mean_judge_std"})
    )

    q1a_rows: list[dict] = []

    for metric in metrics:
        sub_j = judge_by_record[judge_by_record["metric"] == metric]
        trial_cols = ["model_id", "record_id", "std"] + (["domain"] if "domain" in trial_var.columns else [])
        sub_t = trial_var[trial_var["metric"] == metric][trial_cols].rename(columns={"std": "trial_std"})
        model_ids = sub_j["model_id"].unique()
        bonf_factor = max(len(model_ids), 1)
        merge_keys = ["record_id"] + (["domain"] if "domain" in sub_j.columns and "domain" in sub_t.columns else [])

        for model_id in model_ids:
            j_sub = sub_j[sub_j["model_id"] == model_id]
            t_sub = sub_t[sub_t["model_id"] == model_id]

            merged = pd.merge(
                j_sub[merge_keys + ["mean_judge_std"]],
                t_sub[merge_keys + ["trial_std"]],
                on=merge_keys,
            ).dropna()

            if len(merged) < 5:
                continue

            diffs = merged["mean_judge_std"] - merged["trial_std"]
            median_delta = float(diffs.median())
            direction = "judge > trial" if median_delta > 0 else ("judge < trial" if median_delta < 0 else "equal")

            try:
                with np.errstate(divide="ignore", invalid="ignore"):
                    stat, p_raw = scipy_stats.wilcoxon(
                        merged["mean_judge_std"].values,
                        merged["trial_std"].values,
                        alternative="two-sided",
                        zero_method="wilcox",
                    )
                p_bonf = min(float(p_raw) * bonf_factor, 1.0)
                q1a_rows.append(
                    {
                        "metric": metric,
                        "model": model_id,
                        "n_records": int(len(merged)),
                        "median_judge_std": float(merged["mean_judge_std"].median()),
                        "median_trial_std": float(merged["trial_std"].median()),
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
                        "model": model_id,
                        "n_records": int(len(merged)),
                        "median_judge_std": float(merged["mean_judge_std"].median()),
                        "median_trial_std": float(merged["trial_std"].median()),
                        "median_delta": median_delta,
                        "W": float("nan"),
                        "p_raw": float("nan"),
                        "p_bonferroni": float("nan"),
                        "significant": False,
                        "direction": direction,
                    }
                )

    # ── Q1b: K-W on per-record deltas across models ───────────────────────────
    q1b_merge_keys = ["model_id", "metric", "record_id"] + (
        ["domain"] if "domain" in judge_var.columns and "domain" in trial_var.columns else []
    )
    trial_cols = ["model_id", "metric", "record_id", "std"] + (["domain"] if "domain" in trial_var.columns else [])
    merged_all = pd.merge(
        judge_by_record[judge_by_record["metric"].isin(metrics)],
        trial_var[trial_var["metric"].isin(metrics)][trial_cols].rename(columns={"std": "trial_std"}),
        on=q1b_merge_keys,
    ).dropna(subset=["mean_judge_std", "trial_std"])
    merged_all["delta"] = merged_all["mean_judge_std"] - merged_all["trial_std"]

    q1b_rows: list[dict] = []
    for metric in metrics:
        sub = merged_all[merged_all["metric"] == metric]
        model_ids = sub["model_id"].unique()
        groups_delta = [sub[sub["model_id"] == m]["delta"].values for m in model_ids]
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
    # Same structure as Q2 but on per-record trial stds (trial_var["std"]).
    # Trial variance is already at record level (one std dev per record), so
    # no aggregation step is needed — the groups are smaller than Q2's (N records
    # rather than N×K (record,trial) pairs) but the test is identical.
    q3_kw_rows: list[dict] = []
    q3_pw_rows: list[dict] = []

    for metric in metrics:
        sub = trial_var[trial_var["metric"] == metric]
        model_ids_t = sub["model_id"].unique()

        t_groups: dict[str, np.ndarray] = {}
        t_labels: dict[str, str] = {}
        for r in model_ids_t:
            mask = sub["model_id"] == r
            t_groups[r] = sub.loc[mask, "std"].dropna().values
            t_labels[r] = r

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
    trial_var: pd.DataFrame,
    run_type_map: dict[str, str],
    metrics: list[str],
) -> dict[str, dict[str, pd.DataFrame]]:
    """Run Q2/Q3 Kruskal-Wallis + pairwise tests separately within each model type.

    Prevents cross-type comparisons (cascade vs S2S) from dominating the K-W
    signal and masking within-type differences.

    Args:
        judge_var: Output of compute_judge_variance()
        trial_var: Output of compute_trial_variance()
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
            sub = trial_var[(trial_var["metric"] == metric) & (trial_var["run_id"].isin(run_ids_for_type))]
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
    trial_var: pd.DataFrame,
    metrics: list[str],
    domain: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Q0: Is variance significantly greater than zero?

    One-sample Wilcoxon signed-rank test (alternative="greater") against 0.
    Std devs are bounded at zero and right-skewed, so a non-parametric test is used.
    A significant result means the median std dev is reliably above zero — i.e. there
    is genuine variance to study, not just measurement noise.

    Run pooled across models (per metric) and per (metric, model).

    Args:
        judge_var: Output of compute_judge_variance() — per (run, metric, record, trial) std devs
        trial_var: Output of compute_trial_variance() — per (run, metric, record) std devs
        metrics: Metrics to test
        domain: If set, filter to this domain; if None, pool across all domains.

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
            if len(all_vals) == 0:
                continue  # metric absent from this dataset (e.g. transcription for S2S-only view)
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

            # Per model — pool across run_ids that share the same model_id
            for model_id in sub["model_id"].unique():
                model_sub = sub[sub["model_id"] == model_id]
                vals = model_sub[std_col].values
                W_m, p_m = _wilcoxon_vs_zero(vals)
                per_model_rows.append(
                    {
                        "metric": metric,
                        "model": model_id,
                        "n": int(len(vals)),
                        "median_std": float(np.median(vals)),
                        "mean_std": float(np.mean(vals)),
                        "W": W_m,
                        "p": p_m,
                        "significant": bool(not np.isnan(p_m) and p_m < alpha),
                    }
                )

        return pooled_rows, per_model_rows

    j = _filter_and_relabel(judge_var[judge_var["metric"].isin(metrics)], domain)
    t = _filter_and_relabel(trial_var[trial_var["metric"].isin(metrics)], domain)
    j_pooled, j_per_model = _run_q0(j)
    t_pooled, t_per_model = _run_q0(t)

    return {
        "q0_judge_pooled": pd.DataFrame(j_pooled),
        "q0_judge_per_model": pd.DataFrame(j_per_model),
        "q0_trial_pooled": pd.DataFrame(t_pooled),
        "q0_trial_per_model": pd.DataFrame(t_per_model),
    }


def main(config_path: Path = CONFIG_PATH) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    project_root = config_path.parent.parent.parent
    base_dir = project_root / config["output_dir"]
    data_dir = base_dir / "data"
    stats_dir = base_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    metrics: list[str] = config["metrics"]
    runs: dict = config["runs"]
    run_type_map: dict[str, str] = {cfg["run_id"]: cfg.get("type", "cascade") for cfg in runs.values()}

    for name in ("scores.csv", "judge_var.csv", "trial_var.csv"):
        p = data_dir / name
        if not p.exists() or p.stat().st_size <= 1:
            raise FileNotFoundError(f"{name} not found or empty in {data_dir}. Run run_data.py first.")

    print(f"Loading data CSVs from {data_dir} ...")
    scores_df = pd.read_csv(data_dir / "scores.csv")
    judge_var = pd.read_csv(data_dir / "judge_var.csv")
    trial_var = pd.read_csv(data_dir / "trial_var.csv")
    print(f"  {len(scores_df):,} score rows loaded")

    domains_present: list[str] = []
    if "domain" in scores_df.columns:
        domains_present = sorted(scores_df["domain"].dropna().unique().tolist())

    print("Computing variance budget (4-level decomposition) ...")
    variance_budget = compute_variance_budget(scores_df, metrics)
    variance_budget.to_csv(stats_dir / "variance_budget.csv", index=False)

    # ── ICC: per-domain only for per_model and pooled_centered (residual
    # inflation when pooling across domains makes the cross-domain view
    # misleading); pooled_twoway is already domain-stratified internally.
    print("Computing ICC (per-domain + cross-domain twoway) ...")
    if domains_present:
        for d in domains_present:
            icc_d = compute_icc(scores_df, metrics, domain=d)
            icc_d["per_model"].to_csv(stats_dir / f"icc_per_model_{d}.csv", index=False)
            icc_d["pooled_centered"].to_csv(stats_dir / f"icc_pooled_centered_{d}.csv", index=False)
        # pooled_twoway uses ALL domain data (it stratifies internally)
        icc_all = compute_icc(scores_df, metrics, domain=None)
        icc_all["pooled_twoway"].to_csv(stats_dir / "icc_pooled_twoway.csv", index=False)
    else:
        icc_all = compute_icc(scores_df, metrics, domain=None)
        icc_all["per_model"].to_csv(stats_dir / "icc_per_model.csv", index=False)
        icc_all["pooled_centered"].to_csv(stats_dir / "icc_pooled_centered.csv", index=False)
        icc_all["pooled_twoway"].to_csv(stats_dir / "icc_pooled_twoway.csv", index=False)

    # ── Q0: pooled view (single answer per metric, all domains pooled). The
    # per_model output groups by actual model_id and pools across domains.
    print("Computing Q0 (pooled across domains) ...")
    q0_pooled = compute_q0_tests(judge_var, trial_var, metrics, domain=None)
    q0_pooled["q0_judge_pooled"].to_csv(stats_dir / "q0_judge_pooled.csv", index=False)
    q0_pooled["q0_judge_per_model"].to_csv(stats_dir / "q0_judge_per_model.csv", index=False)
    q0_pooled["q0_trial_pooled"].to_csv(stats_dir / "q0_trial_pooled.csv", index=False)
    q0_pooled["q0_trial_per_model"].to_csv(stats_dir / "q0_trial_per_model.csv", index=False)

    # ── Q1a, Q1b: pooled (primary) + per-domain (drill-down).
    # Q2, Q3: per-domain only (within-domain comparisons; the pooled-12-cell
    # version was confounded and is no longer written).
    print("Computing Q1 pooled + Q1/Q2/Q3 per-domain ...")
    pooled_tests = compute_statistical_tests(judge_var, trial_var, metrics, domain=None)
    pooled_tests["q1a"].to_csv(stats_dir / "q1a.csv", index=False)
    pooled_tests["q1b"].to_csv(stats_dir / "q1b.csv", index=False)
    # Note: q2_kw / q3_kw / q2_pairwise / q3_pairwise from pooled_tests are
    # intentionally NOT written — they conflate model and domain.

    if domains_present:
        for d in domains_present:
            d_tests = compute_statistical_tests(judge_var, trial_var, metrics, domain=d)
            d_tests["q1a"].to_csv(stats_dir / f"q1a_{d}.csv", index=False)
            d_tests["q1b"].to_csv(stats_dir / f"q1b_{d}.csv", index=False)
            d_tests["q2_kw"].to_csv(stats_dir / f"q2_kw_{d}.csv", index=False)
            d_tests["q2_pairwise"].to_csv(stats_dir / f"q2_pairwise_{d}.csv", index=False)
            d_tests["q3_kw"].to_csv(stats_dir / f"q3_kw_{d}.csv", index=False)
            d_tests["q3_pairwise"].to_csv(stats_dir / f"q3_pairwise_{d}.csv", index=False)
    else:
        pooled_tests["q2_kw"].to_csv(stats_dir / "q2_kw.csv", index=False)
        pooled_tests["q2_pairwise"].to_csv(stats_dir / "q2_pairwise.csv", index=False)
        pooled_tests["q3_kw"].to_csv(stats_dir / "q3_kw.csv", index=False)
        pooled_tests["q3_pairwise"].to_csv(stats_dir / "q3_pairwise.csv", index=False)

    print("Computing within-type tests ...")
    within_type = compute_within_type_tests(judge_var, trial_var, run_type_map, metrics)
    for rtype, type_results in within_type.items():
        for key, df in type_results.items():
            df.to_csv(stats_dir / f"within_type_{rtype}_{key}.csv", index=False)

    n_files = 10 + sum(len(v) for v in within_type.values())
    print(f"Wrote {n_files} CSV files to {stats_dir}")


if __name__ == "__main__":
    main()
