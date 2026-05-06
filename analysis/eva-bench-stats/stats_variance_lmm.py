# analysis/eva-bench-stats/stats_variance_lmm.py
# Config: local/eva-bench-stats/variance_config.yaml
#
# Reads from: output_processed/eva-bench-stats/variance/data/scores.csv
# Writes to:  output_processed/eva-bench-stats/variance/stats/lmm/
#
# Pipeline:
#   _detect_judge_metrics          → classify metrics as judge-graded or deterministic
#   fit_lmm_pooled                 → one REML model per metric (all 4 models together)
#     extract_variance_components  → σ², proportions, CIs, fixed effects, convergence
#   fit_lmm_per_model              → one REML model per (metric, model_id)
#     extract_variance_components  → same extraction
#   results_to_dataframes          → assemble 5 output CSVs from result dicts
#   main                           → load scores.csv, run pipeline, write lmm/ CSVs
#
# FUTURE IMPLEMENTATION: Variance decomposition (interaction)
# This analysis extends the current model to include a model x scenario
# random interaction effect, requiring pymer4 (lme4 via rpy2).
# Planned as a separate script and new app tab: "Variance decomp (interaction)".
# Do not implement here. See local/superpowers/specs/ for the future spec.
# Shared context: same 4-model subset, same metrics, same app structure.
"""REML mixed effects variance decomposition for EVA-Bench metrics.

Fits a linear mixed effects model to partition score variance into:
  - Domain (fixed effect)
  - Model (fixed effect, pooled analysis only)
  - Scenario (random intercept)
  - Trial (random intercept, nested within scenario)
  - Judge (random intercept, nested within trial; judge-graded metrics only)
  - Residual

Uses statsmodels.MixedLM with REML estimation and sum-to-zero (effects) coding
for fixed effects via patsy Sum contrasts.
"""

import warnings
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "local" / "eva-bench-stats" / "variance_config.yaml"


def _model_id_from_label(run_label: str) -> str:
    """Strip the trailing ' — <domain>' suffix from a per-domain run_label."""
    return run_label.split(" — ")[0].strip()


def _detect_judge_metrics(df: pd.DataFrame, metrics: list[str]) -> set[str]:
    """Identify judge-graded metrics by checking for score variation across iterations.

    A metric is judge-graded if any (record_id, trial) cell has more than one
    distinct normalized_score value across iterations. Deterministic metrics
    always produce the same score for the same conversation.
    """
    judge: set[str] = set()
    for metric in metrics:
        sub = df[df["metric"] == metric]
        if sub.empty:
            continue
        # Group by run_id (not just record_id) because record_ids are domain-scoped
        # namespaces — the same integer record_id can appear in multiple domains.
        # Within a single run, if any (record, trial) cell has more than one distinct
        # score across iterations, the metric is judge-graded.
        nunique = sub.groupby(["run_id", "record_id", "trial"])["normalized_score"].nunique()
        if (nunique > 1).any():
            judge.add(metric)
    return judge


def extract_variance_components(
    model_fit,
    metric_name: str,
    has_judge: bool,
) -> dict:
    """Extract variance components, proportions, CIs, and fixed effects from a fitted MixedLM.

    Parameters
    ----------
    model_fit : statsmodels.regression.mixed_linear_model.MixedLMResults
        Fitted REML model returned by MixedLM.fit().
    metric_name : str
        Name of the metric being fitted (used in warning messages).
    has_judge : bool
        Whether judge random effect is included in this model.

    Returns
    -------
    dict with keys:
        variance_components : dict mapping component name to σ²
        proportions         : dict mapping component name to fraction of total
        component_cis       : dict mapping component name to (lower, upper) CI
        fixed_effects       : DataFrame with columns term, coef, ci_lower, ci_upper
        total_variance      : float
        log_likelihood      : float
        converged           : bool
        convergence_note    : str

    Notes
    -----
    Variance component CIs are approximate Wald (likelihood-based) intervals
    derived from the REML Hessian. Bootstrap CIs may be added in a future revision.
    """
    try:
        # ── Variance components ────────────────────────────────────────────
        # cov_re holds the groups-level (scenario) random intercept variance (absolute σ²).
        # params stores vc_formula entries as "<key> Var" = component_σ² / residual_scale.
        # Multiply by scale to convert to absolute σ². Using params by name is more robust
        # than vcomp indices because statsmodels sorts vc_formula keys alphabetically
        # in vcomp (e.g., "Judge" < "Trial"), not in insertion order.
        residual_scale = float(model_fit.scale)
        params_dict = dict(model_fit.params)

        sigma2_scenario = float(model_fit.cov_re.iloc[0, 0])
        sigma2_trial = params_dict.get("Trial Var", 0.0) * residual_scale
        sigma2_judge = params_dict.get("Judge Var", 0.0) * residual_scale if has_judge else None
        sigma2_residual = residual_scale

        components: dict[str, float] = {
            "scenario": sigma2_scenario,
            "trial": sigma2_trial,
            "residual": sigma2_residual,
        }
        if has_judge:
            components["judge"] = sigma2_judge  # type: ignore[assignment]

        sigma2_total = sum(v for v in components.values())
        proportions = {k: (v / sigma2_total if sigma2_total > 0 else 0.0) for k, v in components.items()}

        # ── Confidence intervals ───────────────────────────────────────────
        # conf_int() returns Wald CIs for all free parameters.
        # VC params use "<key> Var" naming (e.g. "Group Var", "Trial Var", "Judge Var").
        # Like params, VC CIs are in ratio-to-residual scale; multiply by residual_scale
        # to get absolute σ² CIs. Clip lower bounds to 0.
        all_ci = model_fit.conf_int()
        fe_names = list(model_fit.fe_params.index)
        vc_ci_rows = all_ci.drop(index=fe_names, errors="ignore")

        component_cis: dict[str, tuple[float, float]] = {}
        for label in vc_ci_rows.index:
            lo = max(0.0, float(vc_ci_rows.loc[label, 0])) * residual_scale
            hi = max(0.0, float(vc_ci_rows.loc[label, 1])) * residual_scale
            lbl = str(label).lower()
            if "group" in lbl:
                component_cis["scenario"] = (lo, hi)
            elif "trial" in lbl:
                component_cis["trial"] = (lo, hi)
            elif "judge" in lbl:
                component_cis["judge"] = (lo, hi)
            elif "scale" in lbl or "residual" in lbl:
                component_cis["residual"] = (lo, hi)

        # ── Fixed effects ──────────────────────────────────────────────────
        fe_ci = all_ci.loc[fe_names]
        fixed_effects = pd.DataFrame(
            {
                "term": fe_names,
                "coef": model_fit.fe_params.values,
                "ci_lower": fe_ci.iloc[:, 0].values,
                "ci_upper": fe_ci.iloc[:, 1].values,
            }
        )

        return {
            "variance_components": components,
            "proportions": proportions,
            "component_cis": component_cis,
            "fixed_effects": fixed_effects,
            "total_variance": sigma2_total,
            "log_likelihood": float(model_fit.llf),
            "converged": bool(model_fit.converged),
            "convergence_note": "",
        }

    except Exception as exc:
        warnings.warn(f"[LMM] extract_variance_components failed for {metric_name!r}: {exc}", stacklevel=2)
        return {
            "variance_components": {},
            "proportions": {},
            "component_cis": {},
            "fixed_effects": pd.DataFrame(),
            "total_variance": float("nan"),
            "log_likelihood": float("nan"),
            "converged": False,
            "convergence_note": str(exc),
        }


def fit_lmm_pooled(
    df: pd.DataFrame,
    metrics: list[str],
    judge_metrics: set[str],
) -> dict[str, dict]:
    """Fit one REML mixed effects model per metric using all models' data.

    model_id and domain are fixed effects with sum-to-zero coding.
    Scenario, trial, and judge (where applicable) are nested random intercepts.

    For cascade-only metrics (e.g. transcription_accuracy_key_entities), only
    rows with data are used; the model_id fixed effect has fewer levels.

    Parameters
    ----------
    df : pd.DataFrame
        Scores DataFrame from scores.csv.
    metrics : list[str]
        Metric names to fit.
    judge_metrics : set[str]
        Metrics that include a judge random effect.

    Returns
    -------
    dict mapping metric name → extract_variance_components result dict
    """
    import statsmodels.formula.api as smf

    work = df.copy()
    work["model_id"] = work["run_label"].map(_model_id_from_label)
    work["scenario_uid"] = work["domain"].astype(str) + "::" + work["record_id"].astype(str)
    work["trial_uid"] = work["scenario_uid"] + "::" + work["trial"].astype(str)
    work["judge_uid"] = work["trial_uid"] + "::" + work["iteration"].astype(str)

    results: dict[str, dict] = {}

    for metric in metrics:
        sub = work[work["metric"] == metric].copy()
        if sub.empty or sub["model_id"].nunique() < 2 or sub["scenario_uid"].nunique() < 3:
            results[metric] = {
                "variance_components": {},
                "proportions": {},
                "component_cis": {},
                "fixed_effects": pd.DataFrame(),
                "total_variance": float("nan"),
                "log_likelihood": float("nan"),
                "converged": False,
                "convergence_note": "insufficient data for pooled LMM",
            }
            continue

        has_judge = metric in judge_metrics
        vcf: dict[str, str] = {"Trial": "0 + C(trial_uid)"}
        if has_judge:
            vcf["Judge"] = "0 + C(judge_uid)"

        formula = "normalized_score ~ C(model_id, Sum) + C(domain, Sum)"

        try:
            model = smf.mixedlm(
                formula,
                sub,
                groups=sub["scenario_uid"],
                re_formula="~1",
                vc_formula=vcf,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit = model.fit(reml=True, method="powell")
                conv_note = (
                    "; ".join(str(w.message) for w in caught if issubclass(w.category, Warning))
                    if not fit.converged
                    else ""
                )
            results[metric] = extract_variance_components(fit, metric, has_judge)
            if conv_note and not results[metric]["convergence_note"]:
                results[metric]["convergence_note"] = conv_note
        except Exception as exc:
            warnings.warn(f"[LMM pooled] fit failed for {metric!r}: {exc}", stacklevel=2)
            results[metric] = {
                "variance_components": {},
                "proportions": {},
                "component_cis": {},
                "fixed_effects": pd.DataFrame(),
                "total_variance": float("nan"),
                "log_likelihood": float("nan"),
                "converged": False,
                "convergence_note": str(exc),
            }

    return results


def _fit_deterministic_per_model(sub: pd.DataFrame, metric_name: str) -> dict:
    """Two-level REML model for deterministic (non-judge-graded) metrics.

    Deterministic metrics produce identical scores across iterations for a given
    (model, scenario, trial), so fitting the three-level model with a Trial vc_formula
    creates a degenerate within-trial variance (singular matrix).

    Instead: collapse to one row per (scenario, trial) and fit a two-level model.
    In this model the residual is entirely within-scenario variation across trials,
    so scale IS σ²_trial. We return two components: scenario and trial.
    """
    import statsmodels.formula.api as smf

    sub_fit = sub.drop_duplicates(subset=["scenario_uid", "trial_uid"]).copy()
    try:
        model = smf.mixedlm(
            "normalized_score ~ C(domain, Sum)",
            sub_fit,
            groups=sub_fit["scenario_uid"],
            re_formula="~1",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = model.fit(reml=True, method="powell")
            conv_note = (
                "; ".join(str(w.message) for w in caught if issubclass(w.category, Warning))
                if not fit.converged
                else ""
            )

        residual_scale = float(fit.scale)
        sigma2_scenario = float(fit.cov_re.iloc[0, 0])
        sigma2_trial = residual_scale  # residual IS trial variance in this two-level model
        sigma2_total = sigma2_scenario + sigma2_trial
        proportions = {
            k: (v / sigma2_total if sigma2_total > 0 else 0.0)
            for k, v in {"scenario": sigma2_scenario, "trial": sigma2_trial}.items()
        }

        all_ci = fit.conf_int()
        fe_names = list(fit.fe_params.index)
        vc_ci_rows = all_ci.drop(index=fe_names, errors="ignore")
        scenario_ci: tuple[float, float] = (float("nan"), float("nan"))
        for label in vc_ci_rows.index:
            if "group" in str(label).lower():
                lo = max(0.0, float(vc_ci_rows.loc[label, 0])) * residual_scale
                hi = max(0.0, float(vc_ci_rows.loc[label, 1])) * residual_scale
                scenario_ci = (lo, hi)

        fe_ci = all_ci.loc[fe_names]
        fixed_effects = pd.DataFrame(
            {
                "term": fe_names,
                "coef": fit.fe_params.values,
                "ci_lower": fe_ci.iloc[:, 0].values,
                "ci_upper": fe_ci.iloc[:, 1].values,
            }
        )
        return {
            "variance_components": {"scenario": sigma2_scenario, "trial": sigma2_trial},
            "proportions": proportions,
            "component_cis": {"scenario": scenario_ci, "trial": (float("nan"), float("nan"))},
            "fixed_effects": fixed_effects,
            "total_variance": sigma2_total,
            "log_likelihood": float(fit.llf),
            "converged": bool(fit.converged),
            "convergence_note": conv_note,
        }
    except Exception as exc:
        warnings.warn(f"[LMM det per-model] fit failed for {metric_name!r}: {exc}", stacklevel=2)
        return {
            "variance_components": {},
            "proportions": {},
            "component_cis": {},
            "fixed_effects": pd.DataFrame(),
            "total_variance": float("nan"),
            "log_likelihood": float("nan"),
            "converged": False,
            "convergence_note": str(exc),
        }


def fit_lmm_per_model(
    df: pd.DataFrame,
    metrics: list[str],
    judge_metrics: set[str],
) -> dict[tuple[str, str], dict]:
    """Fit one REML model per (metric, model_id) with domain as the only fixed effect.

    Shows whether variance structure is consistent across models.

    Parameters
    ----------
    df : pd.DataFrame
        Scores DataFrame from scores.csv.
    metrics : list[str]
        Metric names to fit.
    judge_metrics : set[str]
        Metrics that include a judge random effect.

    Returns
    -------
    dict mapping (metric, model_id) → extract_variance_components result dict
    """
    import statsmodels.formula.api as smf

    work = df.copy()
    work["model_id"] = work["run_label"].map(_model_id_from_label)
    work["scenario_uid"] = work["domain"].astype(str) + "::" + work["record_id"].astype(str)
    work["trial_uid"] = work["scenario_uid"] + "::" + work["trial"].astype(str)
    work["judge_uid"] = work["trial_uid"] + "::" + work["iteration"].astype(str)

    results: dict[tuple[str, str], dict] = {}
    model_ids = sorted(work["model_id"].unique())

    for metric in metrics:
        has_judge = metric in judge_metrics

        for model_id in model_ids:
            sub = work[(work["metric"] == metric) & (work["model_id"] == model_id)].copy()
            if sub.empty or sub["scenario_uid"].nunique() < 3:
                results[(metric, model_id)] = {
                    "variance_components": {},
                    "proportions": {},
                    "component_cis": {},
                    "fixed_effects": pd.DataFrame(),
                    "total_variance": float("nan"),
                    "log_likelihood": float("nan"),
                    "converged": False,
                    "convergence_note": "insufficient data",
                }
                continue

            if not has_judge:
                # Deterministic metric: identical iterations → two-level model
                results[(metric, model_id)] = _fit_deterministic_per_model(sub, metric)
                continue

            # Judge-graded metric: use Trial vc_formula only (no Judge).
            # judge_uid has 1 obs per level in a single-model fit, making judge variance
            # and residual unidentifiable. Dropping Judge lets trial_uid (3 obs per level
            # from 3 iterations) be identified, and the residual = judge stochasticity.
            vcf: dict[str, str] = {"Trial": "0 + C(trial_uid)"}
            try:
                model = smf.mixedlm(
                    "normalized_score ~ C(domain, Sum)",
                    sub,
                    groups=sub["scenario_uid"],
                    re_formula="~1",
                    vc_formula=vcf,
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    fit = model.fit(reml=True, method="powell")
                    conv_note = (
                        "; ".join(str(w.message) for w in caught if issubclass(w.category, Warning))
                        if not fit.converged
                        else ""
                    )
                results[(metric, model_id)] = extract_variance_components(fit, metric, has_judge=False)
                if conv_note and not results[(metric, model_id)]["convergence_note"]:
                    results[(metric, model_id)]["convergence_note"] = conv_note
            except Exception as exc:
                warnings.warn(f"[LMM per-model] fit failed for {metric!r} / {model_id!r}: {exc}", stacklevel=2)
                results[(metric, model_id)] = {
                    "variance_components": {},
                    "proportions": {},
                    "component_cis": {},
                    "fixed_effects": pd.DataFrame(),
                    "total_variance": float("nan"),
                    "log_likelihood": float("nan"),
                    "converged": False,
                    "convergence_note": str(exc),
                }

    return results


def results_to_dataframes(
    pooled: dict[str, dict],
    per_model: dict[tuple[str, str], dict],
) -> dict[str, pd.DataFrame]:
    """Assemble the 5 output CSVs from fit result dicts.

    Returns
    -------
    dict with keys:
        lmm_variance_components         — pooled: one row per (metric, component)
        lmm_fixed_effects               — pooled: one row per (metric, term)
        lmm_convergence                 — all fits: one row per (metric, analysis)
        lmm_per_model_variance_components — per-model: one row per (metric, model_id, component)
        lmm_per_model_fixed_effects     — per-model: one row per (metric, model_id, term)
    """
    vc_rows, fe_rows, conv_rows = [], [], []
    pm_vc_rows, pm_fe_rows = [], []

    # ── Pooled results ─────────────────────────────────────────────────────
    for metric, res in pooled.items():
        conv_rows.append(
            {
                "metric": metric,
                "analysis": "pooled",
                "converged": res["converged"],
                "log_likelihood": res["log_likelihood"],
                "convergence_note": res["convergence_note"],
            }
        )
        for comp, sigma2 in res["variance_components"].items():
            ci = res["component_cis"].get(comp, (float("nan"), float("nan")))
            vc_rows.append(
                {
                    "metric": metric,
                    "component": comp,
                    "sigma2": sigma2,
                    "proportion": res["proportions"].get(comp, float("nan")),
                    "ci_lower": ci[0],
                    "ci_upper": ci[1],
                    "total_variance": res["total_variance"],
                }
            )
        if not res["fixed_effects"].empty:
            for _, fe_row in res["fixed_effects"].iterrows():
                fe_rows.append({"metric": metric, **fe_row.to_dict()})

    # ── Per-model results ──────────────────────────────────────────────────
    for (metric, model_id), res in per_model.items():
        conv_rows.append(
            {
                "metric": metric,
                "analysis": model_id,
                "converged": res["converged"],
                "log_likelihood": res["log_likelihood"],
                "convergence_note": res["convergence_note"],
            }
        )
        for comp, sigma2 in res["variance_components"].items():
            ci = res["component_cis"].get(comp, (float("nan"), float("nan")))
            pm_vc_rows.append(
                {
                    "metric": metric,
                    "model_id": model_id,
                    "component": comp,
                    "sigma2": sigma2,
                    "proportion": res["proportions"].get(comp, float("nan")),
                    "ci_lower": ci[0],
                    "ci_upper": ci[1],
                    "total_variance": res["total_variance"],
                }
            )
        if not res["fixed_effects"].empty:
            for _, fe_row in res["fixed_effects"].iterrows():
                pm_fe_rows.append({"metric": metric, "model_id": model_id, **fe_row.to_dict()})

    return {
        "lmm_variance_components": pd.DataFrame(vc_rows),
        "lmm_fixed_effects": pd.DataFrame(fe_rows),
        "lmm_convergence": pd.DataFrame(conv_rows),
        "lmm_per_model_variance_components": pd.DataFrame(pm_vc_rows),
        "lmm_per_model_fixed_effects": pd.DataFrame(pm_fe_rows),
    }


def main(config_path: Path = CONFIG_PATH) -> None:
    """Load scores.csv, fit all LMM models, write CSVs to variance/stats/lmm/."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    project_root = config_path.parent.parent.parent
    data_dir = project_root / config["output_dir"] / "data"
    lmm_dir = project_root / config["output_dir"] / "stats" / "lmm"
    lmm_dir.mkdir(parents=True, exist_ok=True)

    scores_path = data_dir / "scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"scores.csv not found at {scores_path}. Run run_data.py first.")

    metrics: list[str] = config["metrics"]
    print(f"Loading scores from {scores_path} ...")
    df = pd.read_csv(scores_path)
    print(f"  {len(df):,} rows, {df['metric'].nunique()} metrics")

    print("Detecting judge vs deterministic metrics ...")
    judge_metrics = _detect_judge_metrics(df, metrics)
    det_metrics = set(metrics) - judge_metrics
    print(f"  Judge-graded: {sorted(judge_metrics)}")
    print(f"  Deterministic: {sorted(det_metrics)}")

    print("Fitting pooled LMM (all models) ...")
    pooled = fit_lmm_pooled(df, metrics, judge_metrics)
    n_conv = sum(1 for r in pooled.values() if r["converged"])
    print(f"  {n_conv}/{len(pooled)} metrics converged")

    print("Fitting per-model LMM ...")
    per_model = fit_lmm_per_model(df, metrics, judge_metrics)
    n_pm_conv = sum(1 for r in per_model.values() if r["converged"])
    print(f"  {n_pm_conv}/{len(per_model)} (metric, model) fits converged")

    dfs = results_to_dataframes(pooled, per_model)
    for name, out_df in dfs.items():
        path = lmm_dir / f"{name}.csv"
        out_df.to_csv(path, index=False)
        print(f"  Wrote {path.name} ({len(out_df)} rows)")

    print(f"Done. Results in {lmm_dir}")


if __name__ == "__main__":
    main()
