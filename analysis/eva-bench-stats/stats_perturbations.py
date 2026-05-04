# Config: local/eva-bench-stats/perturbations_config.yaml
#
# trial_scores_path: output/eva-bench-stats/trial_scores.csv
# output_dir: output_processed/eva-bench-stats/perturbations
# random_seed: 42
# metrics:
#   - EVA-A_mean
#   - EVA-X_mean
#   - EVA-overall_mean
#   - task_completion
#   - faithfulness
#   - agent_speech_fidelity
#   - conversation_progression
#   - turn_taking
#   - conciseness
# alpha: 0.05
# n_permutations: 10000
# n_bootstrap: 1000
#
# models:
#   <display_label>:
#     alias: "<system_alias from trial_scores.csv>"
#     conditions:
#       A: accent
#       B: background_noise
#       "A+B": both

"""Statistical tests for perturbation analysis.

Pure computation: takes DataFrames, returns DataFrames. No file I/O, no plotting.

Pipeline:
  1. permutation_test  — paired sign-flip permutation test on scenario-level deltas
  2. bootstrap_ci      — bootstrapped 95% CI on mean delta (resample across scenarios)
  3. run_analysis      — applies both + Holm-Bonferroni correction across conditions
                         within each model × metric combination
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from statsmodels.stats.multitest import multipletests

from stats_utils import bootstrap_ci  # noqa: F401 (re-exported for backward compatibility)

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "local" / "eva-bench-stats" / "perturbations_config.yaml"


def permutation_test(
    deltas: np.ndarray,
    n_perm: int = 10000,
    seed: int = 42,
) -> float:
    """Two-sided paired sign-flip permutation test.

    For each permutation, independently flip the sign of each delta with p=0.5,
    compute the mean. P-value = fraction of permutations where |permuted mean|
    >= |observed mean|.

    Args:
        deltas: 1-D array of scenario-level (perturbation - baseline) deltas.
        n_perm: Number of permutations.
        seed: RNG seed for reproducibility.

    Returns:
        Two-sided p-value in [0, 1].
    """
    deltas = np.asarray(deltas, dtype=float)
    n = len(deltas)
    observed = np.mean(deltas)

    if observed == 0.0 and np.all(deltas == 0.0):
        return 1.0

    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    permuted_means = (signs * deltas).mean(axis=1)

    p = np.mean(np.abs(permuted_means) >= np.abs(observed))
    return float(p)


def run_analysis(
    deltas_df: pd.DataFrame,
    config: dict,
    correction_groupby: list[str] | None = None,
) -> pd.DataFrame:
    """Run full perturbation analysis: permutation test + bootstrap CI + Holm-Bonferroni.

    For each group defined by correction_groupby, runs one permutation test and
    one bootstrap CI per (perturbation_condition × domain) combination, then applies
    Holm-Bonferroni correction across all tests in that group.

    Args:
        deltas_df: DataFrame with columns:
            model_label, perturbation_condition, domain, scenario_id,
            metric, delta, baseline_mean, perturb_mean
        config: Parsed perturbations_config.yaml, must have keys:
            alpha, n_permutations, n_bootstrap, random_seed
        correction_groupby: Columns that define one Holm-Bonferroni family.
            Defaults to ["model_label", "metric", "domain"] (pooled analysis:
            3 conditions per family). Use ["model_label", "metric"] for per-domain
            analysis where domain is one of the varying dimensions, giving
            3 conditions × 3 domains = 9 tests per family.

    Returns:
        DataFrame with one row per (model_label, metric, domain, perturbation_condition)
        and columns:
            model_label, metric, domain, perturbation_condition,
            observed_mean_delta, ci_lower, ci_upper, raw_p, corrected_p, reject
    """
    if correction_groupby is None:
        correction_groupby = ["model_label", "metric", "domain"]

    alpha: float = config["alpha"]
    n_perm: int = config["n_permutations"]
    n_boot: int = config["n_bootstrap"]
    seed: int = config["random_seed"]

    result_rows: list[dict] = []

    # Within each correction group, compute p-values for all (condition × domain) cells
    cell_keys = ["perturbation_condition", "domain"]
    varying_keys = [k for k in cell_keys if k not in correction_groupby]

    for group_vals, group_df in deltas_df.groupby(correction_groupby, sort=False):
        if isinstance(group_vals, str):
            group_vals = (group_vals,)
        group_meta = dict(zip(correction_groupby, group_vals))

        # Enumerate all (condition, domain) cells within this correction group
        cell_group_keys = ["perturbation_condition"] + varying_keys
        cell_results: list[dict] = []

        for cell_vals, cell_df in group_df.groupby(cell_group_keys, sort=False):
            if isinstance(cell_vals, str):
                cell_vals = (cell_vals,)
            cell_meta = dict(zip(cell_group_keys, cell_vals))

            cond = cell_meta["perturbation_condition"]
            domain = cell_meta.get("domain", group_meta.get("domain", "pooled"))

            d = cell_df["delta"].to_numpy()
            observed_mean = float(d.mean())

            cell_seed = seed + hash(f"{group_meta}:{cond}:{domain}") % (2**31)

            p_val = permutation_test(d, n_perm=n_perm, seed=cell_seed)
            ci_lower, ci_upper = bootstrap_ci(d, n_boot=n_boot, seed=cell_seed, alpha=alpha)

            cell_results.append(
                {
                    **group_meta,
                    "domain": domain,
                    "perturbation_condition": cond,
                    "observed_mean_delta": observed_mean,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "raw_p": p_val,
                }
            )

        # Holm-Bonferroni correction across all cells in this correction group
        raw_ps = [r["raw_p"] for r in cell_results]
        if len(raw_ps) > 1:
            reject_arr, corrected_ps, _, _ = multipletests(raw_ps, alpha=alpha, method="holm")
        else:
            corrected_ps = raw_ps
            reject_arr = [raw_ps[0] < alpha]

        for r, corr_p, rej in zip(cell_results, corrected_ps, reject_arr):
            result_rows.append({**r, "corrected_p": float(corr_p), "reject": bool(rej)})

    return pd.DataFrame(
        result_rows,
        columns=[
            "model_label",
            "metric",
            "domain",
            "perturbation_condition",
            "observed_mean_delta",
            "ci_lower",
            "ci_upper",
            "raw_p",
            "corrected_p",
            "reject",
        ],
    )


def run_pairwise_analysis(
    pairwise_deltas_df: pd.DataFrame,
    config: dict,
    correction_groupby: list[str] | None = None,
) -> pd.DataFrame:
    """Pairwise analysis: permutation test + bootstrap CI + Holm-Bonferroni.

    Parallel structure to run_analysis. Operates on comparisons
    accent_vs_background_noise, accent_vs_both, background_noise_vs_both.
    H-B correction is the secondary family — applied separately from the
    primary (vs-baseline) family.
    """
    if correction_groupby is None:
        correction_groupby = ["model_label", "metric", "domain"]

    alpha: float = config["alpha"]
    n_perm: int = config["n_permutations"]
    n_boot: int = config["n_bootstrap"]
    seed: int = config["random_seed"]

    pairwise_comparisons = {
        "accent_vs_background_noise", "accent_vs_both", "background_noise_vs_both"
    }
    df = pairwise_deltas_df[pairwise_deltas_df["comparison"].isin(pairwise_comparisons)]

    result_rows: list[dict] = []
    cell_keys = ["comparison", "domain"]
    varying_keys = [k for k in cell_keys if k not in correction_groupby]

    for group_vals, group_df in df.groupby(correction_groupby, sort=False):
        if isinstance(group_vals, str):
            group_vals = (group_vals,)
        group_meta = dict(zip(correction_groupby, group_vals))

        cell_group_keys = ["comparison"] + varying_keys
        cell_results: list[dict] = []

        for cell_vals, cell_df in group_df.groupby(cell_group_keys, sort=False):
            if isinstance(cell_vals, str):
                cell_vals = (cell_vals,)
            cell_meta = dict(zip(cell_group_keys, cell_vals))

            comp = cell_meta["comparison"]
            domain = cell_meta.get("domain", group_meta.get("domain", "pooled"))

            d = cell_df["delta"].to_numpy()
            observed_mean = float(d.mean())
            cell_seed = seed + hash(f"{group_meta}:{comp}:{domain}") % (2**31)

            p_val = permutation_test(d, n_perm=n_perm, seed=cell_seed)
            ci_lower, ci_upper = bootstrap_ci(d, n_boot=n_boot, seed=cell_seed, alpha=alpha)

            cell_results.append({
                **group_meta,
                "domain": domain,
                "comparison": comp,
                "observed_mean_delta": observed_mean,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "raw_p": p_val,
            })

        raw_ps = [r["raw_p"] for r in cell_results]
        if len(raw_ps) > 1:
            reject_arr, corrected_ps, _, _ = multipletests(raw_ps, alpha=alpha, method="holm")
        else:
            corrected_ps = raw_ps
            reject_arr = [raw_ps[0] < alpha] if raw_ps else []

        for r, corr_p, rej in zip(cell_results, corrected_ps, reject_arr):
            result_rows.append({**r, "corrected_p": float(corr_p), "reject": bool(rej)})

    return pd.DataFrame(
        result_rows,
        columns=[
            "model_label", "metric", "domain", "comparison",
            "observed_mean_delta", "ci_lower", "ci_upper",
            "raw_p", "corrected_p", "reject",
        ],
    )


def run_additivity_analysis(
    pairwise_deltas_df: pd.DataFrame,
    config: dict,
    correction_groupby: list[str] | None = None,
) -> pd.DataFrame:
    """Additivity residual test: one uncorrected permutation test per cell.

    No H-B correction — there is exactly one test per (model, metric, domain).
    reject is determined directly by raw_p < alpha.
    """
    if correction_groupby is None:
        correction_groupby = ["model_label", "metric", "domain"]

    alpha: float = config["alpha"]
    n_perm: int = config["n_permutations"]
    n_boot: int = config["n_bootstrap"]
    seed: int = config["random_seed"]

    df = pairwise_deltas_df[pairwise_deltas_df["comparison"] == "additivity"]

    result_rows: list[dict] = []
    for group_vals, group_df in df.groupby(correction_groupby, sort=False):
        if isinstance(group_vals, str):
            group_vals = (group_vals,)
        group_meta = dict(zip(correction_groupby, group_vals))

        domain = group_meta.get("domain", "pooled")
        d = group_df["delta"].to_numpy()
        observed_mean = float(d.mean())
        cell_seed = seed + hash(f"{group_meta}:additivity:{domain}") % (2**31)

        p_val = permutation_test(d, n_perm=n_perm, seed=cell_seed)
        ci_lower, ci_upper = bootstrap_ci(d, n_boot=n_boot, seed=cell_seed, alpha=alpha)

        result_rows.append({
            **group_meta,
            "observed_mean_delta": observed_mean,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "raw_p": p_val,
            "reject": p_val < alpha,
        })

    return pd.DataFrame(
        result_rows,
        columns=[
            "model_label", "metric", "domain",
            "observed_mean_delta", "ci_lower", "ci_upper", "raw_p", "reject",
        ],
    )


def compute_cld(
    sig_matrix: np.ndarray,
    condition_names: list[str],
) -> dict[str, str]:
    """Compact letter display from pairwise significance matrix.

    Assigns letters so that conditions sharing a letter are not significantly
    different from each other. Uses maximal-clique insert-absorption: finds all
    maximal subsets of conditions where no pair is significantly different,
    assigns each subset a letter (a, b, c ...), and gives each condition all
    letters of subsets it belongs to.
    """
    from itertools import combinations as _comb

    n = len(condition_names)
    sig = np.asarray(sig_matrix, dtype=bool)

    valid: list[frozenset] = []
    for size in range(1, n + 1):
        for subset in _comb(range(n), size):
            if all(not sig[i, j] for i, j in _comb(subset, 2)):
                valid.append(frozenset(subset))

    maximal = [s for s in valid if not any(s < other for other in valid)]

    letters: dict[str, list[str]] = {name: [] for name in condition_names}
    for idx, group in enumerate(maximal):
        letter = chr(ord("a") + idx)
        for i in group:
            letters[condition_names[i]].append(letter)

    return {name: "".join(sorted(letters[name])) for name in condition_names}


def main(config_path: Path = CONFIG_PATH) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    project_root = config_path.parent.parent.parent
    output_dir = project_root / config["output_dir"]

    deltas_path = output_dir / "scenario_deltas.csv"
    if not deltas_path.exists():
        raise FileNotFoundError(f"scenario_deltas.csv not found at {deltas_path}. Run run_data.py first.")

    print(f"Loading deltas from {deltas_path} ...")
    deltas_df = pd.read_csv(deltas_path)
    print(f"  {len(deltas_df):,} rows loaded")

    print("Running per-domain analysis ...")
    results_per_domain = run_analysis(deltas_df, config, correction_groupby=["model_label", "metric"])

    print("Running pooled analysis ...")
    pooled_df = deltas_df.copy()
    pooled_df["domain"] = "pooled"
    results_pooled = run_analysis(pooled_df, config)

    per_domain_path = output_dir / "results_per_domain.csv"
    pooled_path = output_dir / "results_pooled.csv"

    results_per_domain.to_csv(per_domain_path, index=False)
    results_pooled.to_csv(pooled_path, index=False)

    print(f"Wrote {len(results_per_domain):,} per-domain rows → {per_domain_path}")
    print(f"Wrote {len(results_pooled):,} pooled rows → {pooled_path}")

    # ── Pairwise analysis ─────────────────────────────────────────────────
    pairwise_path = output_dir / "scenario_pairwise_deltas.csv"
    if not pairwise_path.exists():
        raise FileNotFoundError(
            f"scenario_pairwise_deltas.csv not found at {pairwise_path}. Run run_data.py first."
        )

    print(f"Loading pairwise deltas from {pairwise_path} ...")
    pairwise_df = pd.read_csv(pairwise_path)
    print(f"  {len(pairwise_df):,} rows loaded")

    print("Running pairwise per-domain analysis ...")
    pairwise_per_domain = run_pairwise_analysis(
        pairwise_df, config, correction_groupby=["model_label", "metric"]
    )

    print("Running pairwise pooled analysis ...")
    pairwise_pooled_input = pairwise_df.copy()
    pairwise_pooled_input["domain"] = "pooled"
    pairwise_pooled = run_pairwise_analysis(pairwise_pooled_input, config)

    # ── CLD lookup tables ─────────────────────────────────────────────────
    _COND_NAMES = ["accent", "background_noise", "both"]
    _COMP_TO_PAIR = {
        "accent_vs_background_noise": (0, 1),
        "accent_vs_both": (0, 2),
        "background_noise_vs_both": (1, 2),
    }

    def _build_cld_lookup(pairwise_results: pd.DataFrame) -> pd.DataFrame:
        cld_rows: list[dict] = []
        for group_vals, group_df in pairwise_results.groupby(
            ["model_label", "metric", "domain"], sort=False
        ):
            model, metric, domain = group_vals
            sig = np.zeros((3, 3), dtype=bool)
            for _, row in group_df.iterrows():
                pair = _COMP_TO_PAIR.get(row["comparison"])
                if pair is not None and row["reject"]:
                    i, j = pair
                    sig[i, j] = sig[j, i] = True
            cld = compute_cld(sig, _COND_NAMES)
            for cond, letter in cld.items():
                cld_rows.append({
                    "model_label": model, "metric": metric, "domain": domain,
                    "perturbation_condition": cond, "cld_letter": letter,
                })
        return pd.DataFrame(cld_rows)

    cld_pooled = _build_cld_lookup(pairwise_pooled)
    cld_per_domain = _build_cld_lookup(pairwise_per_domain)

    pairwise_pooled_path = output_dir / "results_pairwise_pooled.csv"
    pairwise_per_domain_path = output_dir / "results_pairwise_per_domain.csv"
    cld_pooled_path = output_dir / "cld_pooled.csv"
    cld_per_domain_path = output_dir / "cld_per_domain.csv"

    pairwise_pooled.to_csv(pairwise_pooled_path, index=False)
    pairwise_per_domain.to_csv(pairwise_per_domain_path, index=False)
    cld_pooled.to_csv(cld_pooled_path, index=False)
    cld_per_domain.to_csv(cld_per_domain_path, index=False)

    print(f"Wrote {len(pairwise_pooled):,} pairwise pooled rows → {pairwise_pooled_path}")
    print(f"Wrote {len(pairwise_per_domain):,} pairwise per-domain rows → {pairwise_per_domain_path}")
    print(f"Wrote CLD lookups → {cld_pooled_path}, {cld_per_domain_path}")

    # ── Additivity analysis ───────────────────────────────────────────────
    print("Running additivity per-domain analysis ...")
    additivity_per_domain = run_additivity_analysis(
        pairwise_df, config, correction_groupby=["model_label", "metric", "domain"]
    )

    print("Running additivity pooled analysis ...")
    additivity_pooled_input = pairwise_df.copy()
    additivity_pooled_input["domain"] = "pooled"
    additivity_pooled = run_additivity_analysis(
        additivity_pooled_input, config, correction_groupby=["model_label", "metric", "domain"]
    )

    additivity_pooled_path = output_dir / "results_additivity_pooled.csv"
    additivity_per_domain_path = output_dir / "results_additivity_per_domain.csv"

    additivity_pooled.to_csv(additivity_pooled_path, index=False)
    additivity_per_domain.to_csv(additivity_per_domain_path, index=False)

    print(f"Wrote {len(additivity_pooled):,} additivity pooled rows → {additivity_pooled_path}")
    print(f"Wrote {len(additivity_per_domain):,} additivity per-domain rows → {additivity_per_domain_path}")


if __name__ == "__main__":
    main()
