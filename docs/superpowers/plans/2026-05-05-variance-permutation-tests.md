# Variance Permutation Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Wilcoxon Q1a test with a sign-flip permutation test in a new "Judge vs. trial variance (b)" tab, keeping the original Wilcoxon tab intact for comparison.

**Architecture:** `permutation_test` moves from `stats_perturbations.py` to `stats_utils.py`; a new `compute_permutation_tests()` in `stats_variance.py` produces three new CSV families; `app.py` gets a new tab inserted at index 3 with all tab indices 3–11 shifting to 4–12.

**Tech Stack:** Python, pandas, scipy.stats (binomtest), numpy, Streamlit, Plotly.

---

## Pre-implementation state (read before coding)

These findings come from reading the files *before* any edits. Verify key line numbers before touching code.

- **Tab count is 12, not 11.** The spec was written against 11 tabs, but `tabs[8]` ("Variance decomp (LMM)") was added concurrently. After inserting the new tab, there will be 13 tabs. Tabs 3–11 shift to 4–12.
- `permutation_test` lives in `stats_perturbations.py` starting at line 51 — not yet in `stats_utils.py`.
- `stats_perturbations.py` already imports: `from stats_utils import bootstrap_ci  # noqa: F401 (re-exported for backward compatibility)` — add `permutation_test` to this line and remove the function body.
- `stats_variance.py` has a duplicate `_filter_and_relabel` (lines 385–419) — pre-existing, do not fix.
- Tab list in app.py is at line 884. `with tabs[0]:` starts at 902, `with tabs[2]:` at 1124, `with tabs[3]:` at 1425 (new tab inserted here). `with tabs[11]:` (Statistical tests) is at line 2580.
- Q1a/Q1b loading in app.py: lines 801–802 (pooled) and 871–872 (within-domain).

---

## File Map

| File | Change |
|---|---|
| `analysis/eva-bench-stats/stats_utils.py` | Add `permutation_test()` |
| `analysis/eva-bench-stats/stats_perturbations.py` | Remove `permutation_test()`, add it to the `from stats_utils import …` re-export line |
| `analysis/eva-bench-stats/stats_variance.py` | Add `compute_permutation_tests()`; update `main()` |
| `analysis/eva-bench-stats/app.py` | Load new CSVs; add Tab 1 validity note; add Tab 2 comparison note; insert Tab 2b; update tab indices 3–11 → 4–12; update Statistical tests tab |
| `tests/unit/eva_bench_stats/test_stats_utils.py` | Add `permutation_test` tests |
| `local/eva-bench-stats/tests/test_stats_variance.py` | Add `compute_permutation_tests` tests |

---

### Task 1: Move `permutation_test` to `stats_utils.py`

**Files:**
- Modify: `analysis/eva-bench-stats/stats_utils.py`
- Modify: `analysis/eva-bench-stats/stats_perturbations.py`
- Test: `tests/unit/eva_bench_stats/test_stats_utils.py`

- [ ] **Step 1.1: Add failing tests for `permutation_test` in `test_stats_utils.py`**

Append to the end of `tests/unit/eva_bench_stats/test_stats_utils.py`:

```python
def test_permutation_test_all_zero_deltas():
    from stats_utils import permutation_test
    p = permutation_test(np.zeros(20), seed=0)
    assert p == 1.0


def test_permutation_test_large_positive_effect_low_p():
    from stats_utils import permutation_test
    deltas = np.full(50, 0.3)
    p = permutation_test(deltas, n_perm=1000, seed=42)
    assert p < 0.05


def test_permutation_test_deterministic():
    from stats_utils import permutation_test
    rng = np.random.default_rng(0)
    deltas = rng.normal(0.1, 0.2, 30)
    p1 = permutation_test(deltas, seed=7)
    p2 = permutation_test(deltas, seed=7)
    assert p1 == p2


def test_permutation_test_p_in_unit_interval():
    from stats_utils import permutation_test
    rng = np.random.default_rng(1)
    deltas = rng.normal(0, 0.1, 20)
    p = permutation_test(deltas, n_perm=200, seed=1)
    assert 0.0 <= p <= 1.0
```

- [ ] **Step 1.2: Run tests to confirm they fail**

```bash
cd /Users/lindsay.brin/Projects/eva
python -m pytest tests/unit/eva_bench_stats/test_stats_utils.py::test_permutation_test_all_zero_deltas -v
```

Expected: FAIL with `ImportError` or `AttributeError` (function not in stats_utils yet).

- [ ] **Step 1.3: Add `permutation_test` to `stats_utils.py`**

Append to the end of `analysis/eva-bench-stats/stats_utils.py` (after `bootstrap_slope_ci`):

```python


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
        deltas: 1-D array of scenario-level deltas.
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
```

- [ ] **Step 1.4: Update `stats_perturbations.py` — add to re-export, remove function body**

Find the existing import line in `stats_perturbations.py`:
```python
from stats_utils import bootstrap_ci  # noqa: F401 (re-exported for backward compatibility)
```

Replace it with:
```python
from stats_utils import bootstrap_ci, permutation_test  # noqa: F401 (re-exported for backward compatibility)
```

Then delete the `permutation_test` function body in `stats_perturbations.py` (lines 51–82, the entire function definition from `def permutation_test(` through the final `return float(p)`).

- [ ] **Step 1.5: Run all stats_utils and stats_perturbations tests to confirm they pass**

```bash
cd /Users/lindsay.brin/Projects/eva
python -m pytest tests/unit/eva_bench_stats/test_stats_utils.py local/eva-bench-stats/tests/test_stats_perturbations.py -v
```

Expected: All tests PASS.

- [ ] **Step 1.6: Commit**

```bash
git add analysis/eva-bench-stats/stats_utils.py analysis/eva-bench-stats/stats_perturbations.py tests/unit/eva_bench_stats/test_stats_utils.py
git commit -m "refactor(eva-bench-stats): move permutation_test to stats_utils; re-export from stats_perturbations"
```

---

### Task 2: Add `compute_permutation_tests` to `stats_variance.py`

**Files:**
- Modify: `analysis/eva-bench-stats/stats_variance.py`
- Test: `local/eva-bench-stats/tests/test_stats_variance.py`

- [ ] **Step 2.1: Add failing tests to `test_stats_variance.py`**

Append to the end of `local/eva-bench-stats/tests/test_stats_variance.py`:

```python
# ── helpers for permutation test ──────────────────────────────────────────────

def _make_judge_var_domain(
    models=("model_a", "model_b"),
    records=tuple(f"r{i:02d}" for i in range(20)),
    trials=(1, 2, 3),
    domain="itsm",
    metric="faithfulness",
    std_val=0.05,
) -> pd.DataFrame:
    rows = []
    for model in models:
        for rec in records:
            for trial in trials:
                rows.append({
                    "run_id": model,
                    "run_label": model,
                    "metric": metric,
                    "domain": domain,
                    "record_id": rec,
                    "trial": trial,
                    "std": std_val + np.random.default_rng(abs(hash((model, rec, trial)))).normal(0, 0.005),
                })
    return pd.DataFrame(rows)


def _make_trial_var_domain(
    models=("model_a", "model_b"),
    records=tuple(f"r{i:02d}" for i in range(20)),
    domain="itsm",
    metric="faithfulness",
    std_val=0.05,
) -> pd.DataFrame:
    rows = []
    for model in models:
        for rec in records:
            rows.append({
                "run_id": model,
                "run_label": model,
                "metric": metric,
                "domain": domain,
                "record_id": rec,
                "std": std_val + np.random.default_rng(abs(hash((model, rec)))).normal(0, 0.005),
            })
    return pd.DataFrame(rows)


# ── compute_permutation_tests ─────────────────────────────────────────────────

def test_compute_permutation_tests_returns_dict():
    from stats_variance import compute_permutation_tests
    jv = _make_judge_var_domain()
    tv = _make_trial_var_domain()
    result = compute_permutation_tests(jv, tv, judge_metrics=["faithfulness"])
    assert isinstance(result, dict)
    assert "per_model" in result
    assert "pooled" in result


def test_compute_permutation_tests_per_model_columns():
    from stats_variance import compute_permutation_tests
    jv = _make_judge_var_domain()
    tv = _make_trial_var_domain()
    result = compute_permutation_tests(jv, tv, judge_metrics=["faithfulness"])
    pm = result["per_model"]
    expected = {
        "metric", "model", "n_scenarios",
        "permutation_mean_delta", "permutation_p_value", "permutation_significant",
        "n_positive_deltas", "sign_test_p_value", "sign_test_significant",
    }
    assert expected.issubset(set(pm.columns)), f"Missing columns: {expected - set(pm.columns)}"


def test_compute_permutation_tests_pooled_columns():
    from stats_variance import compute_permutation_tests
    jv = _make_judge_var_domain()
    tv = _make_trial_var_domain()
    result = compute_permutation_tests(jv, tv, judge_metrics=["faithfulness"])
    pl = result["pooled"]
    expected = {
        "metric", "n_scenarios",
        "permutation_mean_delta", "permutation_p_value", "permutation_significant",
        "sign_test_p_value", "sign_test_significant",
    }
    assert expected.issubset(set(pl.columns)), f"Missing columns: {expected - set(pl.columns)}"


def test_compute_permutation_tests_skips_non_judge_metrics():
    """Metrics not in judge_metrics should produce no rows."""
    from stats_variance import compute_permutation_tests
    jv = _make_judge_var_domain(metric="task_completion")
    tv = _make_trial_var_domain(metric="task_completion")
    result = compute_permutation_tests(jv, tv, judge_metrics=["faithfulness"])
    assert result["per_model"].empty
    assert result["pooled"].empty


def test_compute_permutation_tests_detects_large_trial_effect():
    """When trial SD >> judge SD, permutation test should be significant."""
    from stats_variance import compute_permutation_tests
    jv = _make_judge_var_domain(std_val=0.01)
    tv = _make_trial_var_domain(std_val=0.25)
    result = compute_permutation_tests(jv, tv, judge_metrics=["faithfulness"], n_perm=500)
    pm = result["per_model"]
    assert not pm.empty
    assert pm["permutation_significant"].any()


def test_compute_permutation_tests_p_values_in_unit_interval():
    from stats_variance import compute_permutation_tests
    jv = _make_judge_var_domain()
    tv = _make_trial_var_domain()
    result = compute_permutation_tests(jv, tv, judge_metrics=["faithfulness"], n_perm=200)
    pm = result["per_model"]
    pl = result["pooled"]
    assert ((pm["permutation_p_value"] >= 0) & (pm["permutation_p_value"] <= 1)).all()
    assert ((pl["permutation_p_value"] >= 0) & (pl["permutation_p_value"] <= 1)).all()


def test_compute_permutation_tests_min_n_guard():
    """With fewer than 5 records, no row should be produced."""
    from stats_variance import compute_permutation_tests
    jv = _make_judge_var_domain(records=("r0", "r1", "r2", "r3"))  # 4 records
    tv = _make_trial_var_domain(records=("r0", "r1", "r2", "r3"))
    result = compute_permutation_tests(jv, tv, judge_metrics=["faithfulness"], n_perm=100)
    assert result["per_model"].empty
```

- [ ] **Step 2.2: Run tests to confirm they fail**

```bash
cd /Users/lindsay.brin/Projects/eva
python -m pytest local/eva-bench-stats/tests/test_stats_variance.py::test_compute_permutation_tests_returns_dict -v
```

Expected: FAIL with `ImportError` (function not yet defined).

- [ ] **Step 2.3: Implement `compute_permutation_tests` in `stats_variance.py`**

Add the following function after `compute_statistical_tests` and before `compute_within_type_tests` in `analysis/eva-bench-stats/stats_variance.py` (insert before line 827):

```python
def compute_permutation_tests(
    judge_var: pd.DataFrame,
    trial_var: pd.DataFrame,
    judge_metrics: list[str],
    domain: str | None = None,
    n_perm: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, pd.DataFrame]:
    """Sign-flip permutation test: is trial SD systematically greater than judge SD?

    H₁ (one-sided): mean(trial_SD − judge_SD) > 0.
    One-sided p derived from two-sided sign-flip result:
      observed mean ≥ 0 → p_one = p_two / 2
      observed mean < 0 → p_one = 1 − p_two / 2

    Args:
        judge_var: Per-(run, metric, record, trial) judge std devs.
        trial_var: Per-(run, metric, record) trial std devs.
        judge_metrics: Judge-graded metric names to test (caller detects from variance_budget).
        domain: If set, restrict to this domain; if None, pool across all domains.
        n_perm: Number of sign-flip permutations.
        seed: Base RNG seed (perturbed per metric×model combination).
        alpha: Significance threshold.

    Returns:
        Dict with:
          "per_model" — one row per (metric, model), columns:
              metric, model, n_scenarios, permutation_mean_delta,
              permutation_p_value, permutation_significant,
              n_positive_deltas, sign_test_p_value, sign_test_significant
          "pooled"    — one row per metric (deltas averaged per scenario across models), columns:
              metric, n_scenarios, permutation_mean_delta,
              permutation_p_value, permutation_significant,
              sign_test_p_value, sign_test_significant
    """
    from stats_utils import permutation_test
    from scipy.stats import binomtest

    judge_var = _filter_and_relabel(judge_var, domain)
    trial_var = _filter_and_relabel(trial_var, domain)

    judge_var = judge_var[judge_var["metric"].isin(judge_metrics)]
    trial_var = trial_var[trial_var["metric"].isin(judge_metrics)]

    # Aggregate judge std to per-(model_id, metric, [domain,] record_id)
    j_keys = ["model_id", "metric", "record_id"]
    if "domain" in judge_var.columns:
        j_keys = ["model_id", "metric", "domain", "record_id"]
    judge_by_record = (
        judge_var
        .groupby(j_keys, dropna=False)["std"]
        .mean()
        .reset_index()
        .rename(columns={"std": "judge_sd"})
    )

    t_cols = ["model_id", "metric", "record_id", "std"]
    if "domain" in trial_var.columns:
        t_cols = ["model_id", "metric", "domain", "record_id", "std"]
    trial_by_record = trial_var[t_cols].rename(columns={"std": "trial_sd"})

    merge_keys = ["model_id", "metric", "record_id"]
    if "domain" in judge_by_record.columns and "domain" in trial_by_record.columns:
        merge_keys = ["model_id", "metric", "domain", "record_id"]

    merged = pd.merge(judge_by_record, trial_by_record, on=merge_keys).dropna(
        subset=["judge_sd", "trial_sd"]
    )
    merged["delta"] = merged["trial_sd"] - merged["judge_sd"]

    # ── Per-model ─────────────────────────────────────────────────────────────
    per_model_rows: list[dict] = []
    for metric in judge_metrics:
        metric_sub = merged[merged["metric"] == metric]
        for model_id in metric_sub["model_id"].unique():
            sub = metric_sub[metric_sub["model_id"] == model_id]
            deltas = sub["delta"].values
            n = len(deltas)
            if n < 5:
                continue

            cell_seed = seed + hash(f"{metric}:{model_id}") % (2**31)
            p_two = permutation_test(deltas, n_perm=n_perm, seed=cell_seed)
            obs_mean = float(deltas.mean())
            p_one = p_two / 2 if obs_mean >= 0 else 1 - p_two / 2

            n_pos = int((deltas > 0).sum())
            sign_p = float(binomtest(n_pos, n=n, p=0.5, alternative="greater").pvalue)

            per_model_rows.append({
                "metric": metric,
                "model": model_id,
                "n_scenarios": n,
                "permutation_mean_delta": obs_mean,
                "permutation_p_value": p_one,
                "permutation_significant": bool(p_one < alpha),
                "n_positive_deltas": n_pos,
                "sign_test_p_value": sign_p,
                "sign_test_significant": bool(sign_p < alpha),
            })

    # ── Pooled across models ──────────────────────────────────────────────────
    scenario_key_cols = ["domain", "record_id"] if "domain" in merged.columns else ["record_id"]
    pooled_rows: list[dict] = []
    for metric in judge_metrics:
        metric_sub = merged[merged["metric"] == metric]
        if metric_sub.empty:
            continue
        avg_deltas = metric_sub.groupby(scenario_key_cols)["delta"].mean().reset_index()
        deltas = avg_deltas["delta"].values
        n = len(deltas)
        if n < 5:
            continue

        cell_seed = seed + hash(f"{metric}:pooled") % (2**31)
        p_two = permutation_test(deltas, n_perm=n_perm, seed=cell_seed)
        obs_mean = float(deltas.mean())
        p_one = p_two / 2 if obs_mean >= 0 else 1 - p_two / 2

        n_pos = int((deltas > 0).sum())
        sign_p = float(binomtest(n_pos, n=n, p=0.5, alternative="greater").pvalue)

        pooled_rows.append({
            "metric": metric,
            "n_scenarios": n,
            "permutation_mean_delta": obs_mean,
            "permutation_p_value": p_one,
            "permutation_significant": bool(p_one < alpha),
            "sign_test_p_value": sign_p,
            "sign_test_significant": bool(sign_p < alpha),
        })

    return {
        "per_model": pd.DataFrame(per_model_rows),
        "pooled": pd.DataFrame(pooled_rows),
    }
```

- [ ] **Step 2.4: Run tests to confirm they pass**

```bash
cd /Users/lindsay.brin/Projects/eva
python -m pytest local/eva-bench-stats/tests/test_stats_variance.py -k "permutation" -v
```

Expected: All 7 permutation tests PASS.

- [ ] **Step 2.5: Run full stats_variance test suite to check for regressions**

```bash
cd /Users/lindsay.brin/Projects/eva
python -m pytest local/eva-bench-stats/tests/test_stats_variance.py -v
```

Expected: All tests PASS.

- [ ] **Step 2.6: Commit**

```bash
git add analysis/eva-bench-stats/stats_variance.py local/eva-bench-stats/tests/test_stats_variance.py
git commit -m "feat(eva-bench-stats): add compute_permutation_tests to stats_variance"
```

---

### Task 3: Update `main()` in `stats_variance.py` to produce permutation CSVs

**Files:**
- Modify: `analysis/eva-bench-stats/stats_variance.py`

- [ ] **Step 3.1: Add `compute_permutation_tests` calls to `main()`**

In `main()` in `analysis/eva-bench-stats/stats_variance.py`, find the existing block that starts:
```python
    print("Computing Q1 pooled + Q1/Q2/Q3 per-domain ...")
    pooled_tests = compute_statistical_tests(judge_var, trial_var, metrics, domain=None)
    pooled_tests["q1a"].to_csv(stats_dir / "q1a.csv", index=False)
    pooled_tests["q1b"].to_csv(stats_dir / "q1b.csv", index=False)
```

After `pooled_tests["q1b"].to_csv(stats_dir / "q1b.csv", index=False)` and before the `if domains_present:` block, insert:

```python
    judge_metrics = (
        variance_budget[variance_budget["metric_type"] == "judge_graded"]["metric"]
        .dropna()
        .unique()
        .tolist()
    )
    print(f"Computing permutation tests for judge metrics: {judge_metrics} ...")
    perm_pooled = compute_permutation_tests(judge_var, trial_var, judge_metrics, domain=None)
    perm_pooled["per_model"].to_csv(stats_dir / "q1a_perm.csv", index=False)
    perm_pooled["pooled"].to_csv(stats_dir / "q1a_perm_pooled.csv", index=False)
```

Then, inside the `if domains_present:` block, after the existing `d_tests["q1b"].to_csv(...)` line inside the `for d in domains_present:` loop, add:

```python
            d_perm = compute_permutation_tests(judge_var, trial_var, judge_metrics, domain=d)
            d_perm["per_model"].to_csv(stats_dir / f"q1a_perm_{d}.csv", index=False)
```

The resulting main() section (showing context and insertion) should look like:

```python
    print("Computing Q1 pooled + Q1/Q2/Q3 per-domain ...")
    pooled_tests = compute_statistical_tests(judge_var, trial_var, metrics, domain=None)
    pooled_tests["q1a"].to_csv(stats_dir / "q1a.csv", index=False)
    pooled_tests["q1b"].to_csv(stats_dir / "q1b.csv", index=False)
    # Note: q2_kw / q3_kw / q2_pairwise / q3_pairwise from pooled_tests are
    # intentionally NOT written — they conflate model and domain.

    judge_metrics = (
        variance_budget[variance_budget["metric_type"] == "judge_graded"]["metric"]
        .dropna()
        .unique()
        .tolist()
    )
    print(f"Computing permutation tests for judge metrics: {judge_metrics} ...")
    perm_pooled = compute_permutation_tests(judge_var, trial_var, judge_metrics, domain=None)
    perm_pooled["per_model"].to_csv(stats_dir / "q1a_perm.csv", index=False)
    perm_pooled["pooled"].to_csv(stats_dir / "q1a_perm_pooled.csv", index=False)

    if domains_present:
        for d in domains_present:
            d_tests = compute_statistical_tests(judge_var, trial_var, metrics, domain=d)
            d_tests["q1a"].to_csv(stats_dir / f"q1a_{d}.csv", index=False)
            d_tests["q1b"].to_csv(stats_dir / f"q1b_{d}.csv", index=False)
            d_tests["q2_kw"].to_csv(stats_dir / f"q2_kw_{d}.csv", index=False)
            d_tests["q2_pairwise"].to_csv(stats_dir / f"q2_pairwise_{d}.csv", index=False)
            d_tests["q3_kw"].to_csv(stats_dir / f"q3_kw_{d}.csv", index=False)
            d_tests["q3_pairwise"].to_csv(stats_dir / f"q3_pairwise_{d}.csv", index=False)
            d_perm = compute_permutation_tests(judge_var, trial_var, judge_metrics, domain=d)
            d_perm["per_model"].to_csv(stats_dir / f"q1a_perm_{d}.csv", index=False)
    else:
        pooled_tests["q2_kw"].to_csv(stats_dir / "q2_kw.csv", index=False)
        ...
```

- [ ] **Step 3.2: Commit**

```bash
git add analysis/eva-bench-stats/stats_variance.py
git commit -m "feat(eva-bench-stats): run compute_permutation_tests in main(); write q1a_perm CSVs"
```

---

### Task 4: Load permutation CSVs in `app.py`

**Files:**
- Modify: `analysis/eva-bench-stats/app.py`

- [ ] **Step 4.1: Add permutation CSV loads to the data loading section**

In `app.py`, find the block at approximately line 801:
```python
    q1a_pooled = _read_stat("q1a.csv")
    q1b_pooled = _read_stat("q1b.csv")
```

After these two lines, add:
```python
    q1a_perm_pooled_df = _read_stat("q1a_perm.csv")
    q1a_perm_pooled_scenarios = _read_stat("q1a_perm_pooled.csv")
```

Then find the within-domain stats loading block (approximately line 871):
```python
    q1a_within = _read_stat(f"q1a{_d_suffix}.csv") if selected_domain else pd.DataFrame()
    q1b_within = _read_stat(f"q1b{_d_suffix}.csv") if selected_domain else pd.DataFrame()
```

After these two lines, add:
```python
    q1a_perm_within = _read_stat(f"q1a_perm{_d_suffix}.csv") if selected_domain else pd.DataFrame()
```

- [ ] **Step 4.2: Verify app.py still imports (syntax check)**

```bash
cd /Users/lindsay.brin/Projects/eva/analysis/eva-bench-stats
python -c "import ast; ast.parse(open('app.py').read()); print('syntax OK')"
```

Expected: `syntax OK`

- [ ] **Step 4.3: Commit**

```bash
git add analysis/eva-bench-stats/app.py
git commit -m "feat(eva-bench-stats): load q1a_perm CSVs in app.py data section"
```

---

### Task 5: Add validity note in Tab 1 (Variance overview)

**Files:**
- Modify: `analysis/eva-bench-stats/app.py`

- [ ] **Step 5.1: Insert warning above the Q0 section in Tab 1**

In `app.py`, find the exact line (approximately 1077):
```python
        st.subheader("Is variance significantly greater than zero?")
```

Replace it with:
```python
        st.subheader("Is variance significantly greater than zero?")
        st.warning(
            "**Note:** The one-sample Wilcoxon signed-rank test applied here has validity "
            "concerns for non-negative bounded values (standard deviations). Results should "
            "be interpreted with caution."
        )
```

- [ ] **Step 5.2: Syntax check**

```bash
cd /Users/lindsay.brin/Projects/eva/analysis/eva-bench-stats
python -c "import ast; ast.parse(open('app.py').read()); print('syntax OK')"
```

Expected: `syntax OK`

- [ ] **Step 5.3: Commit**

```bash
git add analysis/eva-bench-stats/app.py
git commit -m "feat(eva-bench-stats): add validity note above Q0 Wilcoxon in Tab 1"
```

---

### Task 6: Add comparison note to Tab 2 (original)

**Files:**
- Modify: `analysis/eva-bench-stats/app.py`

- [ ] **Step 6.1: Insert warning at the top of Tab 2 content**

In `app.py`, find the block (approximately line 1124):
```python
    # ── Tab 2: Judge vs. trial variance ──────────────────────────────────────
    with tabs[2]:
        st.header("Judge vs. trial variance")
        if selected_domain:
```

After `st.header("Judge vs. trial variance")` and before `if selected_domain:`, add:

```python
        st.warning(
            "⚠️ This tab uses Wilcoxon signed-rank and is kept for comparison with the "
            "updated statistical approach in the **(b)** tab."
        )
```

- [ ] **Step 6.2: Syntax check**

```bash
cd /Users/lindsay.brin/Projects/eva/analysis/eva-bench-stats
python -c "import ast; ast.parse(open('app.py').read()); print('syntax OK')"
```

Expected: `syntax OK`

- [ ] **Step 6.3: Commit**

```bash
git add analysis/eva-bench-stats/app.py
git commit -m "feat(eva-bench-stats): add Wilcoxon comparison note to Tab 2"
```

---

### Task 7: Insert new Tab 2b and update tab indices

**Files:**
- Modify: `analysis/eva-bench-stats/app.py`

This is the largest single task. Do it in two sub-steps: first update the tab list and shift indices, then insert the new tab content.

- [ ] **Step 7.1: Update the `st.tabs()` call and all `with tabs[N]:` references**

**7.1a — Update tab list**

Find the `st.tabs(...)` call at approximately line 884:
```python
    tabs = st.tabs(
        [
            "Overview",                     # 0
            "Variance overview",            # 1
            "Judge vs. trial variance",     # 2
            "Judge variance",               # 3
            "Trial variance",               # 4
            "EVA score stability",          # 5
            "Borderline scenarios",         # 6
            "Intraclass correlation",       # 7
            "Variance decomp (LMM)",        # 8  ← NEW
            "Variance budget",              # 9  ← was 8
            "Per-metric deep dive",         # 10 ← was 9
            "Statistical tests",            # 11 ← was 10
        ]
    )
```

Replace with:
```python
    tabs = st.tabs(
        [
            "Overview",                         # 0
            "Variance overview",                # 1
            "Judge vs. trial variance",         # 2
            "Judge vs. trial variance (b)",     # 3  ← NEW
            "Judge variance",                   # 4
            "Trial variance",                   # 5
            "EVA score stability",              # 6
            "Borderline scenarios",             # 7
            "Intraclass correlation",           # 8
            "Variance decomp (LMM)",            # 9
            "Variance budget",                  # 10
            "Per-metric deep dive",             # 11
            "Statistical tests",                # 12
        ]
    )
```

**7.1b — Shift tab indices 3–11 → 4–12**

For each of the following `with tabs[N]:` comment+block headers in the variance section, update the index:

| Old line (approx) | Old text | New text |
|---|---|---|
| ~1425 | `# ── Tab 3: Judge variance` → `with tabs[3]:` | `# ── Tab 4: Judge variance` → `with tabs[4]:` |
| ~1633 | `# ── Tab 4: Trial variance` → `with tabs[4]:` | `# ── Tab 5: Trial variance` → `with tabs[5]:` |
| ~1818 | `# ── Tab 5: EVA score stability` → `with tabs[5]:` | `# ── Tab 6: EVA score stability` → `with tabs[6]:` |
| ~1886 | `# ── Tab 6: Borderline scenarios` → `with tabs[6]:` | `# ── Tab 7: Borderline scenarios` → `with tabs[7]:` |
| ~2080 | `# ── Tab 7: Intraclass correlation` → `with tabs[7]:` | `# ── Tab 8: Intraclass correlation` → `with tabs[8]:` |
| ~2184 | `# ── Tab 8:` → `with tabs[8]:` | `# ── Tab 9:` → `with tabs[9]:` |
| ~2451 | `# ── Tab 9:` → `with tabs[9]:` | `# ── Tab 10:` → `with tabs[10]:` |
| ~2545 | `# ── Tab 10: Per-metric deep dive` → `with tabs[10]:` | `# ── Tab 11: Per-metric deep dive` → `with tabs[11]:` |
| ~2580 | `# ── Tab 11: Statistical tests` → `with tabs[11]:` | `# ── Tab 12: Statistical tests` → `with tabs[12]:` |

Verify: grep for remaining `with tabs[3]:` through `with tabs[11]:` to confirm all have been updated.

```bash
grep -n "with tabs\[" /Users/lindsay.brin/Projects/eva/analysis/eva-bench-stats/app.py
```

Expected output should show no `tabs[3]` through `tabs[11]` remaining in the variance section (other than the one we're about to add in 7.2).

- [ ] **Step 7.2: Insert new Tab 3 content**

After the end of Tab 2 (after the closing `st.dataframe(...)` of the Q0 per-model section, approximately line 1423, before the `# ── Tab 4: Judge variance` comment), insert:

```python
    # ── Tab 3: Judge vs. trial variance (b) ──────────────────────────────────
    with tabs[3]:
        st.header("Judge vs. trial variance (b)")
        if selected_domain:
            st.caption(
                f"Plots show {_domain_label}. Permutation tests shown below are **{_pooled_label}** "
                "(per model, pooling its data across all 3 domains)."
            )
        st.write("""
        **What this measures:** Same comparison as the **(a)** tab, using a sign-flip permutation
        test instead of Wilcoxon signed-rank.

        **Why the updated approach?** Standard deviations are non-negative and bounded at zero,
        which can violate the symmetry assumption of the Wilcoxon signed-rank test. The sign-flip
        permutation test is distribution-free and requires no such assumption.

        **H₁ (one-sided):** Mean trial SD > mean judge SD (i.e., trial variance dominates).
        """)

        st.subheader("Judge vs. trial variance across models")
        st.caption(
            "Error bars = std dev of per-(record,trial) std devs across the group. "
            "Asterisks indicate a significant difference (sign-flip permutation test, one-sided): "
            "\\* p < 0.05, \\*\\* p < 0.01, \\*\\*\\* p < 0.001."
        )

        run_labels_cmp_b = list(combined["run_label"].unique())
        fig_b = make_subplots(rows=len(run_labels_cmp_b), cols=1, subplot_titles=run_labels_cmp_b, shared_xaxes=True)
        for row_idx, run_label in enumerate(run_labels_cmp_b, start=1):
            run_data = combined[combined["run_label"] == run_label]
            for source, color in COLORS.items():
                is_judge = source == "Judge (stochasticity)"
                fig_b.add_trace(
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
            fig_b.update_yaxes(range=[0, global_var_ymax], row=row_idx, col=1)

        q1a_perm_sig = (
            q1a_perm_pooled_df[q1a_perm_pooled_df["permutation_significant"]]
            if not q1a_perm_pooled_df.empty
            else pd.DataFrame()
        )
        y_pad_b = global_var_ymax * 0.04
        for row_idx, run_label in enumerate(run_labels_cmp_b, start=1):
            run_data = combined[combined["run_label"] == run_label]
            xref_str = "x" if row_idx == 1 else f"x{row_idx}"
            yref_str = "y" if row_idx == 1 else f"y{row_idx}"
            model_id = run_label.split(" — ")[0].strip()
            model_sig_b = (
                q1a_perm_sig[q1a_perm_sig["model"] == model_id]
                if not q1a_perm_sig.empty
                else pd.DataFrame()
            )
            for _, qrow in model_sig_b.iterrows():
                metric = qrow["metric"]
                mdata = run_data[run_data["metric"] == metric]
                if mdata.empty:
                    continue
                y_top = max(
                    mdata["judge_std"].iloc[0] + mdata["judge_std_err"].fillna(0).iloc[0],
                    mdata["trial_std"].iloc[0] + mdata["trial_std_err"].fillna(0).iloc[0],
                )
                p_val = qrow["permutation_p_value"]
                stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
                fig_b.add_annotation(
                    x=metric,
                    y=y_top + y_pad_b,
                    text=f"<b>{stars}</b>",
                    showarrow=False,
                    xref=xref_str,
                    yref=yref_str,
                    font={"size": 14, "color": "#444"},
                )

        fig_b.update_layout(
            barmode="group",
            height=200 * max(len(run_labels_cmp_b), 1),
            legend={"orientation": "v", "yanchor": "middle", "y": 0.5, "xanchor": "left", "x": 1.02},
            margin={"t": 40, "r": 160},
        )
        for ann in fig_b.layout.annotations:
            if ann.yref == "paper":
                ann.font.size = 12
        st.plotly_chart(fig_b, width="stretch")

        st.dataframe(display_combined.round(4), width="stretch")
        download_button(display_combined, "variance_comparison_b.csv")

        if not judge_dom.empty:
            st.warning(
                f"**Key finding:** Judge variance exceeds trial variance for "
                f"{len(judge_dom)} metric/run combinations: "
                f"{', '.join(judge_dom['metric'].unique())}."
            )
        else:
            st.info("**Key finding:** Trial variance exceeds judge variance for all metric/run combinations.")

        st.subheader("Statistical test: Is trial variance significantly greater than judge variance?")
        st.caption(
            f"**Primary view: {_pooled_label}** — each model's data is pooled across all 3 task domains."
        )
        if q1a_perm_pooled_df.empty:
            st.info("Not enough data to run permutation tests (need ≥ 5 paired records per model × metric). "
                    "Run the stats pipeline first.")
        else:
            st.markdown(
                "**Q1a — Sign-flip permutation test (per model × metric, one-sided, H₁: trial SD > judge SD)**"
            )
            q1a_perm_disp = q1a_perm_pooled_df[
                [
                    "metric",
                    "model",
                    "n_scenarios",
                    "permutation_mean_delta",
                    "permutation_p_value",
                    "permutation_significant",
                    "n_positive_deltas",
                    "sign_test_p_value",
                    "sign_test_significant",
                ]
            ].copy()
            q1a_perm_disp["model"] = q1a_perm_disp["model"].apply(llm_name)
            q1a_perm_disp["permutation_p_value"] = q1a_perm_disp["permutation_p_value"].map(fmt_p)
            q1a_perm_disp["sign_test_p_value"] = q1a_perm_disp["sign_test_p_value"].map(fmt_p)
            st.dataframe(
                q1a_perm_disp.round({"permutation_mean_delta": 4}),
                width="stretch",
            )
            download_button(q1a_perm_pooled_df, "stat_q1a_perm.csv")

            if not q1a_perm_pooled_scenarios.empty:
                st.markdown("**Q1a pooled — Permutation test on scenario-averaged deltas (across models)**")
                st.caption(
                    "Deltas averaged per (domain, scenario) across all 4 models → 213 independent observations. "
                    "Avoids pseudo-replication from the same scenarios appearing in all models."
                )
                pooled_scen_disp = q1a_perm_pooled_scenarios[
                    [
                        "metric",
                        "n_scenarios",
                        "permutation_mean_delta",
                        "permutation_p_value",
                        "permutation_significant",
                        "sign_test_p_value",
                        "sign_test_significant",
                    ]
                ].copy()
                pooled_scen_disp["permutation_p_value"] = pooled_scen_disp["permutation_p_value"].map(fmt_p)
                pooled_scen_disp["sign_test_p_value"] = pooled_scen_disp["sign_test_p_value"].map(fmt_p)
                st.dataframe(
                    pooled_scen_disp.round({"permutation_mean_delta": 4}),
                    width="stretch",
                )
                download_button(q1a_perm_pooled_scenarios, "stat_q1a_perm_pooled.csv")

            if not q1b.empty:
                st.markdown("**Q1b — Does the gap vary by model? (Kruskal-Wallis on per-record deltas)**")
                q1b_disp_b = q1b[["metric", "H", "p", "significant"]].copy()
                q1b_disp_b["H"] = q1b_disp_b["H"].round(2)
                q1b_disp_b["p"] = q1b_disp_b["p"].map(fmt_p)
                st.dataframe(q1b_disp_b, width="stretch")

            if selected_domain and not q1a_perm_within.empty:
                with st.expander(f"Within-domain drill-down ({_domain_label})"):
                    st.caption(
                        "Same permutation test, restricted to the selected domain only. "
                        "Use this to check whether the pooled conclusion holds inside this domain."
                    )
                    st.markdown("**Q1a (permutation) — within-domain**")
                    q1a_pw_disp = q1a_perm_within[
                        [
                            "metric",
                            "model",
                            "n_scenarios",
                            "permutation_mean_delta",
                            "permutation_p_value",
                            "permutation_significant",
                            "n_positive_deltas",
                            "sign_test_p_value",
                            "sign_test_significant",
                        ]
                    ].copy()
                    q1a_pw_disp["model"] = q1a_pw_disp["model"].apply(llm_name)
                    q1a_pw_disp["permutation_p_value"] = q1a_pw_disp["permutation_p_value"].map(fmt_p)
                    q1a_pw_disp["sign_test_p_value"] = q1a_pw_disp["sign_test_p_value"].map(fmt_p)
                    st.dataframe(
                        q1a_pw_disp.round({"permutation_mean_delta": 4}),
                        width="stretch",
                    )

            st.markdown("**Plain-English interpretation:**")
            _perm_metrics = sorted(q1a_perm_pooled_df["metric"].unique()) if not q1a_perm_pooled_df.empty else []

            def _perm_model_list(rows):
                return ", ".join(
                    f"{llm_name(r['model'])} (p={fmt_p(r['permutation_p_value'])})" for r in rows
                )

            for metric in _perm_metrics:
                q1a_perm_sub = q1a_perm_pooled_df[q1a_perm_pooled_df["metric"] == metric]
                q1b_row = (
                    q1b[q1b["metric"] == metric].iloc[0]
                    if (not q1b.empty and (q1b["metric"] == metric).any())
                    else None
                )

                sig_rows = [r for _, r in q1a_perm_sub.iterrows() if r["permutation_significant"]]
                not_sig_rows = [r for _, r in q1a_perm_sub.iterrows() if not r["permutation_significant"]]

                sentences = []
                if sig_rows:
                    sentences.append(
                        f"Trial variance significantly exceeds judge variance for {_perm_model_list(sig_rows)}"
                    )
                if not_sig_rows:
                    sentences.append(
                        f"No significant evidence that trial variance exceeds judge variance for "
                        f"{_perm_model_list(not_sig_rows)}"
                    )

                q1b_txt = ""
                if q1b_row is not None:
                    if q1b_row["significant"]:
                        q1b_txt = (
                            f" The size of the gap varies significantly across models "
                            f"(K-W p={fmt_p(q1b_row['p'])})."
                        )
                    else:
                        q1b_txt = f" The gap is consistent across models (K-W p={fmt_p(q1b_row['p'])})."

                st.markdown(f"- **{metric}**: " + ". ".join(sentences) + "." + q1b_txt)

            with st.expander("Methodology"):
                st.markdown("""
**Why sign-flip permutation (Q1a)?**
Standard deviations are non-negative and bounded at zero. Pairing by record removes the
scenario-difficulty confound (same as the Wilcoxon approach), but the Wilcoxon requires
the *differences* to be symmetrically distributed around zero — an assumption that can be
violated when values are bounded. The sign-flip permutation test has no such requirement.

**Calculation steps (Q1a — permutation):**
1. For each (record, trial, metric, model): compute judge std dev across the 3 judge iterations.
2. Average those judge std devs over trials → one judge-variance estimate per (record, model, metric).
3. delta = trial_SD − judge_SD per record.
4. Observed test statistic = mean(delta).
5. Permutation null: independently flip each delta's sign with p=0.5, repeat 10,000 times,
   compute permuted mean each time.
6. Two-sided p = fraction of |permuted means| ≥ |observed mean|.
7. One-sided p (H₁: mean delta > 0):
   - If observed mean ≥ 0: p_one = p_two / 2
   - If observed mean < 0: p_one = 1 − p_two / 2
   (When mean delta < 0, p_one > 0.5 — no evidence for H₁.)
8. No Bonferroni correction across models — each model is an independent reporting unit.

**Sign test (complementary):**
Tests directional consistency: does P(delta > 0) > 0.5?
scipy.stats.binomtest(n_positive_deltas, n=n, p=0.5, alternative='greater').
Robust to noisy SD estimates; makes no symmetry assumption.

**Pooled-across-models test:**
To avoid pseudo-replication (same scenarios appear in all 4 models), deltas are averaged
per (domain, scenario) across models first → 213 independent observations — then the
permutation and sign tests are applied to those averages.

**Why Kruskal-Wallis for Q1b (unchanged)?**
Q1b tests whether the *gap* between judge and trial variance is consistent across models.
Deltas (trial_SD − judge_SD) are unbounded and can be negative, so the Kruskal-Wallis
test is valid here (no bounded-at-zero concern). No correction is applied: Q1b is one
test per metric.
""")
```

- [ ] **Step 7.3: Syntax check and quick render test**

```bash
cd /Users/lindsay.brin/Projects/eva/analysis/eva-bench-stats
python -c "import ast; ast.parse(open('app.py').read()); print('syntax OK')"
```

Expected: `syntax OK`

- [ ] **Step 7.4: Commit**

```bash
git add analysis/eva-bench-stats/app.py
git commit -m "feat(eva-bench-stats): insert Tab 3 (Judge vs. trial variance (b)); shift old tabs 3-11 → 4-12"
```

---

### Task 8: Update Statistical tests tab (now Tab 12)

**Files:**
- Modify: `analysis/eva-bench-stats/app.py`

- [ ] **Step 8.1: Add reference note to Q1a expander in Tab 12**

Find in `app.py` (approximately line 2656, now tab 12):
```python
        with st.expander("Q1a — Paired Wilcoxon: judge vs. trial variance (→ Judge vs. trial variance tab)"):
            if q1a.empty:
```

Immediately after `with st.expander(...):`  and before `if q1a.empty:`, insert:

```python
            st.warning(
                "⚠️ This describes the original Wilcoxon approach. Results are retained for "
                "reference; the updated analysis uses sign-flip permutation "
                "(see **Judge vs. trial variance (b)** tab)."
            )
```

- [ ] **Step 8.2: Add reference note to Q1b expander**

Find (approximately line 2698):
```python
        with st.expander(
            "Q1b — Kruskal-Wallis: does the judge-vs-trial gap vary across models? (→ Judge vs. trial variance tab)"
        ):
            if q1b.empty:
```

Immediately after the `with st.expander(...):`  and before `if q1b.empty:`, insert:

```python
            st.warning(
                "⚠️ Kept for reference alongside the original Wilcoxon results. "
                "The (b) tab also shows Q1b unchanged — K-W on unbounded deltas is valid."
            )
```

- [ ] **Step 8.3: Add new permutation methodology expander**

After the entire Q1b expander block (which ends with the `with st.expander("Q1b full methodology"):` nested expander and its closing), and before the `# ── Q2` comment, insert a new top-level expander:

```python
        # ── Q1a permutation (new) ─────────────────────────────────────────────────
        with st.expander(
            "Q1a — Sign-flip permutation test (updated approach) (→ Judge vs. trial variance (b) tab)"
        ):
            st.markdown("**Per-model results**")
            if q1a_perm_pooled_df.empty:
                st.info("No results (run stats pipeline first).")
            else:
                _q1a_perm_disp = q1a_perm_pooled_df.copy()
                _q1a_perm_disp["permutation_p_value"] = _q1a_perm_disp["permutation_p_value"].map(fmt_p)
                _q1a_perm_disp["sign_test_p_value"] = _q1a_perm_disp["sign_test_p_value"].map(fmt_p)
                _q1a_perm_disp["model"] = _q1a_perm_disp["model"].apply(llm_name)
                st.dataframe(
                    _q1a_perm_disp.round({"permutation_mean_delta": 4}),
                    width="stretch",
                )
                download_button(q1a_perm_pooled_df, "stat_q1a_perm.csv")

            st.markdown("**Pooled across models (scenario-averaged)**")
            if q1a_perm_pooled_scenarios.empty:
                st.info("No results.")
            else:
                _q1a_perm_sc_disp = q1a_perm_pooled_scenarios.copy()
                _q1a_perm_sc_disp["permutation_p_value"] = _q1a_perm_sc_disp["permutation_p_value"].map(fmt_p)
                _q1a_perm_sc_disp["sign_test_p_value"] = _q1a_perm_sc_disp["sign_test_p_value"].map(fmt_p)
                st.dataframe(
                    _q1a_perm_sc_disp.round({"permutation_mean_delta": 4}),
                    width="stretch",
                )
                download_button(q1a_perm_pooled_scenarios, "stat_q1a_perm_pooled.csv")

            with st.expander("Q1a permutation full methodology"):
                st.markdown("""
**Test choice:** Sign-flip permutation test (10,000 permutations, one-sided).

**Permutation test:** For each permutation, the sign of each per-scenario delta
(trial SD − judge SD) was independently flipped with probability 0.5, and the mean
was recomputed. The two-sided p-value is the fraction of permuted means where
|permuted mean| ≥ |observed mean|. The one-sided p-value (H₁: mean trial SD > mean judge SD)
is p_two / 2 when observed mean ≥ 0, and 1 − p_two / 2 otherwise.

**Why not Bonferroni across models:** Each per-model test is an independent
directional hypothesis about that specific model's behavior — not a family of pairwise
comparisons. This differs from the original Wilcoxon Q1a, which was two-sided and applied
Bonferroni correction across all models simultaneously. Do not directly compare raw
permutation p-values to the Wilcoxon's corrected p-values.

**Sign test:** Exact binomial test (scipy.stats.binomtest), H₁: P(delta > 0) > 0.5.
Tests directional consistency — whether more than half of scenarios have trial SD > judge SD.
Robust to noisy SD estimates; no symmetry assumption.

**Pooled-across-models test:** Deltas averaged per (domain, scenario) across models first
→ 213 independent observations. Avoids pseudo-replication from the same scenarios
appearing in all 4 models. Permutation + sign test applied to the 213 averaged deltas.
""")
```

- [ ] **Step 8.4: Syntax check**

```bash
cd /Users/lindsay.brin/Projects/eva/analysis/eva-bench-stats
python -c "import ast; ast.parse(open('app.py').read()); print('syntax OK')"
```

Expected: `syntax OK`

- [ ] **Step 8.5: Commit**

```bash
git add analysis/eva-bench-stats/app.py
git commit -m "feat(eva-bench-stats): update Statistical tests tab with permutation methodology and reference notes"
```

---

## Self-Review Checklist

### Spec coverage

| Spec requirement | Task covering it |
|---|---|
| Move `permutation_test` to `stats_utils` | Task 1 |
| `stats_perturbations.py` keeps backward-compat re-export | Task 1.4 |
| `compute_permutation_tests()` with correct signature | Task 2.3 |
| Judge metrics detected from `variance_budget` at runtime | Task 3.1 |
| `q1a_perm.csv` — per-model, pooled across domains | Task 3.1 |
| `q1a_perm_pooled.csv` — scenario-averaged across models | Task 3.1 |
| `q1a_perm_{domain}.csv` — per-domain | Task 3.1 |
| Delta = trial_SD − judge_SD (positive = trial dominates) | Task 2.3 |
| One-sided p from two-sided: p/2 or 1−p/2 | Task 2.3 |
| Sign test via binomtest | Task 2.3 |
| Seed strategy: `seed + hash(f"{metric}:{model_id}") % (2**31)` | Task 2.3 |
| Min n=5 guard | Task 2.3 |
| Pooled: average delta per scenario across models first | Task 2.3 |
| Tab 1 validity note above Q0 Wilcoxon section | Task 5 |
| Tab 2 comparison note (Wilcoxon kept for reference) | Task 6 |
| New Tab 2b with all required elements | Task 7.2 |
| Tab indices 3–11 → 4–12 | Task 7.1 |
| Bar chart asterisks in Tab 2b use permutation_p_value | Task 7.2 |
| Q1b unchanged in Tab 2b (same q1b.csv) | Task 7.2 |
| Within-domain drill-down using q1a_perm_{domain}.csv | Task 7.2 |
| Pooled table in Tab 2b (below per-model, before Q1b) | Task 7.2 |
| Statistical tests tab: Q1a note, Q1b note, new expander | Task 8 |

### No placeholders
All code steps include complete code. No TBD, TODO, or "add appropriate handling" entries.

### Type consistency
- `compute_permutation_tests` returns `dict[str, pd.DataFrame]` with keys `"per_model"` and `"pooled"` — consistent with references in Tasks 3, 4, 7, 8.
- Column names in `per_model`: `metric`, `model`, `n_scenarios`, `permutation_mean_delta`, `permutation_p_value`, `permutation_significant`, `n_positive_deltas`, `sign_test_p_value`, `sign_test_significant` — consistent across all tasks.
- Column names in `pooled`: same but without `model` and `n_positive_deltas` — consistent across Tasks 3, 7, 8.
- App variable names: `q1a_perm_pooled_df` (per-model) and `q1a_perm_pooled_scenarios` (pooled across models) — used consistently in Tasks 4, 7, 8.
