import math
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "analysis" / "eva-bench-stats"))
from paper_config import ModelEntry, PaperConfig
from paper_tables import format_cell, shade_index, lookup_pooled, write_accuracy_table, write_experience_table


def test_format_cell_symmetric() -> None:
    assert format_cell(0.428, 0.393, 0.463) == "0.428 {\\scriptsize $\\pm$0.035}"


def test_format_cell_asymmetric_uses_max_halfwidth() -> None:
    # point - lo = 0.080, hi - point = 0.020 → max = 0.080
    assert format_cell(0.500, 0.420, 0.520) == "0.500 {\\scriptsize $\\pm$0.080}"


def test_format_cell_missing() -> None:
    assert format_cell(None, None, None) == "--"
    assert format_cell(float("nan"), float("nan"), float("nan")) == "--"


def test_shade_index_clamps_and_buckets() -> None:
    # min=0.1, max=0.5 → 7 even buckets across range
    assert shade_index(0.1, lo=0.1, hi=0.5, n_steps=7) == 0
    assert shade_index(0.5, lo=0.1, hi=0.5, n_steps=7) == 6
    assert shade_index(0.05, lo=0.1, hi=0.5, n_steps=7) == 0  # clamp low
    assert shade_index(0.6, lo=0.1, hi=0.5, n_steps=7) == 6   # clamp high
    # degenerate range
    assert shade_index(0.3, lo=0.3, hi=0.3, n_steps=7) == 0


def _pooled_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"model_label": "M1", "metric": "EVA-A_pass", "domain": "pooled",
         "point_estimate": 0.40, "ci_lower": 0.35, "ci_upper": 0.45},
        {"model_label": "M1", "metric": "task_completion", "domain": "pooled",
         "point_estimate": 0.60, "ci_lower": 0.55, "ci_upper": 0.65},
        # non-pooled rows must be ignored
        {"model_label": "M1", "metric": "EVA-A_pass", "domain": "itsm",
         "point_estimate": 0.99, "ci_lower": 0.98, "ci_upper": 1.0},
    ])


def test_lookup_pooled_hits() -> None:
    df = _pooled_df()
    point, lo, hi = lookup_pooled(df, "M1", "EVA-A_pass")
    assert (point, lo, hi) == (0.40, 0.35, 0.45)


def test_lookup_pooled_misses_returns_nones() -> None:
    df = _pooled_df()
    assert lookup_pooled(df, "M1", "EVA-A_pass_at_k") == (None, None, None)
    assert lookup_pooled(df, "Ghost", "EVA-A_pass") == (None, None, None)


def _cfg() -> PaperConfig:
    return PaperConfig(
        output_dir="ignored",
        accuracy_aggregate={
            "pass_at_1": "EVA-A_pass",
            "pass_at_k": "EVA-A_pass_at_k",
            "pass_power_k": "EVA-A_pass_power_k",
        },
        accuracy_submetrics={
            "task_completion": "Task Completion",
            "faithfulness": "Faithfulness",
            "agent_speech_fidelity": "Agent Speech Fidelity",
        },
        experience_aggregate={
            "pass_at_1": "EVA-X_pass",
            "pass_at_k": "EVA-X_pass_at_k",
            "pass_power_k": "EVA-X_pass_power_k",
        },
        experience_submetrics={
            "turn_taking": "Turn-Taking",
            "conciseness": "Conciseness",
            "conversation_progression": "Conv. Progression",
        },
        scatter={},
        models={
            "Sys A": ModelEntry(label="Sys A", alias="a", arch="cascade"),
            "Sys B": ModelEntry(label="Sys B", alias="b", arch="s2s"),
        },
    )


def _pooled_with_pass1_only() -> pd.DataFrame:
    rows = []
    for m, p in [("Sys A", 0.42), ("Sys B", 0.61)]:
        for metric, val in [
            ("EVA-A_pass", p), ("EVA-X_pass", p + 0.05),
            ("task_completion", p), ("faithfulness", p),
            ("agent_speech_fidelity", p), ("turn_taking", p),
            ("conciseness", p), ("conversation_progression", p),
        ]:
            rows.append({
                "model_label": m, "metric": metric, "domain": "pooled",
                "point_estimate": val, "ci_lower": val - 0.04, "ci_upper": val + 0.04,
            })
    return pd.DataFrame(rows)


def test_write_accuracy_table_emits_expected_structure(tmp_path: Path) -> None:
    out = tmp_path / "accuracy_table.tex"
    write_accuracy_table(_pooled_with_pass1_only(), _cfg(), out)
    text = out.read_text()
    assert "\\begin{table}" in text
    assert "EVA-A" in text
    assert "Task Completion" in text
    # pass_at_1 present, pass_at_k missing → "--" appears in pass_at_k cells
    assert "0.420 {\\scriptsize $\\pm$0.040}" in text
    assert "--" in text
    # arch grouping
    assert "Cascade" in text
    assert "S2S" in text


def test_write_experience_table_uses_eva_x(tmp_path: Path) -> None:
    out = tmp_path / "experience_table.tex"
    write_experience_table(_pooled_with_pass1_only(), _cfg(), out)
    text = out.read_text()
    assert "EVA-X" in text
    assert "Turn-Taking" in text
    assert "0.470 {\\scriptsize $\\pm$0.040}" in text  # 0.42 + 0.05
