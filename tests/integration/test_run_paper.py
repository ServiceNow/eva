import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis" / "eva-bench-stats"))
import run_paper  # noqa: E402


def _build_pooled_csv(path: Path) -> None:
    rows = []
    metrics = [
        "EVA-A_pass", "EVA-X_pass",
        "task_completion", "faithfulness", "agent_speech_fidelity",
        "turn_taking", "conciseness", "conversation_progression",
    ]
    for m, base in [("Sys A", 0.40), ("Sys B", 0.62), ("Sys C", 0.25)]:
        for metric in metrics:
            rows.append({
                "model_label": m, "metric": metric, "domain": "pooled", "n": "pooled",
                "point_estimate": base,
                "ci_lower": base - 0.04,
                "ci_upper": base + 0.04,
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def _build_config(path: Path, output_dir: str) -> None:
    path.write_text(yaml.safe_dump({
        "models": {
            "Sys A": {"alias": "a", "type": "cascade"},
            "Sys B": {"alias": "b", "type": "s2s"},
            "Sys C": {"alias": "c", "type": "hybrid"},
        },
        "paper": {
            "output_dir": output_dir,
            "accuracy": {
                "aggregate": {
                    "pass_at_1": "EVA-A_pass",
                    "pass_at_k": "EVA-A_pass_at_k",
                    "pass_power_k": "EVA-A_pass_power_k",
                },
                "submetrics": {
                    "task_completion": "Task Completion",
                    "faithfulness": "Faithfulness",
                    "agent_speech_fidelity": "Agent Speech Fidelity",
                },
            },
            "experience": {
                "aggregate": {
                    "pass_at_1": "EVA-X_pass",
                    "pass_at_k": "EVA-X_pass_at_k",
                    "pass_power_k": "EVA-X_pass_power_k",
                },
                "submetrics": {
                    "turn_taking": "Turn-Taking",
                    "conciseness": "Conciseness",
                    "conversation_progression": "Conv. Progression",
                },
            },
            "scatter": {
                "pass_at_1": {"x": "EVA-A_pass", "y": "EVA-X_pass"},
                "pass_at_k": {"x": "EVA-A_pass_at_k", "y": "EVA-X_pass_at_k"},
                "pass_power_k": {"x": "EVA-A_pass_power_k", "y": "EVA-X_pass_power_k"},
            },
        },
    }))


def test_run_paper_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = tmp_path / "CIs_config.yaml"
    pooled_path = tmp_path / "results_pooled.csv"
    out_rel = "out/paper"
    _build_config(cfg_path, out_rel)
    _build_pooled_csv(pooled_path)
    monkeypatch.setattr(run_paper, "PROJECT_ROOT", tmp_path)

    run_paper.run(cfg_path, pooled_path)

    out_dir = tmp_path / out_rel
    tex_a = (out_dir / "accuracy_table.tex").read_text()
    tex_x = (out_dir / "experience_table.tex").read_text()
    assert "\\begin{table}" in tex_a and "EVA-A" in tex_a and "Cascade" in tex_a
    assert "\\begin{table}" in tex_x and "EVA-X" in tex_x

    pdf = out_dir / "accuracy_vs_experience_pass_at_1.pdf"
    assert pdf.exists() and pdf.stat().st_size > 0
    # pass@k variants are absent from the pooled CSV → no PDF written.
    assert not (out_dir / "accuracy_vs_experience_pass_at_k.pdf").exists()

    payload = json.loads((out_dir / "pareto_frontier.json").read_text())
    assert set(payload.keys()) == {"pass_at_1", "pass_at_k", "pass_power_k"}
    assert payload["pass_at_k"] == []
    assert len(payload["pass_at_1"]) >= 1
