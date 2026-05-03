import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "analysis" / "eva-bench-stats"))
from paper_config import ModelEntry, PaperConfig
from paper_frontier import write_frontier_json


def _cfg() -> PaperConfig:
    return PaperConfig(
        output_dir="ignored",
        accuracy_aggregate={}, accuracy_submetrics={},
        experience_aggregate={}, experience_submetrics={},
        scatter={
            "pass_at_1":    {"x": "EVA-A_pass",         "y": "EVA-X_pass"},
            "pass_at_k":    {"x": "EVA-A_pass_at_k",    "y": "EVA-X_pass_at_k"},
            "pass_power_k": {"x": "EVA-A_pass_power_k", "y": "EVA-X_pass_power_k"},
        },
        models={
            "Sys A": ModelEntry(label="Sys A", alias="a", arch="cascade"),
            "Sys B": ModelEntry(label="Sys B", alias="b", arch="s2s"),
            "Sys C": ModelEntry(label="Sys C", alias="c", arch="cascade"),
        },
    )


def _pooled() -> pd.DataFrame:
    rows = []
    # pass@1: Sys A=(0.4,0.5), Sys B=(0.6,0.3), Sys C=(0.3,0.2)
    # Frontier: A and B (both non-dominated); C is dominated by A.
    for m, ax, ay in [("Sys A", 0.4, 0.5), ("Sys B", 0.6, 0.3), ("Sys C", 0.3, 0.2)]:
        rows.append({"model_label": m, "metric": "EVA-A_pass", "domain": "pooled",
                     "point_estimate": ax, "ci_lower": ax - 0.02, "ci_upper": ax + 0.02})
        rows.append({"model_label": m, "metric": "EVA-X_pass", "domain": "pooled",
                     "point_estimate": ay, "ci_lower": ay - 0.03, "ci_upper": ay + 0.03})
    return pd.DataFrame(rows)


def test_write_frontier_json_shape(tmp_path: Path) -> None:
    out = tmp_path / "pareto_frontier.json"
    write_frontier_json(_pooled(), _cfg(), out)
    payload = json.loads(out.read_text())
    assert set(payload.keys()) == {"pass_at_1", "pass_at_k", "pass_power_k"}
    pass1 = payload["pass_at_1"]
    assert {e["system"] for e in pass1} == {"Sys A", "Sys B"}
    a = next(e for e in pass1 if e["system"] == "Sys A")
    assert a["system_type"] == "cascade"
    assert a["eva_a"] == {"point": 0.4, "ci_low": 0.38, "ci_high": 0.42}
    assert a["eva_x"] == {"point": 0.5, "ci_low": 0.47, "ci_high": 0.53}
    # missing variants → empty list
    assert payload["pass_at_k"] == []
    assert payload["pass_power_k"] == []
    # sorted by eva_a.point ascending
    assert [e["eva_a"]["point"] for e in pass1] == sorted(e["eva_a"]["point"] for e in pass1)
