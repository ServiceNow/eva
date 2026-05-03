from pathlib import Path

import pytest
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "analysis" / "eva-bench-stats"))
from paper_config import PaperConfig, load_paper_config, ARCH_ORDER


def test_load_paper_config_parses_sections(tmp_path: Path) -> None:
    cfg_path = tmp_path / "CIs_config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "models": {
            "Sys A": {"alias": "alias-a", "type": "cascade"},
            "Sys B": {"alias": "alias-b", "type": "s2s"},
        },
        "paper": {
            "output_dir": "out/paper",
            "accuracy": {
                "aggregate": {"pass_at_1": "EVA-A_pass"},
                "submetrics": {"task_completion": "Task Completion"},
            },
            "experience": {
                "aggregate": {"pass_at_1": "EVA-X_pass"},
                "submetrics": {"turn_taking": "Turn-Taking"},
            },
            "scatter": {"pass_at_1": {"x": "EVA-A_pass", "y": "EVA-X_pass"}},
        },
    }))
    cfg = load_paper_config(cfg_path)
    assert isinstance(cfg, PaperConfig)
    assert cfg.output_dir == "out/paper"
    assert cfg.accuracy_aggregate == {"pass_at_1": "EVA-A_pass"}
    assert cfg.accuracy_submetrics == {"task_completion": "Task Completion"}
    assert cfg.scatter == {"pass_at_1": {"x": "EVA-A_pass", "y": "EVA-X_pass"}}
    assert cfg.models["Sys A"].arch == "cascade"


def test_arch_order_constant() -> None:
    assert ARCH_ORDER == ("cascade", "hybrid", "s2s")
