"""Configuration loader and shared constants for the paper-figure generators."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ARCH_ORDER: tuple[str, str, str] = ("cascade", "hybrid", "s2s")

# Heatmap palettes (HTML hex, light → dark) — must match the existing tables.
ACCENT_PALETTE = [
    ("acc1", "edeaf4"), ("acc2", "d9d2e6"), ("acc3", "bfb3d4"),
    ("acc4", "9d8dbb"), ("acc5", "7a679f"), ("acc6", "584981"),
    ("acc7", "3b3060"),
]
TEAL_PALETTE = [
    ("tel1", "b8dede"), ("tel2", "90cece"), ("tel3", "62b8b8"),
    ("tel4", "3a9e9e"), ("tel5", "1e8484"), ("tel6", "0f6b6b"),
    ("tel7", "075656"),
]
PINK_PALETTE = [
    ("pnk1", "fde4ec"), ("pnk2", "fac4d4"), ("pnk3", "f59ab5"),
    ("pnk4", "ed6f95"), ("pnk5", "db4577"), ("pnk6", "b82d5c"),
    ("pnk7", "8c1f44"),
]
# Cells in the bottom 3 palette steps use black text; top 4 use white text.
LIGHT_TEXT_THRESHOLD_INDEX = 3


@dataclass(frozen=True)
class ModelEntry:
    label: str
    alias: str
    arch: str  # one of ARCH_ORDER


@dataclass(frozen=True)
class PaperConfig:
    output_dir: str
    accuracy_aggregate: dict[str, str]
    accuracy_submetrics: dict[str, str]
    experience_aggregate: dict[str, str]
    experience_submetrics: dict[str, str]
    scatter: dict[str, dict[str, str]]
    models: dict[str, ModelEntry]


def load_paper_config(config_path: Path) -> PaperConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    paper = raw["paper"]
    models_raw = raw.get("models") or {}
    models = {
        label: ModelEntry(
            label=label,
            alias=spec["alias"],
            arch=spec.get("type", "cascade"),
        )
        for label, spec in models_raw.items()
    }
    return PaperConfig(
        output_dir=paper["output_dir"],
        accuracy_aggregate=dict(paper["accuracy"]["aggregate"]),
        accuracy_submetrics=dict(paper["accuracy"]["submetrics"]),
        experience_aggregate=dict(paper["experience"]["aggregate"]),
        experience_submetrics=dict(paper["experience"]["submetrics"]),
        scatter={k: dict(v) for k, v in paper["scatter"].items()},
        models=models,
    )


def sort_models(models: dict[str, ModelEntry]) -> list[ModelEntry]:
    """Sorted by (arch order, label). Used for stable row order in tables and scatter."""
    arch_index = {a: i for i, a in enumerate(ARCH_ORDER)}
    return sorted(models.values(), key=lambda m: (arch_index.get(m.arch, 99), m.label.lower()))
