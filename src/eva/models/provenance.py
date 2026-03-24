"""Provenance model for tracking run artifacts and environment."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ArtifactInfo(BaseModel):
    """Hash and path information for a single artifact."""

    path: str
    sha256: str


class BaseProvenance(BaseModel):
    """Shared provenance fields for code state, artifacts, and environment."""

    eva_version: str
    simulation_version: str = ""
    metrics_version: str = ""
    git_commit_sha: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: Optional[bool] = None
    git_diff_hash: Optional[str] = None
    dataset: Optional[ArtifactInfo] = None
    agent_config: Optional[ArtifactInfo] = None
    tool_module: Optional[ArtifactInfo] = None
    scenario_db: Optional[ArtifactInfo] = None
    python_version: str = ""
    platform: str = ""
    captured_at: datetime = Field(default_factory=datetime.now)


class RunProvenance(BaseProvenance):
    """Provenance for a benchmark simulation run."""

    prompts: list[ArtifactInfo] = Field(default_factory=list)


class MetricsProvenance(BaseProvenance):
    """Provenance for a metrics computation run."""

    judge_prompts: list[ArtifactInfo] = Field(default_factory=list)
    metrics_computed: list[str] = Field(default_factory=list)
