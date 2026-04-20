"""Provenance capture utility for benchmark runs."""

import hashlib
import importlib.util
import os
import platform
import subprocess
import sys
from pathlib import Path

import eva
from eva.models.config import RunConfig
from eva.models.provenance import ArtifactInfo, MetricsProvenance, RunProvenance
from eva.utils.hash_utils import hash_directory, hash_file
from eva.utils.logging import get_logger

logger = get_logger(__name__)


def _run_git_command(args: list[str]) -> str | None:
    """Run a git command and return stripped stdout, or None on failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.debug(f"Git command failed: git {' '.join(args)}: {e}")
    return None


def _get_git_info() -> dict:
    """Collect git state information."""
    info: dict[str, str | bool | None] = {
        "git_commit_sha": None,
        "git_branch": None,
        "git_dirty": None,
        "git_diff_hash": None,
    }

    info["git_commit_sha"] = _run_git_command(["rev-parse", "HEAD"]) or os.environ.get("GIT_COMMIT_SHA")
    if info["git_commit_sha"] is None:
        return info

    info["git_branch"] = _run_git_command(["branch", "--show-current"]) or os.environ.get("GIT_BRANCH")

    porcelain = _run_git_command(["status", "--porcelain"])
    if porcelain is not None:
        info["git_dirty"] = len(porcelain) > 0
        if info["git_dirty"]:
            diff_output = _run_git_command(["diff"])
            if diff_output:
                info["git_diff_hash"] = hashlib.sha256(diff_output.encode()).hexdigest()[:12]
    else:
        env_dirty = os.environ.get("GIT_DIRTY")
        if env_dirty is not None:
            info["git_dirty"] = env_dirty.lower() in ("1", "true", "yes")
        info["git_diff_hash"] = os.environ.get("GIT_DIFF_HASH") or None

    return info


def _find_project_root() -> Path | None:
    """Find project root by searching up from this file for pyproject.toml."""
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def resolve_tool_module_file(tool_module_path: str | None) -> Path | None:
    """Resolve a Python module path to its filesystem path."""
    if not tool_module_path:
        return None
    try:
        spec = importlib.util.find_spec(tool_module_path)
        if spec and spec.origin:
            return Path(spec.origin)
    except (ModuleNotFoundError, ValueError) as e:
        logger.warning(f"Could not resolve tool module '{tool_module_path}': {e}")
    return None


def capture_provenance(
    config: RunConfig,
    tool_module_file: Path | None = None,
) -> RunProvenance:
    """Capture full provenance for a benchmark run.

    Args:
        config: The run configuration
        tool_module_file: Resolved filesystem path of the tool module.
            If not provided, will attempt to resolve from config.tool_module_path.

    Returns:
        RunProvenance with all captured metadata
    """
    git_info = _get_git_info()
    project_root = _find_project_root()

    def _relative_path(path: Path) -> str:
        """Convert an absolute path to a project-relative path."""
        if project_root:
            try:
                return str(path.resolve().relative_to(project_root))
            except ValueError:
                logger.debug(f"Path {path} is not relative to project root, using absolute path")
        return str(path)

    dataset = ArtifactInfo(
        path=_relative_path(config.dataset_path),
        sha256=hash_file(config.dataset_path),
    )

    agent_config = ArtifactInfo(
        path=_relative_path(config.agent_config_path),
        sha256=hash_file(config.agent_config_path),
    )

    prompts: list[ArtifactInfo] = []
    prompts_dir = project_root / "configs" / "prompts" if project_root else None
    if prompts_dir and prompts_dir.is_dir():
        for yaml_file in sorted(prompts_dir.glob("*.yaml")):
            prompts.append(
                ArtifactInfo(
                    path=_relative_path(yaml_file),
                    sha256=hash_file(yaml_file),
                )
            )
    else:
        logger.warning("Could not find configs/prompts/ directory for provenance")

    tool_module_info: ArtifactInfo | None = None
    if tool_module_file is None:
        tool_module_file = resolve_tool_module_file(config.tool_module_path)
    if tool_module_file and tool_module_file.exists():
        tool_module_info = ArtifactInfo(
            path=_relative_path(tool_module_file),
            sha256=hash_file(tool_module_file),
        )

    scenario_db = ArtifactInfo(
        path=_relative_path(config.tool_mocks_path),
        sha256=hash_directory(config.tool_mocks_path),
    )

    provenance = RunProvenance(
        eva_version=eva.__version__,
        simulation_version=getattr(eva, "simulation_version", ""),
        metrics_version=getattr(eva, "metrics_version", ""),
        git_commit_sha=git_info.get("git_commit_sha"),
        git_branch=git_info.get("git_branch"),
        git_dirty=git_info.get("git_dirty"),
        git_diff_hash=git_info.get("git_diff_hash"),
        dataset=dataset,
        agent_config=agent_config,
        prompts=prompts,
        tool_module=tool_module_info,
        scenario_db=scenario_db,
        python_version=sys.version,
        platform=platform.platform(),
    )

    commit_short = (provenance.git_commit_sha or "unknown")[:8]
    dirty = " (dirty)" if provenance.git_dirty else ""
    logger.info(
        f"Provenance captured: sim={provenance.simulation_version} "
        f"metrics={provenance.metrics_version} commit={commit_short}{dirty}"
    )

    return provenance


def capture_metrics_provenance(
    metric_names: list[str],
    run_config: dict | None = None,
) -> MetricsProvenance:
    """Capture provenance for a metrics computation run.

    Args:
        metric_names: List of metric names being computed
        run_config: Parsed config.json from the run directory, used to
            re-hash simulation artifacts for drift detection.

    Returns:
        MetricsProvenance with judge prompts, simulation artifacts, and git state
    """
    git_info = _get_git_info()
    project_root = _find_project_root()

    def _make_artifact(path_str: str | None, is_dir: bool = False) -> ArtifactInfo | None:
        if not path_str:
            return None
        path = Path(path_str)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            logger.warning(f"Artifact not found for metrics provenance: {path_str}")
            return None
        sha = hash_directory(path) if is_dir else hash_file(path)
        return ArtifactInfo(path=path_str, sha256=sha)

    dataset_info: ArtifactInfo | None = None
    agent_config_info: ArtifactInfo | None = None
    tool_module_info: ArtifactInfo | None = None
    scenario_db_info: ArtifactInfo | None = None

    if run_config:
        dataset_info = _make_artifact(run_config.get("dataset_path"))
        agent_config_info = _make_artifact(run_config.get("agent_config_path"))
        scenario_db_info = _make_artifact(run_config.get("tool_mocks_path"), is_dir=True)
        tool_module_file = resolve_tool_module_file(run_config.get("tool_module_path"))
        if tool_module_file and tool_module_file.exists():
            rel_path = str(tool_module_file)
            if project_root:
                try:
                    rel_path = str(tool_module_file.resolve().relative_to(project_root))
                except ValueError:
                    pass
            tool_module_info = ArtifactInfo(path=rel_path, sha256=hash_file(tool_module_file))

    judge_prompts: list[ArtifactInfo] = []
    if project_root:
        judge_yaml = project_root / "configs" / "prompts" / "judge.yaml"
        if judge_yaml.exists():
            rel_path = str(judge_yaml.relative_to(project_root))
            judge_prompts.append(ArtifactInfo(path=rel_path, sha256=hash_file(judge_yaml)))

    return MetricsProvenance(
        eva_version=eva.__version__,
        simulation_version=getattr(eva, "simulation_version", ""),
        metrics_version=getattr(eva, "metrics_version", ""),
        git_commit_sha=git_info.get("git_commit_sha"),
        git_branch=git_info.get("git_branch"),
        git_dirty=git_info.get("git_dirty"),
        git_diff_hash=git_info.get("git_diff_hash"),
        dataset=dataset_info,
        agent_config=agent_config_info,
        judge_prompts=judge_prompts,
        tool_module=tool_module_info,
        scenario_db=scenario_db_info,
        metrics_computed=sorted(metric_names),
        python_version=sys.version,
        platform=platform.platform(),
    )
