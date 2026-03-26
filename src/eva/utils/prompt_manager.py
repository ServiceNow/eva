"""Prompt management system for Voice Agent Benchmark.

This module provides a centralized way to manage prompts loaded from YAML files.
Prompts can contain variables in {variable} format that are dynamically replaced
using Python's built-in str.format() method.

Prompts are split across multiple files in configs/prompts/:
  - simulation.yaml: Agent and user simulator prompts (used during benchmark runs)
  - judge.yaml: Judge/evaluation prompts (used during metrics computation)
"""

from pathlib import Path
from typing import Any, Optional

import yaml

from eva.utils.logging import get_logger

logger = get_logger(__name__)


class PromptManager:
    """Manages prompts loaded from YAML files with dynamic variable substitution.

    Loads all .yaml files from a prompts directory and merges them into a single
    namespace.

    Variables in prompts should be written as {variable_name} and will be
    replaced with values provided when retrieving the prompt using str.format().

    Example:
        >>> pm = PromptManager()  # loads from configs/prompts/
        >>> prompt = pm.get_prompt(
        ...     "agent.system_prompt",
        ...     agent_personality="Friendly assistant",
        ... )
    """

    def __init__(self, prompts_path: Optional[Path | str] = None):
        """Initialize the prompt manager.

        Args:
            prompts_path: Path to the prompts directory.
                         If None, uses default location (repo root/configs/prompts/).
        """
        if prompts_path is None:
            repo_root = Path(__file__).parent.parent.parent.parent
            prompts_path = repo_root / "configs" / "prompts"

        self.prompts_path = Path(prompts_path)
        self.prompts: dict[str, Any] = {}
        self.loaded_files: list[Path] = []
        self._load_prompts()

    def _load_prompts(self) -> None:
        """Load prompts from YAML files in the prompts directory."""
        if self.prompts_path.is_dir():
            self._load_from_directory(self.prompts_path)
        else:
            logger.warning(f"Prompts directory not found: {self.prompts_path}")
            self.prompts = {}

    def _load_single_file(self, file_path: Path) -> None:
        """Load prompts from a single YAML file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            self.prompts.update(data)
            self.loaded_files.append(file_path)
            logger.info(f"Loaded prompts from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load prompts from {file_path}: {e}")

    def _load_from_directory(self, directory: Path) -> None:
        """Load and merge all YAML files in a directory."""
        yaml_files = sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml"))
        if not yaml_files:
            logger.warning(f"No YAML files found in {directory}")
            return

        for yaml_file in yaml_files:
            self._load_single_file(yaml_file)

    def get_prompt(self, path: str, **variables) -> str:
        """Get a prompt by its path and substitute variables.

        Args:
            path: Dot-separated path to the prompt (e.g., "orchestrator.system_prompt")
            **variables: Variable values to substitute in the prompt

        Returns:
            The prompt with variables substituted

        Raises:
            KeyError: If the prompt path is not found
            ValueError: If the prompt is not a string
        """
        # Navigate to the prompt using the dot-separated path
        parts = path.split(".")
        value = self.prompts

        for part in parts:
            if not isinstance(value, dict):
                raise KeyError(f"Invalid prompt path: {path} (stopped at {part})")
            if part not in value:
                raise KeyError(f"Prompt not found: {path} (missing key: {part})")
            value = value[part]

        if not isinstance(value, str):
            raise ValueError(f"Prompt at {path} is not a string: {type(value)}")

        # Substitute variables using str.format()
        # Auto-inject global variables from the _shared section (prompt-level vars take precedence)
        shared = self.prompts.get("_shared", {})
        formatted_vars = {
            **{k: v for k, v in shared.items() if isinstance(v, str)},
            **{k: (v if v is not None else "") for k, v in variables.items()},
        }

        try:
            return value.format(**formatted_vars)
        except KeyError as e:
            raise KeyError(
                f"Missing variable {e} for prompt '{path}'. Available variables: {sorted(formatted_vars.keys())}"
            ) from e


# Global singleton instance
_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """Get the global PromptManager instance.

    Returns:
        The global PromptManager singleton
    """
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager
