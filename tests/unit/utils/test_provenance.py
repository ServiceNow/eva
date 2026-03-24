"""Tests for provenance utilities."""

from unittest.mock import patch

from eva.utils.provenance import (
    _get_git_info,
    _run_git_command,
    resolve_tool_module_file,
)


class TestRunGitCommand:
    def test_successful_command(self):
        result = _run_git_command(["rev-parse", "HEAD"])
        assert result is not None
        assert len(result) == 40  # SHA-1 hex

    def test_failed_command_returns_none(self):
        result = _run_git_command(["rev-parse", "nonexistent-ref-abc123"])
        assert result is None

    @patch("eva.utils.provenance.subprocess.run")
    def test_timeout_returns_none(self, mock_run):
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=10)
        result = _run_git_command(["status"])
        assert result is None

    @patch("eva.utils.provenance.subprocess.run")
    def test_file_not_found_returns_none(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        result = _run_git_command(["status"])
        assert result is None


class TestGetGitInfo:
    def test_returns_expected_keys(self):
        info = _get_git_info()
        assert "git_commit_sha" in info
        assert "git_branch" in info
        assert "git_dirty" in info
        assert "git_diff_hash" in info

    def test_commit_sha_is_valid(self):
        info = _get_git_info()
        assert info["git_commit_sha"] is not None
        assert len(info["git_commit_sha"]) == 40

    @patch("eva.utils.provenance._run_git_command")
    def test_no_git_repo(self, mock_git):
        mock_git.return_value = None
        info = _get_git_info()
        assert info["git_commit_sha"] is None
        assert info["git_branch"] is None


class TestResolveToolModuleFile:
    def test_none_input(self):
        assert resolve_tool_module_file(None) is None

    def test_empty_string(self):
        assert resolve_tool_module_file("") is None

    def test_valid_module(self):
        result = resolve_tool_module_file("eva.utils.provenance")
        assert result is not None
        assert result.name == "provenance.py"

    def test_nonexistent_module(self):
        result = resolve_tool_module_file("nonexistent_module_xyz_123")
        assert result is None
