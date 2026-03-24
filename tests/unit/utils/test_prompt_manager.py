"""Tests for PromptManager."""

import pytest

from eva.utils.prompt_manager import PromptManager

JUDGE_YAML = """\
judge:
  task_completion:
    user_prompt: |
      Evaluate task completion for: {user_goal}
      Transcript: {transcript}
  faithfulness:
    user_prompt: "Rate faithfulness: {transcript}"
"""

SIMULATION_YAML = """\
agent:
  system_prompt: "You are {agent_name}. {agent_personality}"
"""

SHARED_VARS_YAML = """\
_shared:
  company: AcmeCorp
prompts:
  greeting: "Welcome to {company}!"
"""

INVALID_YAML = ": : : not valid yaml [[["

SIMPLE_KV_YAML = "key: value\n"


@pytest.fixture
def prompts_dir(tmp_path):
    """Create a temp prompts directory with YAML files."""
    judge = tmp_path / "judge.yaml"
    judge.write_text(JUDGE_YAML)

    simulation = tmp_path / "simulation.yaml"
    simulation.write_text(SIMULATION_YAML)
    return tmp_path


@pytest.fixture
def pm(prompts_dir):
    return PromptManager(prompts_path=prompts_dir)


class TestPromptManager:
    def test_loads_all_yaml_files(self, pm, prompts_dir):
        assert len(pm.loaded_files) == 2

    def test_get_prompt_simple(self, pm):
        result = pm.get_prompt(
            "judge.task_completion.user_prompt",
            user_goal="book a flight",
            transcript="User: hi\nBot: hello",
        )
        assert "book a flight" in result
        assert "User: hi" in result

    def test_get_prompt_across_files(self, pm):
        result = pm.get_prompt(
            "agent.system_prompt",
            agent_name="FlightBot",
            agent_personality="Helpful",
        )
        assert "FlightBot" in result
        assert "Helpful" in result

    def test_get_prompt_missing_key_raises(self, pm):
        with pytest.raises(KeyError, match="missing key"):
            pm.get_prompt("judge.nonexistent.foo")

    def test_get_prompt_invalid_path_raises(self, pm):
        with pytest.raises(KeyError, match="Invalid prompt path"):
            pm.get_prompt("judge.task_completion.user_prompt.extra_level")

    def test_get_prompt_non_string_raises(self, pm):
        with pytest.raises(ValueError, match="not a string"):
            pm.get_prompt("judge.task_completion")  # This is a dict, not a string

    def test_missing_variable_raises_key_error(self, pm):
        # user_prompt expects {user_goal} and {transcript}; only provide one
        with pytest.raises(KeyError, match="Missing variable"):
            pm.get_prompt(
                "judge.task_completion.user_prompt",
                user_goal="book a flight",
            )

    def test_none_variable_replaced_with_empty_string(self, pm):
        result = pm.get_prompt(
            "judge.faithfulness.user_prompt",
            transcript=None,
        )
        assert "Rate faithfulness: " in result

    def test_shared_variables(self, tmp_path):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(SHARED_VARS_YAML)
        pm = PromptManager(prompts_path=tmp_path)
        result = pm.get_prompt("prompts.greeting")
        assert result == "Welcome to AcmeCorp!"

    def test_explicit_var_overrides_shared(self, tmp_path):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(SHARED_VARS_YAML)
        pm = PromptManager(prompts_path=tmp_path)
        result = pm.get_prompt("prompts.greeting", company="BetaCorp")
        assert result == "Welcome to BetaCorp!"


class TestPromptManagerEdgeCases:
    def test_empty_directory(self, tmp_path):
        pm = PromptManager(prompts_path=tmp_path)
        assert pm.prompts == {}

    def test_nonexistent_directory(self, tmp_path):
        pm = PromptManager(prompts_path=tmp_path / "nope")
        assert pm.prompts == {}

    def test_invalid_yaml_is_skipped(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(INVALID_YAML)
        good = tmp_path / "good.yaml"
        good.write_text(SIMPLE_KV_YAML)

        pm = PromptManager(prompts_path=tmp_path)
        # Good file should still load
        assert pm.prompts.get("key") == "value"

    def test_empty_yaml_file(self, tmp_path):
        f = tmp_path / "empty.yaml"
        f.write_text("")
        pm = PromptManager(prompts_path=tmp_path)
        assert pm.prompts == {}

    def test_yml_extension_supported(self, tmp_path):
        f = tmp_path / "test.yml"
        f.write_text("mykey: myval\n")
        pm = PromptManager(prompts_path=tmp_path)
        assert pm.prompts.get("mykey") == "myval"
