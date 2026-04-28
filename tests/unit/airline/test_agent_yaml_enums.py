"""Drift test: YAML tool `enum:` lists must match the Pydantic StrEnums.

For every param in `configs/agents/airline_agent.yaml` whose Pydantic field is
backed by a StrEnum, the YAML must declare an `enum:` with the same values. And
vice versa: any YAML `enum:` must correspond to a real StrEnum on the Pydantic
side. This catches the case where someone adds a value to a StrEnum but forgets
to update the YAML (or removes one only from Pydantic).
"""

import types
import typing
from enum import StrEnum
from pathlib import Path

import pytest
import yaml

from eva.assistant.tools import airline_params

AGENT_YAML = Path(__file__).parents[3] / "configs" / "agents" / "airline_agent.yaml"


def _tool_name_to_params_class(tool_name: str) -> type | None:
    """Convert 'get_reservation' → GetReservationParams class on airline_params."""
    pascal = "".join(part.capitalize() for part in tool_name.split("_"))
    return getattr(airline_params, pascal + "Params", None)


def _extract_str_enum(annotation: object) -> type[StrEnum] | None:
    """Return the StrEnum type from an annotation like `FareClass` or `FareClass | None`."""
    if isinstance(annotation, type) and issubclass(annotation, StrEnum):
        return annotation
    # Unwrap Union / Optional / PEP 604 `X | None`
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        for arg in typing.get_args(annotation):
            enum_cls = _extract_str_enum(arg)
            if enum_cls is not None:
                return enum_cls
    return None


@pytest.fixture(scope="module")
def agent_yaml() -> dict:
    with open(AGENT_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _iter_tool_params(agent_yaml: dict):
    """Yield (tool_name, param_dict) for every param (required + optional) in the YAML."""
    for tool in agent_yaml.get("tools", []):
        tool_name = tool["name"]
        for param in tool.get("required_parameters", []) + tool.get("optional_parameters", []):
            if isinstance(param, dict):
                yield tool_name, param


def test_yaml_enum_values_match_pydantic_str_enums(agent_yaml):
    """Every YAML `enum:` must match the values of the underlying Pydantic StrEnum."""
    mismatches = []
    for tool_name, param in _iter_tool_params(agent_yaml):
        yaml_enum = param.get("enum")
        if yaml_enum is None:
            continue

        params_cls = _tool_name_to_params_class(tool_name)
        assert params_cls is not None, f"No *Params class found for tool '{tool_name}'"

        field = params_cls.model_fields.get(param["name"])
        assert field is not None, f"Param '{param['name']}' not found on {params_cls.__name__}"

        enum_cls = _extract_str_enum(field.annotation)
        if enum_cls is None:
            mismatches.append(
                f"{tool_name}.{param['name']}: YAML declares enum but Pydantic field is not a StrEnum "
                f"(annotation: {field.annotation})"
            )
            continue

        expected = {v.value for v in enum_cls}
        actual = set(yaml_enum)
        if expected != actual:
            mismatches.append(
                f"{tool_name}.{param['name']}: YAML={sorted(actual)} vs {enum_cls.__name__}={sorted(expected)}"
            )

    assert not mismatches, "YAML enum drift from Pydantic StrEnums:\n  " + "\n  ".join(mismatches)


def test_pydantic_str_enum_fields_have_yaml_enum(agent_yaml):
    """Every Pydantic field backed by a StrEnum must have a corresponding `enum:` in the YAML."""
    missing = []
    for tool_name, param in _iter_tool_params(agent_yaml):
        params_cls = _tool_name_to_params_class(tool_name)
        if params_cls is None:
            continue
        field = params_cls.model_fields.get(param["name"])
        if field is None:
            continue
        enum_cls = _extract_str_enum(field.annotation)
        if enum_cls is not None and param.get("enum") is None:
            missing.append(
                f"{tool_name}.{param['name']}: Pydantic uses {enum_cls.__name__} but YAML has no `enum:` list"
            )

    assert not missing, "Pydantic StrEnums not reflected in YAML:\n  " + "\n  ".join(missing)
