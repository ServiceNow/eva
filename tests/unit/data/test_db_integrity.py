"""Tests for culture.py placeholder resolution and end-to-end DB integrity.

The DB integrity test (``test_db_integrity_replay``) replays each record's
expected tool-call trace against a fresh, language-resolved scenario DB and
verifies the final DB hash matches the language-resolved expected_scenario_db.

This catches data corruption from any of: bad placeholder replacement, broken
alias index, tool logic regressions, or trace/DB drift — across every language.
"""

import copy
import json
from pathlib import Path

import pytest

from eva.assistant.tools.tool_executor import ToolExecutor
from eva.utils.culture import (
    _inject_aliases,
    _load_aliases_index,
    _names_for,
    _phone_for,
    _replace_in,
    _resolve_locations,
    resolve_scenario_db,
    resolve_user_config,
    resolve_user_goal,
)
from eva.utils.hash_utils import compute_db_diff, get_dict_hash

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

OVERRIDES = {
    "fr": {"first_name": "Éric", "last_name": "Dupont", "phone": "+33 6 12 34 56 78"},
    "en": {"first_name": "John", "last_name": "Smith", "phone": "+1 415-555-0100"},
}

ROMANIZED = {
    "fr": {"first_name": "Eric", "last_name": "Dupont"},
}

STARTING_UTTERANCES = {
    "fr": "Bonjour, je m'appelle <FIRST_NAME> <LAST_NAME>.",
    "en": "Hi, my name is <FIRST_NAME> <LAST_NAME>.",
}


# ---------------------------------------------------------------------------
# _replace_in
# ---------------------------------------------------------------------------


def test_replace_in_string():
    result = _replace_in("<FIRST_NAME> <LAST_NAME>", "Éric", "Dupont", "Eric", "Dupont")
    assert result == "Éric Dupont"


def test_replace_in_romanized_is_lowercased():
    result = _replace_in(
        "<FIRST_NAME_ROMANIZED>.<LAST_NAME_ROMANIZED>@example.com",
        "Éric",
        "Dupont",
        "Eric",
        "Dupont",
    )
    assert result == "eric.dupont@example.com"


def test_replace_in_phone():
    result = _replace_in("call <PHONE>", "A", "B", "A", "B", phone="+1 555-0100")
    assert result == "call +1 555-0100"


def test_replace_in_phone_skipped_when_empty():
    result = _replace_in("call <PHONE>", "A", "B", "A", "B", phone="")
    assert result == "call <PHONE>"


def test_replace_in_companion():
    result = _replace_in(
        "with <COMPANION_FIRST_NAME> (<COMPANION_FIRST_NAME_ROMANIZED>)",
        "A",
        "B",
        "A",
        "B",
        companion_first="Chloé",
        companion_first_rom="Chloe",
    )
    assert result == "with Chloé (chloe)"


def test_replace_in_nested():
    obj = {"greeting": "<FIRST_NAME>", "items": ["<LAST_NAME>", {"x": "<FIRST_NAME>"}]}
    result = _replace_in(obj, "Éric", "Dupont", "eric", "dupont")
    assert result == {"greeting": "Éric", "items": ["Dupont", {"x": "Éric"}]}


# ---------------------------------------------------------------------------
# _names_for
# ---------------------------------------------------------------------------


def test_names_for_fr_uses_romanized():
    first, last, first_rom, last_rom = _names_for(OVERRIDES, ROMANIZED, "fr")
    assert first == "Éric" and last == "Dupont"
    assert first_rom == "Eric" and last_rom == "Dupont"


def test_names_for_en_falls_back_to_same():
    first, last, first_rom, last_rom = _names_for(OVERRIDES, ROMANIZED, "en")
    assert first == "John" and first_rom == "John"


def test_names_for_missing_language_raises():
    with pytest.raises(KeyError, match="de"):
        _names_for(OVERRIDES, ROMANIZED, "de")


# ---------------------------------------------------------------------------
# _phone_for
# ---------------------------------------------------------------------------


def test_phone_for_present():
    assert _phone_for(OVERRIDES, "fr") == "+33 6 12 34 56 78"


def test_phone_for_missing_language():
    assert _phone_for(OVERRIDES, "de") == ""


def test_phone_for_none_overrides():
    assert _phone_for(None, "fr") == ""


# ---------------------------------------------------------------------------
# resolve_user_goal
# ---------------------------------------------------------------------------


def test_resolve_user_goal_injects_name_and_utterance():
    goal = {"description": "Help <FIRST_NAME> <LAST_NAME>"}
    resolved = resolve_user_goal(goal, OVERRIDES, "fr", ROMANIZED, STARTING_UTTERANCES)
    assert resolved["description"] == "Help Éric Dupont"
    assert resolved["starting_utterance"] == "Bonjour, je m'appelle Éric Dupont."


def test_resolve_user_goal_missing_utterance_raises():
    goal = {"description": "x"}
    with pytest.raises(KeyError, match="en"):
        resolve_user_goal(goal, OVERRIDES, "en", ROMANIZED, {"fr": "..."})


def test_resolve_user_goal_does_not_mutate_original():
    goal = {"description": "<FIRST_NAME>"}
    resolve_user_goal(goal, OVERRIDES, "en", ROMANIZED, STARTING_UTTERANCES)
    assert goal["description"] == "<FIRST_NAME>"


# ---------------------------------------------------------------------------
# resolve_user_config
# ---------------------------------------------------------------------------


def test_resolve_user_config():
    config = {"name": "<FIRST_NAME> <LAST_NAME>", "email": "<FIRST_NAME_ROMANIZED>@co.com"}
    resolved = resolve_user_config(config, OVERRIDES, "fr", ROMANIZED)
    assert resolved["name"] == "Éric Dupont"
    assert resolved["email"] == "eric@co.com"


# ---------------------------------------------------------------------------
# resolve_scenario_db
# ---------------------------------------------------------------------------


def test_resolve_scenario_db_basic():
    db = {"passenger": {"first_name": "<FIRST_NAME>", "last_name": "<LAST_NAME>"}}
    resolved = resolve_scenario_db(db, OVERRIDES, "fr", ROMANIZED)
    assert resolved["passenger"] == {"first_name": "Éric", "last_name": "Dupont"}


def test_resolve_scenario_db_injects_aliases(tmp_path):
    alias_file = tmp_path / "downtown.json"
    alias_file.write_text(
        json.dumps(
            {
                "name": "Downtown",
                "translatable": True,
                "base": ["downtown building"],
                "translations": {"fr": ["centre-ville", "bureau centre"]},
            }
        )
    )
    db = {"location": {"name": "Downtown"}}
    resolved = resolve_scenario_db(db, OVERRIDES, "fr", ROMANIZED, aliases_dir=tmp_path)
    assert resolved["location"]["name_aliases"] == ["downtown building", "centre-ville", "bureau centre"]


def test_resolve_scenario_db_no_aliases_dir():
    db = {"location": {"name": "Downtown"}}
    resolved = resolve_scenario_db(db, OVERRIDES, "fr", ROMANIZED, aliases_dir=None)
    assert "name_aliases" not in resolved["location"]


# ---------------------------------------------------------------------------
# _load_aliases_index / _inject_aliases
# ---------------------------------------------------------------------------


def test_load_aliases_index_missing_dir():
    index = _load_aliases_index("/nonexistent/path/aliases")
    assert index == {}


def test_load_aliases_index_reads_files(tmp_path):
    (tmp_path / "garage_a.json").write_text(
        json.dumps(
            {
                "name": "Garage A",
                "translatable": True,
                "base": ["a garage"],
                "translations": {"fr": ["parking a"]},
            }
        )
    )
    index = _load_aliases_index(str(tmp_path))
    assert "Garage A" in index
    assert index["Garage A"]["base"] == ["a garage"]


def test_inject_aliases_deduplication(tmp_path):
    index = {
        "North Surface Lot": {
            "base": ["north lot", "north parking"],
            "translations": {"fr": ["north lot", "lot nord"]},
        }
    }
    obj = {"name": "North Surface Lot"}
    _inject_aliases(obj, index, "fr")
    aliases = obj["name_aliases"]
    assert aliases == ["north lot", "north parking", "lot nord"]
    assert len(aliases) == len(set(aliases))


def test_inject_aliases_unknown_name_left_alone():
    obj = {"name": "Unknown Place"}
    _inject_aliases(obj, {}, "fr")
    assert "name_aliases" not in obj


def test_inject_aliases_language_without_translation(tmp_path):
    index = {
        "Figma": {
            "base": ["figma design"],
            "translations": {},
        }
    }
    obj = {"name": "Figma"}
    _inject_aliases(obj, index, "de")
    assert obj["name_aliases"] == ["figma design"]


# ---------------------------------------------------------------------------
# End-to-end DB integrity replay
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]

REPLAY_DOMAINS = ["itsm", "medical_hr", "airline"]
RESOLVE_DOMAINS = ["itsm", "medical_hr", "airline"]


def _domain_paths(domain: str) -> dict:
    return {
        "dataset": REPO_ROOT / "data" / f"{domain}_dataset.json",
        "scenarios": REPO_ROOT / "data" / f"{domain}_scenarios",
        "aliases": REPO_ROOT / "data" / f"{domain}_aliases",
        "agent_config": REPO_ROOT / "configs" / "agents" / f"{domain}_agent.yaml",
        "tool_module": f"eva.assistant.tools.{domain}_tools",
    }


def _load_dataset(domain: str) -> list[dict]:
    return json.loads(_domain_paths(domain)["dataset"].read_text())


def _load_scenario(domain: str, record_id) -> dict:
    return json.loads((_domain_paths(domain)["scenarios"] / f"{record_id}.json").read_text())


def _resolve_params(params, culture_overrides, language, romanized_culture_overrides, aliases_index=None):
    """Resolve name and location placeholders in tool call params."""
    resolved = resolve_user_config(params, culture_overrides, language, romanized_culture_overrides)
    if aliases_index:
        resolved = _resolve_locations(resolved, aliases_index, language)
    return resolved


def _make_executor(domain: str, sample_record_id) -> ToolExecutor:
    paths = _domain_paths(domain)
    return ToolExecutor(
        tool_config_path=str(paths["agent_config"]),
        scenario_db_path=str(paths["scenarios"] / f"{sample_record_id}.json"),
        tool_module_path=paths["tool_module"],
        current_date_time="2026-01-01 00:00 EST",
    )


def _apply_subs(obj, subs: dict[str, str]):
    """Substitute scalar string values in ``obj`` according to ``subs`` (trace ID → actual ID)."""
    if not subs:
        return obj
    if isinstance(obj, str):
        return subs.get(obj, obj)
    if isinstance(obj, list):
        return [_apply_subs(x, subs) for x in obj]
    if isinstance(obj, dict):
        return {k: _apply_subs(v, subs) for k, v in obj.items()}
    return obj


def _collect_id_subs(trace_response: dict, actual_response: dict, subs: dict[str, str]) -> None:
    """Compare a trace ``tool_response`` with the live tool result and record any ID drift.

    Some tool functions generate content-hashed IDs (e.g. ``REQ-FAC-<hash>``) that the
    hand-authored trace can't predict. We chain the actual IDs forward by mapping the
    trace's placeholder ID → the live ID, then substituting in subsequent tool params.
    """
    if not isinstance(trace_response, dict) or not isinstance(actual_response, dict):
        return
    for key, trace_val in trace_response.items():
        actual_val = actual_response.get(key)
        if isinstance(trace_val, str) and isinstance(actual_val, str) and trace_val != actual_val:
            subs[trace_val] = actual_val
        elif isinstance(trace_val, dict) and isinstance(actual_val, dict):
            _collect_id_subs(trace_val, actual_val, subs)


async def _replay_record(executor: ToolExecutor, record: dict, language: str, paths: dict) -> dict:
    """Replay all tool_calls from expected_trace into a fresh resolved DB.

    Returns dict with ``actual_db`` and ``expected_db`` (both resolved for ``language``).
    Transient IDs (request_id, case_id, etc.) generated by tool calls are chained
    forward so trace-authored placeholders don't break downstream tool params.
    """
    record_id = record["id"]
    co = record["culture_overrides"]
    rco = record.get("romanized_culture_overrides")

    initial_raw = json.loads((paths["scenarios"] / f"{record_id}.json").read_text())
    initial_db = resolve_scenario_db(initial_raw, co, language, rco, aliases_dir=paths["aliases"])
    expected_db = resolve_scenario_db(
        record["ground_truth"]["expected_scenario_db"], co, language, rco, aliases_dir=paths["aliases"]
    )

    executor.db = copy.deepcopy(initial_db)
    executor.db["_current_date"] = record["current_date_time"].split(" ")[0]
    executor._tool_call_counts = {}

    aliases_index = _load_aliases_index(str(paths["aliases"])) if paths["aliases"].exists() else {}

    trace = record["ground_truth"]["expected_trace"]["trace"]
    subs: dict[str, str] = {}
    # Pair tool_calls with their immediate following tool_response (standard trace layout).
    i = 0
    while i < len(trace):
        event = trace[i]
        if event.get("event_type") != "tool_call":
            i += 1
            continue
        tool_name = event["tool_name"]
        params = _resolve_params(event.get("params", {}), co, language, rco, aliases_index)
        params = _apply_subs(params, subs)
        actual_response = await executor.execute(tool_name, params)
        # Build trace_response with the same culture resolution + sub chain so the
        # comparison only flags genuinely different IDs (not placeholder mismatches).
        if i + 1 < len(trace) and trace[i + 1].get("event_type") == "tool_response":
            trace_response = _apply_subs(
                _resolve_params(trace[i + 1].get("response", {}), co, language, rco, aliases_index), subs
            )
            _collect_id_subs(trace_response, actual_response, subs)
        i += 1

    return {"actual_db": executor.db, "expected_db": expected_db}


def _short_json(v) -> str:
    return json.dumps(v, default=str, ensure_ascii=False)[:300]


def _format_record_diff(rdiff: dict, indent: str) -> list[str]:
    """Render the nested record/field diff structure produced by _compute_record_diff."""
    lines: list[str] = []
    if "type" in rdiff and "expected" in rdiff and "actual" in rdiff:
        lines.append(f"{indent}expected={_short_json(rdiff['expected'])}")
        lines.append(f"{indent}actual=  {_short_json(rdiff['actual'])}")
        return lines
    for field in rdiff.get("fields_added", []) or []:
        lines.append(f"{indent}+ field {field!r}")
    for field in rdiff.get("fields_removed", []) or []:
        lines.append(f"{indent}- field {field!r}")
    for field, nested in (rdiff.get("fields_modified") or {}).items():
        if isinstance(nested, dict) and "type" in nested and "expected" in nested and "actual" in nested:
            lines.append(
                f"{indent}~ {field}: expected={_short_json(nested['expected'])}  actual={_short_json(nested['actual'])}"
            )
        else:
            lines.append(f"{indent}~ {field}:")
            lines.extend(_format_record_diff(nested, indent + "    "))
    return lines


def _format_diff(diff: dict, indent: str = "    ") -> str:
    """Render a compute_db_diff result as a readable, line-oriented block."""
    lines: list[str] = []
    if diff.get("tables_added"):
        lines.append(f"{indent}tables only in actual: {diff['tables_added']}")
    if diff.get("tables_removed"):
        lines.append(f"{indent}tables only in expected: {diff['tables_removed']}")
    for table, tdiff in diff.get("tables_modified", {}).items():
        lines.append(f"{indent}table {table!r}:")
        if isinstance(tdiff, dict) and tdiff.get("type") == "non_dict_table":
            lines.append(f"{indent}  expected: {_short_json(tdiff.get('expected'))}")
            lines.append(f"{indent}  actual:   {_short_json(tdiff.get('actual'))}")
            continue
        for rid in tdiff.get("records_added") or []:
            lines.append(f"{indent}  + record {rid!r} (only in actual)")
        for rid in tdiff.get("records_removed") or []:
            lines.append(f"{indent}  - record {rid!r} (only in expected)")
        for rid, rdiff in (tdiff.get("records_modified") or {}).items():
            lines.append(f"{indent}  ~ record {rid!r}:")
            lines.extend(_format_record_diff(rdiff, indent + "      "))
    return "\n".join(lines)


@pytest.mark.asyncio
@pytest.mark.parametrize("domain", REPLAY_DOMAINS)
async def test_db_integrity_replay_multilingual_parity(domain):
    """Per-record replay outcome must be identical across every language.

    For each record, replay the expected trace once per language. The pass/fail
    outcome must match across all languages — if ``en`` passes but ``fr`` fails
    (or vice versa), the culture/alias machinery has broken the data flow for
    that language. Pre-existing data drift that fails across every language is
    surfaced by ``test_db_integrity_replay_english_baseline`` instead.
    """
    paths = _domain_paths(domain)
    dataset = _load_dataset(domain)
    executor = _make_executor(domain, dataset[0]["id"])

    # record_id -> {language: (passed, diff_or_none)}
    outcomes: dict[str, dict[str, tuple[bool, dict | None]]] = {}
    for record in dataset:
        languages = sorted(record.get("culture_overrides", {}).keys())
        if len(languages) < 2:
            continue
        outcomes[record["id"]] = {}
        for language in languages:
            result = await _replay_record(executor, record, language, paths)
            passed = get_dict_hash(result["actual_db"]) == get_dict_hash(result["expected_db"])
            diff = None if passed else compute_db_diff(expected_db=result["expected_db"], actual_db=result["actual_db"])
            outcomes[record["id"]][language] = (passed, diff)

    blocks: list[str] = []
    for rid, by_lang in outcomes.items():
        passes = {lang: passed for lang, (passed, _) in by_lang.items()}
        if len(set(passes.values())) <= 1:
            continue
        lines = [f"\n[{domain}/{rid}] language-divergent outcome: {passes}"]
        for lang, (passed, diff) in by_lang.items():
            if not passed:
                lines.append(f"  {lang} diff:")
                lines.append(_format_diff(diff, indent="    "))
        blocks.append("\n".join(lines))

    if blocks:
        pytest.fail(
            f"\n{len(blocks)} records have language-divergent replay outcomes in {domain} "
            "(culture/alias resolution broke a non-EN language):\n" + "\n\n".join(blocks)
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("domain", REPLAY_DOMAINS)
async def test_db_integrity_replay_english_baseline(domain):
    """English replay of every record's expected trace must reproduce expected_scenario_db.

    This is the strict data-integrity check: replay the trace with no culture
    machinery and assert the final DB matches expected_scenario_db exactly.
    Failures here mean the dataset has drifted from the tool implementation —
    e.g. the trace was authored before a tool started requiring OTP auth — and
    the record needs to be regenerated.
    """
    paths = _domain_paths(domain)
    dataset = _load_dataset(domain)
    executor = _make_executor(domain, dataset[0]["id"])

    blocks: list[str] = []
    for record in dataset:
        if "en" not in record.get("culture_overrides", {}):
            continue
        result = await _replay_record(executor, record, "en", paths)
        if get_dict_hash(result["actual_db"]) == get_dict_hash(result["expected_db"]):
            continue
        diff = compute_db_diff(expected_db=result["expected_db"], actual_db=result["actual_db"])
        blocks.append(f"\n[{domain}/{record['id']}]\n" + _format_diff(diff, indent="    "))

    if blocks:
        pytest.fail(
            f"\n{len(blocks)} records have trace/DB drift in {domain} "
            "(replaying the expected trace does not reproduce expected_scenario_db):\n" + "\n".join(blocks)
        )


@pytest.mark.parametrize("domain", RESOLVE_DOMAINS)
def test_resolve_scenario_db_no_unresolved_placeholders(domain):
    """Every record's initial + expected DB must resolve cleanly with no leftover placeholders.

    Catches missing culture entries, missing romanized overrides, and stale
    placeholder strings without needing a full replay (so it covers airline too).
    """
    paths = _domain_paths(domain)
    dataset = _load_dataset(domain)
    placeholders = ("<FIRST_NAME>", "<LAST_NAME>", "<FIRST_NAME_ROMANIZED>", "<LAST_NAME_ROMANIZED>", "<PHONE>")

    failures = []
    for record in dataset:
        co = record["culture_overrides"]
        rco = record.get("romanized_culture_overrides")
        initial_raw = json.loads((paths["scenarios"] / f"{record['id']}.json").read_text())
        expected_raw = record["ground_truth"]["expected_scenario_db"]

        for language in sorted(co.keys()):
            for label, raw in (("initial", initial_raw), ("expected", expected_raw)):
                resolved = resolve_scenario_db(raw, co, language, rco, aliases_dir=paths["aliases"])
                serialized = json.dumps(resolved, ensure_ascii=False)
                leftover = [p for p in placeholders if p in serialized]
                if leftover:
                    failures.append(f"  {domain}/{record['id']}/{language}/{label}: leftover {leftover}")

    if failures:
        pytest.fail(f"\n{len(failures)} unresolved placeholders:\n" + "\n".join(failures[:20]))
