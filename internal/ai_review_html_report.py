"""Generate EVA Error Analysis Highlights — a shareable HTML report with audio examples.

Produces a ZIP containing report.html and an audio/ folder with conversation recordings.
Each issue includes trace snippets and inline audio players.
"""

import html
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ai_review_pipeline import (
    ALL_ANALYSIS_METRICS,
    EVA_A_METRICS,
    EVA_X_METRICS,
    display_name,
    format_metric_name,
    get_display_combo_label,
    get_record_directories,
    get_valid_trial_dirs,
    normalize_entity_type,
)

# Metrics ordered by proximity to TTS/STT
VOICE_METRIC_ORDER = [
    "agent_speech_fidelity",
    "transcription_accuracy_key_entities",
    "turn_taking",
    "faithfulness",
    "task_completion",
    "conversation_progression",
    "conciseness",
]

VOICE_METRIC_LABELS = {
    "agent_speech_fidelity": "Agent Speech Fidelity (TTS)",
    "transcription_accuracy_key_entities": "Transcription Accuracy (STT)",
    "turn_taking": "Turn Taking",
    "faithfulness": "Faithfulness",
    "task_completion": "Task Completion",
    "conversation_progression": "Conversation Progression",
    "conciseness": "Conciseness",
}

METRIC_TAG_CLASS = {
    "agent_speech_fidelity": "tts",
    "transcription_accuracy_key_entities": "stt",
}

# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root {{
    --bg: #fafbfc; --fg: #1a1a2e; --muted: #5f6b7a;
    --border: #e2e8f0; --card-bg: #ffffff;
    --accent: #2563eb; --accent-light: #dbeafe;
    --danger: #dc2626; --danger-light: #fee2e2;
    --warn: #ea580c; --warn-light: #fff7ed;
    --success: #16a34a; --success-light: #dcfce7;
    --info: #0284c7; --info-light: #e0f2fe;
    --radius: 10px;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 4px 6px rgba(0,0,0,0.05), 0 2px 4px rgba(0,0,0,0.04);
  }}

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: var(--fg); background: var(--bg); line-height: 1.65;
    max-width: 1080px; margin: 0 auto; padding: 3rem 2rem;
    -webkit-font-smoothing: antialiased;
  }}

  /* Header */
  .report-header {{
    text-align: center; margin-bottom: 2.5rem; padding-bottom: 2rem;
    border-bottom: 2px solid var(--border);
  }}
  .report-header h1 {{
    font-size: 2rem; font-weight: 800; letter-spacing: -0.03em;
    background: linear-gradient(135deg, var(--accent), #7c3aed);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.75rem;
  }}
  .report-header .subtitle {{
    color: var(--muted); font-size: 0.95rem; max-width: 680px; margin: 0 auto 1.5rem;
    line-height: 1.7;
  }}

  /* Pipeline cards in header */
  .pipeline-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 0.75rem; margin-top: 1.25rem; text-align: left;
  }}
  .pipeline-card {{
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 0.85rem 1rem;
    box-shadow: var(--shadow-sm); transition: box-shadow 0.2s;
  }}
  .pipeline-card:hover {{ box-shadow: var(--shadow-md); }}
  .pipeline-card .pipe-num {{
    font-weight: 700; color: var(--accent); font-size: 0.8rem;
    text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.3rem;
  }}
  .pipeline-card .component {{
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 0.85rem; color: var(--fg);
  }}
  .pipeline-card .comp-label {{
    color: var(--muted); font-size: 0.75rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.04em;
  }}

  /* Sections */
  .section {{
    margin: 3rem 0 0;
  }}
  .section-title {{
    font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em;
    margin-bottom: 0.5rem; color: var(--fg);
    display: flex; align-items: center; gap: 0.5rem;
  }}
  .section-title .icon {{ font-size: 1.2rem; }}
  .section-desc {{
    color: var(--muted); font-size: 0.9rem; margin-bottom: 1.5rem; line-height: 1.6;
  }}

  /* Pipeline collapsible sections */
  details.pipeline-details {{
    margin: 1.5rem 0 0; border: 1px solid var(--border);
    border-radius: var(--radius); overflow: hidden;
    box-shadow: var(--shadow-sm);
  }}
  details.pipeline-details[open] {{ box-shadow: var(--shadow-md); }}
  details.pipeline-details > summary {{
    font-size: 1.1rem; font-weight: 700; padding: 0.75rem 1.25rem;
    background: var(--accent-light); color: var(--accent);
    cursor: pointer; list-style: none; display: flex;
    align-items: center; justify-content: space-between;
    user-select: none;
  }}
  details.pipeline-details > summary::-webkit-details-marker {{ display: none; }}
  details.pipeline-details > summary::after {{
    content: ''; display: inline-block; width: 10px; height: 10px;
    border-right: 2px solid var(--accent); border-bottom: 2px solid var(--accent);
    transform: rotate(45deg); transition: transform 0.2s; margin-top: -3px;
  }}
  details.pipeline-details[open] > summary::after {{ transform: rotate(225deg); margin-top: 2px; }}
  details.pipeline-details > .pipeline-body {{ padding: 0.75rem 1.25rem 1.25rem; }}

  /* Nested details for non-voice issues */
  details.other-issues {{
    margin: 1rem 0 0; border: 1px solid var(--border);
    border-radius: var(--radius); overflow: hidden;
  }}
  details.other-issues > summary {{
    font-size: 0.92rem; font-weight: 600; padding: 0.6rem 1rem;
    background: var(--bg); color: var(--muted); cursor: pointer;
    list-style: none; display: flex; align-items: center;
    justify-content: space-between; user-select: none;
  }}
  details.other-issues > summary::-webkit-details-marker {{ display: none; }}
  details.other-issues > summary::after {{
    content: ''; display: inline-block; width: 8px; height: 8px;
    border-right: 2px solid var(--muted); border-bottom: 2px solid var(--muted);
    transform: rotate(45deg); transition: transform 0.2s; margin-top: -2px;
  }}
  details.other-issues[open] > summary::after {{ transform: rotate(225deg); margin-top: 2px; }}
  details.other-issues > .other-body {{ padding: 0.5rem 1rem 1rem; }}

  /* Metric sub-headings */
  .metric-heading {{
    font-size: 0.95rem; font-weight: 600; margin: 1.25rem 0 0.5rem;
    display: flex; align-items: center; gap: 0.5rem;
  }}

  /* Tags */
  .tag {{
    display: inline-flex; align-items: center; padding: 2px 10px;
    border-radius: 20px; font-size: 0.7rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.05em;
  }}
  .tag.tts {{ background: var(--warn-light); color: var(--warn); }}
  .tag.stt {{ background: var(--danger-light); color: var(--danger); }}
  .tag.other {{ background: var(--info-light); color: var(--info); }}
  .tag.severity {{ font-size: 0.65rem; padding: 1px 7px; margin-left: 0.25rem; }}
  .tag.high {{ background: var(--danger-light); color: var(--danger); }}
  .tag.cascade {{ background: var(--warn-light); color: var(--warn); }}
  .tag.common {{ background: #f3e8ff; color: #7c3aed; }}

  /* Summary bullets */
  .summary-finding {{
    padding: 0.5rem 0; font-size: 0.9rem; line-height: 1.6;
  }}
  .summary-finding strong {{ color: var(--fg); }}
  .summary-best {{
    background: var(--success-light); border: 1px solid #86efac;
    border-radius: var(--radius); padding: 0.75rem 1rem;
    margin: 0.75rem 0; font-size: 0.9rem;
  }}
  .count-badge {{
    font-family: monospace; font-size: 0.8rem; color: var(--muted);
    font-weight: 400;
  }}

  /* Issue items */
  .issue-card {{
    padding: 1rem 0; border-bottom: 1px solid var(--border);
  }}
  .issue-card:last-child {{ border-bottom: none; }}
  .issue-card h4 {{
    font-size: 0.95rem; font-weight: 700; margin-bottom: 0.3rem;
    display: flex; align-items: baseline; gap: 0.5rem;
  }}
  .issue-card h4 code {{
    font-size: 0.75rem; color: var(--muted); font-weight: 500;
    background: var(--bg); padding: 2px 6px; border-radius: 4px;
  }}
  .issue-card > p {{ font-size: 0.88rem; color: var(--muted); margin-bottom: 0.5rem; }}

  /* Evidence block */
  .evidence {{
    margin-top: 0.75rem; padding: 0.75rem; background: var(--bg);
    border-radius: 8px; border: 1px solid var(--border);
  }}
  .sample-label {{
    font-weight: 700; color: var(--accent); font-size: 0.8rem;
    text-transform: uppercase; letter-spacing: 0.03em;
  }}
  .trace {{
    background: #1e293b; color: #e2e8f0; padding: 0.75rem 1rem;
    border-radius: 8px; font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 0.8rem; line-height: 1.7; margin: 0.5rem 0;
    overflow-x: auto; white-space: pre-wrap;
  }}
  .example-text {{
    font-size: 0.88rem; color: var(--muted); margin: 0.3rem 0;
    font-style: italic;
  }}
  audio {{
    width: 100%; margin-top: 0.5rem; height: 40px; border-radius: 8px;
  }}

  /* Info boxes */
  .info-box {{
    background: var(--info-light); border: 1px solid #7dd3fc;
    border-radius: var(--radius); padding: 1rem 1.25rem;
    margin: 1.25rem 0; font-size: 0.88rem; line-height: 1.7;
  }}
  .info-box strong {{ color: var(--info); }}

  .finding-box {{
    background: var(--success-light); border: 1px solid #86efac;
    border-radius: var(--radius); padding: 1rem 1.25rem;
    margin: 1.25rem 0; font-size: 0.9rem; line-height: 1.6;
    font-style: italic;
  }}

  /* Charts */
  .chart-container {{
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 0.5rem;
    margin: 1.5rem 0; box-shadow: var(--shadow-sm);
    overflow: hidden;
  }}

  /* Tables */
  table {{
    width: 100%; border-collapse: separate; border-spacing: 0;
    margin: 1rem 0; font-size: 0.88rem;
    background: var(--card-bg); border-radius: var(--radius);
    overflow: hidden; box-shadow: var(--shadow-sm);
    border: 1px solid var(--border);
  }}
  th {{
    background: var(--bg); font-weight: 600; text-align: left;
    padding: 10px 14px; border-bottom: 2px solid var(--border);
    font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.03em;
    color: var(--muted);
  }}
  td {{ padding: 10px 14px; border-bottom: 1px solid var(--border); }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #f8fafc; }}

  /* Causal chains */
  .causal-chain {{
    border-left: 4px solid var(--warn); padding: 0.75rem 1rem;
    margin: 0.75rem 0; background: var(--card-bg);
    border-radius: 0 var(--radius) var(--radius) 0;
    box-shadow: var(--shadow-sm);
  }}
  .causal-chain strong {{ font-size: 0.92rem; }}
  .causal-chain p {{ font-size: 0.88rem; margin-top: 0.25rem; color: var(--muted); }}
  .causal-chain .meta {{
    font-size: 0.8rem; color: var(--muted); margin-top: 0.3rem;
  }}

  /* Dividers */
  .divider {{
    border: none; border-top: 1px solid var(--border); margin: 3rem 0;
  }}

  /* Footer */
  .footer {{
    color: var(--muted); font-size: 0.8rem; margin-top: 4rem;
    padding-top: 1.5rem; border-top: 1px solid var(--border);
    text-align: center;
  }}

  @media print {{
    body {{ padding: 1rem; background: white; }}
    .chart-container {{ page-break-inside: avoid; box-shadow: none; }}
    audio {{ display: none; }}
    .pipeline-card, .causal-chain {{ box-shadow: none; }}
  }}

  @media (max-width: 640px) {{
    body {{ padding: 1.5rem 1rem; }}
    .pipeline-grid {{ grid-template-columns: 1fr; }}
    .report-header h1 {{ font-size: 1.5rem; }}
  }}
</style>
</head>
<body>
{body}
<div class="footer">Generated on {timestamp} &nbsp;&middot;&nbsp; EVA Error Analysis</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------


def _plotly_div(fig: go.Figure, height: int = 450) -> str:
    fig.update_layout(
        height=height,
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Inter, -apple-system, sans-serif", "size": 12},
    )
    inner = fig.to_html(full_html=False, include_plotlyjs=False)
    return f'<div class="chart-container">{inner}</div>'


def _esc(text: str) -> str:
    return html.escape(str(text))


def _replace_raw_labels(text: str, combos: dict) -> str:
    """Replace raw combo labels in free text with display names."""
    for raw_label in combos:
        disp = get_display_combo_label(raw_label)
        if disp != raw_label:
            text = text.replace(raw_label, disp)
    return text


def _md_to_html(text: str) -> str:
    """Convert simple markdown (tables, bold, bullets) to HTML."""

    def _cell_html(cell_text: str) -> str:
        """Escape a cell then convert **bold** markers."""
        return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", _esc(cell_text))

    lines = text.strip().split("\n")
    out: list[str] = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        # Markdown table row
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            # Skip separator rows (|---|---|)
            if all(set(c) <= {"-", ":", " "} for c in cells):
                continue
            if not in_table:
                out.append('<table style="width:100%;border-collapse:collapse;margin:8px 0;font-size:0.9em;">')
                out.append("<thead><tr>")
                for c in cells:
                    out.append(
                        f'<th style="text-align:left;padding:6px 10px;border-bottom:2px solid var(--border);">'
                        f"{_cell_html(c)}</th>"
                    )
                out.append("</tr></thead><tbody>")
                in_table = True
            else:
                out.append("<tr>")
                for c in cells:
                    out.append(
                        f'<td style="padding:5px 10px;border-bottom:1px solid var(--border);">{_cell_html(c)}</td>'
                    )
                out.append("</tr>")
        else:
            if in_table:
                out.append("</tbody></table>")
                in_table = False
            if stripped:
                converted = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", _esc(stripped))
                out.append(f"<p>{converted}</p>")

    if in_table:
        out.append("</tbody></table>")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------


def _build_record_lookup(combos: dict) -> dict[str, dict[str, Path]]:
    """Build {combo_label: {record_id: data_path}} lookup."""
    lookup: dict[str, dict[str, Path]] = {}
    for label, report in combos.items():
        records: dict[str, Path] = {}
        for rd_str in report.get("run_dirs", []):
            run_dir = Path(rd_str)
            for record_dir in get_record_directories(run_dir):
                rec_id = record_dir.name
                for _trial, data_path in get_valid_trial_dirs(record_dir):
                    # Store multiple keys: plain record_id and record_id/trial
                    if rec_id not in records:
                        records[rec_id] = data_path
                    trial_key = f"{rec_id}/{_trial}" if _trial else rec_id
                    if trial_key not in records:
                        records[trial_key] = data_path
        lookup[label] = records
    return lookup


def _resolve_sample_path(records: dict[str, Path], sample_id: str) -> Path | None:
    """Resolve a sample_id to a data_path, handling record_id/trial format."""
    path = records.get(sample_id)
    if path:
        return path
    # Try stripping trial suffix: "123/trial_0" -> "123"
    base_id = sample_id.split("/")[0] if "/" in sample_id else sample_id
    return records.get(base_id)


def _load_metric_details(data_path: Path, metric_name: str) -> dict:
    """Load metric details dict from metrics.json."""
    metrics_path = data_path / "metrics.json"
    if not metrics_path.exists():
        return {}
    try:
        with open(metrics_path) as f:
            raw = json.load(f)
        metrics = raw.get("metrics", raw)
        return metrics.get(metric_name, {}).get("details", {})
    except Exception:
        return {}


def _parse_intended_turns(judge_prompt: str) -> dict[str, str]:
    """Extract intended turn text from the judge_prompt string.

    The prompt contains lines like 'Turn 0: Hello, how can I help you?'
    """
    intended: dict[str, str] = {}
    for match in re.finditer(r"^Turn (\d+): (.+)$", judge_prompt, re.MULTILINE):
        intended[match.group(1)] = match.group(2).strip()
    return intended


def _format_speech_fidelity_trace(details: dict) -> str:
    """Format agent_speech_fidelity per-turn failures with intended text.

    Only shows turns that failed (rating=0 on the 0-1 scale).
    """
    per_turn_exp = details.get("per_turn_explanations", {})
    per_turn_ratings = details.get("per_turn_ratings", {})
    per_turn_modes = details.get("per_turn_failure_modes", {})
    intended = _parse_intended_turns(details.get("judge_prompt", ""))
    lines: list[str] = []
    for tid, explanation in per_turn_exp.items():
        rating = per_turn_ratings.get(tid, per_turn_ratings.get(str(tid), "?"))
        # Skip passing turns — agent_speech_fidelity uses 0 (fail) / 1 (pass)
        if isinstance(rating, (int, float)) and rating >= 1:
            continue
        modes = per_turn_modes.get(tid, per_turn_modes.get(str(tid), []))
        mode_str = f"  [{', '.join(modes)}]" if modes else ""
        intended_text = intended.get(str(tid), "")
        header = f"Turn {tid} (rating={rating}){mode_str}"
        if intended_text:
            lines.append(header)
            lines.append(f'  Intended: "{intended_text}"')
            lines.append(f"  Issue:    {explanation}")
        else:
            lines.append(f"{header}: {explanation}")
    return "\n".join(lines[:12])


def _format_entity_accuracy_trace(details: dict) -> str:
    """Format transcription_accuracy_key_entities failures."""
    per_turn = details.get("per_turn_entity_details", {})
    lines: list[str] = []
    for tid, turn_data in per_turn.items():
        for entity in turn_data.get("entities", []):
            if entity.get("skipped") or entity.get("correct"):
                continue
            expected = entity.get("value", "?")
            actual = entity.get("transcribed_value", "?")
            etype = entity.get("type", "unknown")
            lines.append(f'Turn {tid}: "{actual}" \u2192 expected "{expected}" ({etype})')
    return "\n".join(lines[:8])


def _format_trace(metric_name: str, details: dict) -> str:
    """Dispatch to metric-specific trace formatter."""
    if metric_name == "agent_speech_fidelity":
        return _format_speech_fidelity_trace(details)
    if metric_name == "transcription_accuracy_key_entities":
        return _format_entity_accuracy_trace(details)
    return ""


def _count_entity_totals_and_failures(
    combos: dict,
) -> tuple[dict[str, dict[str, dict[str, int]]], dict[str, dict]]:
    """Count total and failed entities per type per pipeline from metrics on disk.

    Returns:
        entity_data: {pipeline_label: {entity_type: {"total": N, "failed": N}}}
        pipeline_stats: {pipeline_label: {"conversations": N, "per_convo_turns": [...], "per_convo_entities": [...]}}
    """
    entity_data: dict[str, dict[str, dict[str, int]]] = {}
    pipeline_stats: dict[str, dict] = {}
    for label, report in combos.items():
        type_counts: dict[str, dict[str, int]] = {}
        per_convo_turns: list[int] = []
        per_convo_entities: list[int] = []
        for rd_str in report.get("run_dirs", []):
            run_dir = Path(rd_str)
            for record_dir in get_record_directories(run_dir):
                for _trial, data_path in get_valid_trial_dirs(record_dir):
                    metrics_path = data_path / "metrics.json"
                    if not metrics_path.exists():
                        continue
                    try:
                        with open(metrics_path) as f:
                            data = json.load(f)
                        metrics_dict = data.get("metrics", data)
                        ms = metrics_dict.get("transcription_accuracy_key_entities", {})
                        details = ms.get("details", {})
                    except Exception:
                        continue
                    per_turn = details.get("per_turn_entity_details", {})
                    if not per_turn:
                        continue
                    per_convo_turns.append(len(per_turn))
                    convo_entity_count = 0
                    for _tid, turn_data in per_turn.items():
                        for entity in turn_data.get("entities", []):
                            if entity.get("skipped"):
                                continue
                            etype = normalize_entity_type(entity.get("type", "unknown"))
                            if etype not in type_counts:
                                type_counts[etype] = {"total": 0, "failed": 0}
                            type_counts[etype]["total"] += 1
                            convo_entity_count += 1
                            if not entity.get("correct"):
                                type_counts[etype]["failed"] += 1
                    per_convo_entities.append(convo_entity_count)
        entity_data[label] = type_counts
        pipeline_stats[label] = {
            "conversations": len(per_convo_turns),
            "per_convo_turns": per_convo_turns,
            "per_convo_entities": per_convo_entities,
        }
    return entity_data, pipeline_stats


# ---------------------------------------------------------------------------
# Example collection (first pass — determines audio and trace data)
# ---------------------------------------------------------------------------


def _collect_examples(
    pipeline_result: dict, max_issues_per_metric: int = 3, max_audio: int = 30
) -> tuple[dict[str, list[tuple[str, dict]]], dict[str, Path]]:
    """Collect issue examples with trace snippets and audio paths.

    Returns:
        examples: {combo_label: [(metric_name, issue_info), ...]}
        audio_map: {audio_key: audio_file_path} for ZIP packaging
    """
    combos = pipeline_result.get("combos", {})
    record_lookup = _build_record_lookup(combos)

    examples: dict[str, list[tuple[str, dict]]] = {}
    audio_map: dict[str, Path] = {}
    audio_count = 0

    for label, report in combos.items():
        pass1 = report.get("pass1", {})
        records = record_lookup.get(label, {})
        combo_examples: list[tuple[str, dict]] = []

        # Build severity lookup from pass3 critical issues
        severity_map: dict[str, list[str]] = {}
        for ci in report.get("pass3", {}).get("critical_issues", []):
            sev = ci.get("severity", [])
            for merged_title in ci.get("merged_from", []):
                severity_map[merged_title.lower().strip()] = sev

        for metric in VOICE_METRIC_ORDER:
            p1 = pass1.get(metric, {})
            issues = p1.get("issues", [])
            if not issues:
                continue

            for issue in issues[:max_issues_per_metric]:
                samples = issue.get("affected_samples", [])
                title = issue.get("title", "Untitled")

                # Look up severity from pass3
                severity = severity_map.get(title.lower().strip(), [])

                # Find first sample with audio on disk
                example_info: dict[str, Any] | None = None
                for sid in samples[:8]:
                    sid_str = str(sid)
                    data_path = _resolve_sample_path(records, sid_str)
                    if not data_path:
                        continue

                    details = _load_metric_details(data_path, metric)
                    trace = _format_trace(metric, details)

                    audio_key: str | None = None
                    audio_path = data_path / "audio_mixed.wav"
                    if audio_path.exists() and audio_count < max_audio:
                        safe_label = label.replace(" ", "_").replace("/", "-")
                        safe_sid = sid_str.replace("/", "_")
                        audio_key = f"{safe_label}_{safe_sid}"
                        audio_map[audio_key] = audio_path
                        audio_count += 1

                    example_info = {
                        "sample_id": sid_str,
                        "trace": trace,
                        "example_text": issue.get("example", ""),
                        "audio_key": audio_key,
                    }
                    break

                combo_examples.append(
                    (
                        metric,
                        {
                            "title": title,
                            "description": issue.get("description", ""),
                            "sample_count": issue.get("sample_count", 0),
                            "severity": severity,
                            "example": example_info,
                        },
                    )
                )

        examples[label] = combo_examples

    return examples, audio_map


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _build_header(pipeline_result: dict, recipient: str = "") -> str:
    combos = pipeline_result.get("combos", {})
    n = len(combos)
    total_records = sum(r.get("total_samples", 0) for r in combos.values())

    heading = "EVA Error Analysis Highlights"
    if recipient:
        heading = f"{_esc(recipient)} &mdash; {heading}"
    parts = [
        '<div class="report-header">',
        f"<h1>{heading}</h1>",
        '<p class="subtitle">'
        "This report highlights error patterns identified in EVA voice agent evaluation runs. "
        "We summarize STT and TTS related issues, entity recognition failures, and how they "
        "cascade into downstream task completion."
        "</p>",
        f'<p style="font-size:0.92rem;">We evaluate the following <strong>{n}</strong> system(s) '
        f"across <strong>{total_records}</strong> conversations:</p>",
        '<div class="pipeline-grid">',
    ]

    for idx, (_label, report) in enumerate(combos.items(), 1):
        pc = report.get("pipeline_config", {})
        llm = display_name(pc.get("llm_model", "?"), "llm")
        stt = display_name(pc.get("stt_model", "?"), "stt")
        tts = display_name(pc.get("tts_model", "?"), "tts")
        parts.append(
            f'<div class="pipeline-card">'
            f'<div class="pipe-num">System {idx}</div>'
            f'<div><span class="comp-label">STT</span> <span class="component">{_esc(stt)}</span></div>'
            f'<div><span class="comp-label">LLM</span> <span class="component">{_esc(llm)}</span></div>'
            f'<div><span class="comp-label">TTS</span> <span class="component">{_esc(tts)}</span></div>'
            f"</div>"
        )

    parts.append("</div>")  # pipeline-grid
    parts.append("</div>")  # report-header
    return "\n".join(parts)


def _build_system_summary(pipeline_result: dict) -> str:
    """Build executive summary: key findings, best system, component insights."""
    cross_combo = pipeline_result.get("cross_combo", {})
    executive = pipeline_result.get("executive_summary", {})
    component_isolation = pipeline_result.get("component_isolation", {})

    combos = pipeline_result.get("combos", {})
    key_findings = executive.get("key_findings", [])
    best = cross_combo.get("best_overall", "")
    reasoning = _replace_raw_labels(cross_combo.get("best_overall_reasoning", ""), combos)
    shared = cross_combo.get("shared_failures", [])

    if not key_findings and not best and not component_isolation:
        return ""

    parts = [
        '<div class="section">',
        '<div class="section-title">System Summary</div>',
        '<p class="section-desc">'
        "High-level findings synthesized across all systems, including component-level insights."
        "</p>",
    ]

    # Best overall — value is already a display name from the LLM
    if best:
        display_best = _replace_raw_labels(best, combos)
        parts.append(f'<div class="summary-best"><strong>Best overall:</strong> {_esc(display_best)}')
        if reasoning:
            parts.append(f" &mdash; {_esc(reasoning)}")
        parts.append("</div>")

    # Key findings
    if key_findings:
        parts.append("<h3>Key Findings</h3>")
        for i, finding in enumerate(key_findings, 1):
            finding = _replace_raw_labels(finding, combos)
            if " | " in finding:
                title, desc = finding.split(" | ", 1)
                desc_html = _md_to_html(desc.strip())
                parts.append(f'<div class="summary-finding"><strong>{i}. {_esc(title)}</strong>{desc_html}</div>')
            else:
                parts.append(f'<div class="summary-finding"><strong>{i}.</strong> {_esc(finding)}</div>')

    # Component insights
    comp_parts: list[str] = []
    for comp_key, comp_report in component_isolation.items():
        if not isinstance(comp_report, dict):
            continue
        summary = comp_report.get("summary", "")
        if "No comparable groups found" in summary:
            continue
        comp_label = comp_key.replace("_model", "").upper()
        best_comp = comp_report.get("best_component", "")
        if best_comp or summary:
            display_comp = display_name(best_comp, comp_key.replace("_model", "")) if best_comp else ""
            line = f"<strong>{comp_label}:</strong> "
            if display_comp:
                line += f"Best = {_esc(display_comp)}. "
            if summary:
                line += _esc(_replace_raw_labels(summary, combos))
            comp_parts.append(f'<div class="summary-finding">{line}</div>')

    if comp_parts:
        parts.append("<h3>Component Insights</h3>")
        parts.extend(comp_parts)

    # Shared failures
    if shared:
        parts.append("<h3>Shared Issues Across All Systems</h3>")
        parts.append('<ul style="margin:0.5rem 0;padding-left:1.25rem;font-size:0.88rem;color:var(--muted);">')
        for f in shared:
            parts.append(f"<li>{_esc(f)}</li>")
        parts.append("</ul>")

    parts.append("</div>")  # section
    return "\n".join(parts)


def _build_score_heatmap(pipeline_result: dict) -> str:
    """Build EVA-A, EVA-X and per-metric score comparison heatmap."""
    combos = pipeline_result.get("combos", {})
    if not combos:
        return ""

    rows = []
    for label, report in combos.items():
        ms = report.get("metric_scores", {})
        row: dict[str, Any] = {"System": get_display_combo_label(label)}
        row["EVA-A Mean"] = report.get("eva_a_mean")
        row["EVA-A Pass@1"] = report.get("eva_a_pass1")
        row["EVA-X Mean"] = report.get("eva_x_mean")
        row["EVA-X Pass@1"] = report.get("eva_x_pass1")
        for m in ALL_ANALYSIS_METRICS:
            row[format_metric_name(m)] = ms.get(m)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("System")
    if df.empty:
        return ""

    parts = [
        '<div class="section">',
        '<div class="section-title">Score Overview</div>',
        '<p class="section-desc">'
        "EVA-A (Accuracy) = "
        + ", ".join(format_metric_name(m) for m in EVA_A_METRICS)
        + ". EVA-X (Experience) = "
        + ", ".join(format_metric_name(m) for m in EVA_X_METRICS)
        + ". <strong>Mean</strong> = average of per-sample scores. "
        "<strong>Pass@1</strong> = fraction of samples where all metrics in the group score 1.0. "
        "All scores normalized 0\u20131."
        "</p>",
        '<details class="other-issues" style="margin-bottom:0.75rem;">'
        "<summary>Metric scales &amp; normalization</summary>"
        '<div class="other-body" style="font-size:0.85rem;">'
        "<table>"
        "<tr><th>Metric</th><th>Raw Scale</th><th>Normalized</th></tr>"
        "<tr><td>Task Completion</td><td>F1 score (0\u20131)</td><td>Used as-is</td></tr>"
        "<tr><td>Faithfulness</td><td>1\u20133 (integer)</td><td>1\u21920, 2\u21920.5, 3\u21921</td></tr>"
        "<tr><td>Agent Speech Fidelity</td><td>1\u20133 (integer)</td><td>1\u21920, 2\u21920.5, 3\u21921</td></tr>"
        "<tr><td>Entity Accuracy</td><td>Correct/total entities</td><td>Accuracy ratio (0\u20131)</td></tr>"
        "<tr><td>Turn Taking</td><td>\u22122 to +2 (integer)</td><td>Mapped to 0\u20131</td></tr>"
        "<tr><td>Conversation Progression</td><td>1\u20133 (integer)</td><td>1\u21920, 2\u21920.5, 3\u21921</td></tr>"
        "<tr><td>Conciseness</td><td>1\u20133 (integer)</td><td>1\u21920, 2\u21920.5, 3\u21921</td></tr>"
        "</table>"
        "</div></details>",
    ]

    metrics = list(df.columns)
    colors = px.colors.qualitative.Plotly
    fig = go.Figure()
    for idx, system_name in enumerate(df.index):
        values = [df.loc[system_name, m] for m in metrics]
        color = colors[idx % len(colors)]
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=[v if v is not None else 0 for v in values],
                name=system_name,
                text=[f"{v:.3f}" if v is not None else "" for v in values],
                textposition="outside",
                textfont={"size": 10},
                marker_color=color,
            )
        )
    fig.update_layout(
        barmode="group",
        yaxis={"range": [0, 1.12], "title": "Score", "dtick": 0.25},
        xaxis={"tickangle": -40},
        height=400,
        margin={"l": 50, "r": 30, "t": 10, "b": 120},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.45, "xanchor": "center", "x": 0.5},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    parts.append(f'<div class="chart-container">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>')

    parts.append("</div>")  # section
    return "\n".join(parts)


def _render_issue_cards(issue_list: list[tuple[str, dict]], combos: dict, label: str, parts: list[str]) -> None:
    """Render a list of (metric, issue_data) tuples as issue cards."""
    current_metric: str | None = None
    for metric, issue_data in issue_list:
        if metric != current_metric:
            current_metric = metric
            metric_label = VOICE_METRIC_LABELS.get(metric, format_metric_name(metric))
            tag_cls = METRIC_TAG_CLASS.get(metric, "other")
            parts.append(
                f'<div class="metric-heading">'
                f'<span class="tag {tag_cls}">{tag_cls.upper()}</span> '
                f"{_esc(metric_label)}"
                f"</div>"
            )

        title = issue_data["title"]
        desc = issue_data["description"]
        count = issue_data["sample_count"]
        severity = issue_data.get("severity", [])

        # Build severity tags
        sev_html = ""
        sev_cls_map = {
            "high_impact": ("high", "HIGH"),
            "cascading": ("cascade", "CASCADE"),
            "common": ("common", "COMMON"),
        }
        for sev_key in severity:
            cls, text = sev_cls_map.get(sev_key, ("", sev_key))
            if cls:
                sev_html += f' <span class="tag severity {cls}">{text}</span>'

        parts.append('<div class="issue-card">')
        parts.append(f"<h4>{_esc(title)} <code>{count} samples</code>{sev_html}</h4>")
        parts.append(f"<p>{_esc(desc)}</p>")

        ex = issue_data.get("example")
        if ex:
            parts.append('<div class="evidence">')
            parts.append(f'<span class="sample-label">Sample #{_esc(ex["sample_id"])}</span>')

            if ex.get("trace"):
                parts.append(f'<div class="trace">{_esc(ex["trace"])}</div>')
            elif ex.get("example_text"):
                parts.append(f'<p class="example-text">{_esc(ex["example_text"])}</p>')

            if ex.get("audio_key"):
                parts.append(
                    f'<audio controls preload="none">'
                    f'<source src="audio/{ex["audio_key"]}.wav" type="audio/wav">'
                    f"Your browser does not support audio playback.</audio>"
                )
            parts.append("</div>")  # evidence

        parts.append("</div>")  # issue-card


# Metrics that count as voice/speech issues (shown directly)
_VOICE_METRICS = {"agent_speech_fidelity", "transcription_accuracy_key_entities"}


def _build_pipeline_issues(examples: dict[str, list], combos: dict) -> str:
    """Build the key issues section — per pipeline, ordered by voice relevance."""
    parts = [
        '<div class="section">',
        '<div class="section-title">Key Issues by System</div>',
        '<p class="section-desc">'
        "Issues are ordered by proximity to the speech pipeline: TTS fidelity issues first, "
        "then STT transcription errors, followed by downstream metric failures."
        "</p>",
    ]

    for idx, (label, combo_examples) in enumerate(examples.items()):
        voice_issues = [(m, d) for m, d in combo_examples if m in _VOICE_METRICS]

        n_voice = len(voice_issues)
        open_attr = " open" if idx == 0 else ""
        parts.append(f'<details class="pipeline-details"{open_attr}>')
        display_label = get_display_combo_label(label)
        parts.append(f'<summary>{_esc(display_label)} <span class="count-badge">{n_voice} issues</span></summary>')
        parts.append('<div class="pipeline-body">')

        if not voice_issues:
            parts.append('<p style="color:var(--muted);">No issues identified.</p>')
            parts.append("</div></details>")
            continue

        _render_issue_cards(voice_issues, combos, label, parts)

        parts.append("</div></details>")  # pipeline-body, details

    parts.append("</div>")  # section
    return "\n".join(parts)


def _build_radar(pipeline_result: dict) -> str:
    """Build combined radar overlay chart."""
    combos = pipeline_result.get("combos", {})
    if not combos:
        return ""

    pipeline_labels = list(combos.keys())
    colors = px.colors.qualitative.Plotly
    metric_names = [format_metric_name(m) for m in ALL_ANALYSIS_METRICS]
    polar_axis = {
        "range": [0, 1],
        "showticklabels": True,
        "tickvals": [0, 0.25, 0.5, 0.75, 1.0],
        "gridcolor": "#e2e8f0",
    }

    parts = [
        '<div class="section">',
        '<div class="section-title">Metric Profiles</div>',
        '<p class="section-desc">'
        "Radar overlay comparing normalized scores (0\u20131) across all metrics. "
        "Larger area indicates better overall performance."
        "</p>",
    ]

    fig = go.Figure()
    for idx, label in enumerate(pipeline_labels):
        ms = combos[label].get("metric_scores", {})
        values = [ms.get(m, 0) for m in ALL_ANALYSIS_METRICS]
        values_closed = values + [values[0]]
        theta_closed = metric_names + [metric_names[0]]
        color = colors[idx % len(colors)]
        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=theta_closed,
                fill="toself",
                fillcolor=color,
                opacity=0.15,
                line={"color": color, "width": 2.5},
                name=get_display_combo_label(label),
            )
        )
    fig.update_layout(
        polar={"radialaxis": polar_axis, "angularaxis": {"gridcolor": "#e2e8f0"}},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.2, "xanchor": "center", "x": 0.5},
        margin={"l": 80, "r": 80, "t": 30, "b": 90},
    )
    parts.append(_plotly_div(fig, 520))
    parts.append("</div>")  # section
    return "\n".join(parts)


def _build_entity_heatmap(pipeline_result: dict) -> str:
    """Build entity type failure chart with summary stats and grouped stacked bars."""
    combos = pipeline_result.get("combos", {})
    entity_data, pipeline_stats = _count_entity_totals_and_failures(combos)

    rows = []
    for label, type_counts in entity_data.items():
        for etype, counts in type_counts.items():
            if counts["total"] > 0:
                rate = counts["failed"] / counts["total"] * 100
                rows.append(
                    {
                        "System": get_display_combo_label(label),
                        "Entity Type": etype,
                        "Failure Rate (%)": round(rate, 1),
                        "Failed": counts["failed"],
                        "Total": counts["total"],
                    }
                )
    if not rows:
        return ""

    parts = [
        '<div class="section">',
        '<div class="section-title">Entity Type Failure Patterns</div>',
        '<p class="section-desc">'
        "Failure rate = failed entities \u00f7 total entities of each type. "
        "Shows which entity types (names, dates, codes, etc.) are hardest for each system's STT to transcribe."
        "</p>",
    ]

    # Summary table with medians
    parts.append(
        '<p style="font-size:0.82rem;color:var(--muted);margin:0 0 0.5rem;">'
        "Entity totals vary between pipelines because each conversation unfolds differently — "
        "longer conversations surface more entities for STT to transcribe."
        "</p>"
    )
    td_s = 'style="text-align:right;padding:4px 8px;border-bottom:1px solid var(--border);"'
    th_s = 'style="text-align:right;padding:4px 8px;border-bottom:2px solid var(--border);font-weight:600;"'
    tbl = ['<table style="width:100%;border-collapse:collapse;font-size:0.85rem;margin-bottom:1rem;">']
    tbl.append("<thead><tr>")
    tbl.append(
        '<th style="text-align:left;padding:4px 8px;border-bottom:2px solid var(--border);font-weight:600;">Pipeline</th>'
    )
    for col in (
        "Conversations",
        "Median Turns/Convo",
        "Total Entities",
        "Median Entities/Convo",
        "Failed",
        "Failure Rate",
    ):
        tbl.append(f"<th {th_s}>{col}</th>")
    tbl.append("</tr></thead><tbody>")

    for label, type_counts in entity_data.items():
        total = sum(c["total"] for c in type_counts.values())
        failed = sum(c["failed"] for c in type_counts.values())
        stats = pipeline_stats.get(label, {})
        n_convos = stats.get("conversations", 0)
        convo_turns = stats.get("per_convo_turns", [])
        convo_entities = stats.get("per_convo_entities", [])
        med_turns = float(np.median(convo_turns)) if convo_turns else 0
        med_entities = float(np.median(convo_entities)) if convo_entities else 0
        rate = f"{failed / total * 100:.1f}%" if total > 0 else "\u2014"
        dl = _esc(get_display_combo_label(label))
        tbl.append(f'<tr><td style="padding:4px 8px;border-bottom:1px solid var(--border);"><strong>{dl}</strong></td>')
        tbl.append(f"<td {td_s}>{n_convos}</td>")
        tbl.append(f"<td {td_s}>{med_turns:.1f}</td>")
        tbl.append(f"<td {td_s}>{total}</td>")
        tbl.append(f"<td {td_s}>{med_entities:.1f}</td>")
        tbl.append(f"<td {td_s}>{failed}</td>")
        tbl.append(f"<td {td_s}>{rate}</td></tr>")

    tbl.append("</tbody></table>")
    parts.append("".join(tbl))

    # Grouped stacked bar chart: one bar per pipeline per entity type
    df = pd.DataFrame(rows)
    type_order = df.groupby("Entity Type")["Total"].sum().sort_values(ascending=False).index.tolist()

    pipelines = sorted(df["System"].unique().tolist())
    palette_base = ["#93c5fd", "#86efac", "#fde68a", "#c4b5fd", "#fbcfe8"]
    palette_fail = ["#2563eb", "#16a34a", "#ca8a04", "#7c3aed", "#db2777"]

    fig = go.Figure()
    for i, pipeline in enumerate(pipelines):
        pdf = df[df["System"] == pipeline].set_index("Entity Type").reindex(type_order).fillna(0)
        passed = pdf["Total"] - pdf["Failed"]
        failed = pdf["Failed"]

        fig.add_trace(
            go.Bar(
                name=f"{pipeline}",
                x=type_order,
                y=passed,
                marker_color=palette_base[i % len(palette_base)],
                text=passed.astype(int),
                textposition="inside",
                legendgroup=pipeline,
                offsetgroup=i,
            )
        )
        fig.add_trace(
            go.Bar(
                name=f"{pipeline} (failed)",
                x=type_order,
                y=failed,
                marker_color=palette_fail[i % len(palette_fail)],
                text=failed.astype(int),
                textposition="inside",
                legendgroup=pipeline,
                offsetgroup=i,
                base=passed,
            )
        )

    fig.update_layout(
        barmode="group",
        height=480,
        yaxis_title="Entity Count",
        xaxis_title="Entity Type",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "center", "x": 0.5},
        margin={"l": 50, "r": 30, "t": 50, "b": 80},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    parts.append(f'<div class="chart-container">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>')

    # Entity type definitions dropdown
    parts.append('<details class="other-issues">')
    parts.append("<summary>Entity type definitions</summary>")
    parts.append('<div class="other-body">')
    parts.append(
        "<table>"
        "<tr><th>Entity Type</th><th>Description</th><th>Examples</th></tr>"
        "<tr><td><strong>identifier</strong></td><td>Confirmation codes, booking IDs, "
        "reference numbers</td><td>ABC123, ZK3FFW</td></tr>"
        "<tr><td><strong>name</strong></td><td>People's names, including spelled-out "
        "names</td><td>John Smith, S-M-I-T-H</td></tr>"
        "<tr><td><strong>flight</strong></td><td>Flight numbers and airline names</td>"
        "<td>UA 204, Delta Airlines</td></tr>"
        "<tr><td><strong>airport</strong></td><td>Airport codes, airport names, and "
        "flight routes</td><td>JFK, LAX, Austin-Bergstrom</td></tr>"
        "<tr><td><strong>location</strong></td><td>Cities, destinations, origins, "
        "and place names</td><td>Austin, New York</td></tr>"
        "<tr><td><strong>date</strong></td><td>Specific dates and years</td>"
        "<td>December 15th, 2026-03-18</td></tr>"
        "<tr><td><strong>time</strong></td><td>Specific times, durations, and "
        "time zones</td><td>3:45 PM, 2 hours, EST</td></tr>"
        "<tr><td><strong>amount</strong></td><td>Monetary amounts, prices, and "
        "currencies</td><td>$120, 50 dollars</td></tr>"
        "<tr><td><strong>seat</strong></td><td>Seat numbers, cabin classes, and "
        "fare classes</td><td>12A, Business, Economy</td></tr>"
        "<tr><td><strong>contact</strong></td><td>Email addresses and phone "
        "numbers</td><td>john@example.com, 555-867-5309</td></tr>"
        "<tr><td><strong>voucher</strong></td><td>Voucher codes and travel credit "
        "codes</td><td>CREDIT-500, VC12345</td></tr>"
        "</table>"
    )
    parts.append("</div></details>")

    parts.append("</div>")  # section
    return "\n".join(parts)


def _build_error_propagation(pipeline_result: dict) -> str:
    """Build error propagation section: explanation, table, chart, causal chains."""
    combos = pipeline_result.get("combos", {})
    has_data = any(r.get("error_propagation", {}).get("stats", {}).get("available") for r in combos.values())
    if not has_data:
        return ""

    parts = [
        '<div class="section">',
        '<div class="section-title">Error Propagation Analysis</div>',
        '<p class="section-desc">'
        "How STT entity transcription errors cascade into downstream metric failures. "
        "We split records into two groups — those where STT mis-transcribed at least one key entity "
        "vs. those with no entity errors — and compare metric scores and failure rates between them."
        "</p>",
    ]

    prop_idx = 0
    for label, report in combos.items():
        prop = report.get("error_propagation", {})
        stats = prop.get("stats", {})
        analysis = prop.get("analysis", {})
        if not stats.get("available"):
            continue

        display_label = get_display_combo_label(label)
        n_errors = stats.get("n_entity_errors", 0)
        n_clean = stats.get("n_clean", 0)
        open_attr = " open" if prop_idx == 0 else ""
        prop_idx += 1

        parts.append(f'<details class="pipeline-details"{open_attr}>')
        parts.append(
            f"<summary>{_esc(display_label)} "
            f'<span class="count-badge">{n_errors} records with entity errors &middot; {n_clean} clean records</span>'
            f"</summary>"
        )
        parts.append('<div class="pipeline-body">')

        # Impact table
        error_means = stats.get("error_means", {})
        clean_means = stats.get("clean_means", {})
        cofailure_rates = stats.get("cofailure_rates", {})
        baseline_rates = stats.get("baseline_rates", {})

        if error_means and clean_means:
            parts.append(
                "<table><tr><th>Metric</th>"
                "<th>Mean Score (records with entity errors)</th>"
                "<th>Mean Score (clean records)</th>"
                "<th>Delta</th>"
                "<th>Failure Rate (records with entity errors)</th>"
                "<th>Failure Rate (clean records)</th></tr>"
            )
            for m in error_means:
                err = error_means.get(m, 0)
                clean = clean_means.get(m, 0)
                delta = clean - err
                cofail = cofailure_rates.get(m, 0)
                baseline = baseline_rates.get(m, 0)
                delta_cls = "color:var(--success)" if delta >= 0 else "color:var(--danger)"
                parts.append(
                    f"<tr><td>{format_metric_name(m)}</td><td>{err:.3f}</td><td>{clean:.3f}</td>"
                    f'<td style="{delta_cls};font-weight:600">{delta:+.3f}</td>'
                    f"<td>{cofail:.0%}</td><td>{baseline:.0%}</td></tr>"
                )
            parts.append("</table>")
            parts.append(
                '<div style="background:#f5f7fa;border-radius:6px;padding:0.65rem 0.9rem;margin:0.5rem 0;font-size:0.82rem;color:#475569;line-height:1.7;">'
                "<strong>Mean Score</strong> columns show the average normalized metric score (0\u20131) for each group.<br>"
                "<strong>Failure Rate</strong> columns show what percentage of records in each group score below the passing threshold.<br>"
                "A large gap indicates entity errors are a likely cause of downstream failures."
                "</div>"
            )

        # Co-failure bar chart
        if cofailure_rates and baseline_rates:
            bar_rows = []
            for m in cofailure_rates:
                bar_rows.append(
                    {"Metric": format_metric_name(m), "Rate": cofailure_rates[m], "Group": "Records with entity errors"}
                )
                bar_rows.append({"Metric": format_metric_name(m), "Rate": baseline_rates[m], "Group": "Clean records"})
            fig = px.bar(
                pd.DataFrame(bar_rows),
                x="Metric",
                y="Rate",
                color="Group",
                barmode="group",
                title="Failure rate: records with entity errors vs clean records",
                color_discrete_map={"Records with entity errors": "#EF553B", "Clean records": "#636EFA"},
            )
            fig.update_layout(yaxis_tickformat=".0%")
            parts.append(_plotly_div(fig, 380))

        # Overall finding
        overall = analysis.get("overall_finding", "")
        if overall:
            parts.append(f'<div class="finding-box">{_esc(overall)}</div>')

        # Causal chains
        chains = analysis.get("causal_chains", [])
        if chains:
            parts.append('<h4 style="margin-top:1.25rem;">Causal Chains</h4>')
            for chain_idx, chain in enumerate(chains, 1):
                entity_types = chain.get("entity_types_involved", [])
                downstream = chain.get("downstream_effects", [])
                mechanism = chain.get("mechanism", "")

                parts.append('<div class="causal-chain">')
                parts.append(f"<strong>{chain_idx}. {_esc(chain.get('pattern', 'Unknown'))}</strong>")
                parts.append(f"<p>{_esc(mechanism)}</p>")
                meta_parts = []
                if entity_types:
                    meta_parts.append(f"Entity types: {', '.join(entity_types)}")
                if downstream:
                    meta_parts.append(f"Downstream: {', '.join(downstream)}")
                if meta_parts:
                    parts.append(f'<p class="meta">{" &nbsp;|&nbsp; ".join(meta_parts)}</p>')
                parts.append("</div>")

        parts.append("</div></details>")  # pipeline-body, details

    parts.append("</div>")  # section
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def generate_report_zip(pipeline_result: dict, max_audio_samples: int = 30, recipient: str = "") -> bytes:
    """Generate a ZIP containing report.html and audio/ folder.

    Args:
        pipeline_result: The full dict returned by run_full_pipeline().
        max_audio_samples: Maximum number of audio files to include.
        recipient: Optional name prepended to the report title (e.g. "ElevenLabs").

    Returns:
        Bytes of the ZIP file.
    """
    examples, audio_map = _collect_examples(pipeline_result, max_audio=max_audio_samples)
    html_str = _generate_html(pipeline_result, examples, audio_map, recipient=recipient)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("report.html", html_str)
        for key, audio_path in audio_map.items():
            zf.write(audio_path, f"audio/{key}.wav")

    return buf.getvalue()


def generate_html_report(pipeline_result: dict, recipient: str = "") -> str:
    """Generate HTML report string (no audio files)."""
    examples, _ = _collect_examples(pipeline_result, max_audio=0)
    return _generate_html(pipeline_result, examples, audio_map={}, recipient=recipient)


def _generate_html(
    pipeline_result: dict,
    examples: dict[str, list],
    audio_map: dict[str, Path],
    recipient: str = "",
) -> str:
    """Assemble the full HTML document."""
    combos = pipeline_result.get("combos", {})

    sections = [
        _build_header(pipeline_result, recipient=recipient),
        _build_system_summary(pipeline_result),
        '<hr class="divider">',
        _build_score_heatmap(pipeline_result),
        '<hr class="divider">',
        _build_pipeline_issues(examples, combos),
        '<hr class="divider">',
        _build_radar(pipeline_result),
        '<hr class="divider">',
        _build_entity_heatmap(pipeline_result),
        '<hr class="divider">',
        _build_error_propagation(pipeline_result),
    ]

    body = "\n".join(s for s in sections if s)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    base_title = "EVA Error Analysis Highlights"
    if recipient:
        title = f"{recipient} \u2014 {base_title}"
    else:
        labels = list(combos.keys())
        title = f"EVA Error Analysis \u2014 {', '.join(labels)}" if labels else base_title

    return _HTML_TEMPLATE.format(title=title, body=body, timestamp=timestamp)
