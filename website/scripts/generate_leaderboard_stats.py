#!/usr/bin/env python3
"""Generate website/src/data/leaderboardStats.json from output_processed/eva-bench-stats/all_scores.json.

Run with: /opt/miniconda3/bin/python3 website/scripts/generate_leaderboard_stats.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "output_processed" / "eva-bench-stats" / "all_scores.json"
DST = ROOT / "website" / "src" / "data" / "leaderboardStats.json"

CLEAN_METRICS = [
    "EVA-A_mean", "EVA-A_pass", "EVA-A_pass_at_k", "EVA-A_pass_power_k",
    "EVA-X_mean", "EVA-X_pass", "EVA-X_pass_at_k", "EVA-X_pass_power_k",
    "task_completion", "agent_speech_fidelity", "faithfulness",
    "turn_taking", "conciseness", "conversation_progression",
    "response_speed",
]
PERTURBATION_METRICS = [
    "task_completion", "agent_speech_fidelity", "faithfulness",
    "turn_taking", "conciseness", "conversation_progression",
    "EVA-A_pass", "EVA-X_pass", "conversation_correctly_finished",
]
DOMAINS = ["airline", "itsm", "medical_hr"]

# Intentionally excluded from the website surface:
#   - response_speed.no_tool_calls / response_speed.with_tool_calls
#   - turn_taking.early_rate / late_rate / on_time_rate / agent_interruption.rate
#   - transcription_accuracy_key_entities (cascade-only, perturbation-only)
# These breakdowns live in the paper / dev docs; the leaderboard surfaces only
# top-level metrics. Add them here if a future page needs them.


# System metadata — author-curated. Edit when new systems are added.
# type: 'cascade' | 's2s' | '2-part'
# Architectures sourced from EVA-Bench paper Appendix B (Tables 3, 4, 5) and
# Figures 8/9/10 captions (pages 54-55) which list systems by pipeline class:
#   Cascade: Cohere + Gemma-4-26B + Voxtral; ElevenAgents (Scribe + Gemini-3-Flash + Conversational v3);
#            Ink + Haiku-4.5 + Sonic; Nova + GPT-5.4 + Sonic; Nova + GPT-5.4-mini + Aura;
#            Parakeet + Gemma-4-31B + Kokoro; Whisper + Qwen3.5-27B + Voxtral.
#   Hybrid (2-part): Gemini-3-Flash + Gemini-3.1-Flash-TTS (merged from two source rows
#     carrying clean and perturbation_delta separately); Ultravox.
#   S2S: Gemini-3.1-Flash-Live; GPT-Realtime-1.5; GPT-Realtime-mini.
SYSTEM_META = {
    "Ultravox":                                {"type": "2-part",  "stt": "-",                "llm": "Ultravox-Realtime", "tts": "-"},
    "GPT Realtime":                            {"type": "s2s",     "stt": "-",                "llm": "gpt-realtime",      "tts": "-"},
    "GPT Realtime 1.5":                        {"type": "s2s",     "stt": "-",                "llm": "gpt-realtime-1.5",  "tts": "-"},
    "GPT Realtime 2":                          {"type": "s2s",     "stt": "-",                "llm": "gpt-realtime-2",    "tts": "-"},
    "GPT Realtime Mini":                       {"type": "s2s",     "stt": "-",                "llm": "gpt-realtime-mini", "tts": "-"},
    "ElevenAgent":                             {"type": "cascade", "stt": "Scribe v2.2 Realtime", "llm": "Gemini 3 Flash", "tts": "TTS Conversational v3", "display_name": "Scribe v2.2 Realtime + Gemini 3 Flash + TTS Conversational v3 (ElevenAgents)"},
    "Gemini 3.1 Flash Live":                   {"type": "s2s",     "stt": "-",                "llm": "Gemini 3.1 Flash Live", "tts": "-"},
    "Gemini 3 Flash + Gemini 3.1 Flash TTS":   {"type": "2-part",  "stt": "-",                "llm": "Gemini 3 Flash", "tts": "Gemini 3.1 Flash TTS"},
    "Cohere + Gemma-4-26B + Voxtral":          {"type": "cascade", "stt": "Cohere Transcribe", "llm": "Gemma-4-26B", "tts": "Voxtral 4B TTS"},
    "Ink Whisper + Haiku 4.5 + Sonic 3":       {"type": "cascade", "stt": "Ink Whisper",       "llm": "Haiku 4.5",   "tts": "Sonic 3"},
    "Nova 3 + GPT-5.4 + Sonic 3":              {"type": "cascade", "stt": "Nova 3",            "llm": "GPT-5.4",     "tts": "Sonic 3"},
    "Nova 3 + GPT-5.4-mini + Aura 2":          {"type": "cascade", "stt": "Nova 3",            "llm": "GPT-5.4-mini", "tts": "Aura 2"},
    "Parakeet 1.1 + Gemma 31B + Kokoro":       {"type": "cascade", "stt": "Parakeet 1.1",      "llm": "Gemma-4-31B", "tts": "Kokoro"},
    "Whisper Large v3 + Qwen35-27B + Voxtral": {"type": "cascade", "stt": "Whisper Large v3",  "llm": "Qwen3.5-27B", "tts": "Voxtral 4B TTS"},
}


def slugify(name: str) -> str:
    return (
        name.lower()
        .replace("+", "plus").replace(".", "-").replace(" ", "-")
        .replace("/", "-").replace("(", "").replace(")", "")
    )


def _pt(node):
    if node is None:
        return None
    out = {"point": node["point"], "ci_lower": node["ci_lower"], "ci_upper": node["ci_upper"]}
    if "n" in node:
        out["n"] = node["n"]
    return out


def extract_clean(system_clean: dict) -> dict:
    out = {}
    for m in CLEAN_METRICS:
        if m not in system_clean:
            continue
        node = system_clean[m]
        out[m] = {
            "pooled": _pt(node.get("pooled")),
            "per_domain": {d: _pt(node.get("per_domain", {}).get(d)) for d in DOMAINS if node.get("per_domain", {}).get(d)},
        }
    return out


def _pt_pert(node):
    if node is None:
        return None
    return {
        "point": node["point"],
        "ci_lower": node["ci_lower"],
        "ci_upper": node["ci_upper"],
        "corrected_p": node.get("corrected_p"),
        "raw_p": node.get("raw_p"),
        "reject": node.get("reject"),
    }


def extract_pert(system_pert: dict) -> dict:
    out = {}
    for m in PERTURBATION_METRICS:
        if m not in system_pert:
            continue
        out[m] = {}
        for pert_name, pert_node in system_pert[m].items():
            out[m][pert_name] = {
                "pooled": _pt_pert(pert_node.get("pooled")),
                "per_domain": {d: _pt_pert(pert_node.get("per_domain", {}).get(d)) for d in DOMAINS if pert_node.get("per_domain", {}).get(d)},
            }
    return out


def main():
    raw = json.loads(SRC.read_text())
    # Merge accidentally-split Gemini hybrid rows (one carries `clean`, the
    # other carries `perturbation_delta`). Keep the canonical name.
    GEMINI_MERGE = {
        "Gemini 3 Flash + Gemini 3.1 Flash TTS": "Gemini 3 Flash + Gemini Flash TTS",
    }
    for canon, alias in GEMINI_MERGE.items():
        if alias in raw:
            extra = raw.pop(alias)
            if "perturbation_delta" in extra:
                raw[canon].setdefault("perturbation_delta", {}).update(extra["perturbation_delta"])
            if "clean" in extra:
                raw[canon].setdefault("clean", {}).update(extra["clean"])
    systems = []
    for name, payload in raw.items():
        meta = SYSTEM_META.get(name)
        if meta is None:
            raise SystemExit(f"Missing SYSTEM_META for {name!r}. Add it to generate_leaderboard_stats.py and re-run.")
        systems.append({
            "id": slugify(name),
            "name": meta.get("display_name", name),
            "type": meta["type"],
            "stt": meta["stt"], "llm": meta["llm"], "tts": meta["tts"],
            "clean": extract_clean(payload.get("clean", {})),
            "perturbation_delta": extract_pert(payload.get("perturbation_delta", {})),
        })
    DST.write_text(json.dumps({"systems": systems}, indent=2))
    print(f"wrote {len(systems)} systems to {DST}")


if __name__ == "__main__":
    main()
