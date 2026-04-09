#!/usr/bin/env python3
"""Extract all audio clips from an LLM request dump as playable WAV files.

Usage:
    python scripts/extract_audio.py <path_to_llm_request.json> [output_dir]

If output_dir is omitted, extracts to an 'extracted_audio' folder next to the input file.
"""

import argparse
import base64
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Extract audio clips from an LLM request dump.")
    parser.add_argument("dump_path", type=Path, help="Path to the LLM request JSON file")
    parser.add_argument("output_dir", type=Path, nargs="?", default=None,
                        help="Output directory (default: extracted_audio/ next to input file)")
    args = parser.parse_args()

    out_dir = args.output_dir or (args.dump_path.parent / "extracted_audio")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dump_path) as f:
        data = json.load(f)

    # Support both formats: top-level array or {"messages": [...], "response": {...}}
    messages = data if isinstance(data, list) else data.get("messages", [])

    audio_count = 0
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if not isinstance(content, list):
            continue
        for chunk in content:
            if chunk.get("type") != "audio_url":
                continue
            url = chunk["audio_url"]["url"]
            # data:audio/wav;base64,<data>
            _, b64_data = url.split(",", 1)
            wav_bytes = base64.b64decode(b64_data)
            filename = f"msg_{i:02d}_{role}_{audio_count:02d}.wav"
            (out_dir / filename).write_bytes(wav_bytes)
            audio_count += 1
            print(f"  Wrote {filename} ({len(wav_bytes)} bytes)")

    print(f"\nExtracted {audio_count} audio files to {out_dir}")


if __name__ == "__main__":
    main()
