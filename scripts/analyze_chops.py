"""Detect choppy-audio artifacts (dropped/spliced chunks) in EVA recordings.

A "chop" is a missing or spliced audio chunk. Because EVA streams fixed-size
chunks (20ms = 160 µ-law bytes, or 250ms), a dropped chunk leaves a specific,
machine-detectable fingerprint: a run of digital-zero samples that severs
continuous speech, with a HARD CUT on BOTH edges — full-amplitude speech
jumping straight to zero and back. A natural pause is the opposite: speech
tapers to ~0 before the silence, so its edge amplitude is tiny. Requiring both
edges to be hard cuts is what separates chops from pauses (validated against
labeled recordings: 0 false positives on a clean file, all real chops caught).
Chops are bucketed "short" (< 150ms, ~one dropped chunk) vs "long" (>= 150ms).

Two modes:

  characterize
      Given a WAV and labeled chop timestamps, print the discriminating
      features inside each labeled window vs. the file's baseline, so we can
      SEE what separates chops from normal speech/pauses and lock thresholds.

  scan
      Run the calibrated detector over every audio_mixed.wav under a run
      directory and emit a CSV of candidate chops sorted by confidence.

Label file format (JSON) for characterize:
    {
      "audio_file": "output/<run>/records/<id>/audio_mixed.wav",
      "chops": [
        {"t": 12.34},                     # a point in the chop
        {"start": 30.10, "end": 30.35}    # or an explicit span (seconds)
      ]
    }

Usage:
    python scripts/analyze_chops.py characterize labels1.json labels2.json
    python scripts/analyze_chops.py scan output/<run_id> --out chops.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

# --- Framing / feature parameters -------------------------------------------
FRAME_MS = 10.0  # analysis frame length (baseline stats only)
HOP_MS = 5.0  # analysis hop (baseline stats only)

# --- Detector thresholds (calibrated against labeled EVA recordings) ---------
# A chop is a dropped/spliced chunk: a run of digital-zero samples that severs
# continuous speech, leaving a HARD CUT on BOTH edges (full-amplitude speech
# jumping straight to zero and back). A natural pause differs on exactly this
# point — speech tapers to ~0 before the silence, so its "cut" amplitude is
# tiny. Requiring both edges to be hard is what separates chops from pauses
# (validated: 0 false positives on a clean recording, catches all real chops).
SILENCE_EPS = 1e-4  # |sample| below this counts as digital silence (float32 -1..1)
MIN_ZERO_MS = 20.0  # ignore sub-chunk blips (one 20ms chunk is the smallest drop)
EDGE_WIN_MS = 5.0  # window just outside the silence used to measure cut amplitude
EDGE_STEP_MIN = 0.05  # BOTH edges' amplitude must exceed this (hard splice)
LONG_CHOP_MS = 150.0  # >= this is a "long" chop; below is "short"


@dataclass
class Chop:
    start: float
    end: float
    kind: str  # "gap" | "click"
    confidence: float
    detail: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV as float32 mono in [-1, 1]. Stereo is averaged across channels."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    return mono, sr


def longest_zero_run_ms(audio: np.ndarray, sr: int, lo: int, hi: int) -> float:
    """Longest contiguous digital-silence run (in ms) within sample range [lo, hi)."""
    seg = np.abs(audio[lo:hi]) < SILENCE_EPS
    if not seg.any():
        return 0.0
    best = run = 0
    for v in seg:
        run = run + 1 if v else 0
        best = max(best, run)
    return best * 1000.0 / sr


def detect_chops(audio: np.ndarray, sr: int) -> list[Chop]:
    """Find dropped/spliced chunks: digital-zero runs with a hard cut on both edges.

    A chop severs continuous speech, so the sample just before the silence and
    the sample just after are both at speech amplitude (a splice). Natural
    pauses taper to ~0 before the silence, so at least one edge is tiny — those
    are rejected by requiring MIN of the two edge amplitudes to exceed the
    threshold.
    """
    silent = np.abs(audio) < SILENCE_EPS
    w = max(1, int(sr * EDGE_WIN_MS / 1000.0))
    min_run = int(sr * MIN_ZERO_MS / 1000.0)
    chops: list[Chop] = []
    n = len(audio)
    i = 0
    while i < n:
        if not silent[i]:
            i += 1
            continue
        j = i
        while j < n and silent[j]:
            j += 1
        if (j - i) >= min_run:
            pre = float(np.max(np.abs(audio[max(0, i - w) : i]))) if i > 0 else 0.0
            post = float(np.max(np.abs(audio[j : j + w]))) if j < n else 0.0
            edge = min(pre, post)
            if edge >= EDGE_STEP_MIN:
                dur_ms = (j - i) * 1000.0 / sr
                chops.append(
                    Chop(
                        start=i / sr,
                        end=j / sr,
                        kind="long" if dur_ms >= LONG_CHOP_MS else "short",
                        confidence=round(min(1.0, edge / 0.25), 3),
                        detail={"dur_ms": round(dur_ms, 1), "pre": round(pre, 3), "post": round(post, 3)},
                    )
                )
        i = j
    return chops


def detect_all(path: Path) -> list[Chop]:
    audio, sr = load_mono(path)
    return sorted(detect_chops(audio, sr), key=lambda c: c.start)


# ---------------------------------------------------------------------------
# characterize
# ---------------------------------------------------------------------------
def _label_spans(chops: list[dict]) -> list[tuple[float, float]]:
    spans = []
    for c in chops:
        if "start" in c and "end" in c:
            spans.append((float(c["start"]), float(c["end"])))
        else:
            t = float(c["t"])
            spans.append((t - 0.4, t + 0.4))  # ±400ms window (labels may be slightly misaligned)
    return spans


def _window_stats(audio: np.ndarray, sr: int, a: float, b: float) -> tuple[float, float, float]:
    """Return (min frame RMS, longest zero-run ms, max sample-diff) over [a, b] seconds."""
    lo, hi = max(0, int(a * sr)), min(len(audio), int(b * sr))
    seg = audio[lo:hi]
    if seg.size == 0:
        return 0.0, 0.0, 0.0
    fr = max(1, int(sr * FRAME_MS / 1000))
    hp = max(1, int(sr * HOP_MS / 1000))
    rmss = [np.sqrt(np.mean(seg[k : k + fr] ** 2)) for k in range(0, max(1, seg.size - fr), hp)]
    min_rms = float(min(rmss)) if rmss else 0.0
    max_diff = float(np.max(np.abs(np.diff(seg)))) if seg.size > 1 else 0.0
    return min_rms, longest_zero_run_ms(audio, sr, lo, hi), max_diff


def characterize(label_files: list[Path]) -> None:
    print(f"{'file/window':<38} {'min_rms':>9} {'zero_ms':>9} {'max_diff':>9} {'detected':>18}")
    print("-" * 90)
    for lf in label_files:
        spec = json.loads(lf.read_text())
        wav = Path(spec["audio_file"])
        if not wav.is_absolute():
            wav = (lf.parent / wav).resolve()
        audio, sr = load_mono(wav)
        detected = detect_all(wav)

        print(f"[{wav.name}] labeled chops:")
        for a, b in _label_spans(spec.get("chops", [])):
            mr, zr, md = _window_stats(audio, sr, a, b)
            near = [d for d in detected if d.start <= b + 0.1 and d.end >= a - 0.1]
            tag = ",".join(f"{d.kind}({d.confidence})" for d in near) or "MISS"
            print(f"  {a:6.2f}-{b:6.2f}s{'':<20} {mr:9.4f} {zr:9.1f} {md:9.4f} {tag:>18}")

        # Baseline: sample random speech windows NOT overlapping any label.
        spans = _label_spans(spec.get("chops", []))
        rng = np.random.default_rng(0)
        dur = len(audio) / sr
        shown = 0
        for _ in range(200):
            a = float(rng.uniform(0, max(0.001, dur - 0.3)))
            b = a + 0.3
            if any(a < s2 and b > s1 for s1, s2 in spans):
                continue
            mr, zr, md = _window_stats(audio, sr, a, b)
            if mr > 0.02:  # only report active-speech baselines (min RMS above the noise floor)
                print(f"  baseline {a:6.2f}-{b:6.2f}s{'':<11} {mr:9.4f} {zr:9.1f} {md:9.4f} {'':>18}")
                shown += 1
            if shown >= 5:
                break
        print()


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------
def scan(run_dir: Path, out_csv: Path, min_conf: float) -> None:
    wavs = sorted(run_dir.rglob("audio_mixed.wav"))
    if not wavs:
        print(f"No audio_mixed.wav found under {run_dir}", file=sys.stderr)
        return
    rows = []
    for wav in wavs:
        # e.g. records/<record_id>/<trial>/audio_mixed.wav -> "<record_id>/<trial>"
        record = f"{wav.parent.parent.name}/{wav.parent.name}"
        try:
            chops = [c for c in detect_all(wav) if c.confidence >= min_conf]
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {wav}: {e}", file=sys.stderr)
            continue
        for c in chops:
            rows.append(
                {
                    "record": record,
                    "file": str(wav),
                    "start_s": round(c.start, 3),
                    "end_s": round(c.end, 3),
                    "duration_ms": round(c.duration * 1000, 1),
                    "kind": c.kind,
                    "confidence": c.confidence,
                    "detail": json.dumps(c.detail),
                }
            )
        print(f"{record}: {len(chops)} candidate chop(s)")
    rows.sort(key=lambda r: r["confidence"], reverse=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["record"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} candidate chops across {len(wavs)} files -> {out_csv}")


def detect_overlap(user_path: Path, asst_path: Path, speech_rms: float = 0.02, hop_s: float = 0.02, min_run_s: float = 0.20) -> dict:
    """Measure cross-channel overlap (both user and assistant speaking at once).

    Overlap is present in normal conversation (interruptions), so the useful
    signal for a *drift/misalignment* bug is whether overlap grows late in the
    conversation vs early — a drift accumulates over turns, natural overlap does
    not. Returns totals plus an early(first 40%)/late(last 40%) split.
    """
    u, sr = load_mono(user_path)
    a, _ = load_mono(asst_path)
    n = min(len(u), len(a))
    u, a = u[:n], a[:n]
    h = max(1, int(sr * hop_s))
    nf = n // h
    both = np.array(
        [
            (np.sqrt(np.mean(u[i * h : (i + 1) * h] ** 2)) > speech_rms)
            and (np.sqrt(np.mean(a[i * h : (i + 1) * h] ** 2)) > speech_rms)
            for i in range(nf)
        ]
    )

    def runs_secs(mask: np.ndarray) -> float:
        total = 0.0
        i = 0
        m = len(mask)
        while i < m:
            if mask[i]:
                j = i
                while j < m and mask[j]:
                    j += 1
                if (j - i) * hop_s >= min_run_s:
                    total += (j - i) * hop_s
                i = j
            else:
                i += 1
        return total

    q = max(1, nf // 5)
    return {
        "total_s": round(runs_secs(both), 2),
        "early_s": round(runs_secs(both[: 2 * q]), 2),
        "late_s": round(runs_secs(both[3 * q :]), 2),
    }


def scan_overlap(run_dir: Path, out_csv: Path) -> None:
    """Report per-record chops (user channel) and user/assistant overlap."""
    users = sorted(run_dir.rglob("audio_user.wav"))
    rows = []
    for u in users:
        a = u.parent / "audio_assistant.wav"
        if not a.exists():
            continue
        record = f"{u.parent.parent.name}/{u.parent.name}"
        try:
            chops = detect_all(u)
            ov = detect_overlap(u, a)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {u}: {e}", file=sys.stderr)
            continue
        # Separate audible LONG chops (>=150ms) from benign 20ms single-frame
        # micro-gaps, which are a pre-existing baseline artifact and otherwise
        # dominate (and distort) the raw count.
        chops_long = sum(c.kind == "long" for c in chops)
        chops_short = sum(c.kind == "short" for c in chops)
        rows.append(
            {
                "record": record,
                "chops_long": chops_long,
                "chops_short": chops_short,
                **{f"overlap_{k}": v for k, v in ov.items()},
            }
        )
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["record"])
        w.writeheader()
        w.writerows(rows)
    tcl = sum(r["chops_long"] for r in rows)
    tcs = sum(r["chops_short"] for r in rows)
    to = sum(r["overlap_total_s"] for r in rows)
    te = sum(r["overlap_early_s"] for r in rows)
    tl = sum(r["overlap_late_s"] for r in rows)
    print(
        f"{len(rows)} records: LONG chops={tcl} (audible)  short/20ms={tcs} (benign)  "
        f"overlap total={to:.1f}s (early={te:.1f}s late={tl:.1f}s) -> {out_csv}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("characterize", help="Inspect features in labeled chop windows vs baseline")
    c.add_argument("labels", nargs="+", type=Path, help="Label JSON file(s)")

    s = sub.add_parser("scan", help="Detect chops across all audio_mixed.wav in a run dir")
    s.add_argument("run_dir", type=Path)
    s.add_argument("--out", type=Path, default=Path("chops.csv"))
    s.add_argument("--min-conf", type=float, default=0.0)

    o = sub.add_parser("overlap", help="Report chops + user/assistant overlap (early/late) per record")
    o.add_argument("run_dir", type=Path)
    o.add_argument("--out", type=Path, default=Path("overlap.csv"))

    args = ap.parse_args()
    if args.cmd == "characterize":
        characterize(args.labels)
    elif args.cmd == "scan":
        scan(args.run_dir, args.out, args.min_conf)
    elif args.cmd == "overlap":
        scan_overlap(args.run_dir, args.out)


if __name__ == "__main__":
    main()
