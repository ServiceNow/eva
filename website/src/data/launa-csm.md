# Launa CSM-only evaluation (provisional)

This submission evaluates Launa with Cartesia Sonic 3 TTS on clean English CSM. It contains no ITSM, HR, pooled, or perturbation result. The public model name maps to the internal sonic-v3 run.

There are 50 scenarios and 250 planned trials. 244 trials passed simulator validation; six invalid trials affect four scenarios. Valid model failures remain in the scores. Five-trial measures use the 46 scenarios with all five valid trials. This incomplete-run policy and evaluation version are submitted for maintainer review.

## Aggregation

For pass@1, composite means, and component metrics, average valid observations within each scenario and then weight scenarios equally. A legitimately skipped component is omitted for that component. `n` is the number of contributing scenarios, not trials. `pooled` is null and other domain fields are absent.

For pass@5, score 1 if a complete scenario has at least one successful trial. For estimated pass^5, raise each complete scenario's success fraction to the fifth power. Do not use the observed all-five-success fraction in place of estimated pass^5.

For every metric, recompute 95% percentile confidence intervals with EVA's `src/eva/utils/bootstrap.py` (2,000 scenario resamples and `run_seed` of the run ID). Point estimates and confidence intervals use the same scenario population. The accompanying `launa-csm-evidence.json` supplies scenario means and success counts to reproduce the exported values.

This differs from the original report's trial-weighted pass@1 (49.59% EVA-A / 55.33% EVA-X) because incomplete scenarios have fewer valid trials. The submitted scenario-weighted values are 48.70% / 55.03%; composite means are 64.19% / 75.36%. Five-trial results are unchanged: pass@5 80.43% / 95.65%, estimated pass^5 29.09% / 18.14%.

## Evaluation setup

- Run date: August 18, 2026.
- EVA 2.1.0; simulation 2.0.2; metrics 2.2.2.
- Base commit: `227690fb6b5cade308d9aba1830050ed4272ed54`.
- Recorded tracked-diff hash: `8e0ebf21c233`; matched against the retained patch.
- Architecture: audio-language model served through vLLM + Cartesia Sonic 3 TTS; Pipecat, Silero VAD, turn analyzer.
- Thinking disabled; temperature 1.0, top-p 0.95, top-k 64, max output tokens 12,000.
- Streaming endpoint responses were reassembled before pipeline consumption (`llm_streaming=false`).
- Local integration changes: gateway authentication, streaming-response reassembly, diagnostics, and conversation-start pacing. No tracked scoring-code edits.
- The run combines a single-trial run and two two-trial runs with matching recorded evaluation settings, except endpoint and trial count. The retained manifest records the source of every trial.

Per-trial audio, transcripts, tool logs, database snapshots, metric outputs, and merge manifest are retained. This PR includes sanitized numeric scenario evidence; credentials, private endpoints, and raw logs are excluded.

Maintainer decisions requested: acceptance of CSM-only coverage, handling of the six invalid trials, and whether metrics 2.2.2 requires re-scoring on the current revision. No pooled ranking is claimed.
