# Agent Speech Fidelity Development

## Starting point: basic entity-focused evaluation

The first version of the Agent speech fidelity judge established the core concept: compare audio against intended text with special attention to TTS-critical entities (confirmation codes, flight numbers, dollar amounts, seat numbers, times, dates, names). It used a binary per-turn rating (1 = high fidelity, 0 = low fidelity) with an overall rating that fails if any turn fails. It supported two evaluation modes — single-turn and multi-turn conversation — via a Jinja2 conditional.

The error/ignore boundary was straightforward: entity errors, missing/added words that change meaning = rating 0; minor pronunciation variations, filler words, end-of-audio cutoff, pacing differences = acceptable.

## Adding comparison rules to prevent entity-vs-text confusion

The second revision added an "IMPORTANT: Comparison Rules" section at the top of the evaluation criteria. The judge was incorrectly treating entity values as substitutes for the intended text, for example, if the intended text said, "the flight" but the audio said, "SkyWay seven zero three" (the actual flight number), the judge was calling it a match because the entity was correct. The new rules made it explicit: always compare against the literal text in "Text:" for each turn, not against what the entities represent. Concrete examples were added to illustrate the distinction.

## Handling non-spoken tags in the intended text

The same revision added a "Understanding the Intended Text" section addressing two types of non-spoken content that were causing false positives:

- **Audio-direction tags:** Tags like [slow], [firm], [annoyed] describe how words were meant to be spoken; they are not spoken aloud and should never be expected in the audio.
- **Interruption tags:** Six specific tag types were documented ([assistant interrupts], [user interrupts], [likely cut off by user], [speaker likely cut itself off], [likely interruption], [assistant starts replying - user interrupts]). These are metadata markers from post-processing that describe what happened during overlapping speech. The key principle established: if a tag indicates text was likely never spoken (due to interruption or cut-off), do not penalize for those words being missing. Each tag received a specific definition explaining which portions of text before/after it were likely spoken vs. not.

This was necessary because the judge was flagging "missing words" errors when the intended text contained words that the speaker never actually said due to being interrupted or cut off.

## Tightening entity verification specificity

Spelled-out codes and reference/voucher IDs received more precise verification guidance. For spelled-out codes, the instruction was added to "verify EVERY letter and digit individually," with an example showing that "K O L T S F" vs "K O L T S S F" is an error. For reference IDs, "verify each segment" was added, with "M E L" vs "M E A L" as an example error. This addressed the judge being too lenient on partial matches in alphanumeric sequences.

## Reducing judge hallucination in evaluations

Two additions addressed cases where the judge itself was fabricating or assuming what was spoken:

- The "What to ignore" list was expanded to include "Interruption tags or metadata markers not present in audio" — explicitly telling the judge not to expect these in the audio.
- A new instruction was added: "IMPORTANT: Only rate what you clearly hear. If you cannot clearly make out a word or entity, note the uncertainty in your explanation rather than guessing. Do not fabricate or assume what was spoken." This was driven by cases where the judge was confidently asserting it heard specific letters or digits when the audio was ambiguous.

A minor pronunciation example was added ("Ms." vs "Miss" is acceptable) to calibrate the judge on what does vs. does not count as an identity-changing variation.
