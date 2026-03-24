# Faithfulness Development

## Starting point: binary scoring

The first version of the faithfulness judge used a simple binary score: 1 (faithful) or 0 (violations present). It evaluated four dimensions — Information Accuracy, Tool Call Grounding, Policy & Instruction Adherence, and Disambiguation & Conflict Resolution.

## Moving to a 1/2/3 scale

The first major change was replacing the binary scale with a three-tier rating: 3 (no issues), 2 (minor or ambiguous issues), 1 (clear violations). A faithfulness reviewer prompt was also introduced at this point to resolve disagreements between different judge models on the same samples, and improve the prompt automatically based on mistakes that some judges would make.

The motivation was that binary scoring couldn't handle borderline cases that would not affect the user experience as much. For example, a tool call that used a bad parameter but failed harmlessly and was immediately self-corrected is meaningfully different from a clear hallucination that would impact financially the user — but both would have scored 0 under the old scheme. The middle tier gave a place for these ambiguous cases.

## Adding specificity to each dimension

The next round of changes fleshed out each dimension with more precise guidance, driven by patterns in the data where judges were either over- or under-flagging.

### Information Accuracy

Clarified that hedged caveats ("you may want to verify at the counter") and general domain knowledge are not hallucinations. Made financial figures (fares, fees, refunds) an always-flag category given their high-stakes nature.

### Tool Call Grounding

Added that placeholder values like "?", "UNKNOWN", and "N/A" count as fabrication. Added enum/categorical parameter verification — e.g., a 239-minute delay doesn't qualify for a "delay over 4 hours" category since 239 < 240. Added a mitigating factor: an ungrounded call that fails harmlessly and is self-corrected should be rated 2, not 1.

### Policy & Instruction Adherence

Added temporal sequencing as a concept — when instructions say "explain before acting," the assistant must actually pause between read and write operations; summarizing after the fact doesn't satisfy a prior-confirmation requirement. Irreversible actions (cancellations, refunds, rebookings) without disclosing financial implications became explicit rating-1 violations. Also added fabricated policies, brand misidentification, and claiming capabilities not supported by any available tool.

### Disambiguation

Clarified the boundary — failing to call additional tools that could have been useful is a task completeness issue, not a faithfulness issue. Only penalize if the assistant misrepresents information it actually has.

A Key Distinctions section was also added to help choose between ratings 1 and 2: the deciding factor is whether the violation materially affects the outcome (financial consequences, irreversible actions, incorrect information that misleads the customer).

## Restructuring into explicit per-dimension flags

The output schema was then changed to require a flagged/evidence pair per dimension rather than a single overall explanation. This forced the judge to evaluate each dimension independently before producing an overall rating.

Alongside that, the five dimensions were formalized with structured "IS a flag" / "Is NOT a flag" / "Disambiguation from other dimensions" sections for each. A key structural decision was moving hallucination to last, and scoping it explicitly to "information with no source at all — not already covered by the preceding dimensions." This fixed a recurrent false positive: judges were flagging hallucination when the actual issue was a distorted tool result, which belongs under misrepresenting_tool_result. The explicit routing made that distinction clear.

## Adding date/time context

The judge was flagging hallucination when the assistant correctly referenced the current date, because the date didn't appear in any tool result or agent instruction. A current_date_time field was added to the prompt context, making it a named grounded source.

## Reducing false positives

The next revision addressed patterns of false positives that had emerged from reviewing judge outputs. Each dimension received more precise guidance.

### Fabricating tool parameters

Had too broad a definition. Standard geographic/industry mappings ("Chicago O'Hare" to "ORD") are grounded, not fabricated. Parameters derived from policy entitlements (e.g., waiving a change fee when the passenger's elite status entitles them to it) are grounded. Values computed from prior tool results via simple arithmetic are also grounded. A verification checklist was added: before flagging a parameter, confirm it cannot be traced to any source — user statements, tool results, policy, arithmetic, or standard domain mappings.

### Misrepresenting tool results

Had over-flagging on fee omissions and time formats. Fee omission is only a violation when a charge was actually collected — failing to mention a $0 or waived fee is not misrepresentation. Time format conversions (16:40 = 4:40 PM) are not misrepresentation. Arithmetic errors were added as an explicit flag, with a verification requirement: carefully compute fare differences, check 24h/12h conversions, and cross-reference all tool fields before flagging a discrepancy.

### Violating policies

Was generating false positives on the "explain before acting" rule. Temporal sequencing was scoped to consequential actions only (financial or irreversible). Proportionality was made explicit: severity of a violation should match its potential negative impact on the customer. If fees turn out to be $0 or waived, not mentioning them upfront is not a violation. Guidance was also added for when two policy paths could apply: if a flight hasn't departed and the passenger is within the policy window, applying the more favorable path is not a violation.

### Hallucination

Was still triggering on system-provided context. System context (including the date/time field) is a valid grounded source. A critical verification step was added: before flagging hallucination, check all tool responses, user utterances, agent instructions, and system context fields — do not assume fields are empty without verifying.

## Adding examples to failure to disambiguate

In the final round, more examples were added around how the model should behave when faced with conflicting information from the users, typically due to transcript issues. It was specified what is considered a failure to disambiguate vs normal behavior in the face of conflicting information.
