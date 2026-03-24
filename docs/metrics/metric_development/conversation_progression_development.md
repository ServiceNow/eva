# Conversation Progression Development

## Starting point: general checklist with a 1-3 scale

The first version used a simple checklist approach: evaluate the entire conversation for whether the assistant made progress, avoided repetitive tool calls, avoided redundant statements, avoided repetitive questions, maintained context retention, and asked appropriate clarifying questions. A note exempted end-of-conversation recaps from redundancy penalties. The rating scale was 3 (consistently moved forward), 2 (some progress with issues), 1 (failed to move forward). Output was a single explanation + rating JSON.

## Restructuring into four explicit dimensions with flags

The first major revision replaced the general checklist with four mutually exclusive dimensions, each evaluated as a binary flag (issue present or not):

- **Unnecessary Tool Calls:** Tool call actions only. IS/Is NOT flag lists were added. A caveat: 3+ unnecessary tool calls count as 2 flags.
- **Information Loss:** The assistant forgetting or ignoring established facts.
- **Redundant Statements:** The assistant repeating its own prior output.
- **Question Quality:** Poorly formed questions or failure to ask necessary clarifying questions.

Each dimension received explicit "IS a flag" and "Is NOT a flag" sections with concrete examples. Disambiguation rules were added between dimensions e.g., re-asking for user-provided info goes under Information Loss not Redundant Statements; a forgotten tool result leading to an unnecessary tool call goes under Unnecessary Tool Calls.

The rating aggregation was mechanical: 3 = zero flags, 2 = one flag, 1 = two+ flags. The output schema changed to per-dimension flagged/evidence pairs.

## Reordering output schema

A minor change reordered the JSON output to put "evidence" before "flagged" in each dimension. This was to encourage the judge to write its reasoning before committing to a flag decision (chain-of-thought ordering). We observed that sometimes the judge would have outputted a score, and then have a different opinion while reasoning, resulting in an unreliable rating.

## Adding scope boundaries with faithfulness

The next major revision added several important scoping rules:

**Faithfulness boundary:** An explicit "IMPORTANT — Scope boundary with faithfulness" section was added. The prompt now states that conversation progression does NOT evaluate whether the assistant followed policies, complied with user constraints, or acted faithfully to instructions. If an issue is primarily about violating a policy (e.g., rebooking when the user said not to, not disclosing fees), it should NOT be flagged here. Only flag issues where the conversational choices themselves were inefficient or counterproductive.

This addressed widespread over-flagging where the conversation progression judge was penalizing policy violations that belonged under the faithfulness metric.

## Adding voice conversation context

Two new "IMPORTANT" sections were added:

- **Voice conversation context:** Speech recognition errors are common in voice conversations. When the assistant repeats a request because the previous attempt was misheard or garbled, this is expected voice interface behavior, not a progression issue.
- **Interruption tags:** Six specific interruption tag types were documented (same set as TTS fidelity). The judge was told to treat these as natural voice phenomena and only flag an issue if the interruption caused observable consequences (e.g., information loss because critical details in cut-off speech were never restated).

## Refining each dimension

Multiple refinements were made to reduce false positives and clarify edge cases across dimensions:

### Unnecessary Tool Calls

Added two new "Is NOT a flag" items; tool calls executed prematurely (before user confirmed) are a faithfulness issue, not progression; tool calls following standard agent instructions (e.g., automatically carrying over seat assignments when rebooking) are not unnecessary even if the user didn't explicitly request them. The 3+ unnecessary tool calls caveat was changed from "count as 2 flags" to "this dimension should be rated 1."

### Information Loss

Added clarification that when the transcript shows the user clearly provided information (no speech recognition garbling) but the assistant asks them to repeat it, this IS information loss. Added a "Is NOT a flag" item: the assistant acting on information that contradicts what the user said due to a faithfulness/policy violation should be flagged under faithfulness, not here.

New disambiguation rule: if the assistant proceeds with an action contradicting the user's preference (e.g., rebooking instead of standby), this is faithfulness, not information loss - only flag here if the assistant clearly forgot, not if it chose to override.

### Redundant Statements

Multiple recaps were addressed. Only the final recap at the very end is exempt; earlier recaps that restate already-communicated information are still flagged.

New "Is NOT a flag" items:
- Repeating information when the user explicitly requested confirmation
- Re-explaining a policy when the user continues to dispute (but verbatim repetition across turns is still flagged; the assistant should vary phrasing)
- Repeating a request when speech recognition errors clearly caused the previous attempt to fail (with explicit instruction that this exception doesn't apply when there's no evidence of ASR failure in the transcript)

### Question Quality

Added a new flag: taking an irreversible action without confirming when user input is ambiguous or contradicts system data.

Added "Is NOT a flag" items:
- Not disclosing fees before an action is a faithfulness concern, not progression
- Referencing information from agent instructions without a tool call is expected — only flag if the information was genuinely unknown

## Moving from binary flags to per-dimension 1-2-3 ratings

The rating system was substantially reworked. Instead of binary flags with mechanical aggregation, each dimension now receives its own 1-2-3 rating:

- **3** = No issue in this dimension
- **2** = A single isolated issue that doesn't significantly impact flow, or a borderline case
- **1** = Multiple instances of the same issue, or a single severe issue that clearly derailed the conversation

The overall rating considers both severity and breadth:

- **3** = All dimensions rated 3
- **2** = One or two dimensions flagged at rating 2, none at 1
- **1** = Any dimension rated 1, OR three+ dimensions flagged (even if individually minor)

This replaced the pure flag-count aggregation, allowing the judge to distinguish between a single inconsequential redundant statement (rating 2) and repeated information loss that stalled the conversation (rating 1).

## Requiring evidence even when not flagged

The output format was updated to require the "evidence" field to ALWAYS contain 1-2 sentences, even when flagged is false. The instruction was changed from "'None' if not flagged" to "REQUIRED: cite transcript examples if flagged, or explain why clean if not." This forced the judge to demonstrate it actually evaluated each dimension rather than defaulting to no-flag with empty reasoning.

Finally, the prompt was updated to note that the judge should consider both the assistant agent instructions and the evaluation dimensions.
