# Data

## Data Structure

Each test case (aka scenario) in EVA is an evaluation record that specifies:

- User Goal — What the caller is trying to accomplish, with a detailed scenario of a highly specific user goal with an exact decision tree that guides the user simulator through the conversation, leaving no ambiguity about the intended outcome.
- User Persona — How the caller should behave — their speaking style, patience level, and personality traits.
- Scenario Database — The backend data the agent's tools will query.
- Ground Truth — The expected final state of the scenario database after a successful conversation.

This structure makes tests reproducible. The same evaluation record always presents the same scenario, so you can compare different agents or different versions of the same agent on identical tasks.

## Data Domain: Flight Rebooking

We believe flight rebooking is a strong initial domain for voice agent evaluation for several reasons. It is a high-stakes, real-world scenario where users regularly interact with voice agents under time pressure. It requires temporal reasoning (understanding IRROPS vs. voluntary changes, same day vs. future travel), complex policy following (voucher eligibility, rebooking options), and disambiguation (handling user constraints and preferences). And critically, it is heavily dependent on accurate transcription of named entities — confirmation codes, flight numbers, passenger names, dates — making it a demanding and realistic test of the full pipeline.

The initial EVA dataset covers 50 scenarios across the following categories:

- IRROPS rebooking: airline-initiated disruptions where the user is entitled to rebooking at no cost.
- Voluntary changes: user-initiated change requests subject to fare difference and change fees.
- Missed connections: scenarios involving cascading disruptions across legs.
- Same-day changes and standby: time-sensitive requests requiring specific policy handling.
- Adversarial scenarios: users attempting to obtain compensation (meal vouchers, hotel vouchers) they are not entitled to under policy.

Scenarios are designed to test temporal reasoning, policy following, constraint satisfaction, and named entity handling. Each scenario is a JSONL record specifying the user's goal, the initial database state (flights, bookings, disruptions), the expected end state, and the key named entities the user must communicate.

## Data Generation and Validation

Producing a reliable benchmark dataset for voice agents requires more than generating plausible scenarios — it requires guaranteeing that every record is internally consistent, that policies are unambiguous, that user goals have exactly one correct resolution, and that any model failure on the task is attributable to actual model error rather than dataset noise. We invested heavily in this.

Generation pipeline. Scenarios were generated using SyGra - a graph-based data generation pipeline. User goals, mock scenario databases (flights, bookings, disruptions), and expected end-state databases were generated together with frequent human review and regeneration cycles. Generating these jointly is important: the expected end state must be consistent with both the user goal and the scenario database, and small inconsistencies (e.g., a flight in the user's goal that doesn't exist in the database) can silently corrupt evaluation signal.

Human review and editing. Following the generation pipeline, we conducted multiple rounds of comprehensive manual review passes over all 50 records. This included verifying policy consistency across scenarios, checking that user goals were specific enough to admit exactly one correct resolution, confirming that expected end states were correct, and editing records where generated content was ambiguous or subtly inconsistent.

Model-based stress testing. As a final validation step, we ran three frontier models — Claude Opus 4.6, Gemini 2.1 Pro Preview, and GPT-5.2 — on a text-only version of each task (i.e., bypassing the audio pipeline entirely, giving the model the conversation transcript directly). We then carefully examined every record where any model scored 0 on task completion. For each such failure, we investigated whether the failure was due to genuine model error or instead reflected a dataset issue: an ambiguous policy, an underspecified user goal, a bug in the mock tool executor, or an inconsistency between the scenario database and the expected end state. Records with identified dataset issues were corrected or removed. This process gave us high confidence that task completion failures in the full audio evaluation reflect real agent errors — not evaluation artifacts. 
