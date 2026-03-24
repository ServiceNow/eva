# Conciseness Development

## Metric Development Process

The core evaluation question is whether each assistant turn stays within the bounds of what an average listener can comfortably process and retain in working memory during real-time conversation. This metric was developed starting from the central user experience problem of cognitive overload - the point at which a spoken response exceeds what a listener can absorb in a single pass. We identified the key failure patterns that cause this and the prompt was then refined through an iterative process: running conversations across multiple agent configurations, manually reviewing data transcripts to identify turns where agents overwhelmed or lost the listener, discovering new patterns of unfair penalization, and adjusting the prompt to handle them correctly.

## Evaluation Approach

Conciseness is evaluated at the per-turn level using an LLM judge. The full conversation transcript including user turns, tool calls, and tool responses, is provided for context, but only assistant-spoken content is rated. The judge is instructed to evaluate conciseness relative to conversational context. If additional wording or detail is clearly necessary for the user to understand or act on information, modest increases in verbosity are acceptable. Each turn receives a score on a 3-point integer scale:

- **3 (Highly Concise / No Cognitive Overload):** Clear, appropriately scoped for voice, and comfortably digestible in real time. A turn that delivers a few closely related facts as part of a single transactional step (e.g., confirming booking details) still qualifies if the listener can absorb it in one pass.
- **2 (Adequate but Not Optimally Concise):** One minor failure mode is present, but the response remains reasonably processable and does not meaningfully overwhelm the listener. Reserved for turns where specific content should have been omitted or deferred.
- **1 (Not Concise / Causes Cognitive Overload):** One or more significant failure modes are present which materially increases cognitive load for the user and would hinder comprehension in spoken conversation.

Per-turn scores are averaged across the conversation to produce a normalized score (3→1.0, 2→0.5, 1→0.0).

## Failure Modes

- **excess_information_density** - Too many distinct facts/options/numbers at once
- **over_enumeration_or_list_exhaustion** - Reading out long lists instead of summarizing
- **contextually_disproportionate_detail** - More explanation than the situation warrants
- **verbosity_or_filler** - Unnecessary wording, repetition within the same turn, hedging

## Allowed Exceptions (Not Penalized)

- Phonetic confirmation of codes (NATO alphabet) when user misheard
- Delivery of essential reference codes (vouchers, booking references)
- Slightly longer end-of-call wrap-up with recap/confirmation
- Essential information (confirmation codes, voucher numbers) regardless of length
- Interrupted/truncated content caused by user interruptions
