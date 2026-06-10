# Data

## Data Structure

Each test case (aka scenario) in EVA is an evaluation record that specifies:

- User Goal — What the caller is trying to accomplish, with a detailed scenario of a highly specific user goal with an exact decision tree that guides the user simulator through the conversation, leaving no ambiguity about the intended outcome.
- User Persona — How the caller should behave — their speaking style, patience level, and personality traits.
- Scenario Database — The backend data the agent's tools will query.
- Ground Truth — The expected final state of the scenario database after a successful conversation.

This structure makes tests reproducible. The same evaluation record always presents the same scenario, so you can compare different agents or different versions of the same agent on identical tasks.

More details on data domains, and dataset construction and validation can be found in our paper.
