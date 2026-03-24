import demoData from './demo_data.json';

export type EntryType = 'user' | 'assistant' | 'tool_call' | 'tool_response';

export interface ConversationEntry {
  type: EntryType;
  content: string;
  toolName?: string;
  toolParams?: Record<string, unknown>;
  toolResponse?: Record<string, unknown>;
  toolStatus?: 'success' | 'error';
  turnId?: number;
}

export interface UserGoal {
  highLevelGoal: string;
  startingUtterance: string;
  decisionTree: {
    mustHaveCriteria: string[];
    niceToHaveCriteria: string[];
    negotiationBehavior: string[];
    resolutionCondition: string;
    failureCondition: string;
    escalationBehavior: string;
    edgeCases: string[];
  };
  informationRequired: Record<string, unknown>;
}

const rawGoal = demoData.userGoal;

export const userGoal: UserGoal = {
  highLevelGoal: rawGoal.high_level_user_goal,
  startingUtterance: rawGoal.starting_utterance,
  decisionTree: {
    mustHaveCriteria: rawGoal.decision_tree.must_have_criteria,
    niceToHaveCriteria: rawGoal.decision_tree.nice_to_have_criteria ?? [],
    negotiationBehavior: rawGoal.decision_tree.negotiation_behavior,
    resolutionCondition: rawGoal.decision_tree.resolution_condition,
    failureCondition: rawGoal.decision_tree.failure_condition,
    escalationBehavior: rawGoal.decision_tree.escalation_behavior,
    edgeCases: rawGoal.decision_tree.edge_cases,
  },
  informationRequired: rawGoal.information_required,
};

export const userPersona: string = demoData.userPersona;

// Convert conversation_trace to ConversationEntry[]
function convertTrace(trace: typeof demoData.conversationTrace): ConversationEntry[] {
  const entries: ConversationEntry[] = [];

  for (let i = 0; i < trace.length; i++) {
    const item = trace[i];

    if (item.type === 'transcribed') {
      entries.push({
        type: 'user',
        content: (item as { content?: string }).content ?? '',
        turnId: item.turn_id,
      });
    } else if (item.type === 'intended') {
      entries.push({
        type: 'assistant',
        content: (item as { content?: string }).content ?? '',
        turnId: item.turn_id,
      });
    } else if (item.type === 'tool_call') {
      const toolItem = item as { tool_name?: string; parameters?: Record<string, unknown> };
      // Look ahead for the matching tool_response
      const next = trace[i + 1];
      const nextTool = next as { type?: string; tool_name?: string; tool_response?: Record<string, unknown> } | undefined;
      let toolResponse: Record<string, unknown> | undefined;
      let toolStatus: 'success' | 'error' | undefined;

      if (nextTool?.type === 'tool_response' && nextTool?.tool_name === toolItem.tool_name) {
        const resp = nextTool.tool_response ?? {};
        toolResponse = resp;
        toolStatus = (resp as { status?: string }).status === 'success' ? 'success' : 'error';
        // We'll add both tool_call and tool_response as a single entry
        entries.push({
          type: 'tool_call',
          content: '',
          toolName: toolItem.tool_name,
          toolParams: toolItem.parameters,
          turnId: item.turn_id,
        });
        entries.push({
          type: 'tool_response',
          content: '',
          toolName: toolItem.tool_name,
          toolResponse,
          toolStatus,
          turnId: item.turn_id,
        });
        i++; // Skip the tool_response entry
      } else {
        entries.push({
          type: 'tool_call',
          content: '',
          toolName: toolItem.tool_name,
          toolParams: toolItem.parameters,
          turnId: item.turn_id,
        });
      }
    }
    // tool_response entries without a preceding tool_call are skipped (handled above)
  }

  return entries;
}

export const exampleConversation: ConversationEntry[] = convertTrace(demoData.conversationTrace);
