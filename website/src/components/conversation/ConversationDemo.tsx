import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Section } from '../layout/Section';
import { motion } from 'framer-motion';
import { exampleConversation, userGoal, userPersona, type ConversationEntry } from '../../data/conversationData';
import { evaAMetrics, evaXMetrics, diagnosticMetrics, type DemoMetricScore } from '../../data/demoMetricsData';
import {
  User, Bot, Wrench, Volume2, VolumeX, ChevronDown, ChevronRight,
  CheckCircle2, AlertTriangle, MessageSquare, Shield, Target,
  Activity, Search, Play, Pause,
} from 'lucide-react';

/* ─── Audio Player ─── */

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function AudioPlayer({ src }: { src: string }) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const progressRef = useRef<HTMLDivElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isMuted, setIsMuted] = useState(false);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const onTimeUpdate = () => setCurrentTime(audio.currentTime);
    const onLoadedMetadata = () => setDuration(audio.duration);
    const onEnded = () => setIsPlaying(false);

    audio.addEventListener('timeupdate', onTimeUpdate);
    audio.addEventListener('loadedmetadata', onLoadedMetadata);
    audio.addEventListener('ended', onEnded);
    return () => {
      audio.removeEventListener('timeupdate', onTimeUpdate);
      audio.removeEventListener('loadedmetadata', onLoadedMetadata);
      audio.removeEventListener('ended', onEnded);
    };
  }, []);

  const togglePlay = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;
    if (isPlaying) {
      audio.pause();
    } else {
      audio.play();
    }
    setIsPlaying(!isPlaying);
  }, [isPlaying]);

  const toggleMute = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.muted = !isMuted;
    setIsMuted(!isMuted);
  }, [isMuted]);

  const handleProgressClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const audio = audioRef.current;
    const bar = progressRef.current;
    if (!audio || !bar) return;
    const rect = bar.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    audio.currentTime = ratio * duration;
  }, [duration]);

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div className="rounded-xl bg-bg-secondary border border-border-default p-4">
      <audio ref={audioRef} preload="metadata">
        <source src={src} type="audio/wav" />
      </audio>

      <div className="flex items-center gap-3 mb-3">
        <Volume2 className="w-5 h-5 text-purple-light" />
        <span className="text-sm font-semibold text-text-primary">Conversation Audio</span>
        <span className="text-[10px] px-2 py-0.5 rounded-full bg-bg-tertiary text-text-muted border border-border-default">Recording</span>
      </div>

      <div className="flex items-center gap-3">
        {/* Play/Pause */}
        <button
          onClick={togglePlay}
          className="w-10 h-10 rounded-full bg-purple/20 hover:bg-purple/30 flex items-center justify-center transition-colors flex-shrink-0"
        >
          {isPlaying ? (
            <Pause className="w-5 h-5 text-purple-light" />
          ) : (
            <Play className="w-5 h-5 text-purple-light ml-0.5" />
          )}
        </button>

        {/* Time */}
        <span className="text-xs font-mono text-text-muted w-10 text-right flex-shrink-0">
          {formatTime(currentTime)}
        </span>

        {/* Progress bar */}
        <div
          ref={progressRef}
          onClick={handleProgressClick}
          className="flex-1 h-2 bg-bg-tertiary rounded-full cursor-pointer group relative"
        >
          <div
            className="h-full bg-purple rounded-full transition-[width] duration-100 relative"
            style={{ width: `${progress}%` }}
          >
            <div className="absolute right-0 top-1/2 -translate-y-1/2 w-3.5 h-3.5 rounded-full bg-purple-light border-2 border-bg-secondary opacity-0 group-hover:opacity-100 transition-opacity" />
          </div>
        </div>

        {/* Duration */}
        <span className="text-xs font-mono text-text-muted w-10 flex-shrink-0">
          {duration > 0 ? formatTime(duration) : '--:--'}
        </span>

        {/* Mute */}
        <button
          onClick={toggleMute}
          className="w-8 h-8 rounded-lg hover:bg-bg-tertiary flex items-center justify-center transition-colors flex-shrink-0"
        >
          {isMuted ? (
            <VolumeX className="w-4 h-4 text-text-muted" />
          ) : (
            <Volume2 className="w-4 h-4 text-text-muted" />
          )}
        </button>
      </div>
    </div>
  );
}

/* ─── Issue Annotations ─── */

interface TurnIssue {
  message: string;
}

/**
 * Map of conversation entry indices (in exampleConversation) to known issues.
 * These are hand-annotated to highlight specific failure modes in the demo.
 */
function buildIssueMap(entries: ConversationEntry[]): Map<number, TurnIssue> {
  const map = new Map<number, TurnIssue>();
  for (let idx = 0; idx < entries.length; idx++) {
    const e = entries[idx];
    // Turn 3 user: entity transcription error (confirmation code garbled)
    if (e.turnId === 3 && e.type === 'user' && e.content.includes('The code is 6-8-1-1')) {
      map.set(idx, { message: 'Entity transcription error — confirmation code was garbled by STT' });
    }
    // Turn 3 assistant: "(Waiting for the user's response.)"
    if (e.turnId === 3 && e.type === 'assistant' && e.content.includes('Waiting for the user')) {
      map.set(idx, { message: 'Voice agent spoke too soon and said text that should not be spoken out loud' });
    }
    // Turn 4 user: major transcription error (L-A-L-A repeating)
    if (e.turnId === 4 && e.type === 'user' && e.content.includes('L-A-L-A-L-A')) {
      map.set(idx, { message: 'Major transcription error — user speech was severely garbled by STT' });
    }
    // Turn 5 assistant: conciseness issue (first long response with options)
    if (e.turnId === 5 && e.type === 'assistant' && e.content.includes('found a few earlier flights')) {
      map.set(idx, { message: 'Conciseness issue - information overload' });
    }
    // Turn 6 assistant: faithfulness + conciseness issue (second long response)
    if (e.turnId === 6 && e.type === 'assistant' && e.content.includes('Sure, let me list')) {
      map.set(idx, { message: 'Information overload (conciseness issue) and wrong calculation of credit amount (faithfulness issue - misrepresenting tool result)' });
    }
    // Turn 7 tool_call: search_rebooking_options called again with same params
    if (e.turnId === 7 && e.type === 'tool_call' && e.toolName === 'search_rebooking_options') {
      map.set(idx, { message: 'Conversation progression issue — tool already called with the same parameters' });
    }
  }
  return map;
}

function IssueIcon({ issue }: { issue: TurnIssue }) {
  return (
    <div className="relative group/issue inline-flex items-center ml-1.5">
      <AlertTriangle className="w-4 h-4 text-amber cursor-help" />
      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-72 px-3 py-2.5 rounded-lg bg-bg-tertiary border border-border-default text-xs text-text-primary leading-relaxed opacity-0 invisible group-hover/issue:opacity-100 group-hover/issue:visible transition-all z-[100] shadow-xl pointer-events-none">
        {issue.message}
        <div className="absolute top-full left-1/2 -translate-x-1/2 w-2 h-2 bg-bg-tertiary border-r border-b border-border-default rotate-45 -mt-1" />
      </div>
    </div>
  );
}

/* ─── Chat Message ─── */

function ChatMessage({ entry, index, issue }: { entry: ConversationEntry; index: number; issue?: TurnIssue }) {
  const isUser = entry.type === 'user';

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.3, delay: index * 0.03 }}
      className={`flex gap-3 ${isUser ? '' : 'flex-row-reverse'}`}
    >
      <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
        isUser ? 'bg-blue/20' : 'bg-purple/20'
      }`}>
        {isUser ? <User className="w-4 h-4 text-blue-light" /> : <Bot className="w-4 h-4 text-purple-light" />}
      </div>
      <div className={`max-w-[85%] rounded-xl px-4 py-3 ${
        isUser
          ? 'bg-blue/10 border border-blue/20'
          : 'bg-purple/10 border border-purple/20'
      }`}>
        <div className={`text-[10px] font-semibold mb-1 flex items-center ${isUser ? 'text-blue-light' : 'text-purple-light'}`}>
          {isUser ? 'User (Transcribed)' : 'Voice Agent (Intended)'}
          {issue && <IssueIcon issue={issue} />}
        </div>
        <p className="text-sm text-text-primary leading-relaxed whitespace-pre-line">{entry.content}</p>
      </div>
    </motion.div>
  );
}

/* ─── Tool Call Block ─── */

function ToolCallBlock({ entry, responseEntry, index, issue }: { entry: ConversationEntry; responseEntry?: ConversationEntry; index: number; issue?: TurnIssue }) {
  const [showResponse, setShowResponse] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.3, delay: index * 0.03 }}
      className="mx-4 sm:mx-8"
    >
      <div className="rounded-xl bg-bg-primary border border-border-default">
        <div className="flex items-center gap-2 px-4 py-2.5 bg-bg-tertiary border-b border-border-default">
          <Wrench className="w-3.5 h-3.5 text-amber" />
          <span className="text-xs font-semibold font-mono text-amber">{entry.toolName}</span>
          {issue && <IssueIcon issue={issue} />}
          <div className="flex-1" />
          {responseEntry?.toolStatus === 'success' && (
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 font-medium border border-emerald-500/20">
              Success
            </span>
          )}
          {responseEntry?.toolStatus === 'error' && (
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-red-500/10 text-red-400 font-medium border border-red-500/20">
              Error
            </span>
          )}
        </div>
        <div className="p-3">
          <div className="text-[10px] text-text-muted font-semibold uppercase tracking-wider mb-1.5">Parameters</div>
          <div className="space-y-1">
            {entry.toolParams && Object.entries(entry.toolParams).map(([key, value]) => (
              <div key={key} className="flex gap-2 text-xs">
                <span className="text-text-muted font-mono">{key}:</span>
                <span className="text-text-secondary font-mono">{JSON.stringify(value)}</span>
              </div>
            ))}
          </div>
        </div>
        {responseEntry?.toolResponse && (
          <div className="border-t border-border-default/50">
            <button
              onClick={() => setShowResponse(!showResponse)}
              className="w-full flex items-center gap-2 px-3 py-2 text-[10px] text-text-muted font-semibold uppercase tracking-wider hover:bg-bg-hover/30 transition-colors"
            >
              {showResponse ? (
                <ChevronDown className="w-3 h-3" />
              ) : (
                <ChevronRight className="w-3 h-3" />
              )}
              Response
            </button>
            {showResponse && (
              <div className="px-3 pb-3">
                <pre className="text-xs text-text-secondary font-mono leading-relaxed max-h-48 overflow-y-auto overflow-x-auto bg-bg-tertiary rounded-lg p-3">
                  {JSON.stringify(responseEntry.toolResponse, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}
      </div>
    </motion.div>
  );
}

/* ─── Collapsible Section ─── */

function CollapsibleSection({ title, icon: Icon, children, defaultOpen = false }: { title: string; icon: React.ComponentType<{ className?: string }>; children: React.ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div>
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 rounded-lg border border-border-default bg-bg-primary px-3 py-2 hover:bg-bg-hover/30 transition-colors"
      >
        <Icon className="w-3.5 h-3.5 text-text-muted" />
        <span className="text-xs font-semibold text-text-muted uppercase tracking-wider flex-1 text-left">{title}</span>
        <ChevronDown className={`w-3.5 h-3.5 text-text-muted transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        <div className="mt-2 bg-bg-primary rounded-lg p-3">
          {children}
        </div>
      )}
    </div>
  );
}

/* ─── User Goal Card ─── */

function UserGoalCard() {
  const [expanded, setExpanded] = useState(true);
  // Format information_required for display
  const infoEntries = Object.entries(userGoal.informationRequired).map(([key, value]) => {
    const displayKey = key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
    const displayValue = typeof value === 'object' ? JSON.stringify(value) : String(value);
    return [displayKey, displayValue] as const;
  });

  return (
    <div className="rounded-xl border border-border-default bg-bg-secondary p-5">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-3 mb-0 hover:opacity-80 transition-opacity"
      >
        <div className="w-10 h-10 rounded-full bg-blue/20 flex items-center justify-center flex-shrink-0">
          <User className="w-5 h-5 text-blue-light" />
        </div>
        <div className="flex-1 text-left">
          <div className="text-base font-semibold text-text-primary">User Goal</div>
          <div className="text-[10px] text-text-muted uppercase tracking-wider">Scenario Briefing</div>
        </div>
        <ChevronDown className={`w-4 h-4 text-text-muted transition-transform ${expanded ? 'rotate-180' : ''}`} />
      </button>

      {expanded && <div className="mt-4">

      <div className="border-l-2 border-blue/40 pl-4 mb-5">
        <p className="text-sm text-text-primary leading-relaxed">{userGoal.highLevelGoal}</p>
      </div>

      {/* Persona */}
      <div className="mb-4 bg-bg-tertiary rounded-lg p-3">
        <div className="text-[10px] font-semibold text-text-muted uppercase tracking-wider mb-1.5">Persona</div>
        <p className="text-xs text-text-secondary leading-relaxed">{userPersona}</p>
      </div>

      {/* Must-Have Criteria */}
      <div className="mb-4">
        <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2.5">Must-Have Criteria</div>
        <div className="space-y-2">
          {userGoal.decisionTree.mustHaveCriteria.map((criterion, i) => (
            <div key={i} className="flex gap-2 items-start">
              <CheckCircle2 className="w-3.5 h-3.5 text-blue-light mt-0.5 flex-shrink-0" />
              <span className="text-xs text-text-secondary leading-relaxed">{criterion}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Collapsible sections */}
      <div className="space-y-2">
        <CollapsibleSection title="Negotiation Behavior" icon={MessageSquare}>
          <div className="space-y-2.5">
            {userGoal.decisionTree.negotiationBehavior.map((behavior, i) => (
              <p key={i} className="text-xs text-text-secondary leading-relaxed">{behavior}</p>
            ))}
          </div>
        </CollapsibleSection>

        <CollapsibleSection title="Resolution & Failure" icon={Target}>
          <div className="space-y-3">
            <div>
              <div className="text-[10px] font-semibold text-emerald-400 uppercase tracking-wider mb-1">Resolution</div>
              <p className="text-xs text-text-secondary leading-relaxed">{userGoal.decisionTree.resolutionCondition}</p>
            </div>
            <div>
              <div className="text-[10px] font-semibold text-red-400 uppercase tracking-wider mb-1">Failure</div>
              <p className="text-xs text-text-secondary leading-relaxed">{userGoal.decisionTree.failureCondition}</p>
            </div>
          </div>
        </CollapsibleSection>

        <CollapsibleSection title="Escalation" icon={Shield}>
          <p className="text-xs text-text-secondary leading-relaxed">{userGoal.decisionTree.escalationBehavior}</p>
        </CollapsibleSection>

        <CollapsibleSection title="Edge Cases" icon={AlertTriangle}>
          <div className="space-y-2">
            {userGoal.decisionTree.edgeCases.map((edgeCase, i) => (
              <p key={i} className="text-xs text-text-secondary leading-relaxed">{edgeCase}</p>
            ))}
          </div>
        </CollapsibleSection>

        <CollapsibleSection title="Scenario Details" icon={User}>
          <div className="space-y-2">
            {infoEntries.map(([key, value]) => (
              <div key={key} className="flex justify-between gap-3">
                <span className="text-[11px] text-text-muted flex-shrink-0">{key}</span>
                <span className="text-[11px] text-text-primary font-medium text-right break-all">{value}</span>
              </div>
            ))}
          </div>
        </CollapsibleSection>
      </div>
      </div>}
    </div>
  );
}

/* ─── Agent Tools Panel ─── */

interface AgentToolDef {
  name: string;
  description: string;
  toolType: 'read' | 'write' | 'system';
}

const agentTools: AgentToolDef[] = [
  { name: 'get_reservation', description: 'Retrieve flight reservation using confirmation number and passenger last name', toolType: 'read' },
  { name: 'get_flight_status', description: 'Get flight info including status, delays, cancellations, and gate information', toolType: 'read' },
  { name: 'get_disruption_info', description: 'Get detailed disruption info for IRROPS handling and rebooking entitlements', toolType: 'read' },
  { name: 'search_rebooking_options', description: 'Search for available flights to rebook a passenger', toolType: 'read' },
  { name: 'rebook_flight', description: 'Rebook passenger(s) to a new flight (voluntary, IRROPS, or missed flight)', toolType: 'write' },
  { name: 'add_to_standby', description: 'Add passenger to standby list for a flight', toolType: 'write' },
  { name: 'assign_seat', description: 'Assign a seat to a passenger based on preference', toolType: 'write' },
  { name: 'add_baggage_allowance', description: 'Add checked baggage allowance to a flight segment', toolType: 'write' },
  { name: 'add_meal_request', description: 'Add or update special meal request for a passenger', toolType: 'write' },
  { name: 'issue_travel_credit', description: 'Issue a travel credit or future flight voucher', toolType: 'write' },
  { name: 'issue_hotel_voucher', description: 'Issue a hotel voucher for delays or disruptions', toolType: 'write' },
  { name: 'issue_meal_voucher', description: 'Issue a meal voucher for delays or disruptions', toolType: 'write' },
  { name: 'cancel_reservation', description: 'Cancel a flight booking', toolType: 'write' },
  { name: 'process_refund', description: 'Process a refund for a cancelled or eligible reservation', toolType: 'write' },
  { name: 'transfer_to_agent', description: 'Transfer the call to a live human agent', toolType: 'system' },
];

function computeToolUsage(entries: ConversationEntry[]): Map<string, { calls: number; success: number; error: number }> {
  const map = new Map<string, { calls: number; success: number; error: number }>();
  for (const entry of entries) {
    if (entry.type === 'tool_response' && entry.toolName) {
      const existing = map.get(entry.toolName) ?? { calls: 0, success: 0, error: 0 };
      existing.calls++;
      if (entry.toolStatus === 'success') existing.success++;
      else existing.error++;
      map.set(entry.toolName, existing);
    }
  }
  return map;
}

function ToolItem({ tool, isUsed, typeColors }: { tool: AgentToolDef; isUsed: boolean; typeColors: Record<string, string> }) {
  const [showDesc, setShowDesc] = useState(false);

  return (
    <div className={`rounded-lg border ${
      isUsed
        ? 'border-amber/30 bg-amber/5'
        : 'border-border-default bg-bg-primary opacity-60'
    }`}>
      <button
        onClick={() => setShowDesc(!showDesc)}
        className="w-full flex items-center gap-2 px-3 py-2 hover:opacity-80 transition-opacity"
      >
        <Wrench className={`w-3.5 h-3.5 flex-shrink-0 ${isUsed ? 'text-amber' : 'text-text-muted'}`} />
        <span className={`text-xs font-semibold font-mono flex-1 text-left ${isUsed ? 'text-text-primary' : 'text-text-muted'}`}>{tool.name}</span>
        <span className={`text-[9px] px-1.5 py-0.5 rounded-full font-medium border ${typeColors[tool.toolType]}`}>
          {tool.toolType}
        </span>
      </button>
      {showDesc && (
        <div className="px-3 pb-2.5 pt-0">
          <p className="text-xs text-text-secondary leading-relaxed">{tool.description}</p>
        </div>
      )}
    </div>
  );
}

function AgentToolsPanel() {
  const toolUsage = computeToolUsage(exampleConversation);
  const usedCount = agentTools.filter((t) => toolUsage.has(t.name)).length;

  const typeColors: Record<string, string> = {
    read: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20',
    write: 'bg-purple/10 text-purple-light border-purple/20',
    system: 'bg-amber/10 text-amber border-amber/20',
  };

  const [expanded, setExpanded] = useState(true);

  return (
    <div className="rounded-xl border border-border-default bg-bg-secondary p-5">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-3 mb-0 hover:opacity-80 transition-opacity"
      >
        <div className="w-10 h-10 rounded-full bg-amber/20 flex items-center justify-center flex-shrink-0">
          <Wrench className="w-5 h-5 text-amber" />
        </div>
        <div className="flex-1 text-left">
          <div className="text-base font-semibold text-text-primary">Agent Tools</div>
          <div className="text-[10px] text-text-muted uppercase tracking-wider">{usedCount} of {agentTools.length} used in this conversation</div>
        </div>
        <ChevronDown className={`w-4 h-4 text-text-muted transition-transform ${expanded ? 'rotate-180' : ''}`} />
      </button>

      {expanded && <div className="mt-4 space-y-1.5">
        {agentTools.map((tool) => {
          const isUsed = toolUsage.has(tool.name);

          return <ToolItem key={tool.name} tool={tool} isUsed={isUsed} typeColors={typeColors} />;
        })}
      </div>}
    </div>
  );
}

/* ─── Metric Score Badge ─── */

function ScoreBadge({ score, size = 'md' }: { score: number; size?: 'sm' | 'md' }) {
  const colorClass =
    score >= 0.8 ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' :
    score >= 0.5 ? 'bg-amber/10 text-amber border-amber/20' :
    'bg-red-500/10 text-red-400 border-red-500/20';

  const sizeClass = size === 'sm' ? 'text-[10px] px-1.5 py-0.5' : 'text-xs px-2 py-0.5';

  return (
    <span className={`${sizeClass} rounded-full font-semibold border ${colorClass}`}>
      {(score * 100).toFixed(0)}%
    </span>
  );
}

function MetricTypeBadge({ type }: { type: string }) {
  const labels: Record<string, string> = {
    deterministic: 'Deterministic',
    llm_judge: 'LLM Judge',
    lalm_judge: 'Audio Judge',
  };
  const colors: Record<string, string> = {
    deterministic: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20',
    llm_judge: 'bg-purple/10 text-purple-light border-purple/20',
    lalm_judge: 'bg-amber/10 text-amber border-amber/20',
  };

  return (
    <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-medium border ${colors[type] ?? 'bg-bg-tertiary text-text-muted border-border-default'}`}>
      {labels[type] ?? type}
    </span>
  );
}

/* ─── Metric Card ─── */

function MetricCard({ metric }: { metric: DemoMetricScore }) {
  const [expanded, setExpanded] = useState(false);
  const details = metric.details;

  const perTurnRatings = details.per_turn_ratings as Record<string, number> | undefined;
  const perTurnExplanations = details.per_turn_explanations as Record<string, string> | undefined;
  const perTurnTimingRatings = details.per_turn_judge_timing_ratings as Record<string, string> | undefined;
  const perTurnTimingExplanations = details.per_turn_judge_timing_explanations as Record<string, string> | undefined;
  const explanation = details.explanation as string | Record<string, unknown> | undefined;
  const perTurnEntityDetails = details.per_turn_entity_details as Record<string, { entities: Array<{ type: string; value: string; transcribed_value: string; correct: boolean; analysis: string }>; summary: string }> | undefined;

  // For faithfulness / conversation_progression, explanation is an object with dimensions
  const dimensionExplanation = typeof explanation === 'object' && explanation !== null
    ? explanation as Record<string, unknown>
    : undefined;
  const dimensions = dimensionExplanation?.dimensions as Record<string, { evidence: string; flagged: boolean; rating: number }> | undefined;

  return (
    <div className="rounded-xl border border-border-default bg-bg-secondary overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-3 px-4 py-3 hover:bg-bg-hover/30 transition-colors"
      >
        <div className="flex-1 flex items-center gap-3">
          <span className="text-base font-semibold text-text-primary">{metric.displayName}</span>
          <MetricTypeBadge type={metric.type} />
        </div>
        <ScoreBadge score={metric.normalizedScore} />
        <ChevronDown className={`w-4 h-4 text-text-muted transition-transform ${expanded ? 'rotate-180' : ''}`} />
      </button>

      {expanded && (
        <div className="px-5 pb-5 border-t border-border-default/50 pt-4 space-y-5">
          {/* Task completion: simple pass/fail */}
          {metric.name === 'task_completion' && (
            <div className="flex items-center gap-2">
              {details.match ? (
                <CheckCircle2 className="w-5 h-5 text-emerald-400" />
              ) : (
                <AlertTriangle className="w-5 h-5 text-red-400" />
              )}
              <span className="text-base text-text-secondary">{details.message as string}</span>
            </div>
          )}

          {/* Dimension-based explanation (faithfulness, conversation_progression) */}
          {dimensions && (
            <div className="space-y-3">
              {Object.entries(dimensions).map(([dimName, dim]) => {
                const isFaithfulness = metric.name === 'faithfulness';
                let badgeLabel: string;
                let badgeClass: string;
                if (isFaithfulness) {
                  if (dim.rating === 3) {
                    badgeLabel = 'OK';
                    badgeClass = 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
                  } else if (dim.rating === 2) {
                    badgeLabel = 'Minor/Ambiguous Issue';
                    badgeClass = 'bg-amber/10 text-amber border-amber/20';
                  } else {
                    badgeLabel = 'Clear Error';
                    badgeClass = 'bg-red-500/10 text-red-400 border-red-500/20';
                  }
                } else if (metric.name === 'conversation_progression') {
                  if (dim.rating === 3) {
                    badgeLabel = 'OK';
                    badgeClass = 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
                  } else if (dim.rating === 2) {
                    badgeLabel = 'Minor Issue';
                    badgeClass = 'bg-amber/10 text-amber border-amber/20';
                  } else {
                    badgeLabel = 'Clear Issue';
                    badgeClass = 'bg-red-500/10 text-red-400 border-red-500/20';
                  }
                } else {
                  badgeLabel = dim.flagged ? 'Flagged' : 'OK';
                  badgeClass = dim.flagged
                    ? 'bg-amber/10 text-amber border-amber/20'
                    : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
                }

                return (
                  <div key={dimName} className="rounded-lg bg-bg-primary p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-sm font-semibold text-text-primary">
                        {dimName.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
                      </span>
                      <span className={`text-xs px-2 py-0.5 rounded-full font-medium border ${badgeClass}`}>{badgeLabel}</span>
                      <span className="text-xs text-text-muted ml-auto">{dim.rating}/3</span>
                    </div>
                    <p className="text-base text-text-secondary leading-relaxed">{dim.evidence}</p>
                  </div>
                );
              })}
            </div>
          )}

          {/* Per-turn ratings table (conciseness, agent_speech_fidelity, etc.) */}
          {perTurnRatings && !dimensions && !perTurnEntityDetails && (
            <div className="space-y-3">
              <div className="text-xs font-semibold text-text-muted uppercase tracking-wider">Per-Turn Breakdown</div>
              <div className="space-y-3 max-h-[32rem] overflow-y-auto">
                {Object.entries(perTurnRatings).map(([turnId, rating]) => {
                  const turnExplanation = perTurnExplanations?.[turnId];
                  // Skip turns with -1 rating (not applicable)
                  if (rating === -1) return null;
                  return (
                    <div key={turnId} className="rounded-lg bg-bg-primary p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-xs font-semibold text-text-muted">Turn {turnId}</span>
                        <span className={`text-xs px-2 py-0.5 rounded-full font-medium border ${
                          rating >= 3 || rating === 1 && metric.name === 'agent_speech_fidelity'
                            ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                            : rating >= 2 ? 'bg-amber/10 text-amber border-amber/20'
                            : 'bg-red-500/10 text-red-400 border-red-500/20'
                        }`}>
                          {rating}
                        </span>
                      </div>
                      {turnExplanation && (
                        <p className="text-base text-text-secondary leading-relaxed">{turnExplanation}</p>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Turn taking: timing ratings */}
          {perTurnTimingRatings && (
            <div className="space-y-3">
              <div className="text-xs font-semibold text-text-muted uppercase tracking-wider">Per-Turn Timing</div>
              <div className="space-y-3 max-h-[32rem] overflow-y-auto">
                {Object.entries(perTurnTimingRatings).map(([turnId, timingLabel]) => {
                  const timingExplanation = perTurnTimingExplanations?.[turnId];
                  const timingColor =
                    timingLabel === 'On-Time' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' :
                    timingLabel === 'Late' ? 'bg-amber/10 text-amber border-amber/20' :
                    'bg-red-500/10 text-red-400 border-red-500/20';

                  return (
                    <div key={turnId} className="rounded-lg bg-bg-primary p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-xs font-semibold text-text-muted">Turn {turnId}</span>
                        <span className={`text-xs px-2 py-0.5 rounded-full font-medium border ${timingColor}`}>
                          {timingLabel}
                        </span>
                      </div>
                      {timingExplanation && (
                        <p className="text-base text-text-secondary leading-relaxed">{timingExplanation}</p>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Transcription accuracy: entity details */}
          {perTurnEntityDetails && (
            <div className="space-y-3">
              <div className="text-xs font-semibold text-text-muted uppercase tracking-wider">Per-Turn Entity Accuracy</div>
              <div className="space-y-3 max-h-[32rem] overflow-y-auto">
                {Object.entries(perTurnEntityDetails).map(([turnId, turnData]) => {
                  if (!turnData.entities || turnData.entities.length === 0) return null;
                  return (
                    <div key={turnId} className="rounded-lg bg-bg-primary p-4">
                      <div className="text-xs font-semibold text-text-muted mb-2">Turn {turnId}</div>
                      <div className="space-y-2">
                        {turnData.entities.map((entity, i) => (
                          <div key={i} className="flex items-start gap-2 text-base">
                            {entity.correct ? (
                              <CheckCircle2 className="w-4 h-4 text-emerald-400 mt-0.5 flex-shrink-0" />
                            ) : (
                              <AlertTriangle className="w-4 h-4 text-red-400 mt-0.5 flex-shrink-0" />
                            )}
                            <div>
                              <span className="text-text-muted">{entity.type}:</span>{' '}
                              <span className="text-text-secondary">{entity.value}</span>
                              {!entity.correct && (
                                <span className="text-red-400"> → {entity.transcribed_value}</span>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                      <p className="text-xs text-text-muted mt-2">{turnData.summary}</p>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Simple text explanation fallback */}
          {typeof explanation === 'string' && (
            <p className="text-base text-text-secondary leading-relaxed">{explanation}</p>
          )}
        </div>
      )}
    </div>
  );
}

/* ─── Metrics Section ─── */

function MetricsSection() {
  const groups = [
    { label: 'Accuracy (EVA-A)', icon: Target, metrics: evaAMetrics, color: 'text-emerald-400' },
    { label: 'Experience (EVA-X)', icon: Activity, metrics: evaXMetrics, color: 'text-purple-light' },
    { label: 'Relevant Diagnostic Metric', icon: Search, metrics: diagnosticMetrics, color: 'text-cyan-400' },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className="mt-12"
    >


      <div className="space-y-8">
        {groups.map((group) => (
          <div key={group.label}>
            <div className="flex items-center gap-3 mb-4 pb-3 border-b border-border-default">
              <span className="text-lg font-bold text-text-primary">{group.label}</span>
            </div>
            <div className="space-y-2">
              {group.metrics.map((metric) => (
                <MetricCard key={metric.name} metric={metric} />
              ))}
            </div>
          </div>
        ))}
      </div>
    </motion.div>
  );
}

/* ─── Main Component ─── */

export function ConversationDemo() {
  const issueMap = buildIssueMap(exampleConversation);

  const elements: React.ReactElement[] = [];
  let i = 0;

  while (i < exampleConversation.length) {
    const entry = exampleConversation[i];

    if (entry.type === 'user' || entry.type === 'assistant') {
      elements.push(<ChatMessage key={i} entry={entry} index={elements.length} issue={issueMap.get(i)} />);
    } else if (entry.type === 'tool_call') {
      const responseEntry = exampleConversation[i + 1]?.type === 'tool_response' ? exampleConversation[i + 1] : undefined;
      elements.push(<ToolCallBlock key={i} entry={entry} responseEntry={responseEntry} index={elements.length} issue={issueMap.get(i)} />);
      if (responseEntry) i++;
    }

    i++;
  }

  return (
    <Section
      id="demo"
      title="Example Conversation"
      subtitle="A real flight rebooking conversation from an EVA evaluation run, showing the full bot-to-bot interaction with tool calls and audio. Scroll down for evaluation results."
      wide
    >
      {/* 3-column layout */}
      <div className="flex flex-col lg:flex-row lg:items-start gap-6">
        {/* Left column: User Goal */}
        <div className="lg:w-[25%] flex-shrink-0 lg:sticky lg:top-8 lg:max-h-[calc(100vh-4rem)] lg:overflow-y-auto">
          <UserGoalCard />
        </div>

        {/* Center column: Audio + Conversation */}
        <div className="flex-1 min-w-0">
          {/* Audio player */}
          <div className="mb-6">
            <AudioPlayer src={`${import.meta.env.BASE_URL}demo/audio_mixed.wav`} />
          </div>

          {/* Conversation trace */}
          <div className="space-y-3">
            {elements}
          </div>
        </div>

        {/* Right column: Agent Tools */}
        <div className="lg:w-[22%] flex-shrink-0 lg:sticky lg:top-8 lg:max-h-[calc(100vh-4rem)] lg:overflow-y-auto">
          <AgentToolsPanel />
        </div>
      </div>

      {/* Evaluation scores below the conversation */}
      <MetricsSection />
    </Section>
  );
}
