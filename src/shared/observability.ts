/**
 * Observability aggregation.
 *
 * Builds the session-level observability payload from per-message
 * `MessageStats` and the underlying EventStore stream. Pure function —
 * no I/O, no DB access. Tolerates legacy sessions that predate the
 * observability types (optional fields stay undefined).
 */

import type {
  CacheSource,
  CallStatsDataPoint,
  Message,
  MessageStats,
  ModelSessionStats as _ModelSessionStats, // imported for downstream consumers
  ObservabilityCallRow,
  ObservabilityResponseRow,
  ObservabilityStats,
  ObservabilitySummary,
  RetryRecord,
  SessionStats,
  StatsDataPoint,
  ToolActivityEntry,
  ToolActivitySummary,
  ModelObservability,
  CompactionRecord,
} from './types.js'
import { classifyTool, TOOL_CATEGORY_ORDER } from './tool-category.js'
import type { ToolCategory } from './types.js'

// ============================================================================
// Event helpers (small subset, typed for compilation in isolation)
// ============================================================================

interface MinimalEvent {
  type: string
  data: Record<string, unknown>
  timestamp?: number
}

interface CompactionEventData {
  closedWindowId: string
  newWindowId: string
  beforeTokens: number
  afterTokens: number
  summary: string
  subAgentId?: string
  subAgentType?: string
}

interface PatternRetryEventData {
  pattern: string
  field: string
  attempt: number
  maxAttempts: number
}

interface ToolCallEventData {
  messageId: string
  toolCall: { id: string; name: string; arguments?: Record<string, unknown> }
}

interface ToolResultEventData {
  messageId: string
  toolCallId: string
  result: { success: boolean; durationMs: number; error?: string }
}

interface MessageStartEventData {
  messageId: string
  contextWindowId?: string
  subAgentId?: string
  subAgentType?: string
  isSystemGenerated?: boolean
  messageKind?: string
}

// ============================================================================
// Percentile helpers
// ============================================================================

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor((sorted.length - 1) * p)))
  return sorted[idx] ?? 0
}

function pickNumber(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

function computeCompactionRecords(events: MinimalEvent[]): CompactionRecord[] {
  const records: CompactionRecord[] = []
  for (const event of events) {
    if (event.type !== 'context.compacted') continue
    const data = event.data as unknown as CompactionEventData
    const before = pickNumber(data.beforeTokens)
    const after = pickNumber(data.afterTokens)
    const reduction = before - after
    records.push({
      timestamp: event.timestamp ?? 0,
      closedWindowId: data.closedWindowId,
      newWindowId: data.newWindowId,
      beforeTokens: before,
      afterTokens: after,
      reduction,
      reductionPercent: before > 0 ? (reduction / before) * 100 : 0,
      ...(data.subAgentId ? { subAgentId: data.subAgentId } : {}),
      ...(data.subAgentType ? { subAgentType: data.subAgentType } : {}),
    })
  }
  return records
}

function computeRetryRecords(
  events: MinimalEvent[],
  responseByMessageId: Map<string, { responseIndex: number; mode: string }>,
): RetryRecord[] {
  const records: RetryRecord[] = []
  for (const event of events) {
    if (event.type !== 'pattern.retry') continue
    const data = event.data as unknown as PatternRetryEventData
    // pattern.retry references a messageId of the truncated assistant message.
    // We surface the retry on the next response (responseIndex+1) because the
    // retry triggers a continuation message + new response.
    const meta = responseByMessageId.get((event.data['messageId'] as string) ?? '')
    records.push({
      timestamp: event.timestamp ?? 0,
      type: 'pattern',
      pattern: data.pattern,
      field: data.field,
      attempt: data.attempt,
      maxAttempts: data.maxAttempts,
      responseIndex: meta ? meta.responseIndex + 1 : 0,
      ...(meta?.mode ? { reason: `pattern:${data.pattern}` } : {}),
    })
  }
  // Truncation + continuation retries are captured via MessageStats.retryCount
  // We merge them in a later pass to keep attribution per response.
  return records
}

function buildToolActivity(events: MinimalEvent[]): ToolActivitySummary {
  const byTool = new Map<string, ToolActivityEntry>()
  let totalCount = 0
  let totalErrors = 0
  const byCategory = Object.fromEntries(TOOL_CATEGORY_ORDER.map((c) => [c, 0])) as Record<ToolCategory, number>

  // First pass: index durations from tool.result events.
  const resultIndex = new Map<string, { durationMs: number; success: boolean }>()
  for (const event of events) {
    if (event.type !== 'tool.result') continue
    const data = event.data as unknown as ToolResultEventData
    resultIndex.set(data.toolCallId, {
      durationMs: pickNumber(data.result?.durationMs),
      success: data.result?.success !== false,
    })
  }

  for (const event of events) {
    if (event.type !== 'tool.call') continue
    const data = event.data as unknown as ToolCallEventData
    const name = data.toolCall?.name ?? 'unknown'
    const category = classifyTool(name)
    const result = resultIndex.get(data.toolCall.id)
    const duration = result?.durationMs ?? 0
    const isError = result ? !result.success : false
    const existing = byTool.get(name)
    if (existing) {
      existing.count += 1
      existing.totalDurationMs += duration
      if (isError) existing.errorCount += 1
    } else {
      byTool.set(name, { toolName: name, category, count: 1, totalDurationMs: duration, errorCount: isError ? 1 : 0 })
    }
    byCategory[category] = (byCategory[category] ?? 0) + 1
    totalCount += 1
    if (isError) totalErrors += 1
  }

  const byToolList = Array.from(byTool.values()).sort((a, b) => b.count - a.count)
  return { totalCount, totalErrors, byCategory, byTool: byToolList }
}

function computeFollowingTools(events: MinimalEvent[]): Map<string, string> {
  // For each LLM call (keyed by messageId), find the first tool.call event
  // that appears after the message.done event for that messageId.
  // We return a map from "messageId:callIndex" → toolName. The caller can
  // correlate by messageId + callIndex.
  const followingByMessageCall = new Map<string, string>()
  // We can't determine callIndex from events alone without per-call metadata.
  // As a pragmatic fallback, return the first tool name for each assistant
  // messageId, which is sufficient for the dashboard "following action" column.
  const assistantMessageIds = new Set<string>()
  const assistantDoneIndex = new Map<string, number>()
  const toolCallsByIndex = new Map<number, ToolCallEventData[]>()

  events.forEach((e, idx) => {
    if (e.type === 'message.start') {
      const data = e.data as unknown as MessageStartEventData
      if (data.messageId) assistantMessageIds.add(data.messageId)
    } else if (e.type === 'message.done') {
      const data = e.data as unknown as { messageId: string; stats?: MessageStats }
      if (data.messageId && assistantMessageIds.has(data.messageId)) {
        assistantDoneIndex.set(data.messageId, idx)
      }
    } else if (e.type === 'tool.call') {
      const data = e.data as unknown as ToolCallEventData
      const arr = toolCallsByIndex.get(idx) ?? []
      arr.push(data)
      toolCallsByIndex.set(idx, arr)
    }
  })

  for (const [messageId, doneIdx] of assistantDoneIndex.entries()) {
    let nextToolName: string | undefined
    for (let i = doneIdx + 1; i < events.length; i += 1) {
      const event = events[i]
      if (!event) continue
      if (event.type === 'tool.call') {
        const data = event.data as unknown as ToolCallEventData
        nextToolName = data.toolCall?.name
        break
      }
      // Stop searching once the assistant moves on to a new message
      if (event.type === 'message.start' || event.type === 'message.done') {
        const startData = event.data as unknown as { messageId: string }
        if (startData.messageId !== messageId && event.type === 'message.start') break
      }
    }
    if (nextToolName) followingByMessageCall.set(messageId, nextToolName)
  }

  return followingByMessageCall
}

/**
 * Compute the real contextBefore / contextAfter evolution across responses.
 *
 * Each response carries `stats.prefillTokens` (final prompt size of the last
 * LLM call in the response) — that becomes the response's `contextAfter`.
 * The next response's `contextBefore` is normally that previous value, unless
 * a compaction happened in between, in which case the context resets to the
 * compaction's `afterTokens` (the new window's starting size).
 *
 * `prefTokenIncrement` is also surfaced as the net new tokens added during the
 * response (sum of (promptTokens - previousContext) across the response's calls).
 */
function computeContextEvolution(
  messages: Message[],
  sortedResponseIndices: number[],
  messageIdToResponseIndex: Map<string, number>,
  compactionsByResponse: Map<number, CompactionRecord[]>,
): Map<number, { contextBefore: number; contextAfter: number; newInput: number }> {
  const result = new Map<number, { contextBefore: number; contextAfter: number; newInput: number }>()
  let runningContext = 0
  let lastSeenResponseIndex = 0
  for (const msg of messages) {
    const stats = msg.stats
    if (!stats) continue
    const responseIndex = messageIdToResponseIndex.get(msg.id)
    if (responseIndex === undefined) continue
    const contextAfter = stats.prefillTokens
    // Sanity-check monotonicity: if the next response's prefillTokens is lower
    // than the current running context (e.g. legacy data with missing compactions),
    // reset running context to 0 BEFORE recording contextBefore so the dashboard
    // doesn't show a negative growth or a stale carry-over.
    if (contextAfter < runningContext) {
      runningContext = 0
    }
    const contextBefore = runningContext
    runningContext = contextAfter
    // Apply any compactions bucketed to this response: after a compaction,
    // the running context is the compaction's afterTokens (new window start).
    const responseCompactions = compactionsByResponse.get(responseIndex) ?? []
    if (responseCompactions.length > 0) {
      // Use the last compaction's afterTokens as the new running context.
      const lastCompaction = responseCompactions[responseCompactions.length - 1]
      if (lastCompaction) {
        runningContext = lastCompaction.afterTokens
      }
    }
    lastSeenResponseIndex = Math.max(lastSeenResponseIndex, responseIndex)
    result.set(responseIndex, {
      contextBefore,
      contextAfter,
      newInput: stats.prefTokenIncrement ?? stats.prefillTokens,
    })
  }
  // Reference sortedResponseIndices to keep the signature stable; the iteration
  // already respects response ordering via sortedByTime.
  void sortedResponseIndices
  void lastSeenResponseIndex
  return result
}

function buildResponseRows(
  messages: Message[],
  messageIdToResponseIndex: Map<string, number>,
  retriesByResponse: Map<number, RetryRecord[]>,
  compactionsByResponse: Map<number, CompactionRecord[]>,
  toolActivityByResponse: Map<number, ToolActivityEntry[]>,
  toolCountByResponse: Map<number, number>,
  cacheByResponse: Map<
    number,
    { rawPrompt: number; cacheRead: number; cacheWrite: number; newInput: number; cacheSource: CacheSource }
  >,
): ObservabilityResponseRow[] {
  // Compute the real context evolution before we iterate so the rows reflect
  // growth → compaction → growth instead of always starting at 0.
  const sortedIndices = Array.from(messageIdToResponseIndex.values()).sort((a, b) => a - b)
  const contextEvolution = computeContextEvolution(
    messages,
    sortedIndices,
    messageIdToResponseIndex,
    compactionsByResponse,
  )

  const rows: ObservabilityResponseRow[] = []
  for (const msg of messages) {
    const stats = msg.stats
    if (!stats) continue
    const responseIndex = messageIdToResponseIndex.get(msg.id)
    if (responseIndex === undefined) continue
    const cache = cacheByResponse.get(responseIndex) ?? {
      rawPrompt: stats.prefillTokens,
      cacheRead: stats.cachedPromptTokens ?? 0,
      cacheWrite: stats.cacheWriteTokens ?? 0,
      newInput: stats.prefTokenIncrement ?? stats.prefillTokens,
      cacheSource: stats.cacheSource ?? 'unavailable',
    }
    const cacheHitRatio =
      cache.cacheSource === 'provider' && cache.rawPrompt > 0 ? cache.cacheRead / cache.rawPrompt : undefined
    const evolution = contextEvolution.get(responseIndex) ?? {
      contextBefore: 0,
      contextAfter: stats.prefillTokens,
      newInput: cache.newInput,
    }
    rows.push({
      responseIndex,
      messageId: msg.id,
      timestamp: msg.timestamp,
      durationSeconds: stats.totalTime,
      llmCalls: stats.llmCalls?.length ?? 1,
      retryCount: stats.retryCount ?? retriesByResponse.get(responseIndex)?.length ?? 0,
      contextBefore: evolution.contextBefore,
      contextAfter: evolution.contextAfter,
      rawPrompt: cache.rawPrompt,
      cacheRead: cache.cacheRead,
      cacheWrite: cache.cacheWrite,
      newInput: cache.newInput,
      ...(cacheHitRatio !== undefined && { cacheHitRatio }),
      cacheSource: cache.cacheSource,
      toolCalls: toolCountByResponse.get(responseIndex) ?? 0,
      toolBreakdown: toolActivityByResponse.get(responseIndex) ?? [],
      retries: retriesByResponse.get(responseIndex) ?? [],
      compactions: compactionsByResponse.get(responseIndex) ?? [],
    })
  }
  return rows.sort((a, b) => a.responseIndex - b.responseIndex)
}

function aggregateCallSource(calls: ObservabilityCallRow[]): CacheSource {
  if (calls.length === 0) return 'unavailable'
  let hasProvider = false
  let hasEstimated = false
  let hasUnavailable = false
  for (const c of calls) {
    if (c.cacheSource === 'provider') hasProvider = true
    else if (c.cacheSource === 'estimated') hasEstimated = true
    else hasUnavailable = true
  }
  if (hasProvider && !hasUnavailable && !hasEstimated) return 'provider'
  if (hasEstimated && !hasProvider && !hasUnavailable) return 'estimated'
  return 'unavailable'
}

function computeSummary(args: {
  calls: ObservabilityCallRow[]
  responses: ObservabilityResponseRow[]
  retries: RetryRecord[]
  compactions: CompactionRecord[]
  toolActivity: ToolActivitySummary
  durationSeconds: number
  cacheSource: CacheSource
}): ObservabilitySummary {
  const { calls, responses, retries, compactions, toolActivity, durationSeconds, cacheSource } = args
  const rawPromptTokens = calls.reduce((sum, c) => sum + c.promptTokens, 0)
  const providerCachedTokens = calls.reduce(
    (sum, c) => sum + (c.cacheSource === 'provider' ? (c.cachedPromptTokens ?? 0) : 0),
    0,
  )
  const cacheWriteTokens = calls.reduce((sum, c) => sum + (c.cacheWriteTokens ?? 0), 0)
  // estimatedNewInputTokens = Σ (promptTokens - cachedPromptTokens) for provider calls,
  //                     + Σ promptTokens for non-provider calls (we don't know).
  // Never substitutes absence of cache info as zero cache.
  const estimatedNewInputTokens = calls.reduce(
    (sum, c) =>
      sum + (c.cacheSource === 'provider' ? Math.max(0, c.promptTokens - (c.cachedPromptTokens ?? 0)) : c.promptTokens),
    0,
  )
  const providerDenominator = calls
    .filter((c) => c.cacheSource === 'provider')
    .reduce((sum, c) => sum + c.promptTokens, 0)
  const providerCacheHitRatio =
    cacheSource === 'provider' && providerDenominator > 0 ? providerCachedTokens / providerDenominator : undefined

  const contexts = calls.map((c) => c.contextSize).sort((a, b) => a - b)
  const contextP50 = percentile(contexts, 0.5)
  const contextP95 = percentile(contexts, 0.95)
  const contextMax = contexts.length > 0 ? (contexts[contexts.length - 1] ?? 0) : 0

  const generationTokens = calls.reduce((sum, c) => sum + c.completionTokens, 0)
  const llmCalls = calls.length
  const generatedPerCall = llmCalls > 0 ? generationTokens / llmCalls : 0

  // Context Amplification Factor:
  //   numerator   = Σ promptTokens across ALL calls (raw prompt load)
  //   denominator = Σ (promptTokens - cachedPromptTokens) across PROVIDER-CACHE calls only
  // The denominator is reliable only when at least one provider-cache call exists.
  // For "partial" sessions (mix of provider and unavailable/estimated), the displayed CAF
  // is a lower bound: the true uncached tokens are at least what we know from the provider
  // subset, so rawPrompt/denominator <= true rawPrompt/true uncached. We still surface the
  // metric rather than hiding it, but mark its source as 'partial'.
  const providerCalls = calls.filter((c) => c.cacheSource === 'provider')
  const cafDenominator = providerCalls.reduce(
    (sum, c) => sum + Math.max(0, c.promptTokens - (c.cachedPromptTokens ?? 0)),
    0,
  )
  let contextAmplificationFactor: number | undefined
  let amplificationSource: 'provider' | 'partial' | 'unavailable' | undefined
  if (cafDenominator > 0 && providerCalls.length > 0) {
    contextAmplificationFactor = rawPromptTokens / cafDenominator
    amplificationSource = cacheSource === 'provider' ? 'provider' : 'partial'
  } else {
    amplificationSource = 'unavailable'
  }

  const subAgentCalls = toolActivity.byCategory['sub-agent'] ?? 0
  const toolCalls = toolActivity.totalCount

  return {
    durationSeconds,
    responses: responses.length,
    llmCalls,
    callsPerResponse: responses.length > 0 ? llmCalls / responses.length : 0,
    rawPromptTokens,
    providerCachedTokens,
    cacheWriteTokens,
    estimatedNewInputTokens,
    ...(providerCacheHitRatio !== undefined && { providerCacheHitRatio }),
    contextP50,
    contextP95,
    contextMax,
    ...(contextAmplificationFactor !== undefined &&
      amplificationSource !== undefined && {
        contextAmplificationFactor,
        amplificationSource,
      }),
    generationTokens,
    generatedPerCall,
    cacheSource,
    retries: retries.length,
    compactions: compactions.length,
    subAgentCalls,
    toolCalls,
  }
}

/**
 * Compute observability stats from message stats + the raw event stream.
 *
 * The event stream is used to derive:
 *   - compactions from `context.compacted` events
 *   - retries from `pattern.retry` events (truncation/continuation are added
 *     via `MessageStats.retryCount` when available)
 *   - tool activity from `tool.call` / `tool.result` events
 *   - "following tool" lookup per assistant message
 *
 * The function is tolerant of legacy sessions that predate the
 * observability types — missing fields are simply left undefined and
 * ratios are not displayed.
 */
export function computeObservability(
  messages: Message[],
  events: MinimalEvent[],
  options?: { sessionId?: string; sessionTitle?: string; generatedAt?: string },
): ObservabilityStats | null {
  const messagesWithStats = messages.filter((m) => m.stats !== undefined && m.stats !== null)
  if (messagesWithStats.length === 0) return null

  // Build messageId → responseIndex map (1-based, only stats messages).
  const sortedByTime = [...messagesWithStats].sort(
    (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime(),
  )
  const messageIdToResponseIndex = new Map<string, number>()
  sortedByTime.forEach((m, i) => messageIdToResponseIndex.set(m.id, i + 1))

  const compactions = computeCompactionRecords(events)
  const compactionsByResponse = new Map<number, CompactionRecord[]>()
  for (const record of compactions) {
    // Compaction events don't reference a response messageId. Approximate by
    // bucketing into the LATEST response whose timestamp is <= the compaction
    // timestamp (most recent response that triggered or completed the compaction).
    // Using findIndex would pick the FIRST matching response, which is wrong
    // when the compaction happens after several responses have accumulated.
    const recordTs = record.timestamp ?? 0
    let lastIdx = -1
    for (let i = 0; i < sortedByTime.length; i += 1) {
      const m = sortedByTime[i]
      if (m && m.timestamp && new Date(m.timestamp).getTime() <= recordTs) {
        lastIdx = i
      } else {
        break
      }
    }
    const idx = lastIdx >= 0 ? lastIdx + 1 : sortedByTime.length
    const list = compactionsByResponse.get(idx) ?? []
    list.push(record)
    compactionsByResponse.set(idx, list)
  }

  // Build retries by response.
  const retries = computeRetryRecords(
    events,
    new Map(Array.from(messageIdToResponseIndex.entries()).map(([id, idx]) => [id, { responseIndex: idx, mode: '' }])),
  )
  const retriesByResponse = new Map<number, RetryRecord[]>()
  for (const r of retries) {
    if (r.responseIndex <= 0) continue
    const list = retriesByResponse.get(r.responseIndex) ?? []
    list.push(r)
    retriesByResponse.set(r.responseIndex, list)
  }

  // Add truncation + continuation retries from MessageStats.retryCount where present.
  for (const msg of sortedByTime) {
    const idx = messageIdToResponseIndex.get(msg.id)
    const stats = msg.stats
    if (!stats || idx === undefined) continue
    const explicitRetries = stats.retryCount ?? 0
    const list = retriesByResponse.get(idx) ?? []
    // The pattern.retry records already cover the pattern path.
    // The leftover (explicitRetries - pattern matches) is split heuristically
    // between truncation and continuation. Without per-call metadata we
    // attribute it to continuation as the most common fallback in agent-loop.
    const leftover = Math.max(0, explicitRetries - list.length)
    for (let k = 0; k < leftover; k += 1) {
      list.push({ timestamp: new Date(msg.timestamp).getTime() || 0, type: 'continuation', responseIndex: idx })
    }
    retriesByResponse.set(idx, list)
  }
  const allRetries = Array.from(retriesByResponse.values()).flat()

  // Tool activity
  const toolActivity = buildToolActivity(events)
  const toolActivityByResponse = new Map<number, ToolActivityEntry[]>()
  const toolCountByResponse = new Map<number, number>()
  for (const event of events) {
    if (event.type !== 'tool.call') continue
    const data = event.data as unknown as ToolCallEventData
    const msgId = data.messageId
    const responseIndex = messageIdToResponseIndex.get(msgId)
    if (responseIndex === undefined) continue
    const list = toolActivityByResponse.get(responseIndex) ?? []
    const name = data.toolCall.name
    const entry = toolActivity.byTool.find((t) => t.toolName === name)
    if (entry) list.push(entry)
    toolActivityByResponse.set(responseIndex, list)
    toolCountByResponse.set(responseIndex, (toolCountByResponse.get(responseIndex) ?? 0) + 1)
  }

  // Cache per response (sum of LLMCallStats values per response)
  const cacheByResponse = new Map<
    number,
    { rawPrompt: number; cacheRead: number; cacheWrite: number; newInput: number; cacheSource: CacheSource }
  >()
  for (const msg of sortedByTime) {
    const stats = msg.stats
    const idx = messageIdToResponseIndex.get(msg.id)
    if (!stats || idx === undefined) continue
    const calls = stats.llmCalls ?? []
    let raw = 0
    let read = 0
    let write = 0
    let uncached = 0
    let hasProvider = false
    let hasOther = false
    for (const call of calls) {
      raw += call.promptTokens
      if (call.cachedPromptTokens !== undefined) read += call.cachedPromptTokens
      if (call.cacheWriteTokens !== undefined) write += call.cacheWriteTokens
      if (call.cacheSource === 'provider') {
        hasProvider = true
        uncached += call.promptTokens - (call.cachedPromptTokens ?? 0)
      } else {
        hasOther = true
        uncached += call.promptTokens
      }
    }
    cacheByResponse.set(idx, {
      rawPrompt: raw,
      cacheRead: read,
      cacheWrite: write,
      newInput: uncached,
      cacheSource: hasProvider && !hasOther ? 'provider' : 'unavailable',
    })
  }

  // Following tool lookup
  const followingTools = computeFollowingTools(events)

  // Build call rows from MessageStats.llmCalls (per-call) or fall back to response-level.
  const calls: ObservabilityCallRow[] = []
  let sessionCallIndex = 0
  const sortedWithStats = sortedByTime
  for (const msg of sortedWithStats) {
    const stats = msg.stats
    if (!stats) continue
    const responseIndex = messageIdToResponseIndex.get(msg.id) ?? 0
    const responseCalls = stats.llmCalls ?? []
    if (responseCalls.length === 0) continue
    for (const call of responseCalls) {
      sessionCallIndex += 1
      const followingTool = followingTools.get(msg.id)
      calls.push({
        sessionCallIndex,
        responseIndex,
        callIndex: call.callIndex,
        messageId: msg.id,
        timestamp: call.timestamp ?? msg.timestamp,
        providerId: call.providerId,
        providerName: call.providerName,
        backend: call.backend,
        model: call.model,
        mode: stats.mode,
        promptTokens: call.promptTokens,
        ...(call.cachedPromptTokens !== undefined && { cachedPromptTokens: call.cachedPromptTokens }),
        ...(call.cacheWriteTokens !== undefined && { cacheWriteTokens: call.cacheWriteTokens }),
        ...(call.cacheSource && { cacheSource: call.cacheSource }),
        completionTokens: call.completionTokens,
        ttft: call.ttft,
        completionTime: call.completionTime,
        totalTime: call.totalTime,
        prefillSpeed: call.prefillSpeed,
        generationSpeed: call.generationSpeed,
        contextSize: call.contextSize ?? call.promptTokens,
        retries: call.retries ?? 0,
        ...(followingTool && { followingTool }),
      })
    }
  }

  const cacheSource = aggregateCallSource(calls)

  // Duration = from earliest message to latest message timestamp.
  let durationSeconds = 0
  if (sortedByTime.length > 0) {
    const first = sortedByTime[0]?.timestamp
    const last = sortedByTime[sortedByTime.length - 1]?.timestamp
    if (first && last) durationSeconds = (new Date(last).getTime() - new Date(first).getTime()) / 1000
  }

  const responseRows = buildResponseRows(
    sortedByTime,
    messageIdToResponseIndex,
    retriesByResponse,
    compactionsByResponse,
    toolActivityByResponse,
    toolCountByResponse,
    cacheByResponse,
  )

  const summary = computeSummary({
    calls,
    responses: responseRows,
    retries: allRetries,
    compactions,
    toolActivity,
    durationSeconds,
    cacheSource,
  })

  // Model breakdown
  const modelBuckets = new Map<string, ObservabilityCallRow[]>()
  for (const c of calls) {
    const key = `${c.providerId}::${c.model}`
    const list = modelBuckets.get(key) ?? []
    list.push(c)
    modelBuckets.set(key, list)
  }
  // Per-model responseIndex sets — used to scope retries, compactions, and
  // tool activity to the responses where each model was actually active.
  const modelResponseIndices = new Map<string, Set<number>>()
  for (const c of calls) {
    const key = `${c.providerId}::${c.model}`
    let set = modelResponseIndices.get(key)
    if (!set) {
      set = new Set<number>()
      modelResponseIndices.set(key, set)
    }
    set.add(c.responseIndex)
  }
  const modelBreakdown: ModelObservability[] = Array.from(modelBuckets.entries()).map(([key, modelCalls]) => {
    const sample = modelCalls[0]
    if (!sample) {
      return {
        key,
        label: key,
        providerId: '',
        providerName: '',
        backend: 'unknown',
        model: '',
        summary: { ...summary, llmCalls: 0 },
        calls: [],
      }
    }
    const modelResponseSet = modelResponseIndices.get(key) ?? new Set<number>()
    const modelRetries = allRetries.filter((r) => modelResponseSet.has(r.responseIndex))
    // Compactions are bucketed by responseIndex; only include those that
    // happened during a response where this model was active. Sub-agent
    // compactions (subAgentId present) are excluded from the top-level model
    // breakdown to avoid double-counting.
    const modelCompactions: CompactionRecord[] = []
    for (const [respIdx, list] of compactionsByResponse.entries()) {
      if (!modelResponseSet.has(respIdx)) continue
      for (const c of list) {
        if (c.subAgentId) continue
        modelCompactions.push(c)
      }
    }
    // Tool activity is scoped to the messages this model produced.
    const modelMessageIds = new Set(modelCalls.map((c) => c.messageId))
    const modelEvents = events.filter((e) => {
      const msgId = (e.data as { messageId?: unknown }).messageId
      return typeof msgId === 'string' && modelMessageIds.has(msgId)
    })
    const modelToolActivity = buildToolActivity(modelEvents)
    const modelSummary = computeSummary({
      calls: modelCalls,
      responses: responseRows.filter((r) => modelResponseSet.has(r.responseIndex)),
      retries: modelRetries,
      compactions: modelCompactions,
      toolActivity: modelToolActivity,
      durationSeconds,
      cacheSource,
    })
    const sampleCall = modelCalls.find((c) => c.providerId === sample.providerId && c.model === sample.model) ?? sample
    return {
      key,
      label: `${sampleCall.providerName} > ${sampleCall.model}`,
      providerId: sampleCall.providerId,
      providerName: sampleCall.providerName,
      backend: sampleCall.backend,
      model: sampleCall.model,
      summary: modelSummary,
      calls: modelCalls,
    }
  })

  return {
    sessionId: options?.sessionId ?? '',
    ...(options?.sessionTitle ? { sessionTitle: options.sessionTitle } : {}),
    generatedAt: options?.generatedAt ?? new Date().toISOString(),
    summary,
    responses: responseRows,
    calls,
    compactions,
    retries: allRetries,
    toolActivity,
    modelBreakdown,
    schemaVersion: 'obs.v1',
  }
}

/**
 * Lightweight observability view derived purely from an upstream `SessionStats`
 * payload (already-fetched data points + per-call points, with cache fields
 * populated by the server). Use this when events are not available — the
 * result covers cache attribution, P50/P95/Max context, and per-call
 * observability, but `compactions`, `retries`, and `toolActivity` are
 * empty (those require the full event stream).
 */
export function computeObservabilityFromSessionStats(
  fullStats: SessionStats,
  options?: { sessionId?: string; sessionTitle?: string },
): ObservabilityStats {
  const calls: ObservabilityCallRow[] = fullStats.callDataPoints.map((dp: CallStatsDataPoint) => ({
    sessionCallIndex: dp.sessionCallIndex,
    responseIndex: dp.responseIndex,
    callIndex: dp.callIndex,
    messageId: dp.messageId,
    timestamp: dp.timestamp,
    providerId: dp.providerId,
    providerName: dp.providerName,
    backend: dp.backend,
    model: dp.model,
    mode: dp.mode,
    promptTokens: dp.promptTokens,
    ...(dp.cachedPromptTokens !== undefined && { cachedPromptTokens: dp.cachedPromptTokens }),
    ...(dp.cacheWriteTokens !== undefined && { cacheWriteTokens: dp.cacheWriteTokens }),
    ...(dp.cacheSource && { cacheSource: dp.cacheSource }),
    completionTokens: dp.completionTokens,
    ttft: dp.ttft,
    completionTime: dp.completionTime,
    totalTime: dp.totalTime,
    prefillSpeed: dp.prefillSpeed,
    generationSpeed: dp.generationSpeed,
    contextSize: dp.contextSize ?? dp.promptTokens,
    retries: dp.retries ?? 0,
  }))

  const responses: ObservabilityResponseRow[] = fullStats.dataPoints.map(
    (dp: StatsDataPoint): ObservabilityResponseRow => {
      const responseCalls = calls.filter((c) => c.messageId === dp.messageId)
      const cachedSum = responseCalls.reduce(
        (sum, c) => sum + (c.cacheSource === 'provider' ? (c.cachedPromptTokens ?? 0) : 0),
        0,
      )
      const writeSum = responseCalls.reduce((sum, c) => sum + (c.cacheWriteTokens ?? 0), 0)
      const hasProvider = responseCalls.some((c) => c.cacheSource === 'provider')
      const hasOther = responseCalls.some((c) => c.cacheSource !== 'provider')
      const cacheSource: CacheSource =
        hasProvider && !hasOther ? 'provider' : hasProvider ? 'unavailable' : 'unavailable'
      const rawPrompt = responseCalls.reduce((sum, c) => sum + c.promptTokens, 0)
      const newInput = responseCalls.reduce(
        (sum, c) =>
          sum + (c.cacheSource === 'provider' ? c.promptTokens - (c.cachedPromptTokens ?? 0) : c.promptTokens),
        0,
      )
      const cacheHitRatio = cacheSource === 'provider' && rawPrompt > 0 ? cachedSum / rawPrompt : undefined
      return {
        responseIndex: dp.responseIndex,
        messageId: dp.messageId,
        timestamp: dp.timestamp,
        durationSeconds: dp.totalTime,
        llmCalls: responseCalls.length || 1,
        retryCount: dp.retryCount ?? 0,
        contextBefore: 0,
        contextAfter: dp.prefillTokens,
        rawPrompt,
        cacheRead: cachedSum,
        cacheWrite: writeSum,
        newInput,
        ...(cacheHitRatio !== undefined && { cacheHitRatio }),
        cacheSource,
        toolCalls: 0,
        toolBreakdown: [],
        retries: [],
        compactions: [],
      }
    },
  )

  // Per-model breakdown using only the data points already in `fullStats`.
  const modelBuckets = new Map<string, { calls: ObservabilityCallRow[]; responses: ObservabilityResponseRow[] }>()
  for (const c of calls) {
    const key = `${c.providerId}::${c.model}`
    let bucket = modelBuckets.get(key)
    if (!bucket) {
      bucket = { calls: [], responses: [] }
      modelBuckets.set(key, bucket)
    }
    bucket.calls.push(c)
  }
  for (const r of responses) {
    const sampleCall = calls.find((c) => c.messageId === r.messageId)
    if (sampleCall) {
      const key = `${sampleCall.providerId}::${sampleCall.model}`
      const bucket = modelBuckets.get(key)
      if (bucket) bucket.responses.push(r)
    }
  }
  const modelBreakdown: ModelObservability[] = Array.from(modelBuckets.entries()).map(([key, bucket]) => {
    const sample = bucket.calls[0]
    const modelSummary = computeSummary({
      calls: bucket.calls,
      responses: bucket.responses,
      retries: [],
      compactions: [],
      toolActivity: emptyToolActivitySummary(),
      durationSeconds: 0,
      cacheSource: aggregateCallSource(bucket.calls),
    })
    return {
      key,
      label: sample ? `${sample.providerName} > ${sample.model}` : key,
      providerId: sample?.providerId ?? '',
      providerName: sample?.providerName ?? '',
      backend: sample?.backend ?? 'unknown',
      model: sample?.model ?? '',
      summary: modelSummary,
      calls: bucket.calls,
    }
  })

  return {
    sessionId: options?.sessionId ?? '',
    ...(options?.sessionTitle ? { sessionTitle: options.sessionTitle } : {}),
    generatedAt: new Date().toISOString(),
    summary: computeSummary({
      calls,
      responses,
      retries: [],
      compactions: [],
      toolActivity: emptyToolActivitySummary(),
      durationSeconds: 0,
      cacheSource: aggregateCallSource(calls),
    }),
    responses,
    calls,
    compactions: [],
    retries: [],
    toolActivity: emptyToolActivitySummary(),
    modelBreakdown,
    schemaVersion: 'obs.v1',
  }
}

function emptyToolActivitySummary(): ToolActivitySummary {
  const byCategory = {} as Record<ToolCategory, number>
  for (const cat of TOOL_CATEGORY_ORDER) byCategory[cat] = 0
  return { totalCount: 0, totalErrors: 0, byCategory, byTool: [] }
}
