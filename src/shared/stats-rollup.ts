/**
 * Server-side session event rollup computation.
 *
 * Sourced strictly from the EventStore stream:
 *   - `context.compacted` → compactions
 *   - `pattern.retry`     → retries
 *   - `tool.call`         → tool calls
 *   - `tool.result`       → tool errors
 *   - `message.start`/`message.done` with `subAgentId`/`subAgentType` → sub-agent activity
 *
 * Never inferred from token metrics. The plugin reads this rollup via the
 * `/api/sessions/:id/stats` endpoint or through the live `registerHook`
 * stream (Plugin API v2).
 */

import type {
  CompactionEventRecord,
  ModelSessionStatsEventRollup,
  RetryEventRecord,
  SessionStatsEventRollup,
  ToolEventEntry,
} from './types.js'
import { classifyTool } from './tool-category.js'

export interface MinimalEvent {
  type: string
  data: unknown
  timestamp?: number
}

interface CompactionEventData {
  closedWindowId: string
  newWindowId: string
  beforeTokens: number
  afterTokens: number
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
  toolCall: { id: string; name: string }
}

interface ToolResultEventData {
  messageId: string
  toolCallId: string
  result: { success?: boolean }
}

interface MessageStartData {
  messageId: string
  subAgentId?: string
  subAgentType?: string
}

function pickNumber(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

function buildCompactions(events: MinimalEvent[]): CompactionEventRecord[] {
  const records: CompactionEventRecord[] = []
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

function buildRetries(events: MinimalEvent[]): RetryEventRecord[] {
  const records: RetryEventRecord[] = []
  for (const event of events) {
    if (event.type !== 'pattern.retry') continue
    const data = event.data as unknown as PatternRetryEventData
    const eventData = event.data as { messageId?: string }
    records.push({
      timestamp: event.timestamp ?? 0,
      type: 'pattern',
      reason: data.pattern,
      pattern: data.pattern,
      ...(eventData.messageId ? { messageId: eventData.messageId } : {}),
      attempt: data.attempt,
      maxAttempts: data.maxAttempts,
    })
  }
  return records
}

function buildToolBreakdown(events: MinimalEvent[]): {
  byName: Map<string, { count: number; errors: number }>
  totalCount: number
  totalErrors: number
} {
  const resultIndex = new Map<string, { success: boolean }>()
  for (const event of events) {
    if (event.type !== 'tool.result') continue
    const data = event.data as unknown as ToolResultEventData
    resultIndex.set(data.toolCallId, { success: data.result?.success !== false })
  }
  const byName = new Map<string, { count: number; errors: number }>()
  let totalCount = 0
  let totalErrors = 0
  for (const event of events) {
    if (event.type !== 'tool.call') continue
    const data = event.data as unknown as ToolCallEventData
    const name = data.toolCall?.name ?? 'unknown'
    const result = resultIndex.get(data.toolCall.id)
    const isError = result ? !result.success : false
    const entry = byName.get(name) ?? { count: 0, errors: 0 }
    entry.count += 1
    if (isError) entry.errors += 1
    byName.set(name, entry)
    totalCount += 1
    if (isError) totalErrors += 1
  }
  return { byName, totalCount, totalErrors }
}

function buildToolBreakdownEntries(byName: Map<string, { count: number; errors: number }>): ToolEventEntry[] {
  const entries: ToolEventEntry[] = []
  for (const [toolName, agg] of byName.entries()) {
    entries.push({
      toolName,
      category: classifyTool(toolName),
      count: agg.count,
      errors: agg.errors,
    })
  }
  return entries.sort((a, b) => b.count - a.count)
}

function countSubAgentCalls(events: MinimalEvent[]): number {
  let count = 0
  for (const event of events) {
    if (event.type !== 'message.start') continue
    const data = event.data as unknown as MessageStartData
    if (data.subAgentId || data.subAgentType) count += 1
  }
  return count
}

export interface BuildSessionStatsEventRollupOptions {
  /**
   * Number of historical compactions known to the snapshot via
   * `snapshot.contextState.compactionCount` but whose per-compaction
   * details (`snapshot.contextWindows`) are unavailable (legacy snapshot
   * with the details pruned). The rollup adds this to the
   * `compactionCount` it reports but does NOT synthesize fake per-record
   * entries in `compactions[]`. `compactionsDetailsAvailable` is set to
   * `false` in that case so consumers can render the count while
   * acknowledging the per-compaction records are absent.
   */
  legacyCompactionCount?: number
}

/**
 * Build the session-wide event rollup from a list of stored events.
 *
 * `messageIdToResponseIndex` is used to backfill `responseIndex` on retry
 * records. When the caller does not have a response index map, retries are
 * returned with `responseIndex: 0`.
 *
 * `options.legacyCompactionCount` lets the caller carry over a known
 * historical compaction count from a legacy snapshot whose per-compaction
 * details (`contextWindows[]`) were pruned. The count is added to the
 * reported `compactionCount`; `compactions[]` stays empty (no fabricated
 * details) and `compactionsDetailsAvailable` becomes `false`.
 */
export function buildSessionStatsEventRollup(
  events: MinimalEvent[],
  messageIdToResponseIndex?: Map<string, number>,
  options?: BuildSessionStatsEventRollupOptions,
): SessionStatsEventRollup {
  const compactions = buildCompactions(events)
  const retries = buildRetries(events)
  if (messageIdToResponseIndex) {
    for (const r of retries) {
      const msgId = (r as { messageId?: string }).messageId
      const idx = msgId ? messageIdToResponseIndex.get(msgId) : undefined
      if (idx !== undefined) (r as { responseIndex: number }).responseIndex = idx + 1
    }
  }
  const { byName, totalCount, totalErrors } = buildToolBreakdown(events)
  const subAgentCalls = countSubAgentCalls(events)
  const legacy = options?.legacyCompactionCount ?? 0
  const detailsAvailable = compactions.length > 0 || legacy === 0
  return {
    compactions,
    retries,
    toolCalls: totalCount,
    toolErrors: totalErrors,
    toolBreakdown: buildToolBreakdownEntries(byName),
    subAgentCalls,
    compactionCount: compactions.length + legacy,
    retryCount: retries.length,
    compactionsDetailsAvailable: detailsAvailable,
  }
}

/**
 * Build per-model event rollups. Tools and compactions are attributed to
 * the model whose LLM call produced the response that triggered them.
 *
 * `modelResponseSet` maps a model key (providerId::model) to the set of
 * response indices driven by that model. When omitted, all events are
 * attributed to a single virtual model.
 */
export function buildModelEventRollup(
  events: MinimalEvent[],
  modelResponseSet: Map<string, Set<number>>,
  messageIdToResponseIndex: Map<string, number>,
): Map<string, ModelSessionStatsEventRollup> {
  const result = new Map<string, ModelSessionStatsEventRollup>()
  for (const key of modelResponseSet.keys()) {
    result.set(key, {
      compactions: [],
      retries: [],
      toolCalls: 0,
      toolErrors: 0,
      subAgentCalls: 0,
      toolBreakdown: [],
    })
  }

  for (const event of events) {
    if (event.type === 'context.compacted') {
      const data = event.data as unknown as CompactionEventData
      const record: CompactionEventRecord = {
        timestamp: event.timestamp ?? 0,
        closedWindowId: data.closedWindowId,
        newWindowId: data.newWindowId,
        beforeTokens: pickNumber(data.beforeTokens),
        afterTokens: pickNumber(data.afterTokens),
        reduction: pickNumber(data.beforeTokens) - pickNumber(data.afterTokens),
        reductionPercent:
          pickNumber(data.beforeTokens) > 0
            ? ((pickNumber(data.beforeTokens) - pickNumber(data.afterTokens)) /
                pickNumber(data.beforeTokens)) *
              100
            : 0,
        ...(data.subAgentId ? { subAgentId: data.subAgentId } : {}),
        ...(data.subAgentType ? { subAgentType: data.subAgentType } : {}),
      }
      // Find the response index just before the compaction event.
      let bestIdx: number | undefined
      for (const [, idx] of messageIdToResponseIndex) {
        if (bestIdx === undefined || idx > bestIdx) bestIdx = idx
      }
      const targetKey =
        bestIdx === undefined
          ? undefined
          : [...modelResponseSet.entries()].find(([_, indices]) => indices.has(bestIdx + 1))?.[0]
      if (targetKey) result.get(targetKey)!.compactions.push(record)
      continue
    }
    if (event.type === 'pattern.retry') {
      const data = event.data as unknown as PatternRetryEventData
      const msgId = (event.data as { messageId?: string }).messageId
      const idx = msgId ? messageIdToResponseIndex.get(msgId) : undefined
      const targetKey = [...modelResponseSet.entries()].find(
        ([_, indices]) => idx !== undefined && indices.has(idx + 1),
      )?.[0]
      if (!targetKey) continue
      const r: RetryEventRecord = {
        timestamp: event.timestamp ?? 0,
        type: 'pattern',
        reason: data.pattern,
        pattern: data.pattern,
        ...(msgId ? { messageId: msgId } : {}),
        responseIndex: (idx ?? -1) + 1,
        attempt: data.attempt,
        maxAttempts: data.maxAttempts,
      }
      result.get(targetKey)!.retries.push(r)
      continue
    }
    if (event.type === 'tool.call') {
      const data = event.data as unknown as ToolCallEventData
      const msgId = data.messageId
      const idx = msgId ? messageIdToResponseIndex.get(msgId) : undefined
      const targetKey = [...modelResponseSet.entries()].find(
        ([_, indices]) => idx !== undefined && indices.has(idx + 1),
      )?.[0]
      if (!targetKey) continue
      const entry = result.get(targetKey)!
      entry.toolCalls += 1
      const name = data.toolCall?.name ?? 'unknown'
      const existing = entry.toolBreakdown.find((b) => b.toolName === name)
      if (existing) existing.count += 1
      else entry.toolBreakdown.push({ toolName: name, category: classifyTool(name), count: 1, errors: 0 })
      continue
    }
    if (event.type === 'tool.result') {
      const data = event.data as unknown as ToolResultEventData
      const msgId = data.messageId
      const idx = msgId ? messageIdToResponseIndex.get(msgId) : undefined
      const targetKey = [...modelResponseSet.entries()].find(
        ([_, indices]) => idx !== undefined && indices.has(idx + 1),
      )?.[0]
      if (!targetKey) continue
      const isError = data.result?.success === false
      if (!isError) continue
      const entry = result.get(targetKey)!
      entry.toolErrors += 1
      const name = (event.data as { toolCall?: { name?: string } }).toolCall?.name
      const existing = name ? entry.toolBreakdown.find((b) => b.toolName === name) : undefined
      if (existing) existing.errors += 1
      continue
    }
    if (event.type === 'message.start') {
      const data = event.data as unknown as MessageStartData
      if (!data.subAgentId && !data.subAgentType) continue
      const msgId = data.messageId
      const idx = msgId ? messageIdToResponseIndex.get(msgId) : undefined
      const targetKey = [...modelResponseSet.entries()].find(
        ([_, indices]) => idx !== undefined && indices.has(idx + 1),
      )?.[0]
      if (!targetKey) continue
      result.get(targetKey)!.subAgentCalls += 1
    }
  }
  return result
}

/** Empty session-wide event rollup (no events yet). */
export function emptyEventRollup(): SessionStatsEventRollup {
  return {
    compactions: [],
    retries: [],
    toolCalls: 0,
    toolErrors: 0,
    toolBreakdown: [],
    subAgentCalls: 0,
    compactionCount: 0,
    retryCount: 0,
    compactionsDetailsAvailable: true,
  }
}

/** Empty per-model event rollup. */
export function emptyModelEventRollup(): ModelSessionStatsEventRollup {
  return {
    compactions: [],
    retries: [],
    toolCalls: 0,
    toolErrors: 0,
    subAgentCalls: 0,
    toolBreakdown: [],
  }
}
