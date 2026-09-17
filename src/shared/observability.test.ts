import { describe, it, expect } from 'vitest'
import { computeObservability } from './observability.js'
import type { Message, MessageStats, LLMCallStats } from './types.js'

function makeCall(overrides: Partial<LLMCallStats> = {}): LLMCallStats {
  return {
    providerId: 'p1',
    providerName: 'Local',
    backend: 'vllm',
    model: 'm',
    callIndex: 1,
    promptTokens: 1000,
    completionTokens: 50,
    ttft: 0.1,
    completionTime: 0.5,
    prefillSpeed: 10000,
    generationSpeed: 100,
    totalTime: 0.6,
    ...overrides,
  }
}

function makeStats(overrides: Partial<MessageStats> = {}): MessageStats {
  return {
    providerId: 'p1',
    providerName: 'Local',
    backend: 'vllm',
    model: 'm',
    mode: 'planner',
    totalTime: 5,
    toolTime: 1,
    prefillTokens: 1000,
    prefillSpeed: 10000,
    generationTokens: 50,
    generationSpeed: 100,
    ...overrides,
  }
}

function makeMessage(id: string, stats: MessageStats, timestamp: string): Message {
  return {
    id,
    role: 'assistant',
    content: 'ok',
    timestamp,
    stats,
  }
}

function makeEvent(
  type: string,
  data: Record<string, unknown>,
  timestamp = 0,
): { type: string; data: Record<string, unknown>; timestamp: number } {
  return { type, data, timestamp }
}

describe('computeObservability', () => {
  it('returns null for empty messages', () => {
    expect(computeObservability([], [])).toBeNull()
  })

  it('returns null when no messages have stats', () => {
    const messages: Message[] = [{ id: 'm1', role: 'assistant', content: 'ok', timestamp: '2024-01-01T00:00:00Z' }]
    expect(computeObservability(messages, [])).toBeNull()
  })

  it('builds summary with provider cache hit ratio when cacheSource=provider', () => {
    const stats = makeStats({
      prefillTokens: 1000,
      cachedPromptTokens: 800,
      cacheSource: 'provider',
      llmCalls: [makeCall({ promptTokens: 1000, cachedPromptTokens: 800, cacheSource: 'provider' })],
    })
    const messages = [makeMessage('m1', stats, '2024-01-01T00:00:00Z')]
    const result = computeObservability(messages, [], { sessionId: 's1' })
    expect(result).not.toBeNull()
    expect(result!.summary.cacheSource).toBe('provider')
    expect(result!.summary.providerCachedTokens).toBe(800)
    expect(result!.summary.rawPromptTokens).toBe(1000)
    expect(result!.summary.providerCacheHitRatio).toBeCloseTo(0.8, 5)
    expect(result!.summary.contextAmplificationFactor).toBeCloseTo(5, 5) // 1000 / (1000-800)
    expect(result!.summary.amplificationSource).toBe('provider')
  })

  it('marks cacheSource as unavailable when no cache data is provided', () => {
    const stats = makeStats({ prefillTokens: 500, llmCalls: [makeCall({ promptTokens: 500 })] })
    const messages = [makeMessage('m1', stats, '2024-01-01T00:00:00Z')]
    const result = computeObservability(messages, [])
    expect(result).not.toBeNull()
    expect(result!.summary.cacheSource).toBe('unavailable')
    expect(result!.summary.providerCacheHitRatio).toBeUndefined()
    expect(result!.summary.contextAmplificationFactor).toBeUndefined()
  })

  it('aggregates compactions from context.compacted events', () => {
    const stats = makeStats({ prefillTokens: 1000, llmCalls: [makeCall({ promptTokens: 1000 })] })
    const messages = [makeMessage('m1', stats, '2024-01-01T00:00:00Z')]
    const events = [
      makeEvent(
        'context.compacted',
        { closedWindowId: 'w1', newWindowId: 'w2', beforeTokens: 175000, afterTokens: 0, summary: 's' },
        Date.parse('2024-01-01T00:01:00Z'),
      ),
    ]
    const result = computeObservability(messages, events)
    expect(result).not.toBeNull()
    expect(result!.compactions).toHaveLength(1)
    expect(result!.compactions[0]).toMatchObject({
      beforeTokens: 175000,
      afterTokens: 0,
      reduction: 175000,
      reductionPercent: 100,
    })
    expect(result!.summary.compactions).toBe(1)
  })

  it('aggregates retries from pattern.retry events', () => {
    const stats = makeStats({ prefillTokens: 100, llmCalls: [makeCall({ promptTokens: 100 })] })
    const messages = [makeMessage('m1', stats, '2024-01-01T00:00:00Z')]
    const events = [
      makeEvent(
        'pattern.retry',
        { pattern: '<tool_call', field: 'content', attempt: 1, maxAttempts: 10, messageId: 'm1' },
        Date.parse('2024-01-01T00:01:00Z'),
      ),
    ]
    const result = computeObservability(messages, events)
    expect(result).not.toBeNull()
    expect(result!.retries.length).toBeGreaterThan(0)
    expect(result!.retries[0]).toMatchObject({ type: 'pattern', pattern: '<tool_call' })
  })

  it('aggregates tool activity from tool.call + tool.result events', () => {
    const stats = makeStats({ prefillTokens: 100, llmCalls: [makeCall({ promptTokens: 100 })] })
    const messages = [makeMessage('m1', stats, '2024-01-01T00:00:00Z')]
    const events = [
      makeEvent('message.start', { messageId: 'm1' }, 0),
      makeEvent('message.done', { messageId: 'm1' }, 1),
      makeEvent('tool.call', { messageId: 'm1', toolCall: { id: 'tc1', name: 'read_file' } }, 2),
      makeEvent('tool.result', { messageId: 'm1', toolCallId: 'tc1', result: { success: true, durationMs: 12 } }, 3),
      makeEvent('tool.call', { messageId: 'm1', toolCall: { id: 'tc2', name: 'run_command' } }, 4),
      makeEvent(
        'tool.result',
        { messageId: 'm1', toolCallId: 'tc2', result: { success: false, durationMs: 8, error: 'x' } },
        5,
      ),
    ]
    const result = computeObservability(messages, events)
    expect(result).not.toBeNull()
    expect(result!.toolActivity.totalCount).toBe(2)
    expect(result!.toolActivity.totalErrors).toBe(1)
    expect(result!.toolActivity.byCategory.read).toBe(1)
    expect(result!.toolActivity.byCategory.shell).toBe(1)
  })

  it('groups multi-model sessions into modelBreakdown', () => {
    const a = makeStats({
      providerId: 'p1',
      providerName: 'Local',
      model: 'm1',
      prefillTokens: 100,
      llmCalls: [makeCall({ providerId: 'p1', model: 'm1', promptTokens: 100 })],
    })
    const b = makeStats({
      providerId: 'p2',
      providerName: 'Cloud',
      model: 'm2',
      prefillTokens: 200,
      llmCalls: [makeCall({ providerId: 'p2', providerName: 'Cloud', model: 'm2', promptTokens: 200 })],
    })
    const messages = [makeMessage('m1', a, '2024-01-01T00:00:00Z'), makeMessage('m2', b, '2024-01-01T00:01:00Z')]
    const result = computeObservability(messages, [])
    expect(result).not.toBeNull()
    expect(result!.modelBreakdown).toHaveLength(2)
    const labels = result!.modelBreakdown.map((mb) => mb.label).sort()
    expect(labels).toEqual(['Cloud > m2', 'Local > m1'])
  })

  it('handles legacy sessions without new stats fields', () => {
    // Legacy MessageStats with no retryCount, no cacheSource, no llmCalls
    const legacy: MessageStats = {
      providerId: 'p1',
      providerName: 'Local',
      backend: 'vllm',
      model: 'm',
      mode: 'planner',
      totalTime: 5,
      toolTime: 0,
      prefillTokens: 100,
      prefillSpeed: 50,
      generationTokens: 10,
      generationSpeed: 100,
    }
    const messages = [makeMessage('m1', legacy, '2024-01-01T00:00:00Z')]
    const result = computeObservability(messages, [])
    expect(result).not.toBeNull()
    expect(result!.summary.llmCalls).toBe(0) // no llmCalls on legacy stats
    expect(result!.summary.cacheSource).toBe('unavailable')
  })

  it('computes P50/P95/Max from call promptTokens distribution', () => {
    const calls = [100, 200, 300, 400, 500, 6000].map((n, i) => makeCall({ callIndex: i + 1, promptTokens: n }))
    const stats = makeStats({ prefillTokens: calls.reduce((s, c) => s + c.promptTokens!, 0), llmCalls: calls })
    const messages = [makeMessage('m1', stats, '2024-01-01T00:00:00Z')]
    const result = computeObservability(messages, [])
    expect(result).not.toBeNull()
    expect(result!.summary.contextMax).toBe(6000)
    expect(result!.summary.contextP50).toBeGreaterThan(0)
    expect(result!.summary.contextP95).toBeGreaterThan(result!.summary.contextP50)
  })

  it('schemaVersion is obs.v1', () => {
    const stats = makeStats({ prefillTokens: 100, llmCalls: [makeCall({ promptTokens: 100 })] })
    const messages = [makeMessage('m1', stats, '2024-01-01T00:00:00Z')]
    const result = computeObservability(messages, [])
    expect(result!.schemaVersion).toBe('obs.v1')
  })

  // ===========================================================================
  // 4 fixes required by delivery:
  //   1. CAF handles mixed provider/unavailable sessions without hiding data.
  //   2. contextBefore reflects real growth→compaction→growth evolution.
  //   3. cacheSource strictly distinguishes provider / estimated / unavailable
  //      and accepts cachedTokens=0 as a valid provider measurement.
  //   4. modelBreakdown scopes compactions and tool activity per model
  //      (no duplication across models).
  // ===========================================================================

  it('CAF surfaces partial attribution for mixed provider/unavailable sessions', () => {
    // Response 1 has provider cache data (e.g. 1000 prompt, 800 cached).
    // Response 2 has no cache data (estimated or unavailable).
    const providerStats = makeStats({
      prefillTokens: 1000,
      cachedPromptTokens: 800,
      cacheSource: 'provider',
      llmCalls: [makeCall({ promptTokens: 1000, cachedPromptTokens: 800, cacheSource: 'provider' })],
    })
    const unavailableStats = makeStats({
      prefillTokens: 1200,
      cacheSource: 'unavailable',
      llmCalls: [makeCall({ promptTokens: 1200, cacheSource: 'unavailable' })],
    })
    const messages = [
      makeMessage('m1', providerStats, '2024-01-01T00:00:00Z'),
      makeMessage('m2', unavailableStats, '2024-01-01T00:01:00Z'),
    ]
    const result = computeObservability(messages, [])
    expect(result).not.toBeNull()
    expect(result!.summary.cacheSource).toBe('unavailable') // mixed → unavailable
    // CAF must still be surfaced as a lower bound on the provider subset
    // (raw=2200, uncached-from-provider=200 → 11x).
    expect(result!.summary.contextAmplificationFactor).toBeCloseTo(11, 5)
    expect(result!.summary.amplificationSource).toBe('partial')
    // providerCacheHitRatio is only 'provider' when ALL calls are provider.
    expect(result!.summary.providerCacheHitRatio).toBeUndefined()
  })

  it('CAF is undefined when no call has provider cache data', () => {
    const stats1 = makeStats({
      prefillTokens: 1000,
      cacheSource: 'unavailable',
      llmCalls: [makeCall({ promptTokens: 1000, cacheSource: 'unavailable' })],
    })
    const stats2 = makeStats({
      prefillTokens: 1500,
      cacheSource: 'unavailable',
      llmCalls: [makeCall({ promptTokens: 1500, cacheSource: 'unavailable' })],
    })
    const messages = [
      makeMessage('m1', stats1, '2024-01-01T00:00:00Z'),
      makeMessage('m2', stats2, '2024-01-01T00:01:00Z'),
    ]
    const result = computeObservability(messages, [])
    expect(result).not.toBeNull()
    expect(result!.summary.contextAmplificationFactor).toBeUndefined()
    expect(result!.summary.amplificationSource).toBeUndefined()
  })

  it('contextBefore tracks real growth → compaction → growth evolution', () => {
    const stats1 = makeStats({
      prefillTokens: 50000,
      llmCalls: [makeCall({ promptTokens: 50000 })],
    })
    const stats2 = makeStats({
      prefillTokens: 175000,
      llmCalls: [makeCall({ promptTokens: 175000 })],
    })
    const stats3 = makeStats({
      prefillTokens: 31000,
      llmCalls: [makeCall({ promptTokens: 31000 })],
    })
    const stats4 = makeStats({
      prefillTokens: 112000,
      llmCalls: [makeCall({ promptTokens: 112000 })],
    })
    const messages = [
      makeMessage('m1', stats1, '2024-01-01T00:00:00Z'),
      makeMessage('m2', stats2, '2024-01-01T00:01:00Z'),
      makeMessage('m3', stats3, '2024-01-01T00:02:00Z'),
      makeMessage('m4', stats4, '2024-01-01T00:03:00Z'),
    ]
    // Compaction happens after response 2 (175k → 0/31k).
    const events = [
      makeEvent(
        'context.compacted',
        {
          closedWindowId: 'w2',
          newWindowId: 'w3',
          beforeTokens: 175000,
          afterTokens: 0,
          summary: 'compact',
        },
        Date.parse('2024-01-01T00:01:30Z'),
      ),
    ]
    const result = computeObservability(messages, events)
    expect(result).not.toBeNull()
    const r = result!.responses
    expect(r).toHaveLength(4)
    // Growth phase.
    expect(r[0]!.contextBefore).toBe(0)
    expect(r[0]!.contextAfter).toBe(50000)
    expect(r[1]!.contextBefore).toBe(50000) // continues from previous
    expect(r[1]!.contextAfter).toBe(175000)
    // Compaction phase: response 3 starts from compaction's afterTokens.
    expect(r[2]!.contextBefore).toBe(0)
    expect(r[2]!.contextAfter).toBe(31000)
    // Growth resumes after compaction.
    expect(r[3]!.contextBefore).toBe(31000)
    expect(r[3]!.contextAfter).toBe(112000)
  })

  it('contextBefore resets to 0 when a non-monotonic prefill is detected (legacy safety)', () => {
    // Legacy / corrupted data: second response's prefillTokens is lower than the first
    // without an intervening compaction. The evolver should clamp to 0 instead of
    // emitting a misleading negative delta.
    const stats1 = makeStats({
      prefillTokens: 100000,
      llmCalls: [makeCall({ promptTokens: 100000 })],
    })
    const stats2 = makeStats({
      prefillTokens: 50000, // regression without compaction
      llmCalls: [makeCall({ promptTokens: 50000 })],
    })
    const messages = [
      makeMessage('m1', stats1, '2024-01-01T00:00:00Z'),
      makeMessage('m2', stats2, '2024-01-01T00:01:00Z'),
    ]
    const result = computeObservability(messages, [])
    expect(result).not.toBeNull()
    // m2 sees runningContext regressed; we clamp to 0 so the dashboard doesn't
    // show a negative growth.
    expect(result!.responses[1]!.contextBefore).toBe(0)
    expect(result!.responses[1]!.contextAfter).toBe(50000)
  })

  it('modelBreakdown scopes compactions and tools per model — no duplication', () => {
    // Model A runs response 1 + 2, performs a compaction, runs 4 tools.
    // Model B runs response 3, no compactions, 2 tools.
    // Without per-model scoping, compactions would appear in both models
    // and tools would be counted 4+2 in each.
    const aStats1 = makeStats({
      providerId: 'pA',
      providerName: 'ProviderA',
      model: 'mA',
      prefillTokens: 50000,
      llmCalls: [makeCall({ providerId: 'pA', model: 'mA', promptTokens: 50000 })],
    })
    const aStats2 = makeStats({
      providerId: 'pA',
      providerName: 'ProviderA',
      model: 'mA',
      prefillTokens: 120000,
      llmCalls: [makeCall({ providerId: 'pA', model: 'mA', promptTokens: 120000 })],
    })
    const bStats = makeStats({
      providerId: 'pB',
      providerName: 'ProviderB',
      model: 'mB',
      prefillTokens: 30000,
      llmCalls: [makeCall({ providerId: 'pB', model: 'mB', promptTokens: 30000 })],
    })
    const messages = [
      makeMessage('m1', aStats1, '2024-01-01T00:00:00Z'),
      makeMessage('m2', aStats2, '2024-01-01T00:01:00Z'),
      makeMessage('m3', bStats, '2024-01-01T00:02:00Z'),
    ]
    const events = [
      // 4 tools under model A (responses 1 & 2).
      makeEvent('message.start', { messageId: 'm1' }, 0),
      makeEvent('message.done', { messageId: 'm1' }, 1),
      makeEvent('tool.call', { messageId: 'm1', toolCall: { id: 'ta1', name: 'read_file' } }, 2),
      makeEvent('tool.result', { messageId: 'm1', toolCallId: 'ta1', result: { success: true, durationMs: 5 } }, 3),
      makeEvent('message.done', { messageId: 'm2' }, 4),
      makeEvent('tool.call', { messageId: 'm2', toolCall: { id: 'ta2', name: 'read_file' } }, 5),
      makeEvent('tool.result', { messageId: 'm2', toolCallId: 'ta2', result: { success: true, durationMs: 5 } }, 6),
      makeEvent('tool.call', { messageId: 'm2', toolCall: { id: 'ta3', name: 'grep' } }, 7),
      makeEvent('tool.result', { messageId: 'm2', toolCallId: 'ta3', result: { success: true, durationMs: 5 } }, 8),
      makeEvent('tool.call', { messageId: 'm2', toolCall: { id: 'ta4', name: 'shell' } }, 9),
      makeEvent('tool.result', { messageId: 'm2', toolCallId: 'ta4', result: { success: true, durationMs: 5 } }, 10),
      // 2 tools under model B (response 3).
      makeEvent('message.done', { messageId: 'm3' }, 11),
      makeEvent('tool.call', { messageId: 'm3', toolCall: { id: 'tb1', name: 'edit_file' } }, 12),
      makeEvent('tool.result', { messageId: 'm3', toolCallId: 'tb1', result: { success: true, durationMs: 5 } }, 13),
      makeEvent('tool.call', { messageId: 'm3', toolCall: { id: 'tb2', name: 'shell' } }, 14),
      makeEvent('tool.result', { messageId: 'm3', toolCallId: 'tb2', result: { success: true, durationMs: 5 } }, 15),
      // Compaction happens during model A's run.
      makeEvent(
        'context.compacted',
        {
          closedWindowId: 'w1',
          newWindowId: 'w2',
          beforeTokens: 120000,
          afterTokens: 0,
          summary: 'compact',
        },
        Date.parse('2024-01-01T00:01:30Z'),
      ),
    ]
    const result = computeObservability(messages, events)
    expect(result).not.toBeNull()
    // Top-level totals are unchanged.
    expect(result!.toolActivity.totalCount).toBe(6) // 4 + 2
    expect(result!.compactions.length).toBe(1)
    // Per-model breakdown is correctly scoped.
    expect(result!.modelBreakdown).toHaveLength(2)
    const modelA = result!.modelBreakdown.find((m) => m.model === 'mA')
    const modelB = result!.modelBreakdown.find((m) => m.model === 'mB')
    expect(modelA).toBeDefined()
    expect(modelB).toBeDefined()
    // Model A scoped: 4 tools, 1 compaction.
    expect(modelA!.summary.toolCalls).toBe(4)
    expect(modelA!.summary.compactions).toBe(1)
    // Model B scoped: 2 tools, 0 compactions.
    expect(modelB!.summary.toolCalls).toBe(2)
    expect(modelB!.summary.compactions).toBe(0)
    // Total of per-model tools == 6 (no duplication).
    const totalToolsPerModel = modelA!.summary.toolCalls + modelB!.summary.toolCalls
    expect(totalToolsPerModel).toBe(result!.toolActivity.totalCount)
    // Total of per-model compactions == 1.
    const totalCompactionsPerModel = modelA!.summary.compactions + modelB!.summary.compactions
    expect(totalCompactionsPerModel).toBe(result!.compactions.length)
  })

  it('modelBreakdown excludes sub-agent compactions from top-level model scope', () => {
    const aStats = makeStats({
      providerId: 'pA',
      providerName: 'ProviderA',
      model: 'mA',
      prefillTokens: 1000,
      llmCalls: [makeCall({ providerId: 'pA', model: 'mA', promptTokens: 1000 })],
    })
    const messages = [makeMessage('m1', aStats, '2024-01-01T00:00:00Z')]
    const events = [
      // Top-level compaction (no subAgentId): counted in model A.
      makeEvent(
        'context.compacted',
        {
          closedWindowId: 'w1',
          newWindowId: 'w2',
          beforeTokens: 1000,
          afterTokens: 0,
          summary: 'top',
        },
        Date.parse('2024-01-01T00:00:30Z'),
      ),
      // Sub-agent compaction (subAgentId present): NOT counted in model A.
      makeEvent(
        'context.compacted',
        {
          closedWindowId: 'sa1',
          newWindowId: 'sa2',
          beforeTokens: 5000,
          afterTokens: 0,
          summary: 'subagent',
          subAgentId: 'sa-id',
          subAgentType: 'explorer',
        },
        Date.parse('2024-01-01T00:00:45Z'),
      ),
    ]
    const result = computeObservability(messages, events)
    expect(result).not.toBeNull()
    // Top-level keeps both.
    expect(result!.compactions).toHaveLength(2)
    // Per-model excludes the sub-agent compaction.
    expect(result!.modelBreakdown[0]!.summary.compactions).toBe(1)
  })
})
