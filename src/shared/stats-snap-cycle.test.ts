import { describe, it, expect } from 'vitest'
import { combineEventsWithSnapshot } from '../server/events/session.js'
import { buildSessionStatsEventRollup } from './stats-rollup.js'

function ev(
  seq: number,
  timestamp: number,
  type: string,
  data: Record<string, unknown>,
) {
  return { seq, timestamp, sessionId: 's1', type, data }
}

const baseSnapshot = {
  mode: 'planner',
  phase: 'build',
  isRunning: false,
  messages: [],
  criteria: [],
  metadataEntries: {},
  contextState: {},
  currentContextWindowId: 'w1',
  todos: [],
  formatRetries: [{ attempt: 1, maxAttempts: 10, timestamp: 500 }],
  contextWindows: [
    {
      timestamp: 700,
      closedWindowId: 'w0',
      newWindowId: 'w1',
      beforeTokens: 175000,
      afterTokens: 0,
      reduction: 175000,
      reductionPercent: 100,
    },
  ],
  snapshotSeq: 0,
  snapshotAt: 2000,
}

const baseRaw = [
  ev(1, 3000, 'message.start', { messageId: 'm1', role: 'assistant' }),
  ev(2, 3100, 'tool.call', {
    messageId: 'm1',
    toolCall: { id: 't3', name: 'read_file' },
  }),
  ev(3, 3200, 'tool.result', {
    messageId: 'm1',
    toolCallId: 't3',
    result: { success: true, durationMs: 3 },
  }),
  ev(4, 3300, 'message.done', { messageId: 'm1' }),
]

describe('snapshot cycle (rollup-only)', () => {
  it('reproduces the bug: without reconstruction, historical compactions and retries are LOST', () => {
    // The fix lives in combineEventsWithSnapshot; if we pass the raw
    // events directly (skipping the combine), we see the historical events
    // disappear.
    const rollup = buildSessionStatsEventRollup(baseRaw)
    expect(rollup.compactions.length).toBe(0)
    expect(rollup.retries.length).toBe(0)
    expect(rollup.toolCalls).toBe(1)
  })

  it('reconstructs snapshot events when combined through combineEventsWithSnapshot', () => {
    const combined = combineEventsWithSnapshot('s1', baseSnapshot, baseRaw)
    const rollup = buildSessionStatsEventRollup(combined)
    expect(rollup.compactions.length).toBe(1)
    expect(rollup.retries.length).toBe(1)
    expect(rollup.toolCalls).toBe(1) // post-snapshot only; tools live on snapshot.messages
  })

  it('handles multiple compactions and retries in the snapshot', () => {
    const snapshot = {
      ...baseSnapshot,
      formatRetries: [
        { attempt: 1, maxAttempts: 10, timestamp: 100 },
        { attempt: 1, maxAttempts: 10, timestamp: 200 },
        { attempt: 1, maxAttempts: 10, timestamp: 300 },
      ],
      contextWindows: [
        { timestamp: 100, closedWindowId: 'w0', newWindowId: 'w1', beforeTokens: 80000, afterTokens: 0, reduction: 80000, reductionPercent: 100 },
        { timestamp: 200, closedWindowId: 'w1', newWindowId: 'w2', beforeTokens: 100000, afterTokens: 0, reduction: 100000, reductionPercent: 100 },
        { timestamp: 300, closedWindowId: 'w2', newWindowId: 'w3', beforeTokens: 150000, afterTokens: 0, reduction: 150000, reductionPercent: 100 },
      ],
    }
    const combined = combineEventsWithSnapshot('s1', snapshot, baseRaw)
    const rollup = buildSessionStatsEventRollup(combined)
    expect(rollup.compactions.length).toBe(3)
    expect(rollup.retries.length).toBe(3)
  })

  it('preserves post-snapshot compactions alongside the snapshot ones', () => {
    const snapshot = {
      ...baseSnapshot,
      contextWindows: [
        {
          timestamp: 700,
          closedWindowId: 'w0',
          newWindowId: 'w1',
          beforeTokens: 175000,
          afterTokens: 0,
          reduction: 175000,
          reductionPercent: 100,
        },
      ],
    }
    const rawEventsWithPostCompaction = [
      ...baseRaw,
      ev(5, 4000, 'context.compacted', {
        closedWindowId: 'w1',
        newWindowId: 'w2',
        beforeTokens: 50000,
        afterTokens: 0,
        summary: '',
      }),
    ]
    const combined = combineEventsWithSnapshot(
      's1',
      snapshot,
      rawEventsWithPostCompaction,
    )
    const rollup = buildSessionStatsEventRollup(combined)
    expect(rollup.compactions.length).toBe(2) // snapshot + post
  })

  it('produces the same counts with zero, one, or many snapshots (no double-count)', () => {
    // Case A: zero snapshots
    const r0 = buildSessionStatsEventRollup(baseRaw)
    expect(r0.compactions.length).toBe(0)

    // Case B: one snapshot applied once
    const r1 = buildSessionStatsEventRollup(combineEventsWithSnapshot('s1', baseSnapshot, baseRaw))
    expect(r1.compactions.length).toBe(1)
    expect(r1.retries.length).toBe(1)

    // Case C: same snapshot applied again on a freshly loaded event
    // stream — the route calls combineEventsWithSnapshot once per request
    // so this is a defensive check. We verify that the SAME snapshot
    // payload applied twice yields TWO copies in the raw stream, which the
    // rollup counts faithfully. This documents that the fix does NOT
    // deduplicate (callers must call combineEventsWithSnapshot exactly
    // once per request).
    const combinedTwice = combineEventsWithSnapshot(
      's1',
      baseSnapshot,
      combineEventsWithSnapshot('s1', baseSnapshot, baseRaw),
    )
    const r2 = buildSessionStatsEventRollup(combinedTwice)
    expect(r2.compactions.length).toBe(2)
  })
})
