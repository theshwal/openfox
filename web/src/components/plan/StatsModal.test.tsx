// @vitest-environment happy-dom
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { StatsModal } from './StatsModal'
import { computeSessionStats, computeSessionStatsSummary } from '@shared/stats.js'
import { authFetch } from '../../lib/api'
import type { Message, SessionStatsSummary } from '@shared/types.js'

vi.mock('../../lib/api', () => ({
  authFetch: vi.fn(),
}))

vi.mock('../shared/SelfContainedModal', () => ({
  Modal: ({ children, isOpen }: { children: React.ReactNode; isOpen?: boolean }) =>
    isOpen ? <div>{children}</div> : null,
}))

vi.mock('../shared/Sparkline', () => ({
  DualSparkline: () => null,
}))

const authFetchMock = vi.mocked(authFetch)

function message(id: string, totalTime: number, toolTime: number, prefill: number, gen: number): Message {
  return {
    id,
    role: 'assistant',
    content: 'done',
    timestamp: `2024-01-01T10:00:${id.length}Z`,
    stats: {
      providerId: 'p1',
      providerName: 'P',
      backend: 'vllm',
      model: 'm1',
      mode: 'builder',
      totalTime,
      toolTime,
      prefillTokens: prefill,
      prefillSpeed: 10000,
      generationTokens: gen,
      generationSpeed: 150,
    },
  }
}

function summaryFor(count: number): SessionStatsSummary {
  const messages = Array.from({ length: count }, (_, i) => message(`m${i}`, 10, 2, 50000, 500))
  return computeSessionStatsSummary(messages)!
}

beforeEach(() => {
  authFetchMock.mockReset()
})

describe('StatsModal', () => {
  it('renders the summary cards from the lean payload without fetching the full log for large sessions', () => {
    const summary = summaryFor(200)

    render(<StatsModal isOpen onClose={() => {}} summary={summary} sessionId="s1" />)

    expect(screen.getByText('Summary')).toBeTruthy()
    expect(screen.getByText('200')).toBeTruthy()
    expect(screen.getByText(/Load full stats/i)).toBeTruthy()
    expect(authFetchMock).not.toHaveBeenCalled()
  })

  it('loads the full response log on demand and renders it once', async () => {
    const summary = summaryFor(200)
    const fullStats = computeSessionStatsSummary(
      Array.from({ length: 200 }, (_, i) => message(`m${i}`, 10, 2, 50000, 500)),
    )!
    authFetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        stats: {
          ...fullStats,
          dataPoints: [],
          callDataPoints: [],
          modelGroups: fullStats.modelGroups.map((g) => ({ ...g, dataPoints: [], callDataPoints: [] })),
        },
      }),
    } as Response)

    const { rerender } = render(<StatsModal isOpen onClose={() => {}} summary={summary} sessionId="s1" />)
    fireEvent.click(screen.getByText(/Load full stats/i))

    await waitFor(() => expect(authFetchMock).toHaveBeenCalledTimes(1))
    expect(authFetchMock).toHaveBeenCalledWith('/api/sessions/s1/stats')

    // Response log renders after the fetch resolves
    await waitFor(() => expect(screen.getByText(/Response Log \(\d+ responses\)/i)).toBeTruthy())

    // Re-open with the same session keeps the cached detail — no second fetch.
    rerender(<StatsModal isOpen onClose={() => {}} summary={summary} sessionId="s1" />)
    expect(authFetchMock).toHaveBeenCalledTimes(1)
  })

  it('auto-loads the full log for small sessions without a button', async () => {
    const summary = summaryFor(3)
    authFetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        stats: {
          ...summary,
          dataPoints: [],
          callDataPoints: [],
          modelGroups: summary.modelGroups.map((g) => ({ ...g, dataPoints: [], callDataPoints: [] })),
        },
      }),
    } as Response)

    render(<StatsModal isOpen onClose={() => {}} summary={summary} sessionId="s1" />)

    await waitFor(() => expect(authFetchMock).toHaveBeenCalledTimes(1))
    expect(screen.queryByText(/Load full stats/i)).toBeNull()
    await waitFor(() => expect(screen.getByText(/Response Log \(\d+ responses\)/i)).toBeTruthy())
  })

  it('surfaces a load failure without crashing', async () => {
    const summary = summaryFor(200)
    authFetchMock.mockResolvedValue({ ok: false, status: 500 } as Response)

    render(<StatsModal isOpen onClose={() => {}} summary={summary} sessionId="s1" />)
    fireEvent.click(screen.getByText(/Load full stats/i))

    await waitFor(() => expect(screen.getByText(/HTTP 500/i)).toBeTruthy())
  })

  it('discards a stale full-stats response that lands after a session switch', async () => {
    const summary = summaryFor(200)
    let resolveFetch!: (value: Response) => void
    authFetchMock.mockReturnValue(
      new Promise<Response>((resolve) => {
        resolveFetch = resolve
      }),
    )

    const { rerender } = render(<StatsModal isOpen onClose={() => {}} summary={summary} sessionId="s1" />)
    fireEvent.click(screen.getByText(/Load full stats/i))

    // Switch sessions while the fetch is still in flight.
    rerender(<StatsModal isOpen onClose={() => {}} summary={summary} sessionId="s2" />)

    resolveFetch({
      ok: true,
      json: async () => ({
        stats: {
          ...summary,
          dataPoints: [],
          callDataPoints: [],
          modelGroups: summary.modelGroups.map((g) => ({ ...g, dataPoints: [], callDataPoints: [] })),
        },
      }),
    } as Response)

    await waitFor(() => expect(authFetchMock).toHaveBeenCalledTimes(1))
    // The stale response must not surface the old session's response log.
    expect(screen.queryByText(/Response Log \(\d+ responses\)/i)).toBeNull()
  })
})

// =============================================================================
// Observability UI rendering tests
// =============================================================================

interface FullStatsOptions {
  cachedPromptTokens?: number
  cacheWriteTokens?: number
  cacheSource?: 'provider' | 'estimated' | 'unavailable'
  prompt?: number
}

function fullStatsWithCache(opts: FullStatsOptions) {
  const prompt = opts.prompt ?? 10000
  const cacheSource: 'provider' | 'estimated' | 'unavailable' =
    opts.cacheSource ?? (opts.cachedPromptTokens !== undefined ? 'provider' : 'unavailable')
  const cachedField = opts.cachedPromptTokens !== undefined ? opts.cachedPromptTokens : undefined
  const writeField = opts.cacheWriteTokens !== undefined ? opts.cacheWriteTokens : undefined
  const messages: Message[] = Array.from({ length: 3 }, (_, i) => {
    const m = message('m' + i, 10, 2, prompt, 500)
    return {
      ...m,
      stats: {
        ...m.stats!,
        cachedPromptTokens: cachedField,
        cacheWriteTokens: writeField,
        cacheSource,
        retryCount: 0,
        compactionCount: 0,
        llmCalls: [
          {
            providerId: 'p1',
            providerName: 'P',
            backend: 'vllm' as const,
            model: 'm1',

            callIndex: 1,
            promptTokens: prompt,
            completionTokens: 500,
            ttft: 0.1,
            completionTime: 0.5,
            prefillSpeed: 10000,
            generationSpeed: 100,
            totalTime: 0.6,
            cachedPromptTokens: cachedField,
            cacheWriteTokens: writeField,
            cacheSource,
            contextSize: prompt,
            retries: 0,
            timestamp: '2024-01-01T10:00:0' + i + 'Z',
          },
        ],
      },
    }
  })
  return computeSessionStats(messages)!
}

async function renderModalWithFullStats(fullStats: unknown) {
  authFetchMock.mockResolvedValue({
    ok: true,
    json: async () => ({ stats: fullStats }),
  } as Response)
  const summary = summaryFor(3)
  const view = render(<StatsModal isOpen onClose={() => {}} summary={summary} sessionId="s1" />)
  // Wait until the lazy fetch resolves and the modal re-renders with observability.
  await waitFor(() => expect(screen.queryAllByText(/Observability overview/i).length).toBeGreaterThan(0))
  return view
}

describe('StatsModal — observability UI', () => {
  it('renders the Provider cache source badge when cache data is available', async () => {
    const full = fullStatsWithCache({
      cachedPromptTokens: 9000,
      cacheWriteTokens: 1000,
      cacheSource: 'provider',
    })
    await renderModalWithFullStats(full)
    // The Provider badge appears next to the overview heading AND next to per-model headings.
    const badges = screen.getAllByText('Provider')
    expect(badges.length).toBeGreaterThan(0)
    // The Overview section shows raw prompt and provider cache read.
    expect(screen.getAllByText(/Raw prompt/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/Provider cache read/i).length).toBeGreaterThan(0)
  })

  it('renders N/A badge and hides the cache hit ratio when no cache data is present', async () => {
    const full = fullStatsWithCache({ cacheSource: 'unavailable' })
    await renderModalWithFullStats(full)
    expect(screen.getAllByText('N/A').length).toBeGreaterThan(0)
    // Cache hit ratio subValue should NOT mention a percentage.
    expect(screen.queryByText(/hit \(provider-reported\)/i)).toBeNull()
  })

  it('shows a "partial" source label in the Context Amplification subValue when cacheSource is mixed', async () => {
    // Mixed session: one call with provider cache, one without.
    const messages: Message[] = [
      {
        ...message('m0', 10, 2, 10000, 500),
        stats: {
          ...message('m0', 10, 2, 10000, 500).stats!,
          cachedPromptTokens: 9000,
          cacheSource: 'provider' as const,
          llmCalls: [
            {
              providerId: 'p1',
              providerName: 'P',
              backend: 'vllm' as const,
              model: 'm1',

              callIndex: 1,
              promptTokens: 10000,
              completionTokens: 500,
              ttft: 0.1,
              completionTime: 0.5,
              prefillSpeed: 10000,
              generationSpeed: 100,
              totalTime: 0.6,
              cachedPromptTokens: 9000,
              cacheSource: 'provider' as const,
              contextSize: 10000,
              retries: 0,
              timestamp: '2024-01-01T10:00:00Z',
            },
          ],
        },
      },
      {
        ...message('m1', 10, 2, 500, 500),
        stats: {
          ...message('m1', 10, 2, 500, 500).stats!,
          cacheSource: 'unavailable' as const,
          llmCalls: [
            {
              providerId: 'p1',
              providerName: 'P',
              backend: 'vllm' as const,
              model: 'm1',

              callIndex: 1,
              promptTokens: 500,
              completionTokens: 500,
              ttft: 0.1,
              completionTime: 0.5,
              prefillSpeed: 5000,
              generationSpeed: 100,
              totalTime: 0.6,
              cacheSource: 'unavailable' as const,
              contextSize: 500,
              retries: 0,
              timestamp: '2024-01-01T10:00:01Z',
            },
          ],
        },
      },
    ]
    const full = computeSessionStats(messages)!
    await renderModalWithFullStats(full)
    // The badge shows N/A because mixed resolves to 'unavailable'.
    expect(screen.getAllByText('N/A').length).toBeGreaterThan(0)
    // The amplification subValue explicitly mentions partial coverage.
    expect(screen.getAllByText(/source: partial/i).length).toBeGreaterThan(0)
  })

  it('renders per-call cache attribution through the CallDataPointRow when expanded', async () => {
    // Build a fullStats with multiple calls per response so the row is expandable.
    const messages: Message[] = [
      {
        ...message('m0', 10, 2, 10000, 500),
        stats: {
          ...message('m0', 10, 2, 10000, 500).stats!,
          cachedPromptTokens: 9000,
          cacheSource: 'provider' as const,
          llmCalls: [
            {
              providerId: 'p1',
              providerName: 'P',
              backend: 'vllm' as const,
              model: 'm1',

              callIndex: 1,
              promptTokens: 10000,
              completionTokens: 500,
              ttft: 0.1,
              completionTime: 0.5,
              prefillSpeed: 10000,
              generationSpeed: 100,
              totalTime: 0.6,
              cachedPromptTokens: 9000,
              cacheSource: 'provider' as const,
              contextSize: 10000,
              retries: 0,
              timestamp: '2024-01-01T10:00:00Z',
            },
            {
              providerId: 'p1',
              providerName: 'P',
              backend: 'vllm' as const,
              model: 'm1',

              callIndex: 2,
              promptTokens: 11000,
              completionTokens: 500,
              ttft: 0.1,
              completionTime: 0.5,
              prefillSpeed: 11000,
              generationSpeed: 100,
              totalTime: 0.6,
              cachedPromptTokens: 10000,
              cacheSource: 'provider' as const,
              contextSize: 11000,
              retries: 0,
              timestamp: '2024-01-01T10:00:01Z',
            },
          ],
        },
      },
    ]
    const full = computeSessionStats(messages)!
    const { container } = await renderModalWithFullStats(full)
    expect(container.textContent).toMatch(/Provider cache read/)
    expect(container.textContent).toMatch(/cached portion/i)
    // The first response row must have at least 1 call carrying cache attribution.
    const firstModelGroup = full.modelGroups[0]!
    expect(firstModelGroup.callDataPoints.length).toBeGreaterThan(0)
    expect(firstModelGroup.callDataPoints[0]!.cachedPromptTokens).toBe(9000)
    expect(firstModelGroup.callDataPoints[0]!.cacheSource).toBe('provider')
  })

  it('does not crash on legacy session without observability fields', async () => {
    // Legacy fullStats: no cache fields at all on any call.
    const prompt = 10000
    const base = computeSessionStatsSummary(Array.from({ length: 3 }, (_, i) => message('m' + i, 10, 2, prompt, 500)))!
    const full = {
      ...base,
      dataPoints: [],
      callDataPoints: [],
      modelGroups: base.modelGroups.map((g) => ({
        ...g,
        dataPoints: [],
        callDataPoints: [
          {
            messageId: 'm0',
            timestamp: '2024-01-01T10:00:00Z',
            providerId: 'p1',
            providerName: 'P',
            backend: 'vllm' as const,
            model: 'm1',

            responseIndex: 1,
            sessionCallIndex: 1,
            callIndex: 1,
            promptTokens: prompt,
            completionTokens: 500,
            ttft: 0.1,
            completionTime: 0.5,
            prefillSpeed: 10000,
            generationSpeed: 100,
            totalTime: 0.6,
            contextSize: prompt,
            retries: 0,
          },
        ],
      })),
    }
    // No exception, all key sections render.
    await renderModalWithFullStats(full)
    expect(screen.getAllByText(/Raw prompt/i).length).toBeGreaterThan(0)
    // Legacy path shows N/A for cache attribution, never crashes.
    expect(screen.getAllByText('N/A').length).toBeGreaterThan(0)
  })
})
