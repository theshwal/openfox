import { ScrollArea } from '../shared/ScrollArea'
import { getLocale } from '@shared/i18n/index.js'
import { useT } from '../../hooks/useT'
import { Fragment, useRef, useCallback, useEffect, useMemo, useState } from 'react'
import { useObservability } from '../../hooks/useObservability'
import { Modal } from '../shared/SelfContainedModal'
import { DualSparkline } from '../shared/Sparkline'
import { buildPerformanceChartData, buildResponseLogRows, type ResponseLogRow } from '@shared/stats-view.js'
import type {
  CallStatsDataPoint,
  CacheSource,
  ModelSessionStats,
  ObservabilityCallRow,
  SessionStats,
  SessionStatsSummary,
} from '@shared/types.js'
import { formatTime } from '../../lib/format-stats'
import { authFetch } from '../../lib/api'

interface StatsModalProps {
  isOpen: boolean
  onClose: () => void
  summary: SessionStatsSummary | null
  sessionId: string
}

/**
 * Sessions at or below this response count auto-load the full response log
 * (it's cheap); larger sessions defer it behind the "Load full stats" button
 * so the always-on payload stays lean.
 */
const AUTO_LOAD_THRESHOLD = 50

/**
 * Format token count with k/M suffix
 */
function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`
  return n.toString()
}

/**
 * Format speed with k suffix
 */
function formatSpeed(n: number): string {
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`
  return n.toFixed(1)
}

function formatContextRange(tokens: number[]): string {
  if (tokens.length === 0) return '0 ctx'

  const minTokens = Math.min(...tokens)
  const maxTokens = Math.max(...tokens)

  if (minTokens === maxTokens) {
    return `${formatTokens(minTokens)} ctx`
  }

  return `${formatTokens(minTokens)}-${formatTokens(maxTokens)} ctx`
}

function formatRate(value: number): string {
  return `${formatSpeed(value)} t/s`
}

/**
 * Format timestamp to time only (HH:MM:SS)
 */
function formatTimestamp(ts: string): string {
  try {
    const date = new Date(ts)
    return date.toLocaleTimeString(getLocale(), { hour12: false })
  } catch {
    return ts
  }
}

/**
 * Create JSON export data
 */
function createExportData(stats: ModelSessionStats, observability?: { summary: { cacheSource: string } } | null) {
  const base = {
    schemaVersion: 'obs.v1',
    exportedAt: new Date().toISOString(),
    providerId: stats.providerId,
    providerName: stats.providerName,
    backend: stats.backend,
    model: stats.model,
    label: stats.label,
    summary: {
      totalTime: stats.totalTime,
      aiTime: stats.aiTime,
      toolTime: stats.toolTime,
      prefillTokens: stats.prefillTokens,
      generationTokens: stats.generationTokens,
      avgPrefillSpeed: stats.avgPrefillSpeed,
      avgGenerationSpeed: stats.avgGenerationSpeed,
      responseCount: stats.responseCount,
      llmCallCount: stats.llmCallCount,
    },
    responses: stats.dataPoints.map((dp) => ({
      responseIndex: dp.responseIndex,
      timestamp: dp.timestamp,
      mode: dp.mode,
      prefillTokens: dp.prefillTokens,
      generationTokens: dp.generationTokens,
      prefillSpeed: dp.prefillSpeed,
      generationSpeed: dp.generationSpeed,
      totalTime: dp.totalTime,
      aiTime: dp.aiTime,
      toolTime: dp.toolTime,
    })),
    llmCalls: stats.callDataPoints.map((dp) => ({
      sessionCallIndex: dp.sessionCallIndex,
      responseIndex: dp.responseIndex,
      callIndex: dp.callIndex,
      timestamp: dp.timestamp,
      mode: dp.mode,
      promptTokens: dp.promptTokens,
      completionTokens: dp.completionTokens,
      cachedPromptTokens: dp.cachedPromptTokens,
      cacheWriteTokens: dp.cacheWriteTokens,
      cacheSource: dp.cacheSource,
      contextSize: dp.contextSize,
      retries: dp.retries,
      ttft: dp.ttft,
      completionTime: dp.completionTime,
      prefillSpeed: dp.prefillSpeed,
      generationSpeed: dp.generationSpeed,
      totalTime: dp.totalTime,
    })),
  }
  // Embed observability summary for cross-model benchmark stability (obs.v1).
  // Note: events-driven fields (compactions, retries, toolActivity) are not
  // available from a SessionStats fetch — they need the full event stream.
  if (observability && observability.summary) {
    return {
      ...base,
      observability: {
        summary: observability.summary,
      },
    }
  }
  return base
}

export function StatsModal({ isOpen, onClose, summary, sessionId }: StatsModalProps) {
  const t = useT()
  const contentRef = useRef<HTMLDivElement>(null)
  const [expandedResponses, setExpandedResponses] = useState<Record<string, boolean>>({})
  const [selectedModelKey, setSelectedModelKey] = useState(() => summary?.modelGroups[0]?.key ?? '')
  const [fullStats, setFullStats] = useState<SessionStats | null>(null)
  const [loadingFull, setLoadingFull] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)
  // Guards against a stale in-flight fetch landing after a session switch.
  const loadRequestRef = useRef(0)

  // Observability view: prefer the lazily-loaded fullStats (carries the
  // server-computed per-call cache attribution) over messages+events which
  // are not available in the modal scope.
  const observability = useObservability([], null, fullStats)

  const modelGroups = fullStats?.modelGroups ?? summary?.modelGroups ?? []

  useEffect(() => {
    if (!modelGroups.some((group) => group.key === selectedModelKey)) {
      setSelectedModelKey(modelGroups[0]?.key ?? '')
    }
  }, [selectedModelKey, modelGroups])

  // The fetched detail belongs to a specific session — drop it when the pane
  // switches, and invalidate any in-flight request so it can't land late.
  useEffect(() => {
    loadRequestRef.current += 1
    setFullStats(null)
    setLoadError(null)
    setLoadingFull(false)
  }, [sessionId])

  const loadFull = useCallback(async () => {
    if (!sessionId || loadingFull) return
    const requestId = loadRequestRef.current + 1
    loadRequestRef.current = requestId
    setLoadingFull(true)
    setLoadError(null)
    try {
      const res = await authFetch(`/api/sessions/${sessionId}/stats`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = (await res.json()) as { stats: SessionStats | null }
      if (loadRequestRef.current === requestId) {
        setFullStats(data.stats)
      }
    } catch (e) {
      if (loadRequestRef.current === requestId) {
        setLoadError(e instanceof Error ? e.message : String(e))
      }
    } finally {
      if (loadRequestRef.current === requestId) {
        setLoadingFull(false)
      }
    }
  }, [sessionId, loadingFull])

  // Small sessions auto-load the full log so a handful of rows never hides
  // behind a button. Once loaded it stays cached across re-opens.
  useEffect(() => {
    if (!isOpen) return
    if (fullStats || loadingFull || !summary) return
    if (summary.responseCount > 0 && summary.responseCount <= AUTO_LOAD_THRESHOLD) {
      void loadFull()
    }
  }, [isOpen, fullStats, loadingFull, summary, loadFull])

  const currentStats = useMemo(() => {
    if (!fullStats) return undefined
    return fullStats.modelGroups.find((group) => group.key === selectedModelKey) ?? fullStats.modelGroups[0]
  }, [fullStats, selectedModelKey])
  const currentSummary = useMemo(
    () => modelGroups.find((group) => group.key === selectedModelKey) ?? modelGroups[0],
    [modelGroups, selectedModelKey],
  )

  const responseRows = useMemo(() => (currentStats ? buildResponseLogRows(currentStats) : []), [currentStats])
  const chartData = useMemo(
    () =>
      currentStats
        ? buildPerformanceChartData(currentStats)
        : { mode: 'responses', xLabel: 'response', prefillLabel: '', generationLabel: '', points: [] },
    [currentStats],
  )

  const toggleResponse = useCallback((messageId: string) => {
    setExpandedResponses((current) => ({
      ...current,
      [messageId]: !current[messageId],
    }))
  }, [])

  // Copy JSON to clipboard
  const handleCopyJson = useCallback(() => {
    if (!currentStats) return

    const data = createExportData(currentStats, observability)
    navigator.clipboard.writeText(JSON.stringify(data, null, 2)).catch((err) => console.error('Failed to copy:', err))
  }, [currentStats])

  // Export PNG (requires html2canvas)
  const handleExportPng = useCallback(async () => {
    if (!contentRef.current) return

    try {
      // Dynamic import to avoid bundling if not used
      const html2canvas = (await import('html2canvas')).default
      const canvas = await html2canvas(contentRef.current, {
        backgroundColor: '#1a1a1a', // bg-bg-primary
        scale: 2, // Higher resolution
      })

      // Download
      const link = document.createElement('a')
      link.download = `openfox-stats-${new Date().toISOString().slice(0, 10)}.png`
      link.href = canvas.toDataURL('image/png')
      link.click()
    } catch (err) {
      console.error('Failed to export PNG:', err)
      // Fallback: show error or just copy JSON
      handleCopyJson()
    }
  }, [handleCopyJson])

  const detailLoaded = fullStats !== null
  const canLoadDetail = summary !== null && summary.responseCount > 0 && !detailLoaded
  // Small sessions auto-load seamlessly — hide the warning/button while that
  // fetch is in flight so it never flashes for them.
  const autoLoading =
    loadingFull && summary !== null && summary.responseCount > 0 && summary.responseCount <= AUTO_LOAD_THRESHOLD
  const showDeferredDetail = canLoadDetail && !autoLoading

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={t({ en: 'Session Stats', fr: 'Statistiques de la session' })}
      size="lg"
    >
      <div ref={contentRef} className="space-y-6">
        {modelGroups.length > 1 && (
          <section>
            <div className="flex flex-wrap gap-2">
              {modelGroups.map((group) => (
                <button
                  key={group.key}
                  onClick={() => setSelectedModelKey(group.key)}
                  className={`px-3 py-1.5 rounded border text-xs transition-colors ${
                    group.key === currentSummary?.key
                      ? 'border-accent-primary bg-accent-primary/10 text-accent-primary'
                      : 'border-border text-text-muted hover:text-text-primary hover:bg-bg-tertiary/40'
                  }`}
                  title={group.label}
                >
                  {group.label}
                </button>
              ))}
            </div>
          </section>
        )}

        {/* Summary Section — always available from the lean payload */}
        {currentSummary && (
          <section>
            <h3 className="text-sm font-semibold text-text-secondary mb-3 uppercase tracking-wide">
              {t({ en: 'Summary', fr: 'Résumé' })}
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              <StatCard label={t({ en: 'AI Time', fr: 'Temps IA' })} value={formatTime(currentSummary.aiTime)} />
              <StatCard
                label={t({ en: 'Total Time', fr: 'Temps total' })}
                value={formatTime(currentSummary.totalTime)}
              />
              <StatCard
                label={t({ en: 'Tool Time', fr: 'Temps outils' })}
                value={formatTime(currentSummary.toolTime)}
              />
              <StatCard
                label={t({ en: 'Responses', fr: 'Réponses' })}
                value={currentSummary.responseCount.toString()}
              />
              <StatCard
                label={t({ en: 'LLM Calls', fr: 'Appels LLM' })}
                value={currentSummary.llmCallCount.toString()}
              />
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-3">
              <StatCard
                label={t({ en: 'Prefill Tokens', fr: 'Jetons de préremplissage' })}
                value={formatTokens(currentSummary.prefillTokens)}
                subValue={`@ ${formatSpeed(currentSummary.avgPrefillSpeed)} tok/s`}
              />
              <StatCard
                label={t({ en: 'Gen Tokens', fr: 'Jetons générés' })}
                value={formatTokens(currentSummary.generationTokens)}
                subValue={`@ ${formatSpeed(currentSummary.avgGenerationSpeed)} tok/s`}
              />
              <StatCard
                label={t({ en: 'Avg PP Speed', fr: 'Vitesse PP moyenne' })}
                value={`${formatSpeed(currentSummary.avgPrefillSpeed)}`}
                subValue="tok/s"
              />
              <StatCard
                label={t({ en: 'Avg TG Speed', fr: 'Vitesse TG moyenne' })}
                value={`${formatSpeed(currentSummary.avgGenerationSpeed)}`}
                subValue="tok/s"
              />
            </div>
          </section>
        )}

        {/* Deferred detail — warning + one-time load */}
        {showDeferredDetail && (
          <section className="rounded border border-border bg-bg-tertiary/40 p-4">
            <p className="text-xs text-text-muted mb-3">
              {t(
                {
                  en: 'The full response log is not loaded. Load it once to see per-response and per-call details ({{n}} responses, {{c}} calls).',
                  fr: 'Le journal complet des réponses n’est pas chargé. Chargez-le une fois pour voir le détail par réponse et par appel ({{n}} réponses, {{c}} appels).',
                },
                { n: summary!.responseCount, c: summary!.llmCallCount },
              )}
            </p>
            {loadError && <p className="text-xs text-accent-error mb-3">{loadError}</p>}
            <button
              onClick={() => void loadFull()}
              disabled={loadingFull}
              className="px-3 py-1.5 rounded bg-accent-primary/25 text-text-primary hover:bg-accent-primary/40 transition-colors text-xs font-medium disabled:opacity-50"
            >
              {loadingFull
                ? t({ en: 'Loading…', fr: 'Chargement…' })
                : t({ en: 'Load full stats', fr: 'Charger toutes les statistiques' })}
            </button>
          </section>
        )}

        {/* Progression Charts */}
        {currentStats && chartData.points.length > 1 && (
          <section>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-text-secondary uppercase tracking-wide">
                {t({ en: 'Performance Progression', fr: 'Progression des performances' })}
              </h3>
              <div className="flex items-center gap-2">
                <button
                  onClick={handleCopyJson}
                  className="px-2 py-1 text-xs text-text-muted hover:text-text-primary hover:bg-bg-tertiary rounded transition-colors"
                >
                  {t({ en: 'Copy JSON', fr: 'Copier le JSON' })}
                </button>
                <button
                  onClick={handleExportPng}
                  className="px-2 py-1 text-xs text-text-muted hover:text-text-primary hover:bg-bg-tertiary rounded transition-colors"
                >
                  {t({ en: 'Save PNG', fr: 'Enregistrer le PNG' })}
                </button>
              </div>
            </div>
            <div className="bg-bg-tertiary/50 rounded p-4">
              <DualSparkline
                data={chartData.points}
                width={50}
                prefillLabel={chartData.prefillLabel}
                generationLabel={chartData.generationLabel}
                xLabel={chartData.xLabel}
              />
            </div>
          </section>
        )}

        {/* Observability Overview — derived from the loaded fullStats. */}
        {observability && (
          <section>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-text-secondary uppercase tracking-wide">
                {t({ en: 'Observability overview', fr: 'Vue d’observabilité' })}
              </h3>
              <CacheSourceBadge source={observability.summary.cacheSource} />
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <StatCard
                label={t({ en: 'Raw prompt', fr: 'Prompt brut' })}
                value={formatTokens(observability.summary.rawPromptTokens)}
                subValue={t(
                  {
                    en: 'Σ promptTokens across {{n}} LLM calls',
                    fr: 'Σ promptTokens sur {{n}} appels LLM',
                  },
                  { n: observability.calls.length },
                )}
              />
              <StatCard
                label={t({ en: 'Provider cache read', fr: 'Cache provider lu' })}
                value={formatTokens(observability.summary.providerCachedTokens)}
                subValue={
                  observability.summary.providerCacheHitRatio !== undefined
                    ? t(
                        {
                          en: '{{pct}}% hit (provider-reported)',
                          fr: '{{pct}}% de cache hit (mesure provider)',
                        },
                        {
                          pct: (observability.summary.providerCacheHitRatio * 100).toFixed(1),
                        },
                      )
                    : t({ en: 'N/A — no provider data', fr: 'N/A — aucune mesure provider' })
                }
              />
              <StatCard
                label={t({ en: 'New provider input', fr: 'Nouvelle entrée provider' })}
                value={formatTokens(observability.summary.estimatedNewInputTokens)}
                subValue={t({
                  en: 'Σ (prompt − cached) over provider calls',
                  fr: 'Σ (prompt − cached) sur les appels provider',
                })}
              />
              <StatCard
                label={t({ en: 'Cache write', fr: 'Cache écrit' })}
                value={formatTokens(observability.summary.cacheWriteTokens)}
                subValue={
                  observability.summary.cacheWriteTokens === 0
                    ? t({ en: 'no provider write info', fr: 'aucune mesure provider' })
                    : undefined
                }
              />
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
              <StatCard
                label={t({ en: 'Context P50', fr: 'Contexte P50' })}
                value={formatTokens(observability.summary.contextP50)}
                subValue={t({ en: 'per LLM call', fr: 'par appel LLM' })}
              />
              <StatCard
                label={t({ en: 'Context P95', fr: 'Contexte P95' })}
                value={formatTokens(observability.summary.contextP95)}
                subValue={t({ en: 'per LLM call', fr: 'par appel LLM' })}
              />
              <StatCard
                label={t({ en: 'Context Max', fr: 'Contexte max' })}
                value={formatTokens(observability.summary.contextMax)}
                subValue={t({ en: 'peak', fr: 'pic' })}
              />
              <StatCard
                label={t({ en: 'Context Amplification', fr: 'Amplification contexte' })}
                value={
                  observability.summary.contextAmplificationFactor !== undefined
                    ? `${observability.summary.contextAmplificationFactor.toFixed(1)}x`
                    : 'N/A'
                }
                subValue={
                  observability.summary.amplificationSource === 'provider'
                    ? t({ en: 'source: provider', fr: 'source : provider' })
                    : observability.summary.amplificationSource === 'partial'
                      ? t({ en: 'source: partial (mixed)', fr: 'source : partiel (mixte)' })
                      : t({ en: 'source: unavailable', fr: 'source : indisponible' })
                }
              />
            </div>
            {observability.summary.cacheSource === 'unavailable' && (
              <p className="text-[10px] text-text-muted mt-2">
                {t({
                  en: 'No provider cache data available for this session. Numbers reflect raw prompt transport only — cache hit % is not displayed.',
                  fr: 'Aucune mesure de cache provider pour cette session. Les chiffres reflètent uniquement le transport brut — le % de cache hit n’est pas affiché.',
                })}
              </p>
            )}
          </section>
        )}

        {/* Context & Cache — chronological SVG chart per LLM call. */}
        {observability && observability.calls.length > 0 && (
          <section>
            <h3 className="text-sm font-semibold text-text-secondary mb-3 uppercase tracking-wide">
              {t({ en: 'Context & Cache', fr: 'Contexte & cache' })}
            </h3>
            <ContextCacheChart calls={observability.calls} compactions={observability.compactions} t={t} />
          </section>
        )}

        {/* Agent Activity — counters + tool category breakdown. */}
        {observability && (
          <section>
            <h3 className="text-sm font-semibold text-text-secondary mb-3 uppercase tracking-wide">
              {t({ en: 'Agent activity', fr: 'Activité agent' })}
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              <MiniStat
                label={t({ en: 'LLM calls', fr: 'Appels LLM' })}
                value={observability.summary.llmCalls.toString()}
              />
              <MiniStat
                label={t({ en: 'Tool calls', fr: 'Appels outils' })}
                value={observability.summary.toolCalls.toString()}
              />
              <MiniStat label={t({ en: 'Retries', fr: 'Relances' })} value={observability.summary.retries.toString()} />
              <MiniStat
                label={t({ en: 'Sub-agent calls', fr: 'Sous-agents' })}
                value={observability.summary.subAgentCalls.toString()}
              />
              <MiniStat
                label={t({ en: 'Compactions', fr: 'Compactions' })}
                value={observability.summary.compactions.toString()}
              />
            </div>
            {observability.toolActivity.byTool.length > 0 && (
              <div className="mt-3 bg-bg-tertiary/30 rounded p-3">
                <h4 className="text-xs uppercase tracking-wide text-text-muted mb-2">
                  {t({ en: 'Tool breakdown', fr: 'Détail outils' })}
                </h4>
                <div className="flex flex-wrap gap-2">
                  {observability.toolActivity.byTool.slice(0, 12).map((t2) => (
                    <span
                      key={t2.toolName}
                      className="inline-flex items-center gap-1.5 px-2 py-1 bg-bg-tertiary rounded text-xs text-text-primary"
                      title={`${t2.toolName}: ${t2.count} calls, ${t2.totalDurationMs}ms, ${t2.errorCount} errors`}
                    >
                      <span className="text-text-muted">{t2.category}</span>
                      <span className="font-mono">{t2.toolName}</span>
                      <span className="text-text-muted">×{t2.count}</span>
                      {t2.errorCount > 0 && <span className="text-red-400">!</span>}
                    </span>
                  ))}
                </div>
              </div>
            )}
            {/* Compactions list (deltas) */}
            {observability.compactions.length > 0 && (
              <div className="mt-3 bg-bg-tertiary/30 rounded p-3">
                <h4 className="text-xs uppercase tracking-wide text-text-muted mb-2">
                  {t({ en: 'Compactions', fr: 'Compactions' })}
                </h4>
                <ul className="text-xs text-text-primary space-y-1">
                  {observability.compactions.map((c, i) => (
                    <li key={`${c.closedWindowId}-${i}`} className="font-mono">
                      {'↓ ' + formatTokens(c.beforeTokens) + ' → ' + formatTokens(c.afterTokens) + ' '}
                      <span className="text-text-muted">
                        {'(−' + formatTokens(c.reduction) + ' '}
                        {t({ en: 'tokens', fr: 'tokens' }) + ', '}
                        {c.reductionPercent >= 0 ? c.reductionPercent.toFixed(0) : '0'}%)
                      </span>
                      {c.subAgentType && (
                        <span className="text-text-muted">
                          {' '}
                          [{t({ en: 'sub-agent', fr: 'sous-agent' })}: {c.subAgentType}]
                        </span>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {/* Retries list */}
            {observability.retries.length > 0 && (
              <div className="mt-3 bg-bg-tertiary/30 rounded p-3">
                <h4 className="text-xs uppercase tracking-wide text-text-muted mb-2">
                  {t({ en: 'Retries', fr: 'Relances' })}
                </h4>
                <ul className="text-xs text-text-primary space-y-1">
                  {observability.retries.slice(0, 10).map((r, i) => (
                    <li key={`retry-${i}`} className="font-mono">
                      {t({ en: 'attempt', fr: 'tentative' })} #{r.attempt ?? '?'} ({r.type})
                      {r.pattern && <span className="text-text-muted"> — {r.pattern}</span>}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </section>
        )}

        {/* Response Log */}
        {currentStats && (
          <section>
            <h3 className="text-sm font-semibold text-text-secondary mb-3 uppercase tracking-wide">
              {t(
                { en: 'Response Log ({{count}} responses)', fr: 'Journal des réponses ({{count}} réponses)' },
                { count: currentStats.responseCount },
              )}
            </h3>
            <ScrollArea className="bg-bg-tertiary/30 rounded">
              <table className="w-full table-fixed border-separate border-spacing-0 text-xs">
                <colgroup>
                  <col className="w-[7%]" />
                  <col className="w-[14%]" />
                  <col className="w-[10%]" />
                  <col className="w-[14%]" />
                  <col className="w-[14%]" />
                  <col className="w-[14%]" />
                  <col className="w-[11%]" />
                  <col className="w-[2%]" />
                </colgroup>
                <thead>
                  <tr className="text-[10px] uppercase tracking-wide text-text-muted/80">
                    <th className="px-3 py-2 text-center font-medium">#</th>
                    <th className="px-2 py-2 text-center font-medium">{t({ en: 'At', fr: 'À' })}</th>
                    <th className="px-2 py-2 text-center font-medium">{t({ en: 'Time', fr: 'Durée' })}</th>
                    <th className="px-2 py-2 text-center font-medium">{t({ en: 'Context', fr: 'Contexte' })}</th>
                    <th className="px-2 py-2 text-center font-medium">PP t/s</th>
                    <th className="px-2 py-2 text-center font-medium">TG t/s</th>
                    <th className="px-2 py-2 text-center font-medium">{t({ en: 'Calls', fr: 'Appels' })}</th>
                    <th className="px-2 py-2" />
                  </tr>
                </thead>
                <tbody>
                  {responseRows.map((row, i) => (
                    <Fragment key={row.messageId}>
                      <ResponseRow
                        row={row}
                        index={i}
                        isExpanded={expandedResponses[row.messageId] ?? false}
                        onToggle={row.isExpandable ? () => toggleResponse(row.messageId) : undefined}
                      />
                      {(expandedResponses[row.messageId] ?? false) &&
                        row.calls.map((call, callIndex) => (
                          <CallDataPointRow
                            key={`${call.messageId}-${call.callIndex}`}
                            dataPoint={call}
                            index={callIndex}
                          />
                        ))}
                    </Fragment>
                  ))}
                </tbody>
              </table>
            </ScrollArea>
          </section>
        )}
      </div>
    </Modal>
  )
}

/**
 * Summary stat card component
 */
function StatCard({ label, value, subValue }: { label: string; value: string; subValue?: string }) {
  return (
    <div className="bg-bg-tertiary/50 rounded p-3">
      <div className="text-text-muted text-xs mb-1">{label}</div>
      <div className="text-text-primary text-lg font-semibold">{value}</div>
      {subValue && <div className="text-text-muted text-xs">{subValue}</div>}
    </div>
  )
}

/**
 * Single row in the response log
 */
function ResponseRow({
  row,
  index,
  isExpanded,
  onToggle,
}: {
  row: ResponseLogRow
  index: number
  isExpanded: boolean
  onToggle?: () => void
}) {
  const contextSummary =
    row.calls.length > 0
      ? formatContextRange(row.calls.map((call) => call.promptTokens))
      : `${formatTokens(row.prefillTokens)} ctx`

  return (
    <tr
      onClick={onToggle}
      className={`${index % 2 === 0 ? 'bg-bg-tertiary/20' : ''} ${onToggle ? 'cursor-pointer hover:bg-bg-tertiary/35 transition-colors' : ''}`}
    >
      <td className="px-3 py-2 text-center text-text-muted align-middle">{row.responseIndex}</td>
      <td className="px-2 py-2 text-center text-text-muted font-mono align-middle whitespace-nowrap">
        {formatTimestamp(row.timestamp)}
      </td>
      <td className="px-2 py-2 text-center text-text-muted align-middle whitespace-nowrap">
        {formatTime(row.totalTime)}
      </td>
      <td className="px-2 py-2 text-center text-text-primary font-mono align-middle whitespace-nowrap">
        {contextSummary.replace(/ ctx$/, '')}
      </td>
      <td className="px-2 py-2 text-center text-text-primary font-mono align-middle whitespace-nowrap">
        {formatRate(row.prefillSpeed)}
      </td>
      <td className="px-2 py-2 text-center text-text-primary font-mono align-middle whitespace-nowrap">
        {formatRate(row.generationSpeed)}
      </td>
      <td className="px-2 py-2 text-center text-text-muted font-mono align-middle whitespace-nowrap">
        {row.callCount}
      </td>
      <td className="px-2 py-2 text-center text-text-muted align-middle whitespace-nowrap">
        {row.isExpandable ? (isExpanded ? 'v' : '>') : ''}
      </td>
    </tr>
  )
}

function CallDataPointRow({ dataPoint, index }: { dataPoint: CallStatsDataPoint; index: number }) {
  const t = useT()
  const hasParams =
    dataPoint.temperature !== undefined ||
    dataPoint.topP !== undefined ||
    dataPoint.topK !== undefined ||
    dataPoint.maxTokens !== undefined
  const hasCache =
    dataPoint.cachedPromptTokens !== undefined ||
    dataPoint.cacheWriteTokens !== undefined ||
    dataPoint.cacheSource !== undefined
  const callNewInput =
    dataPoint.cachedPromptTokens !== undefined
      ? Math.max(0, dataPoint.promptTokens - dataPoint.cachedPromptTokens)
      : dataPoint.contextSize !== undefined
        ? Math.max(0, dataPoint.contextSize - dataPoint.promptTokens)
        : undefined

  return (
    <>
      <tr className={`${index % 2 === 0 ? 'bg-bg-tertiary/10' : 'bg-bg-tertiary/5'}`}>
        <td className="px-3 py-2 pl-6 text-center text-text-muted align-middle border-l border-border/60">
          {`c${dataPoint.callIndex}`}
        </td>
        <td className="px-2 py-2 text-center text-text-muted font-mono align-middle whitespace-nowrap">
          {formatTimestamp(dataPoint.timestamp)}
        </td>
        <td className="px-2 py-2 text-center text-text-muted align-middle whitespace-nowrap">
          {formatTime(dataPoint.totalTime)}
        </td>
        <td className="px-2 py-2 text-center text-text-primary font-mono align-middle whitespace-nowrap">
          {formatTokens(dataPoint.promptTokens)}
        </td>
        <td className="px-2 py-2 text-center text-text-primary font-mono align-middle whitespace-nowrap">
          {formatRate(dataPoint.prefillSpeed)}
        </td>
        <td className="px-2 py-2 text-center text-text-primary font-mono align-middle whitespace-nowrap">
          {formatRate(dataPoint.generationSpeed)}
        </td>
        <td className="px-2 py-2 text-center text-text-muted font-mono align-middle whitespace-nowrap">
          {dataPoint.callIndex}
        </td>
        <td className="px-2 py-2" />
      </tr>
      {(hasCache || hasParams) && (
        <tr className={`${index % 2 === 0 ? 'bg-bg-tertiary/5' : 'bg-bg-tertiary/[2.5%]'}`}>
          <td colSpan={8} className="px-6 py-1.5 border-l border-border/60">
            <div className="flex flex-wrap gap-x-4 gap-y-1 text-[10px] text-text-muted">
              {dataPoint.cachedPromptTokens !== undefined && (
                <span>
                  {t({ en: 'cache', fr: 'cache' })}: {formatTokens(dataPoint.cachedPromptTokens)}
                  {dataPoint.promptTokens > 0
                    ? ' (' + ((dataPoint.cachedPromptTokens / dataPoint.promptTokens) * 100).toFixed(0) + '%)'
                    : ''}
                </span>
              )}
              {callNewInput !== undefined && (
                <span>
                  {t({ en: 'new', fr: 'new' })}: {formatTokens(callNewInput)}
                </span>
              )}
              {dataPoint.cacheWriteTokens !== undefined && (
                <span>
                  {t({ en: 'write', fr: 'write' })}: {formatTokens(dataPoint.cacheWriteTokens)}
                </span>
              )}
              {dataPoint.cacheSource !== undefined && (
                <span>
                  {t({ en: 'source', fr: 'source' })}: {dataPoint.cacheSource}
                </span>
              )}
              {dataPoint.temperature !== undefined && <span>{`temp: ${dataPoint.temperature.toFixed(2)}`}</span>}
              {dataPoint.topP !== undefined && <span>{`topP: ${dataPoint.topP.toFixed(2)}`}</span>}
              {dataPoint.topK !== undefined && <span>{`topK: ${dataPoint.topK}`}</span>}
              {dataPoint.maxTokens !== undefined && <span>{`maxTok: ${dataPoint.maxTokens}`}</span>}
            </div>
          </td>
        </tr>
      )}
    </>
  )
}
/**
 * Cache source attribution badge. Renders the three states with distinct
 * colors so the dashboard never mislabels absence-of-info as zero cache.
 */
function CacheSourceBadge({ source }: { source: CacheSource }) {
  const t = useT()
  const map: Record<CacheSource, { label: string; classes: string; hint: string }> = {
    provider: {
      label: t({ en: 'Provider', fr: 'Provider' }),
      classes: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30',
      hint: t({
        en: 'Cache fields reported by the provider (e.g. prompt_tokens_details.cached_tokens)',
        fr: 'Champs de cache rapportes par le provider (p. ex. prompt_tokens_details.cached_tokens)',
      }),
    },
    estimated: {
      label: t({ en: 'Estimated', fr: 'Estime' }),
      classes: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
      hint: t({
        en: 'Cache value computed from OpenFox-side context tracking (heuristic, lower bound)',
        fr: 'Valeur de cache calculee cote OpenFox (heuristique, limite basse)',
      }),
    },
    unavailable: {
      label: t({ en: 'N/A', fr: 'N/D' }),
      classes: 'bg-gray-500/15 text-gray-300 border-gray-500/30',
      hint: t({
        en: 'No provider cache information in the response — values may be absent',
        fr: 'Aucune mesure de cache dans la reponse provider — valeurs possiblement absentes',
      }),
    },
  }
  const cfg = map[source]
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] border ${cfg.classes}`} title={cfg.hint}>
      {cfg.label}
    </span>
  )
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-bg-tertiary/50 rounded p-2.5">
      <div className="text-text-muted text-[10px] uppercase tracking-wide">{label}</div>
      <div className="text-text-primary text-base font-semibold font-mono">{value}</div>
    </div>
  )
}
/**
 * Context & Cache chart - chronological SVG plot of `promptTokens` per
 * LLM call, with a cached-portion overlay and vertical markers for
 * compactions. Threshold reference lines for 50k/100k/150k help the
 * user spot context growth against typical compaction triggers.
 */
function ContextCacheChart({
  calls,
  compactions,
  t,
}: {
  calls: ObservabilityCallRow[]
  compactions: Array<{ timestamp: number; beforeTokens: number; afterTokens: number }>
  t: (template: { en: string; fr: string }, params?: Record<string, string | number>) => string
}) {
  if (calls.length === 0) return null
  const width = 720
  const height = 160
  const padding = { top: 16, right: 16, bottom: 28, left: 56 }
  const xs = calls.map((c) => c.sessionCallIndex)
  const ys = calls.map((c) => c.contextSize)
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const maxY = Math.max(...ys, 1)
  const xScale = (x: number) =>
    padding.left + ((x - minX) / Math.max(1, maxX - minX)) * (width - padding.left - padding.right)
  const yScale = (y: number) => padding.top + (1 - y / maxY) * (height - padding.top - padding.bottom)
  const linePath = calls
    .map((c, i) => (i === 0 ? 'M' : 'L') + ' ' + xScale(c.sessionCallIndex) + ' ' + yScale(c.contextSize))
    .join(' ')
  const hasCached = calls.some((c) => c.cachedPromptTokens !== undefined && c.cachedPromptTokens > 0)
  const cacheArea = hasCached
    ? calls
        .map((c, i) => {
          const cached = c.cachedPromptTokens ?? 0
          const y1 = yScale(c.contextSize)
          const y2 = yScale(Math.max(0, c.contextSize - cached))
          return (
            (i === 0 ? 'M' : 'L') +
            ' ' +
            xScale(c.sessionCallIndex) +
            ' ' +
            y2 +
            ' L ' +
            xScale(c.sessionCallIndex) +
            ' ' +
            y1 +
            ' Z'
          )
        })
        .join(' ')
    : ''
  return (
    <div>
      <svg
        width="100%"
        viewBox={'0 0 ' + width + ' ' + height}
        preserveAspectRatio="xMidYMid meet"
        className="text-text-primary"
      >
        <rect x="0" y="0" width={width} height={height} fill="transparent" />
        {[50000, 100000, 150000]
          .filter((threshold) => threshold <= maxY)
          .map((threshold) => (
            <g key={threshold}>
              <line
                x1={padding.left}
                y1={yScale(threshold)}
                x2={width - padding.right}
                y2={yScale(threshold)}
                stroke="currentColor"
                strokeOpacity="0.18"
                strokeDasharray="3 3"
              />
              <text
                x={width - padding.right - 4}
                y={yScale(threshold) - 2}
                fontSize="9"
                textAnchor="end"
                fill="currentColor"
                fillOpacity="0.5"
              >
                {formatTokens(threshold)}
              </text>
            </g>
          ))}
        {cacheArea && <path d={cacheArea} fill="currentColor" opacity="0.18" />}
        <path d={linePath} fill="none" stroke="currentColor" strokeWidth="1.5" />
        {calls.map((c) => (
          <circle
            key={c.sessionCallIndex}
            cx={xScale(c.sessionCallIndex)}
            cy={yScale(c.contextSize)}
            r="2"
            fill="currentColor"
          />
        ))}
        {compactions.map((c, i) => {
          if (c.beforeTokens <= 0) return null
          const x =
            padding.left +
            ((i + 1) / Math.max(1, calls.length + compactions.length)) * (width - padding.left - padding.right)
          return (
            <g key={'c-' + i}>
              <line
                x1={x}
                y1={padding.top}
                x2={x}
                y2={height - padding.bottom}
                stroke="#f59e0b"
                strokeDasharray="4 2"
              />
              <text x={x + 4} y={padding.top + 12} fontSize="10" fill="#f59e0b">
                {'↓ ' + formatTokens(c.beforeTokens) + ' → ' + formatTokens(c.afterTokens)}
              </text>
            </g>
          )
        })}
        <text x={padding.left} y={height - 6} fontSize="10" fill="currentColor" opacity="0.6">
          {t({ en: 'LLM call', fr: 'Appel LLM' }) + ' #' + minX}
        </text>
        <text x={width - padding.right} y={height - 6} fontSize="10" textAnchor="end" fill="currentColor" opacity="0.6">
          {'#' + maxX}
        </text>
        <text x={4} y={padding.top + 8} fontSize="10" fill="currentColor" opacity="0.6">
          {formatTokens(maxY)}
        </text>
      </svg>
      <div className="mt-1 flex items-center gap-3 text-[10px] text-text-muted">
        <span className="inline-flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-current inline-block" />
          {t({ en: 'context size', fr: 'taille du contexte' })}
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="w-3 h-3 bg-current opacity-30 inline-block" />
          {t({ en: 'cached portion', fr: 'portion cachee' })}
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="w-3 h-px bg-amber-500 inline-block" />
          {t({ en: 'compaction', fr: 'compaction' })}
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="w-3 h-px border-t border-dashed border-current opacity-40 inline-block" />
          {t({ en: '50k/100k/150k', fr: '50k/100k/150k' })}
        </span>
      </div>
    </div>
  )
}
