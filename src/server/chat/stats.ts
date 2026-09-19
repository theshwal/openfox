/**
 * Message stats computation - single source of truth for the formula.
 */

import type { CacheSource, LLMCallStats, MessageStats, StatsIdentity, ToolMode } from '../../shared/types.js'
import type { StreamTiming } from '../llm/streaming.js'

const roundTo1 = (n: number): number => Math.round(n * 10) / 10

export interface ModelParams {
  temperature?: number
  topP?: number
  topK?: number
  maxTokens?: number
}

/**
 * Provider cache attribution carried through the stats pipeline. All
 * fields are optional and only set when the provider's API response
 * includes the corresponding field. `cacheSource: 'provider'` requires
 * presence of at least one cache field; `cacheSource: 'unavailable'`
 * (default) means the provider did not report cache information.
 */
export interface CacheAttribution {
  cachedPromptTokens?: number
  cacheWriteTokens?: number
  cacheSource?: CacheSource
}

function buildCallStats(input: {
  identity: StatsIdentity
  callIndex: number
  timing: StreamTiming
  promptTokens: number
  completionTokens: number
  prefTokenIncrement?: number
  cache?: CacheAttribution
  timestamp?: string
  modelParams?: ModelParams
}): LLMCallStats {
  const {
    identity,
    callIndex,
    timing,
    promptTokens,
    completionTokens,
    prefTokenIncrement,
    cache,
    timestamp,
    modelParams,
  } = input
  const prefillSource = prefTokenIncrement ?? promptTokens
  return {
    ...identity,
    callIndex,
    promptTokens,
    completionTokens,
    ...(prefTokenIncrement !== undefined && { prefTokenIncrement }),
    ttft: timing.ttft,
    completionTime: timing.completionTime,
    prefillSpeed: timing.ttft > 0 ? roundTo1(prefillSource / timing.ttft) : 0,
    generationSpeed: timing.completionTime > 0 ? roundTo1(completionTokens / timing.completionTime) : 0,
    totalTime: roundTo1(timing.ttft + timing.completionTime),
    ...(timestamp ? { timestamp } : {}),
    ...(modelParams?.temperature !== undefined && { temperature: modelParams.temperature }),
    ...(modelParams?.topP !== undefined && { topP: modelParams.topP }),
    ...(modelParams?.topK !== undefined && { topK: modelParams.topK }),
    ...(modelParams?.maxTokens !== undefined && { maxTokens: modelParams.maxTokens }),
    // Provider cache attribution. Forwarded verbatim — never inferred from
    // prefTokenIncrement. Absence of the field on the API response leaves
    // cacheSource as 'unavailable' (extractor's job), so the dashboard never
    // mislabels absent info as zero cache.
    ...(cache?.cachedPromptTokens !== undefined && { cachedPromptTokens: cache.cachedPromptTokens }),
    ...(cache?.cacheWriteTokens !== undefined && { cacheWriteTokens: cache.cacheWriteTokens }),
    ...(cache?.cacheSource !== undefined && { cacheSource: cache.cacheSource }),
    contextSize: promptTokens,
  }
}

export interface StatsInput {
  identity: StatsIdentity
  mode: ToolMode
  timing: StreamTiming
  usage: {
    promptTokens: number
    completionTokens: number
    cachedPromptTokens?: number
    cacheWriteTokens?: number
    cacheSource?: CacheSource
  }
  /** New (non-cached) tokens that required actual prompt processing */
  prefTokenIncrement?: number
  /** Tool execution time in seconds (default: 0) */
  toolTime?: number
  /** Override totalTime instead of computing from timing + toolTime */
  totalTimeOverride?: number
  timestamp?: string
  modelParams?: ModelParams
}

/**
 * Compute message stats from LLM timing and usage data.
 *
 * For single LLM calls: totalTime = ttft + completionTime + toolTime
 * For multi-call flows: pass totalTimeOverride with wall clock time
 */
export function computeMessageStats(input: StatsInput): MessageStats {
  const {
    identity,
    mode,
    timing,
    usage,
    prefTokenIncrement,
    toolTime = 0,
    totalTimeOverride,
    timestamp,
    modelParams,
  } = input

  const totalTime = totalTimeOverride ?? timing.ttft + timing.completionTime + toolTime
  const prefillSource = prefTokenIncrement ?? usage.promptTokens

  const cache: CacheAttribution = {
    ...(usage.cachedPromptTokens !== undefined && { cachedPromptTokens: usage.cachedPromptTokens }),
    ...(usage.cacheWriteTokens !== undefined && { cacheWriteTokens: usage.cacheWriteTokens }),
    ...(usage.cacheSource !== undefined && { cacheSource: usage.cacheSource }),
  }

  // Roll up per-call cache into the response-level MessageStats only when at
  // least one cache field is present. Empty cache fields stay undefined.
  const hasAnyCache =
    usage.cachedPromptTokens !== undefined || usage.cacheWriteTokens !== undefined || usage.cacheSource !== undefined

  return {
    ...identity,
    mode,
    totalTime,
    toolTime,
    prefillTokens: usage.promptTokens,
    ...(prefTokenIncrement !== undefined && { prefTokenIncrement }),
    prefillSpeed: timing.ttft > 0 ? roundTo1(prefillSource / timing.ttft) : 0,
    generationTokens: usage.completionTokens,
    generationSpeed: timing.completionTime > 0 ? roundTo1(usage.completionTokens / timing.completionTime) : 0,
    ...(hasAnyCache && usage.cachedPromptTokens !== undefined && { cachedPromptTokens: usage.cachedPromptTokens }),
    ...(hasAnyCache && usage.cacheWriteTokens !== undefined && { cacheWriteTokens: usage.cacheWriteTokens }),
    ...(hasAnyCache && usage.cacheSource !== undefined && { cacheSource: usage.cacheSource }),
    llmCalls: [
      buildCallStats({
        identity,
        callIndex: 1,
        timing,
        promptTokens: usage.promptTokens,
        completionTokens: usage.completionTokens,
        ...(prefTokenIncrement !== undefined && { prefTokenIncrement }),
        ...(hasAnyCache && { cache }),
        ...(timestamp ? { timestamp } : {}),
        ...(modelParams && { modelParams }),
      }),
    ],
  }
}

/**
 * Compute stats from aggregated multi-call data (e.g., TurnMetrics).
 * Speeds are computed as averages across all calls.
 */
export function computeAggregatedStats(input: {
  identity: StatsIdentity
  mode: ToolMode
  totalPrefillTokens: number
  totalPrefillIncrement?: number // sum of prefTokenIncrement across all calls; used for accurate prefillSpeed
  totalGenTokens: number
  totalPrefillTime: number // sum of ttft across all calls
  totalGenTime: number // sum of completionTime across all calls
  totalToolTime: number // seconds
  totalTime: number // wall clock seconds
  llmCalls?: LLMCallStats[]
}): MessageStats {
  const {
    identity,
    mode,
    totalPrefillTokens,
    totalPrefillIncrement,
    totalGenTokens,
    totalPrefillTime,
    totalGenTime,
    totalToolTime,
    totalTime,
    llmCalls,
  } = input

  const prefillSource = totalPrefillIncrement ?? totalPrefillTokens

  // Aggregate per-call cache attribution. The last non-unavailable source
  // wins, and totals are summed. If all calls have no cache info, the
  // response-level fields stay undefined.
  let totalCachedPromptTokens: number | undefined
  let totalCacheWriteTokens: number | undefined
  let hasAnyCache = false
  if (llmCalls) {
    for (const call of llmCalls) {
      if (call.cachedPromptTokens !== undefined) {
        totalCachedPromptTokens = (totalCachedPromptTokens ?? 0) + call.cachedPromptTokens
        hasAnyCache = true
      }
      if (call.cacheWriteTokens !== undefined) {
        totalCacheWriteTokens = (totalCacheWriteTokens ?? 0) + call.cacheWriteTokens
        hasAnyCache = true
      }
    }
  }

  return {
    ...identity,
    mode,
    totalTime,
    toolTime: totalToolTime,
    prefillTokens: totalPrefillTokens,
    ...(totalPrefillIncrement !== undefined && { prefTokenIncrement: totalPrefillIncrement }),
    prefillSpeed: totalPrefillTime > 0 ? roundTo1(prefillSource / totalPrefillTime) : 0,
    generationTokens: totalGenTokens,
    generationSpeed: totalGenTime > 0 ? roundTo1(totalGenTokens / totalGenTime) : 0,
    ...(hasAnyCache && totalCachedPromptTokens !== undefined && { cachedPromptTokens: totalCachedPromptTokens }),
    ...(hasAnyCache && totalCacheWriteTokens !== undefined && { cacheWriteTokens: totalCacheWriteTokens }),
    ...(hasAnyCache && { cacheSource: 'provider' as const }),
    ...(llmCalls ? { llmCalls } : {}),
  }
}
