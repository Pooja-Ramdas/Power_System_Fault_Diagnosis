// src/api.ts
// API client with 3.5s timeout, fallback to cached JSON, ?api= override.

import type { CaseData, DiagnoseResponse, EvidenceResponse, ModelId, IndexData } from './schema'
import { DiagnoseResponseSchema, EvidenceResponseSchema } from './schema'

// API URL: from build env var, or ?api= query param override, or relative
function getApiBase(): string {
  const url = new URL(window.location.href)
  const override = url.searchParams.get('api')
  if (override) return override.replace(/\/$/, '')
  const envUrl = import.meta.env.VITE_API_URL
  if (envUrl) return (envUrl as string).replace(/\/$/, '')
  return ''  // relative (no live API in pure static mode)
}

const TIMEOUT_MS = 3500

async function fetchWithTimeout(url: string, opts: RequestInit = {}): Promise<Response> {
  const controller = new AbortController()
  const id = setTimeout(() => controller.abort(), TIMEOUT_MS)
  try {
    return await fetch(url, { ...opts, signal: controller.signal })
  } finally {
    clearTimeout(id)
  }
}

// ── Static cache helpers ───────────────────────────────────────────────────────
export async function fetchIndex(): Promise<IndexData> {
  const r = await fetch('/data/index.json')
  if (!r.ok) throw new Error('index fetch failed')
  return r.json()
}

export async function fetchCase(caseId: string): Promise<CaseData> {
  const r = await fetch(`/data/cases/${caseId}.json`)
  if (!r.ok) throw new Error(`case ${caseId} not found`)
  return r.json()
}

// ── Live API calls with cache fallback ────────────────────────────────────────
export type DiagnoseResult = {
  data: DiagnoseResponse
  fromCache: boolean
}

export async function diagnose(
  caseId: string,
  modelId: ModelId,
  mask: string[] = ['D', 'T', 'R'],
  cacheCase?: CaseData,
): Promise<DiagnoseResult> {
  const apiBase = getApiBase()

  // Try live API first
  if (apiBase) {
    try {
      const resp = await fetchWithTimeout(`${apiBase}/diagnose`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ case_id: caseId, model_id: modelId, mask }),
      })
      if (resp.ok) {
        const raw = await resp.json()
        const parsed = DiagnoseResponseSchema.safeParse(raw)
        if (parsed.success) {
          return { data: parsed.data, fromCache: false }
        }
      }
    } catch {
      // timeout or network error → fall through to cache
    }
  }

  // Fallback: reconstruct from cached case JSON
  if (!cacheCase) {
    cacheCase = await fetchCase(caseId)
  }
  const pred = cacheCase.predictions[modelId]
  const expl = cacheCase.explanations[modelId]
  const ablations = cacheCase.ablations[modelId] ?? {}

  const cached: DiagnoseResponse = {
    prediction: pred,
    explanation: expl,
    ablations,
    latency_ms: 0,
    source: 'cached',
    model_id: modelId,
    case_id: caseId,
  }
  return { data: cached, fromCache: true }
}

export type EvidenceResult = {
  data: EvidenceResponse
  fromCache: boolean
}

export async function fetchEvidence(
  caseId: string,
  modelId: ModelId,
  mask: string[] = ['D', 'T', 'R'],
  cacheCase?: CaseData,
): Promise<EvidenceResult> {
  const apiBase = getApiBase()

  // Try live API first
  if (apiBase) {
    try {
      const resp = await fetchWithTimeout(`${apiBase}/evidence`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ case_id: caseId, model_id: modelId, mask }),
      })
      if (resp.ok) {
        const raw = await resp.json()
        const parsed = EvidenceResponseSchema.safeParse(raw)
        if (parsed.success) {
          return { data: parsed.data, fromCache: false }
        }
      }
    } catch {
      // fallback
    }
  }

  // Cache fallback
  if (!cacheCase) {
    cacheCase = await fetchCase(caseId)
  }
  const expl = cacheCase.explanations[modelId]
  const cached: EvidenceResponse = {
    telemetry_importance: expl.telemetry_importance ?? [],
    report_top_words: expl.report_top_words ?? [],
    word_scores: cacheCase.report.word_scores,
    cam_base64: null,
    latency_ms: 0,
    source: 'cached',
    model_id: modelId,
    case_id: caseId,
  }
  return { data: cached, fromCache: true }
}

export async function checkHealth(apiBase: string): Promise<boolean> {
  try {
    const r = await fetchWithTimeout(`${apiBase}/health`)
    return r.ok
  } catch {
    return false
  }
}
