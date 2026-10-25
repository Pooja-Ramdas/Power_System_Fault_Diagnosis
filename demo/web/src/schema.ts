// src/schema.ts
// Zod schemas matching the case JSON built by demo/export/build_demo_data.py

import { z } from 'zod'

export const ModelId = z.enum(['conflux', 'quorum', 'nexus'])
export type ModelId = z.infer<typeof ModelId>

export const FaultClass = z.enum(['line_fault', 'breaker_fault', 'transformer_fault', 'busbar_fault'])
export type FaultClass = z.infer<typeof FaultClass>

const ContribEntry = z.object({
  share: z.number(),
  signed: z.number(),
})

const PredictionSchema = z.object({
  pred: FaultClass,
  probs: z.record(z.number()),
  confidence: z.number(),
  correct: z.boolean(),
})
export type PredictionData = z.infer<typeof PredictionSchema>

const ExplanationSchema = z.object({
  method: z.string(),
  contrib: z.object({
    D: ContribEntry,
    T: ContribEntry,
    R: ContribEntry,
  }),
  dominant: z.string(),
  telemetry_importance: z.array(z.number()).optional(),
  report_top_words: z.array(z.object({ word: z.string(), score: z.number(), rel_importance: z.number().optional() })).optional(),
  cam: z.string().optional(),
  model_specific: z.record(z.unknown()).optional(),
  narrative: z.array(z.string()),
})
export type ExplanationData = z.infer<typeof ExplanationSchema>

const AblationEntry = z.object({
  pred: z.string(),
  conf: z.number(),
})

export const CaseSchema = z.object({
  id: z.string(),
  event_ts: z.string(),
  element_name: z.string(),
  true_class: FaultClass,
  image: z.object({
    hud: z.string(),
    raw: z.string(),
    width: z.number(),
    height: z.number(),
    highlight: z.unknown().nullable(),
    is_real_infotecnica: z.boolean(),
  }),
  telemetry: z.object({
    columns: z.array(z.string()),
    units: z.array(z.string()),
    t0_index: z.number(),
    rows: z.array(z.object({
      ts: z.string(),
      channels: z.array(z.number()),
    })),
  }),
  report: z.object({
    en: z.string(),
    es: z.string(),
    shown_language: z.string(),
    word_scores: z.array(z.object({ w: z.string(), s: z.number() })),
  }),
  predictions: z.record(PredictionSchema),
  explanations: z.record(ExplanationSchema),
  ablations: z.record(z.record(AblationEntry)),
})
export type CaseData = z.infer<typeof CaseSchema>

const ModelMeta = z.object({
  id: z.string(),
  name: z.string(),
  fusion: z.string(),
  color: z.string(),
  tooltip: z.string(),
  test_accuracy: z.number(),
  test_macro_f1: z.number(),
})
export type ModelMeta = z.infer<typeof ModelMeta>

const ClassMeta = z.object({
  id: z.string(),
  display: z.string(),
  color: z.string(),
  key: z.string(),
})

const CaseIndexEntry = z.object({
  id: z.string(),
  event_ts: z.string(),
  element_name: z.string(),
  true_class: FaultClass,
  predictions: z.record(z.string()),
})
export type CaseIndexEntry = z.infer<typeof CaseIndexEntry>

export const IndexSchema = z.object({
  title: z.string(),
  models: z.array(ModelMeta),
  classes: z.array(ClassMeta),
  cases: z.array(CaseIndexEntry),
})
export type IndexData = z.infer<typeof IndexSchema>

// Live API response schemas
export const DiagnoseResponseSchema = z.object({
  prediction: PredictionSchema,
  explanation: ExplanationSchema,
  ablations: z.record(AblationEntry),
  phi_raw: z.record(z.number()).optional(),
  model_specific: z.record(z.unknown()).optional(),
  latency_ms: z.number(),
  source: z.enum(['live', 'cached']),
  model_id: z.string(),
  case_id: z.string(),
})
export type DiagnoseResponse = z.infer<typeof DiagnoseResponseSchema>

export const EvidenceResponseSchema = z.object({
  telemetry_importance: z.array(z.number()),
  report_top_words: z.array(z.object({ word: z.string(), score: z.number(), rel_importance: z.number().optional() })),
  word_scores: z.array(z.object({ w: z.string(), s: z.number() })),
  cam_base64: z.string().nullable().optional(),
  latency_ms: z.number(),
  source: z.enum(['live', 'cached']),
  model_id: z.string(),
  case_id: z.string(),
})
export type EvidenceResponse = z.infer<typeof EvidenceResponseSchema>
