// src/components/ModelCard.tsx
// CONFLUX / QUORUM / NEXUS diagnosis bubble card with Shapley contribution bars.

import type { ModelId, DiagnoseResponse } from '../schema'
import { classLabel } from '../classLabel'
import type { ModelResult } from '../hooks/useConsole'

const MODEL_META: Record<ModelId, { name: string; fusion: string; color: string; tooltip: string }> = {
  conflux: {
    name: 'CONFLUX', fusion: 'EARLY FUSION', color: 'var(--conflux)',
    tooltip: 'Features from diagram, telemetry and report are merged at input before any decision.',
  },
  quorum: {
    name: 'QUORUM',  fusion: 'LATE FUSION',   color: 'var(--quorum)',
    tooltip: 'Each modality reaches an independent verdict; a learned vote combines them.',
  },
  nexus: {
    name: 'NEXUS',   fusion: 'HYBRID FUSION', color: 'var(--nexus)',
    tooltip: 'Encoders exchange evidence mid-network, then decision-level stage combines the result.',
  },
}

const FAULT_COLORS: Record<string, string> = {
  line_fault:        'var(--fault-sc)',
  breaker_fault:     'var(--fault-oc)',
  transformer_fault: 'var(--fault-ef)',
  busbar_fault:      'var(--fault-oo)',
}

const MOD_COLORS = { D: 'var(--mod-d)', T: 'var(--mod-t)', R: 'var(--mod-r)' }
const MOD_LABELS = { D: 'DIAG', T: 'TELE', R: 'RPT' }

interface Props {
  modelId:  ModelId
  result:   ModelResult
  active:   boolean
  onClick:  () => void
}

export default function ModelCard({ modelId, result, active, onClick }: Props) {
  const meta = MODEL_META[modelId]
  const diag = result.diagnose

  return (
    <div
      className={`model-card ${active ? 'active' : ''}`}
      style={{ color: meta.color }}
      onClick={onClick}
      title={meta.tooltip}
    >
      {/* Header */}
      <div className="model-card__header">
        <div>
          <div className="model-card__name" style={{ color: meta.color }}>{meta.name}</div>
          <div className="model-card__fusion">{meta.fusion}</div>
        </div>
        {/* Source badge */}
        {diag && diag.source === 'live' && (
          <span className="badge badge-live">
            live
          </span>
        )}
        {result.loading && <div className="spinner" style={{ borderTopColor: meta.color }} />}
      </div>

      {/* Confidence + prediction */}
      {diag ? (
        <>
          <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between' }}>
            <div className="model-card__confidence" style={{ color: meta.color }}>
              {(diag.prediction.confidence * 100).toFixed(1)}%
            </div>
            <div style={{ textAlign: 'right' }}>
              <div
                className="model-card__pred"
                style={{ color: FAULT_COLORS[diag.prediction.pred] }}
              >
                {classLabel(diag.prediction.pred)}
              </div>
              <div style={{ fontSize: 9, color: diag.prediction.correct ? 'var(--status-live)' : 'var(--status-err)', marginTop: 2 }}>
                {diag.prediction.correct ? '✓ CORRECT' : '✗ INCORRECT'}
              </div>
              {diag.latency_ms > 0 && (
                <div style={{ fontSize: 8, color: 'var(--text-dim)', marginTop: 1 }}>
                  {`${diag.latency_ms}ms`}
                </div>
              )}
            </div>
          </div>

          {/* Shapley contribution bars */}
          {diag.explanation?.contrib && (
            <div style={{ marginTop: 8 }}>
              {(['D','T','R'] as const).map(m => {
                const c = diag.explanation.contrib[m]
                const isOff = diag.mask ? !diag.mask.includes(m) : ((c?.share ?? 0) === 0 && (c?.signed ?? 0) === 0)
                const pct = isOff ? 0 : Math.round((c?.share ?? 0) * 100)
                const sign = (c?.signed ?? 0) >= 0 ? '+' : '−'
                return (
                  <div key={m} className="contrib-row" style={{ opacity: isOff ? 0.4 : 1 }}>
                    <div className="contrib-label" style={{ color: isOff ? 'var(--text-dim)' : MOD_COLORS[m] }}>
                      {MOD_LABELS[m]}
                    </div>
                    <div className="contrib-bar-bg">
                      <div
                        className="contrib-bar-fill"
                        style={{ width: `${pct}%`, background: isOff ? 'var(--line-dim)' : MOD_COLORS[m] }}
                      />
                    </div>
                    <div className="contrib-pct" style={{ color: isOff ? 'var(--text-dim)' : undefined }}>
                      {isOff ? 'OFF' : `${pct}%`}
                    </div>
                    <div className="contrib-sign" style={{ color: isOff ? 'var(--text-dim)' : ((c?.signed ?? 0) >= 0 ? 'var(--status-live)' : 'var(--status-err)') }}>
                      {isOff ? '0%' : sign}
                    </div>
                  </div>
                )
              })}
            </div>
          )}

          {/* Probability mini bars */}
          <div className="prob-grid">
            {Object.entries(diag.prediction.probs).map(([cls, prob]) => (
              <div key={cls} className="prob-item">
                <div className="prob-item__label">{classLabel(cls)}</div>
                <div className="prob-item__bar">
                  <div
                    className="prob-item__fill"
                    style={{ width: `${Math.round(prob * 100)}%`, background: FAULT_COLORS[cls] }}
                  />
                </div>
                <div className="prob-item__val">{(prob * 100).toFixed(0)}%</div>
              </div>
            ))}
          </div>
        </>
      ) : (
        <div style={{ fontSize: 10, color: 'var(--text-dim)', marginTop: 8, minHeight: 40 }}>
          {result.loading ? 'RUNNING INFERENCE…' : 'AWAITING INJECTION'}
        </div>
      )}
    </div>
  )
}
