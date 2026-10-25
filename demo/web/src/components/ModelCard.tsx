// src/components/ModelCard.tsx
// CONFLUX / QUORUM / NEXUS diagnosis bubble card with Shapley contribution bars.

import type { ModelId, DiagnoseResponse } from '../schema'
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
        {diag && (
          <span className={diag.source === 'live' ? 'badge badge-live' : 'badge badge-cached'}>
            {diag.source}
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
                {diag.prediction.display_pred}
              </div>
              <div style={{ fontSize: 9, color: diag.prediction.correct ? 'var(--status-live)' : 'var(--status-err)', marginTop: 2 }}>
                {diag.prediction.correct ? '✓ CORRECT' : '✗ INCORRECT'}
              </div>
              <div style={{ fontSize: 8, color: 'var(--text-dim)', marginTop: 1 }}>
                {diag.latency_ms > 0 ? `${diag.latency_ms}ms` : 'CACHED'}
              </div>
            </div>
          </div>

          {/* Shapley contribution bars */}
          {diag.explanation?.contrib && (
            <div style={{ marginTop: 8 }}>
              {(['D','T','R'] as const).map(m => {
                const c = diag.explanation.contrib[m]
                const pct = Math.round((c?.share ?? 0) * 100)
                const sign = (c?.signed ?? 0) >= 0 ? '+' : '−'
                return (
                  <div key={m} className="contrib-row">
                    <div className="contrib-label" style={{ color: MOD_COLORS[m] }}>
                      {MOD_LABELS[m]}
                    </div>
                    <div className="contrib-bar-bg">
                      <div
                        className="contrib-bar-fill"
                        style={{ width: `${pct}%`, background: MOD_COLORS[m] }}
                      />
                    </div>
                    <div className="contrib-pct">{pct}%</div>
                    <div className="contrib-sign" style={{ color: (c?.signed ?? 0) >= 0 ? 'var(--status-live)' : 'var(--status-err)' }}>
                      {sign}
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
                <div className="prob-item__label">{cls.replace('_fault','').replace('_',' ')}</div>
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
