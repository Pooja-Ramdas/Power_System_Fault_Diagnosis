// src/components/BottomDock.tsx
// Tabbed bottom panel: Telemetry minimap | Ablation table | Model metadata legend.

import { useState } from 'react'
import type { CaseData, ModelId } from '../schema'
import { classLabel } from '../classLabel'
import type { ModelResult } from '../hooks/useConsole'

const MODEL_IDS: ModelId[] = ['conflux', 'quorum', 'nexus']
const MODEL_COLORS: Record<ModelId, string> = {
  conflux: 'var(--conflux)',
  quorum:  'var(--quorum)',
  nexus:   'var(--nexus)',
}
const MODEL_META: Record<ModelId, { acc: string; f1: string; fusion: string }> = {
  conflux: { acc: '87.97%', f1: '82.20%', fusion: 'EARLY — Joint 384-d representation at input stage' },
  quorum:  { acc: '83.54%', f1: '76.51%', fusion: 'LATE — Weighted vote: Img(0.40)+Txt(0.30)+Tel(0.30)' },
  nexus:   { acc: '82.28%', f1: '73.03%', fusion: 'HYBRID — Feature fusion then decision-level combination' },
}

type Tab = 'telemetry' | 'ablation' | 'legend'

interface Props {
  caseData:    CaseData | null
  activeModel: ModelId
  results:     Record<ModelId, ModelResult>
}

export default function BottomDock({ caseData, activeModel, results }: Props) {
  const [tab, setTab] = useState<Tab>('telemetry')
  const TABS: { id: Tab; label: string }[] = [
    { id: 'telemetry', label: 'TELEMETRY' },
    { id: 'ablation',  label: 'ABLATION TABLE' },
    { id: 'legend',    label: 'MODEL LEGEND' },
  ]

  return (
    <div className="bottom-area dock">
      <div className="dock-tabs">
        {TABS.map(t => (
          <button
            key={t.id}
            className={`dock-tab ${tab === t.id ? 'active' : ''}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
        {caseData && (
          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', padding: '0 12px',
                        fontSize: 9, color: 'var(--text-dim)' }}>
            {caseData.id} · {caseData.element_name} · {classLabel(caseData.true_class)}
          </div>
        )}
      </div>

      <div className="dock-content">
        {tab === 'telemetry' && <TelemetryView caseData={caseData} results={results} activeModel={activeModel} />}
        {tab === 'ablation'  && <AblationView  results={results} />}
        {tab === 'legend'    && <LegendView />}
      </div>
    </div>
  )
}

// ── Telemetry minimap ─────────────────────────────────────────────────────────
function TelemetryView({ caseData, results, activeModel }: { caseData: CaseData | null; results: Record<ModelId, ModelResult>; activeModel: ModelId }) {
  if (!caseData) {
    return <div style={{ color: 'var(--text-dim)', fontSize: 9 }}>No case selected.</div>
  }
  const rows = caseData.telemetry.rows
  const t0   = caseData.telemetry.t0_index
  const importance = results[activeModel]?.evidence?.telemetry_importance ?? []

  // Compute total generation per time step (channel 0 = total)
  const totals = rows.map(r => r.channels[0] ?? 0)
  const maxVal = Math.max(...totals, 1)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <div style={{ fontSize: 9, color: 'var(--text-dim)', display: 'flex', gap: 16 }}>
        <span>TOTAL GENERATION (MWh) · {rows.length} hours · t₀ = h{t0}</span>
        {importance.length > 0 && <span style={{ color: 'var(--mod-t)' }}>■ ATTRIBUTION OVERLAY</span>}
      </div>
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 1, height: 72, overflowX: 'auto' }}>
        {rows.map((r, i) => {
          const h = Math.max(4, Math.round((r.channels[0] / maxVal) * 64))
          const isT0 = i === t0
          const imp  = importance[i] ?? 0
          return (
            <div
              key={i}
              title={`h${i}: ${r.channels[0].toFixed(1)} MWh${imp > 0.2 ? ` | attr:${imp.toFixed(2)}` : ''}`}
              style={{
                width: 5, minWidth: 5, height: h,
                background: isT0 ? 'var(--conflux)' : 'var(--mod-t)',
                opacity: isT0 ? 1 : 0.4 + imp * 0.6,
                borderRadius: '1px 1px 0 0',
                boxShadow: imp > 0.5 ? '0 0 4px var(--mod-t)' : 'none',
                flexShrink: 0,
              }}
            />
          )
        })}
      </div>
      <div style={{ fontSize: 8, color: 'var(--text-dim)' }}>
        Channels: {caseData.telemetry.columns.join(' · ')} (all MWh · hourly generation data)
      </div>
    </div>
  )
}

// ── Ablation table ────────────────────────────────────────────────────────────
function AblationView({ results }: { results: Record<ModelId, ModelResult> }) {
  const COMBOS = ['D','T','R','DT','DR','TR','DTR']
  const COMBO_LABELS: Record<string, string> = {
    D:'Diagram only', T:'Telemetry only', R:'Report only',
    DT:'Diag+Tel', DR:'Diag+Rpt', TR:'Tel+Rpt', DTR:'All modalities'
  }

  const hasData = MODEL_IDS.some(m => results[m].diagnose?.ablations)
  if (!hasData) return <div style={{ color: 'var(--text-dim)', fontSize: 9 }}>Run injection to see ablation table.</div>

  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="abl-table">
        <thead>
          <tr>
            <th>MASK</th>
            {MODEL_IDS.map(m => (
              <th key={m} colSpan={2} style={{ color: MODEL_COLORS[m] }}>{m.toUpperCase()}</th>
            ))}
          </tr>
          <tr>
            <th />
            {MODEL_IDS.flatMap(m => [
              <th key={`${m}-pred`}>PRED</th>,
              <th key={`${m}-conf`}>CONF</th>,
            ])}
          </tr>
        </thead>
        <tbody>
          {COMBOS.map(combo => (
            <tr key={combo}>
              <td title={COMBO_LABELS[combo]}>{combo}</td>
              {MODEL_IDS.flatMap(m => {
                const abl = results[m].diagnose?.ablations?.[combo]
                return [
                  <td key={`${m}-${combo}-pred`} style={{ fontSize: 8 }}>
                    {abl ? classLabel(abl.pred) : '—'}
                  </td>,
                  <td key={`${m}-${combo}-conf`}>
                    {abl ? `${(abl.conf * 100).toFixed(0)}%` : '—'}
                  </td>,
                ]
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ── Legend ────────────────────────────────────────────────────────────────────
function LegendView() {
  return (
    <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
      {MODEL_IDS.map(mid => {
        const m = MODEL_META[mid]
        return (
          <div key={mid} style={{ minWidth: 180 }}>
            <div style={{ color: MODEL_COLORS[mid], fontWeight: 700, fontSize: 11, marginBottom: 4,
                          fontFamily: 'VT323, monospace', letterSpacing: '0.05em' }}>
              {mid.toUpperCase()}
            </div>
            <div style={{ fontSize: 9, color: 'var(--text-sec)' }}>Test Acc: {m.acc}</div>
            <div style={{ fontSize: 9, color: 'var(--text-sec)' }}>Macro-F1: {m.f1}</div>
            <div style={{ fontSize: 8, color: 'var(--text-dim)', marginTop: 4, lineHeight: 1.4 }}>{m.fusion}</div>
          </div>
        )
      })}
      <div style={{ minWidth: 200 }}>
        <div style={{ fontSize: 9, fontWeight: 700, color: 'var(--text-sec)', marginBottom: 4, letterSpacing: '0.1em' }}>
          MODALITY KEY
        </div>
        {[
          { k: 'D', label: 'DIAGRAM — InfoTécnica single-line PDF', color: 'var(--mod-d)' },
          { k: 'T', label: 'TELEMETRY — Hourly generation data (MWh)', color: 'var(--mod-t)' },
          { k: 'R', label: 'REPORT — Operator EAF incident text', color: 'var(--mod-r)' },
        ].map(({ k, label, color }) => (
          <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
            <div style={{ width: 8, height: 8, background: color, borderRadius: 1, flexShrink: 0 }} />
            <div style={{ fontSize: 8, color: 'var(--text-dim)' }}><strong style={{ color }}>{k}</strong> — {label}</div>
          </div>
        ))}
        <div style={{ marginTop: 6, fontSize: 8, color: 'var(--text-dim)' }}>
          Shapley values: share = |φ| / Σ|φ|<br/>
          Efficiency: Σφ = f(all) − f(none)
        </div>
      </div>
    </div>
  )
}
