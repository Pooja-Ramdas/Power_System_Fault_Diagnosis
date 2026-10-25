// src/components/AboutModal.tsx
// Honesty overlay explaining model architectures, data sources, and live vs cached inference.

import React from 'react'

interface Props {
  isOpen: boolean
  onClose: () => void
  apiLive: boolean
}

export default function AboutModal({ isOpen, onClose, apiLive }: Props) {
  if (!isOpen) return null

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(5, 8, 13, 0.85)',
        backdropFilter: 'blur(4px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 9999,
        padding: 16,
      }}
      onClick={onClose}
    >
      <div
        onClick={e => e.stopPropagation()}
        style={{
          background: 'var(--bg-card)',
          border: '1px solid var(--conflux)',
          clipPath: 'var(--chamfer-10)',
          maxWidth: 640,
          width: '100%',
          maxHeight: '85vh',
          overflowY: 'auto',
          padding: '24px 28px',
          boxShadow: '0 0 30px rgba(56, 232, 255, 0.2)',
          color: 'var(--text-pri)',
        }}
      >
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--line-mid)', paddingBottom: 12, marginBottom: 16 }}>
          <div>
            <div className="vt323" style={{ fontSize: 24, color: 'var(--conflux)', letterSpacing: '0.08em' }}>
              TRIAD-13 // SYSTEM MANIFEST & HONESTY DISCLOSURE
            </div>
            <div style={{ fontSize: 9, color: 'var(--text-dim)', letterSpacing: '0.12em' }}>
              Multimodal Reasoning for Power-System Fault Diagnosis · Group 13
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'transparent',
              border: '1px solid var(--line-hi)',
              color: 'var(--text-sec)',
              padding: '4px 8px',
              cursor: 'pointer',
              fontFamily: 'JetBrains Mono',
              fontSize: 11,
            }}
          >
            [ESC / ✕]
          </button>
        </div>

        {/* Content sections */}
        <div style={{ fontSize: 11, lineHeight: 1.6, color: 'var(--text-sec)' }}>
          <p style={{ marginBottom: 12 }}>
            <strong style={{ color: 'var(--text-pri)' }}>Operational Nature:</strong> This console replays fixed held-out test events from the CEN Chile power transmission network. When a fault is injected, diagnosis and explainability are computed <strong style={{ color: apiLive ? 'var(--status-live)' : 'var(--status-cached)' }}>{apiLive ? 'LIVE' : 'FROM PRECOMPUTED CACHE'}</strong> by the backend inference models. No metrics or attributions are fabricated.
          </p>

          <div style={{ background: 'var(--bg-2)', padding: '10px 14px', clipPath: 'var(--chamfer-4)', border: '1px solid var(--line-dim)', marginBottom: 14 }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--conflux)', marginBottom: 6, letterSpacing: '0.08em' }}>
              TRAINED FUSION ARCHITECTURES (Full Test Set Metrics)
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
              <div>
                <strong style={{ color: 'var(--conflux)' }}>CONFLUX</strong>
                <div style={{ fontSize: 9, color: 'var(--text-dim)' }}>Early Fusion (384-d input)</div>
                <div style={{ fontSize: 10, color: 'var(--text-pri)', marginTop: 2 }}>Acc: 87.97% · F1: 82.20%</div>
              </div>
              <div>
                <strong style={{ color: 'var(--quorum)' }}>QUORUM</strong>
                <div style={{ fontSize: 9, color: 'var(--text-dim)' }}>Late Fusion (Learned vote)</div>
                <div style={{ fontSize: 10, color: 'var(--text-pri)', marginTop: 2 }}>Acc: 83.54% · F1: 76.51%</div>
              </div>
              <div>
                <strong style={{ color: 'var(--nexus)' }}>NEXUS</strong>
                <div style={{ fontSize: 9, color: 'var(--text-dim)' }}>Hybrid Fusion (Cross-attention)</div>
                <div style={{ fontSize: 10, color: 'var(--text-pri)', marginTop: 2 }}>Acc: 82.28% · F1: 73.03%</div>
              </div>
            </div>
          </div>

          <div style={{ marginBottom: 12 }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-pri)', marginBottom: 4, letterSpacing: '0.08em' }}>
              REAL GRID DATASET MODALITIES:
            </div>
            <ul style={{ paddingLeft: 16, fontSize: 10, display: 'flex', flexDirection: 'column', gap: 4 }}>
              <li><strong>Diagram (D):</strong> CEN InfoTécnica official substation single-line diagrams (ResNet-18 visual feature embeddings).</li>
              <li><strong>Telemetry (T):</strong> 48-hour hourly generation dispatch (MWh) across 7 channels (Thermal, Solar, Hydro, Wind, BESS, Other, Total). <em>Note: Telemetry represents generation power dispatch, not electrical voltage/current waveforms.</em></li>
              <li><strong>Report (R):</strong> CEN EAF (Estudio de Análisis de Falla) technical incident text reports with bilingual support (ES/EN TF-IDF featurization).</li>
            </ul>
          </div>

          <div style={{ borderTop: '1px solid var(--line-dim)', paddingTop: 10, marginTop: 12, display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 9, color: 'var(--text-dim)' }}>
            <span>Build SHA: <code style={{ color: 'var(--text-sec)' }}>feat/demo-dashboard</code></span>
            <span>Ground Truth: CEN EAF Records</span>
          </div>
        </div>
      </div>
    </div>
  )
}
