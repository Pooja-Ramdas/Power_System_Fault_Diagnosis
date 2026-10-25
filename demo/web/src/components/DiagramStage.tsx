// src/components/DiagramStage.tsx
// Main stage: single-line diagram viewer with HUD tint.
// Injection triggers shockwave ring + scan sweep overlay.

import { useEffect, useRef, useState } from 'react'
import type { CaseData } from '../schema'
import type { InjectionPhase } from '../hooks/useConsole'

interface Props {
  caseData:     CaseData | null
  caseLoading:  boolean
  phase:        InjectionPhase
  selectedCase: { id: string; element_name: string } | null
}

export default function DiagramStage({ caseData, caseLoading, phase, selectedCase }: Props) {
  const [showScan,  setShowScan]  = useState(false)
  const [showShock, setShowShock] = useState(false)
  const [hudMode,   setHudMode]   = useState(true)
  const scanKeyRef  = useRef(0)
  const shockKeyRef = useRef(0)

  // Trigger effects on injection start
  useEffect(() => {
    if (phase === 'injecting') {
      setShowScan(false)
      setShowShock(false)
      requestAnimationFrame(() => {
        scanKeyRef.current++
        shockKeyRef.current++
        setShowScan(true)
        setShowShock(true)
      })
      // Auto-hide after animation
      const t1 = setTimeout(() => setShowScan(false),  1400)
      const t2 = setTimeout(() => setShowShock(false), 900)
      return () => { clearTimeout(t1); clearTimeout(t2) }
    }
  }, [phase])

  const imgSrc = caseData
    ? (hudMode ? caseData.image.hud : caseData.image.raw)
    : null

  return (
    <div className="stage-area crt" style={{ position: 'relative' }}>

      {/* Empty state */}
      {!selectedCase && (
        <div style={{
          position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center', gap: 12,
          color: 'var(--text-dim)', fontSize: 10, letterSpacing: '0.1em'
        }}>
          <div style={{ fontSize: 40, opacity: 0.15 }}>⚡</div>
          <div>SELECT A FAULT CLASS TO BEGIN</div>
          <div style={{ opacity: 0.5 }}>CEN Chile grid · InfoTécnica single-line diagrams</div>
        </div>
      )}

      {/* Loading */}
      {caseLoading && (
        <div style={{
          position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center'
        }}>
          <div className="spinner" />
        </div>
      )}

      {/* Diagram image */}
      {imgSrc && !caseLoading && (
        <div className="diagram-container">
          <img
            key={imgSrc}
            src={imgSrc}
            alt={`Single-line diagram — ${caseData?.element_name}`}
            className="diagram-img fade-in"
            style={{ opacity: phase === 'injecting' ? 0.7 : 1 }}
          />
        </div>
      )}

      {/* Scan sweep */}
      {showScan && (
        <div key={`scan-${scanKeyRef.current}`} className="scan-line" />
      )}

      {/* Shockwave ring */}
      {showShock && (
        <div
          key={`shock-${shockKeyRef.current}`}
          className="shockwave"
          style={{ width: 80, height: 80, left: '50%', top: '50%', marginLeft: -40, marginTop: -40 }}
        />
      )}

      {/* Case info overlay */}
      {caseData && (
        <div style={{
          position: 'absolute', bottom: 8, left: 8, right: 8,
          display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between',
          pointerEvents: 'none',
        }}>
          <div style={{
            background: 'rgba(5,8,13,0.8)', padding: '4px 8px',
            clipPath: 'var(--chamfer-4)', fontSize: 9, color: 'var(--text-sec)'
          }}>
            <div style={{ color: 'var(--text-pri)', fontWeight: 700 }}>{caseData.element_name}</div>
            <div>{caseData.id} · {new Date(caseData.event_ts).toLocaleDateString()}</div>
            <div style={{ color: 'var(--text-dim)' }}>
              {caseData.image.width}×{caseData.image.height}px · InfoTécnica
            </div>
          </div>

          {/* HUD / RAW toggle */}
          <button
            style={{
              pointerEvents: 'all', background: 'rgba(5,8,13,0.8)', border: '1px solid var(--line-mid)',
              color: 'var(--text-sec)', fontSize: 9, padding: '3px 8px',
              clipPath: 'var(--chamfer-4)', cursor: 'pointer', letterSpacing: '0.08em'
            }}
            onClick={() => setHudMode(h => !h)}
          >
            {hudMode ? 'HUD ▾' : 'RAW ▾'}
          </button>
        </div>
      )}
    </div>
  )
}
