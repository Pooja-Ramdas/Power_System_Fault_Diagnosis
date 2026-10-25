// src/components/DiagramStage.tsx
// Main stage: single-line diagram viewer with pan/zoom, diagnosis bubbles,
// HUD/RAW mode, scan sweep, shockwave ring, reticle pulse, and ground truth stamp.

import React, { useEffect, useRef, useState, useCallback } from 'react'
import type { CaseData, ModelId, FaultClass } from '../schema'
import type { InjectionPhase, ModelResult } from '../hooks/useConsole'

const MODEL_COLORS: Record<ModelId, string> = {
  conflux: 'var(--conflux)',
  quorum:  'var(--quorum)',
  nexus:   'var(--nexus)',
}

const FAULT_COLORS: Record<string, string> = {
  line_fault:        'var(--fault-sc)',
  breaker_fault:     'var(--fault-oc)',
  transformer_fault: 'var(--fault-ef)',
  busbar_fault:      'var(--fault-oo)',
}

const FAULT_DISPLAY: Record<string, string> = {
  line_fault:        'SHORT CIRCUIT',
  breaker_fault:     'OPEN CIRCUIT',
  transformer_fault: 'EQUIPMENT FAULT',
  busbar_fault:      'OVERHEATING/OVL',
}

interface Props {
  caseData:     CaseData | null
  caseLoading:  boolean
  phase:        InjectionPhase
  selectedCase: { id: string; element_name: string; true_class: FaultClass } | null
  results:      Record<ModelId, ModelResult>
  activeModel:  ModelId
  onSelectModel: (id: ModelId) => void
  hudMode:      boolean
  onToggleHud:  () => void
  onShuffle:    () => void
}

export default function DiagramStage({
  caseData,
  caseLoading,
  phase,
  selectedCase,
  results,
  activeModel,
  onSelectModel,
  hudMode,
  onToggleHud,
  onShuffle,
}: Props) {
  const [showScan, setShowScan] = useState(false)
  const [showShock, setShowShock] = useState(false)
  const scanKeyRef = useRef(0)
  const shockKeyRef = useRef(0)

  // Pan / Zoom state
  const [scale, setScale] = useState(1.0)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const isDragging = useRef(false)
  const startDrag = useRef({ x: 0, y: 0 })

  // Reset pan/zoom on case change
  useEffect(() => {
    setScale(1.0)
    setPan({ x: 0, y: 0 })
  }, [selectedCase?.id])

  // Trigger theatrics on injection start
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
      const t1 = setTimeout(() => setShowScan(false), 1400)
      const t2 = setTimeout(() => setShowShock(false), 900)
      return () => { clearTimeout(t1); clearTimeout(t2) }
    }
  }, [phase])

  // Wheel zoom
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault()
    const delta = e.deltaY > 0 ? 0.9 : 1.1
    setScale(s => Math.min(Math.max(s * delta, 0.6), 3.5))
  }, [])

  // Mouse pan
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return
    isDragging.current = true
    startDrag.current = { x: e.clientX - pan.x, y: e.clientY - pan.y }
  }, [pan])

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isDragging.current) return
    setPan({
      x: e.clientX - startDrag.current.x,
      y: e.clientY - startDrag.current.y,
    })
  }, [])

  const handleMouseUp = useCallback(() => {
    isDragging.current = false
  }, [])

  const handleDoubleClick = useCallback(() => {
    setScale(1.0)
    setPan({ x: 0, y: 0 })
  }, [])

  const imgSrc = caseData
    ? (hudMode ? caseData.image.hud : caseData.image.raw)
    : null

  // Model predictions for bubbles
  const models: ModelId[] = ['conflux', 'quorum', 'nexus']
  const finishedModels = models.filter(m => results[m].diagnose !== null)
  const distinctPreds = new Set(finishedModels.map(m => results[m].diagnose?.prediction.pred))
  const modelsDisagree = distinctPreds.size > 1

  return (
    <div
      className="stage-area crt"
      style={{ position: 'relative', overflow: 'hidden', cursor: isDragging.current ? 'grabbing' : 'grab' }}
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onDoubleClick={handleDoubleClick}
    >
      {/* Empty State */}
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

      {/* Loading Spinner */}
      {caseLoading && (
        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div className="spinner" />
        </div>
      )}

      {/* Diagram Canvas with Pan/Zoom Transform */}
      {imgSrc && !caseLoading && (
        <div
          className="diagram-container"
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${scale})`,
            transformOrigin: 'center center',
            transition: isDragging.current ? 'none' : 'transform 0.1s ease-out',
          }}
        >
          <img
            key={imgSrc}
            src={imgSrc}
            alt={`Single-line diagram — ${caseData?.element_name}`}
            className="diagram-img fade-in"
            draggable={false}
            style={{ opacity: phase === 'injecting' ? 0.75 : 1 }}
          />

          {/* Reticle Pulse on Element */}
          {phase === 'injecting' && (
            <div
              style={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                width: 60,
                height: 60,
                marginTop: -30,
                marginLeft: -30,
                border: '1px solid var(--conflux)',
                borderRadius: '50%',
                animation: 'pulse-glow 1s infinite alternate',
                pointerEvents: 'none',
              }}
            />
          )}
        </div>
      )}

      {/* Scan Sweep Effect */}
      {showScan && <div key={`scan-${scanKeyRef.current}`} className="scan-line" />}

      {/* Shockwave Ring */}
      {showShock && (
        <div
          key={`shock-${shockKeyRef.current}`}
          className="shockwave"
          style={{ width: 80, height: 80, left: '50%', top: '50%', marginLeft: -40, marginTop: -40 }}
        />
      )}

      {/* Diagnosis Callout Bubbles — stacked along right edge */}
      <div
        aria-live="polite"
        style={{
          position: 'absolute',
          right: 16,
          top: 24,
          display: 'flex',
          flexDirection: 'column',
          gap: 12,
          pointerEvents: 'all',
          zIndex: 30,
        }}
      >
        {models.map(mid => {
          const res = results[mid]
          const diag = res.diagnose
          if (!diag) return null

          const isSelected = activeModel === mid
          const pred = diag.prediction
          const modelColor = MODEL_COLORS[mid]
          const faultColor = FAULT_COLORS[pred.pred] || 'var(--text-pri)'

          return (
            <div
              key={mid}
              onClick={() => onSelectModel(mid)}
              style={{
                position: 'relative',
                background: 'rgba(11, 16, 25, 0.92)',
                border: `1px solid ${isSelected ? modelColor : 'var(--line-mid)'}`,
                boxShadow: isSelected ? `0 0 14px -2px ${modelColor}` : '0 2px 8px rgba(0,0,0,0.5)',
                clipPath: 'var(--chamfer-6)',
                padding: '6px 12px',
                minWidth: 190,
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              {/* Pointer Tail pointing right toward model card */}
              <div
                style={{
                  position: 'absolute',
                  right: -6,
                  top: '50%',
                  transform: 'translateY(-50%) rotate(45deg)',
                  width: 10,
                  height: 10,
                  background: 'rgba(11, 16, 25, 0.92)',
                  borderRight: `1px solid ${isSelected ? modelColor : 'var(--line-mid)'}`,
                  borderTop: `1px solid ${isSelected ? modelColor : 'var(--line-mid)'}`,
                }}
              />

              {/* Bubble Header */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 2 }}>
                <span style={{ fontSize: 9, fontWeight: 700, color: modelColor, letterSpacing: '0.1em' }}>
                  {mid.toUpperCase()} // {{ conflux: 'EARLY', quorum: 'LATE', nexus: 'HYBRID' }[mid]}
                </span>
                <span
                  style={{
                    fontSize: 8,
                    fontWeight: 700,
                    color: diag.source === 'live' ? 'var(--status-live)' : 'var(--status-cached)',
                  }}
                >
                  {diag.source.toUpperCase()}
                </span>
              </div>

              {/* Bubble Body: Fault Class + Confidence */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                <span style={{ fontSize: 10, fontWeight: 700, color: faultColor }}>
                  {pred.display_pred}
                </span>
                <span className="vt323" style={{ fontSize: 18, color: modelColor, marginLeft: 8 }}>
                  {(pred.confidence * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          )
        })}

        {/* Ground Truth Stamp + Models Disagree Flag */}
        {phase === 'done' && selectedCase && (
          <div
            className="fade-in"
            style={{
              background: 'rgba(5, 8, 13, 0.9)',
              border: '1px dashed var(--line-hi)',
              padding: '6px 10px',
              clipPath: 'var(--chamfer-4)',
              fontSize: 9,
              marginTop: 4,
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ color: 'var(--text-dim)', letterSpacing: '0.08em' }}>GROUND TRUTH:</span>
              <span style={{ color: FAULT_COLORS[selectedCase.true_class], fontWeight: 700 }}>
                {FAULT_DISPLAY[selectedCase.true_class]}
              </span>
            </div>
            {modelsDisagree && (
              <div style={{ marginTop: 4, color: 'var(--status-cached)', fontSize: 8, fontWeight: 700, letterSpacing: '0.08em' }}>
                ⚡ MODELS DISAGREE ON VERDICT
              </div>
            )}
          </div>
        )}
      </div>

      {/* Case Info & Controls Overlay */}
      {caseData && (
        <div style={{
          position: 'absolute', bottom: 8, left: 8, right: 8,
          display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between',
          pointerEvents: 'none', zIndex: 20,
        }}>
          {/* Metadata Tag */}
          <div style={{
            background: 'rgba(5,8,13,0.85)', padding: '6px 10px',
            clipPath: 'var(--chamfer-4)', fontSize: 9, color: 'var(--text-sec)',
            border: '1px solid var(--line-dim)',
          }}>
            <div style={{ color: 'var(--text-pri)', fontWeight: 700 }}>{caseData.element_name}</div>
            <div>{caseData.id} · {new Date(caseData.event_ts).toLocaleDateString()}</div>
            <div style={{ color: 'var(--text-dim)' }}>
              {caseData.image.width}×{caseData.image.height}px · InfoTécnica PDF
            </div>
          </div>

          {/* Action Buttons: Shuffle, HUD/RAW, Zoom Reset */}
          <div style={{ display: 'flex', gap: 6, pointerEvents: 'all' }}>
            <button
              onClick={onShuffle}
              style={{
                background: 'rgba(5,8,13,0.85)', border: '1px solid var(--line-mid)',
                color: 'var(--text-sec)', fontSize: 9, padding: '4px 10px',
                clipPath: 'var(--chamfer-4)', cursor: 'pointer', letterSpacing: '0.08em',
              }}
              title="Pick another event of this class"
            >
              ⟳ SHUFFLE
            </button>
            <button
              onClick={onToggleHud}
              style={{
                background: 'rgba(5,8,13,0.85)', border: '1px solid var(--line-mid)',
                color: 'var(--text-sec)', fontSize: 9, padding: '4px 10px',
                clipPath: 'var(--chamfer-4)', cursor: 'pointer', letterSpacing: '0.08em',
              }}
              title="Toggle HUD dark tint vs raw scan (Key: H)"
            >
              {hudMode ? 'HUD' : 'RAW'}
            </button>
            {scale !== 1.0 && (
              <button
                onClick={handleDoubleClick}
                style={{
                  background: 'rgba(5,8,13,0.85)', border: '1px solid var(--conflux)',
                  color: 'var(--conflux)', fontSize: 9, padding: '4px 8px',
                  clipPath: 'var(--chamfer-4)', cursor: 'pointer',
                }}
                title="Reset zoom to 100%"
              >
                1:1
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
