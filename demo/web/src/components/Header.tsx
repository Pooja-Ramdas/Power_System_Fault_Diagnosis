// src/components/Header.tsx
import React from 'react'
import ThreeLogo from '../fx/ThreeLogo'
import type { FaultClass, ModelId } from '../schema'
import type { ActiveMask } from '../hooks/useConsole'

const CLASS_LABELS: Record<string, string> = {
  line_fault:         'LINE FAULT [1]',
  breaker_fault:      'BREAKER FAULT [2]',
  transformer_fault:  'TRANSFORMER FAULT [3]',
  busbar_fault:       'BUSBAR FAULT [4]',
}
const CLASS_COLORS: Record<string, string> = {
  line_fault:        'var(--fault-sc)',
  breaker_fault:     'var(--fault-oc)',
  transformer_fault: 'var(--fault-ef)',
  busbar_fault:      'var(--fault-oo)',
}
const MOD_COLORS: Record<string, string> = {
  D: 'var(--mod-d)',
  T: 'var(--mod-t)',
  R: 'var(--mod-r)',
}

interface Props {
  selectedClass: FaultClass | null
  onSelectClass: (c: FaultClass) => void
  mask: ActiveMask
  onToggleMask: (m: 'D'|'T'|'R') => void
  onInject: () => void
  canInject: boolean
  injecting: boolean
  apiLive: boolean
  onOpenAbout: () => void
}

export default function Header({
  selectedClass, onSelectClass, mask, onToggleMask, onInject, canInject, injecting, apiLive, onOpenAbout
}: Props) {
  const classes: FaultClass[] = ['line_fault','breaker_fault','transformer_fault','busbar_fault']
  return (
    <header className="header-area header">
      {/* Left: wordmark */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <div style={{ cursor: 'pointer' }} onClick={onOpenAbout} title="Click to view system details">
          <div className="header__wordmark vt323">TRIAD-13</div>
          <div className="header__subtitle">Multimodal Fault Console · CEN Chile</div>
        </div>

        {/* Fault class selectors with 1-4 shortcuts */}
        <div style={{ display: 'flex', gap: 4, marginLeft: 16 }}>
          {classes.map(c => (
            <button
              key={c}
              className={`class-btn ${selectedClass === c ? 'active' : ''}`}
              style={selectedClass === c ? { color: CLASS_COLORS[c], borderColor: CLASS_COLORS[c] } : {}}
              onClick={() => onSelectClass(c)}
            >
              {CLASS_LABELS[c]}
            </button>
          ))}
        </div>
      </div>

      {/* Right: modality toggles + inject + about + logo */}
      <div className="header__controls">
        {/* API status badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ fontSize: 9, color: apiLive ? 'var(--status-live)' : 'var(--status-cached)', fontWeight: 700 }}>
            {apiLive ? '● LIVE API' : '▲ CACHED'}
          </span>
          <button
            onClick={onOpenAbout}
            style={{
              background: 'transparent',
              border: '1px solid var(--line-mid)',
              color: 'var(--text-sec)',
              fontSize: 9,
              padding: '2px 6px',
              cursor: 'pointer',
              clipPath: 'var(--chamfer-4)',
              letterSpacing: '0.08em',
            }}
          >
            ABOUT
          </button>
        </div>

        {/* Modality toggles */}
        <div style={{ display: 'flex', gap: 4 }}>
          {(['D','T','R'] as const).map(m => (
            <button
              key={m}
              className={`mod-toggle ${mask.includes(m) ? 'active' : ''}`}
              style={{ color: MOD_COLORS[m], borderColor: mask.includes(m) ? MOD_COLORS[m] : undefined }}
              onClick={() => onToggleMask(m)}
              title={{ D: 'Diagram', T: 'Telemetry', R: 'Report' }[m]}
            >
              {m} · {{ D: 'DIAG', T: 'TEL', R: 'RPT' }[m]}
            </button>
          ))}
        </div>

        {/* Inject button */}
        <button
          className="inject-btn"
          onClick={onInject}
          disabled={!canInject || injecting}
          title="Shortcut: Space / Enter"
        >
          {injecting ? '■ RUNNING…' : '▶ INJECT FAULT'}
        </button>

        {/* Three.js logo (clicking opens ABOUT) */}
        <div onClick={onOpenAbout} style={{ cursor: 'pointer' }} role="img" aria-label="TRIAD-13 logo">
          <ThreeLogo className="header__canvas" />
        </div>
      </div>
    </header>
  )
}
