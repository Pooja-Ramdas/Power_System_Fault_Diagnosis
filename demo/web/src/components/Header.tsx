// src/components/Header.tsx
import ThreeLogo from '../fx/ThreeLogo'
import type { FaultClass, ModelId } from '../schema'
import type { ActiveMask } from '../hooks/useConsole'

const CLASS_LABELS: Record<string, string> = {
  line_fault:         'SHORT CIRCUIT',
  breaker_fault:      'OPEN CIRCUIT',
  transformer_fault:  'EQUIPMENT FAULT',
  busbar_fault:       'OVERHEATING/OVL',
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
}

export default function Header({
  selectedClass, onSelectClass, mask, onToggleMask, onInject, canInject, injecting, apiLive
}: Props) {
  const classes: FaultClass[] = ['line_fault','breaker_fault','transformer_fault','busbar_fault']
  return (
    <header className="header-area header">
      {/* Left: wordmark */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <div>
          <div className="header__wordmark vt323">TRIAD-13</div>
          <div className="header__subtitle">Multimodal Fault Console · CEN Chile</div>
        </div>

        {/* Fault class selectors */}
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

      {/* Right: modality toggles + inject + logo */}
      <div className="header__controls">
        {/* API status dot */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 9, color: apiLive ? 'var(--status-live)' : 'var(--text-dim)' }}>
          <span style={{ fontSize: 8 }}>{apiLive ? '●' : '○'}</span>
          {apiLive ? 'API LIVE' : 'CACHED'}
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
        >
          {injecting ? '■ RUNNING…' : '▶ INJECT FAULT'}
        </button>

        {/* Three.js logo */}
        <ThreeLogo className="header__canvas" />
      </div>
    </header>
  )
}
