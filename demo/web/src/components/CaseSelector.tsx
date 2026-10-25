// src/components/CaseSelector.tsx
// Scrollable case list for the selected fault class.

import type { CaseIndexEntry, FaultClass } from '../schema'
import { classLabel } from '../classLabel'

const FAULT_COLORS: Record<string, string> = {
  line_fault:        'var(--fault-sc)',
  breaker_fault:     'var(--fault-oc)',
  transformer_fault: 'var(--fault-ef)',
  busbar_fault:      'var(--fault-oo)',
}

interface Props {
  cases:       CaseIndexEntry[]
  selected:    CaseIndexEntry | null
  onSelect:    (c: CaseIndexEntry) => void
  faultClass:  FaultClass | null
}

export default function CaseSelector({ cases, selected, onSelect, faultClass }: Props) {
  if (!faultClass) return null
  return (
    <div>
      <div className="section-hdr">
        <span>FAULT EVENTS · {cases.length} cases</span>
        <span style={{ color: FAULT_COLORS[faultClass] }}>
          {classLabel(faultClass)}
        </span>
      </div>
      <div style={{ maxHeight: 120, overflowY: 'auto' }}>
        {cases.map(c => (
          <div
            key={c.id}
            onClick={() => onSelect(c)}
            style={{
              padding: '5px 12px',
              cursor: 'pointer',
              borderBottom: '1px solid var(--line-dim)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              background: selected?.id === c.id ? 'var(--bg-2)' : 'transparent',
              borderLeft: selected?.id === c.id ? `2px solid ${FAULT_COLORS[c.true_class]}` : '2px solid transparent',
              transition: 'background 0.1s',
            }}
          >
            <div>
              <div style={{ fontSize: 10, color: 'var(--text-pri)', fontWeight: 700 }}>{c.id}</div>
              <div style={{ fontSize: 9, color: 'var(--text-dim)' }}>{c.element_name}</div>
            </div>
            <div style={{ fontSize: 8, color: 'var(--text-dim)', textAlign: 'right' }}>
              {new Date(c.event_ts).toLocaleDateString()}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
