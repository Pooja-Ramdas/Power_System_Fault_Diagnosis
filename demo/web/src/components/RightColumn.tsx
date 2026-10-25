// src/components/RightColumn.tsx
// Evidence panel: model cards + narrative + report word highlights.

import ModelCard from './ModelCard'
import CaseSelector from './CaseSelector'
import type { ModelId, CaseData, CaseIndexEntry, FaultClass } from '../schema'
import type { ModelResult, ActiveMask, InjectionPhase } from '../hooks/useConsole'
import { classLabel } from '../classLabel'

const MODEL_IDS: ModelId[] = ['conflux', 'quorum', 'nexus']

const MODEL_ACCURACY: Record<ModelId, string> = {
  conflux: '87.97%',
  quorum:  '83.54%',
  nexus:   '82.28%',
}

interface Props {
  cases:         CaseIndexEntry[]
  selectedCase:  CaseIndexEntry | null
  caseData:      CaseData | null
  onSelectCase:  (c: CaseIndexEntry) => void
  faultClass:    FaultClass | null
  activeModel:   ModelId
  results:       Record<ModelId, ModelResult>
  onSelectModel: (id: ModelId) => void
  phase:         InjectionPhase
  mask:          ActiveMask
}

export default function RightColumn({
  cases, selectedCase, caseData, onSelectCase, faultClass,
  activeModel, results, onSelectModel, phase, mask
}: Props) {
  const activeResult = results[activeModel]
  const diag = activeResult?.diagnose
  const expl = diag?.explanation

  return (
    <div className="right-area" style={{ display: 'flex', flexDirection: 'column' }}>
      {/* Case selector */}
      <CaseSelector
        cases={cases}
        selected={selectedCase}
        onSelect={onSelectCase}
        faultClass={faultClass}
      />

      {/* Model cards */}
      <div className="section-hdr" style={{ marginTop: 4 }}>
        <span>INFERENCE MODELS</span>
        <span style={{ fontSize: 8, color: 'var(--text-dim)' }}>CLICK TO INSPECT</span>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6, padding: '8px 8px' }}>
        {MODEL_IDS.map(mid => (
          <div key={mid}>
            <ModelCard
              modelId={mid}
              result={results[mid]}
              active={activeModel === mid}
              onClick={() => onSelectModel(mid)}
            />
            <div style={{ fontSize: 8, color: 'var(--text-dim)', paddingLeft: 2, marginTop: 1 }}>
              Test acc: {MODEL_ACCURACY[mid]}
            </div>
          </div>
        ))}
      </div>

      {/* Active model evidence: narrative */}
      {expl?.narrative && expl.narrative.length > 0 && (
        <div>
          <div className="section-hdr" style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span>EVIDENCE NARRATIVE · {activeModel.toUpperCase()}</span>
            {expl.dominant && (
              <span style={{ fontSize: 8, color: 'var(--text-live)' }}>
                DOMINANT: {{ D: 'DIAGRAM', T: 'TELEMETRY', R: 'REPORT' }[expl.dominant as 'D'|'T'|'R'] || expl.dominant}
              </span>
            )}
          </div>
          <div className="panel">
            {expl.narrative.map((line, i) => (
              <div key={i} className="narrative-line fade-in">
                {line.split(/(REPORT|TELEMETRY|DIAGRAM|QUORUM|NEXUS|CONFLUX|OFF|DISABLED)/g).map((part, j) =>
                  /^(REPORT|TELEMETRY|DIAGRAM|QUORUM|NEXUS|CONFLUX|OFF|DISABLED)$/.test(part)
                    ? <strong key={j}>{part}</strong>
                    : part
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* QUORUM / NEXUS Specialist and Gating Rows */}
      {expl?.model_specific && (
        <div style={{ marginTop: 4 }}>
          <div className="section-hdr">
            <span>FUSION ARCHITECTURE · {activeModel.toUpperCase()}</span>
          </div>
          <div className="panel" style={{ fontSize: 9 }}>
            {activeModel === 'quorum' && Boolean(expl.model_specific.specialists) && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                {Object.entries(expl.model_specific.specialists as Record<string, any>).map(([mod, sp]) => (
                  <div key={mod} style={{ display: 'flex', justifyContent: 'space-between', opacity: sp.active ? 1 : 0.4 }}>
                    <span style={{ textTransform: 'uppercase' }}>{mod}:</span>
                    <span>
                      {sp.active ? `${classLabel(sp.pred)} (${(sp.conf*100).toFixed(0)}%, w=${sp.weight})` : 'DISABLED (w=0.0)'}
                    </span>
                  </div>
                ))}
              </div>
            )}
            {activeModel === 'nexus' && Boolean(expl.model_specific.branches) && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                {Object.entries(expl.model_specific.branches as Record<string, any>).map(([bName, br]) => (
                  <div key={bName} style={{ display: 'flex', justifyContent: 'space-between', opacity: br.active ? 1 : 0.4 }}>
                    <span style={{ textTransform: 'uppercase' }}>{bName.replace('_', '+')}:</span>
                    <span>
                      {br.active ? `${classLabel(br.pred)} (${(br.conf*100).toFixed(0)}%, w=${br.weight})` : 'DISABLED (w=0.0)'}
                    </span>
                  </div>
                ))}
              </div>
            )}
            {activeModel === 'conflux' && (
              <div style={{ color: 'var(--text-sec)' }}>
                Joint representation fusion (384-d). Active: {mask.join(', ')}.
              </div>
            )}
          </div>
        </div>
      )}

      {/* Report excerpt with word highlights (hidden if R disabled) */}
      {caseData && diag && mask.includes('R') && (
        <div>
          <div className="section-hdr">
            <span>OPERATOR REPORT EXCERPT</span>
            <span style={{ fontSize: 8, color: 'var(--text-dim)' }}>EN</span>
          </div>
          <div className="panel" style={{ maxHeight: 120, overflowY: 'auto' }}>
            <ReportHighlight
              text={caseData.report.en || caseData.report.es}
              wordScores={caseData.report.word_scores}
            />
          </div>
        </div>
      )}
      {caseData && diag && !mask.includes('R') && (
        <div style={{ marginTop: 4 }}>
          <div className="section-hdr">
            <span>OPERATOR REPORT EXCERPT</span>
            <span style={{ fontSize: 8, color: 'var(--text-dim)' }}>OFF</span>
          </div>
          <div className="panel" style={{ color: 'var(--text-dim)', fontSize: 9 }}>
            Report evidence hidden (modality disabled by active mask).
          </div>
        </div>
      )}

      {/* Model tooltip / no-data hint */}
      {phase === 'idle' && !selectedCase && (
        <div className="panel" style={{ color: 'var(--text-dim)', fontSize: 9, marginTop: 8 }}>
          ① Select a fault class above<br/>
          ② Pick a fault event<br/>
          ③ Press INJECT FAULT<br/>
          <br/>
          Models evaluate multimodal inputs upon fault injection.
        </div>
      )}
    </div>
  )
}

// ── Inline sub-component: highlighted report text ─────────────────────────────
function ReportHighlight({ text, wordScores }: { text: string; wordScores: {w:string;s:number}[] }) {
  if (!text) return <div style={{ color: 'var(--text-dim)', fontSize: 9 }}>No report text.</div>

  const scoreMap: Record<string, number> = {}
  for (const { w, s } of wordScores) scoreMap[w.toLowerCase()] = s

  // Tokenize and highlight
  const tokens = text.split(/(\s+|[,;:.!?()\[\]])/g)
  return (
    <div style={{ fontSize: 9, lineHeight: 1.6, color: 'var(--text-sec)' }}>
      {tokens.map((tok, i) => {
        const clean = tok.toLowerCase().replace(/[^a-z0-9]/g, '')
        const score = scoreMap[clean]
        if (score !== undefined && Math.abs(score) > 0.1) {
          return (
            <span
              key={i}
              className={score > 0 ? 'word-hi-pos' : 'word-hi-neg'}
              title={`score: ${score.toFixed(2)}`}
            >
              {tok}
            </span>
          )
        }
        return <span key={i}>{tok}</span>
      })}
    </div>
  )
}
