// src/App.tsx
// Main layout: Header | DiagramStage | RightColumn | BottomDock + BootScene overlay.

import { useState, useEffect, useCallback } from 'react'
import { useConsole } from './hooks/useConsole'
import Header from './components/Header'
import DiagramStage from './components/DiagramStage'
import RightColumn from './components/RightColumn'
import BottomDock from './components/BottomDock'
import BootScene from './fx/BootScene'
import { checkHealth } from './api'

export default function App() {
  const [booted,  setBooted]  = useState(false)
  const [apiLive, setApiLive] = useState(false)

  const {
    state,
    casesForClass,
    selectClass,
    selectCase,
    setModel,
    toggleMask,
    inject,
  } = useConsole()

  // Boot scene
  const handleBoot = useCallback(() => setBooted(true), [])

  // Check API health once after boot
  useEffect(() => {
    if (!booted) return
    const apiBase = import.meta.env.VITE_API_URL as string | undefined
    if (!apiBase) return
    checkHealth(apiBase).then(ok => setApiLive(ok))
  }, [booted])

  const canInject = !!state.selectedCase && !!state.caseData && state.phase !== 'injecting'

  return (
    <>
      {/* Boot overlay */}
      {!booted && <BootScene onDone={handleBoot} />}

      {/* Main layout */}
      <div className="app-layout">
        <Header
          selectedClass={state.selectedClass}
          onSelectClass={selectClass}
          mask={state.mask}
          onToggleMask={toggleMask}
          onInject={inject}
          canInject={canInject}
          injecting={state.phase === 'injecting'}
          apiLive={apiLive}
        />

        <DiagramStage
          caseData={state.caseData}
          caseLoading={state.caseLoading}
          phase={state.phase}
          selectedCase={state.selectedCase}
        />

        <RightColumn
          cases={casesForClass}
          selectedCase={state.selectedCase}
          caseData={state.caseData}
          onSelectCase={selectCase}
          faultClass={state.selectedClass}
          activeModel={state.activeModel}
          results={state.results}
          onSelectModel={setModel}
          phase={state.phase}
        />

        <BottomDock
          caseData={state.caseData}
          activeModel={state.activeModel}
          results={state.results}
        />
      </div>

      {/* Error toast */}
      {state.errorMsg && (
        <div style={{
          position: 'fixed', bottom: 16, left: '50%', transform: 'translateX(-50%)',
          background: 'rgba(255,56,85,0.15)', border: '1px solid var(--status-err)',
          color: 'var(--status-err)', padding: '6px 16px', fontSize: 10,
          clipPath: 'var(--chamfer-6)', zIndex: 9500,
          fontFamily: 'JetBrains Mono, monospace', letterSpacing: '0.05em',
        }}>
          ⚠ {state.errorMsg}
        </div>
      )}
    </>
  )
}
