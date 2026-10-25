// src/App.tsx
// Main layout: Header | DiagramStage | RightColumn | BottomDock + BootScene + AboutModal.
// Supports global keybindings (1-4, Space, Q/W/E, R, H) and ?fast=1/?api= query parameters.

import React, { useState, useEffect, useCallback } from 'react'
import { useConsole } from './hooks/useConsole'
import Header from './components/Header'
import DiagramStage from './components/DiagramStage'
import RightColumn from './components/RightColumn'
import BottomDock from './components/BottomDock'
import AboutModal from './components/AboutModal'
import BootScene from './fx/BootScene'
import { checkHealth } from './api'
import type { FaultClass, ModelId } from './schema'

export default function App() {
  const urlParams = new URLSearchParams(window.location.search)
  const isFast = urlParams.get('fast') === '1'
  const apiOverride = urlParams.get('api')

  const [booted, setBooted] = useState(isFast)
  const [apiLive, setApiLive] = useState(false)
  const [hudMode, setHudMode] = useState(true)
  const [aboutOpen, setAboutOpen] = useState(false)

  const {
    state,
    casesForClass,
    selectClass,
    selectCase,
    setModel,
    toggleMask,
    resetToNominal,
    shuffleCase,
    inject,
  } = useConsole()

  const handleBoot = useCallback(() => setBooted(true), [])
  const toggleHud = useCallback(() => setHudMode(h => !h), [])

  // Check API health periodically
  useEffect(() => {
    if (!booted) return
    const check = () => {
      const apiBase = apiOverride || (import.meta.env.VITE_API_URL as string | undefined) || ''
      if (!apiBase) {
        setApiLive(false)
        return
      }
      checkHealth(apiBase).then(ok => setApiLive(ok))
    }
    check()
    const timer = setInterval(check, 30000)
    return () => clearInterval(timer)
  }, [booted, apiOverride])

  const canInject = !!state.selectedCase && !!state.caseData && state.phase !== 'injecting'

  // Global Keyboard Shortcuts (1-4, Space, Q/W/E, R, H)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if user is inside an input or modal is open
      if (aboutOpen && e.key !== 'Escape') return
      if (['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement)?.tagName)) return

      if (e.key === 'Escape') {
        setAboutOpen(false)
        return
      }

      switch (e.key) {
        case '1':
          e.preventDefault()
          selectClass('line_fault')
          break
        case '2':
          e.preventDefault()
          selectClass('breaker_fault')
          break
        case '3':
          e.preventDefault()
          selectClass('transformer_fault')
          break
        case '4':
          e.preventDefault()
          selectClass('busbar_fault')
          break
        case ' ':
        case 'Enter':
          e.preventDefault()
          if (canInject) inject()
          break
        case 'q':
        case 'Q':
          e.preventDefault()
          setModel('conflux')
          break
        case 'w':
        case 'W':
          e.preventDefault()
          setModel('quorum')
          break
        case 'e':
        case 'E':
          e.preventDefault()
          setModel('nexus')
          break
        case 'r':
        case 'R':
          e.preventDefault()
          resetToNominal()
          break
        case 'h':
        case 'H':
          e.preventDefault()
          toggleHud()
          break
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [canInject, inject, selectClass, setModel, resetToNominal, toggleHud, aboutOpen])

  return (
    <>
      {/* Boot Scene (skipped if ?fast=1) */}
      {!booted && <BootScene onDone={handleBoot} />}

      {/* Main Analogpunk Grid Layout */}
      <div className="app-layout" style={{ height: 'calc(100vh - 20px)' }}>
        <Header
          selectedClass={state.selectedClass}
          onSelectClass={selectClass}
          mask={state.mask}
          onToggleMask={toggleMask}
          onInject={inject}
          canInject={canInject}
          injecting={state.phase === 'injecting'}
          apiLive={apiLive}
          onOpenAbout={() => setAboutOpen(true)}
        />

        <DiagramStage
          caseData={state.caseData}
          caseLoading={state.caseLoading}
          phase={state.phase}
          selectedCase={state.selectedCase}
          results={state.results}
          activeModel={state.activeModel}
          onSelectModel={setModel}
          hudMode={hudMode}
          onToggleHud={toggleHud}
          onShuffle={shuffleCase}
          mask={state.mask}
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
          mask={state.mask}
        />

        <BottomDock
          caseData={state.caseData}
          activeModel={state.activeModel}
          results={state.results}
          mask={state.mask}
        />
      </div>

      {/* Footer Status Bar (Section 3.7 Honesty Line) */}
      <footer
        style={{
          height: 20,
          background: 'var(--bg-0)',
          borderTop: '1px solid var(--line-dim)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 12px',
          fontSize: 9,
          color: 'var(--text-dim)',
          letterSpacing: '0.08em',
        }}
      >
        <div>
          <span>MODE: HELD-OUT TEST EVENT REPLAY · </span>
          <span style={{ color: apiLive ? 'var(--status-live)' : 'var(--status-cached)', fontWeight: 700 }}>
            {apiLive ? 'LIVE INFERENCE' : 'CACHED INFERENCE'}
          </span>
          {apiOverride && (
            <span style={{ marginLeft: 8, color: 'var(--conflux)', border: '1px solid var(--conflux)', padding: '0 4px' }}>
              API OVERRIDE: {apiOverride}
            </span>
          )}
        </div>
        <div>
          <span>KEYS: [1-4] FAULT · [SPACE] INJECT · [Q/W/E] MODEL · [R] RESET · [H] HUD</span>
        </div>
      </footer>

      {/* About / Honesty Modal */}
      <AboutModal
        isOpen={aboutOpen}
        onClose={() => setAboutOpen(false)}
        apiLive={apiLive}
      />

      {/* Error Toast */}
      {state.errorMsg && (
        <div style={{
          position: 'fixed', bottom: 32, left: '50%', transform: 'translateX(-50%)',
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
