// src/hooks/useConsole.ts
// Main state machine + API orchestration for the Fault Console.

import { useCallback, useEffect, useReducer, useRef } from 'react'
import { fetchIndex, fetchCase, diagnose, fetchEvidence } from '../api'
import type { IndexData, CaseData, CaseIndexEntry, DiagnoseResponse, EvidenceResponse, ModelId, FaultClass } from '../schema'

export type ActiveMask = ('D' | 'T' | 'R')[]

export interface ModelResult {
  diagnose:  DiagnoseResponse | null
  evidence:  EvidenceResponse | null
  loading:   boolean
  fromCache: boolean
}

export type InjectionPhase = 'idle' | 'injecting' | 'done' | 'error'

export interface ConsoleState {
  index:          IndexData | null
  indexLoading:   boolean
  selectedClass:  FaultClass | null
  selectedCase:   CaseIndexEntry | null
  caseData:       CaseData | null
  caseLoading:    boolean
  activeModel:    ModelId
  mask:           ActiveMask
  results:        Record<ModelId, ModelResult>
  phase:          InjectionPhase
  errorMsg:       string | null
}

type Action =
  | { type: 'INDEX_LOADED'; payload: IndexData }
  | { type: 'INDEX_ERR'; msg: string }
  | { type: 'SELECT_CLASS'; cls: FaultClass }
  | { type: 'SELECT_CASE'; entry: CaseIndexEntry }
  | { type: 'CASE_LOADED'; data: CaseData }
  | { type: 'CASE_ERR'; msg: string }
  | { type: 'SET_MODEL'; id: ModelId }
  | { type: 'TOGGLE_MASK'; mod: 'D' | 'T' | 'R' }
  | { type: 'INJECT_START' }
  | { type: 'INJECT_RESULT'; modelId: ModelId; diagnose: DiagnoseResponse; fromCache: boolean }
  | { type: 'INJECT_EVIDENCE'; modelId: ModelId; evidence: EvidenceResponse }
  | { type: 'INJECT_DONE' }
  | { type: 'INJECT_ERR'; msg: string }

const MODELS: ModelId[] = ['conflux', 'quorum', 'nexus']

function emptyResult(): ModelResult {
  return { diagnose: null, evidence: null, loading: false, fromCache: false }
}

function initResults(): Record<ModelId, ModelResult> {
  return { conflux: emptyResult(), quorum: emptyResult(), nexus: emptyResult() }
}

function reducer(state: ConsoleState, action: Action): ConsoleState {
  switch (action.type) {
    case 'INDEX_LOADED':
      return { ...state, index: action.payload, indexLoading: false }
    case 'INDEX_ERR':
      return { ...state, indexLoading: false, errorMsg: action.msg }
    case 'SELECT_CLASS':
      return { ...state, selectedClass: action.cls, selectedCase: null, caseData: null, results: initResults(), phase: 'idle' }
    case 'SELECT_CASE':
      return { ...state, selectedCase: action.entry, caseData: null, caseLoading: true, results: initResults(), phase: 'idle' }
    case 'CASE_LOADED':
      return { ...state, caseData: action.data, caseLoading: false }
    case 'CASE_ERR':
      return { ...state, caseLoading: false, errorMsg: action.msg }
    case 'SET_MODEL':
      return { ...state, activeModel: action.id }
    case 'TOGGLE_MASK': {
      const mod = action.mod
      const current = state.mask
      const next = current.includes(mod)
        ? current.filter(m => m !== mod)
        : [...current, mod] as ActiveMask
      // Keep at least one active
      if (next.length === 0) return state
      return { ...state, mask: next }
    }
    case 'INJECT_START':
      return {
        ...state,
        phase: 'injecting',
        results: {
          conflux:  { ...state.results.conflux,  loading: true },
          quorum:   { ...state.results.quorum,   loading: true },
          nexus:    { ...state.results.nexus,    loading: true },
        },
        errorMsg: null,
      }
    case 'INJECT_RESULT': {
      const mid = action.modelId
      return {
        ...state,
        results: {
          ...state.results,
          [mid]: { ...state.results[mid], diagnose: action.diagnose, fromCache: action.fromCache },
        },
      }
    }
    case 'INJECT_EVIDENCE': {
      const mid = action.modelId
      return {
        ...state,
        results: {
          ...state.results,
          [mid]: { ...state.results[mid], evidence: action.evidence, loading: false },
        },
      }
    }
    case 'INJECT_DONE':
      return { ...state, phase: 'done' }
    case 'INJECT_ERR':
      return { ...state, phase: 'error', errorMsg: action.msg,
               results: { conflux: emptyResult(), quorum: emptyResult(), nexus: emptyResult() } }
    default:
      return state
  }
}

const INITIAL: ConsoleState = {
  index: null, indexLoading: true,
  selectedClass: null, selectedCase: null, caseData: null, caseLoading: false,
  activeModel: 'conflux',
  mask: ['D', 'T', 'R'],
  results: initResults(),
  phase: 'idle',
  errorMsg: null,
}

export function useConsole() {
  const [state, dispatch] = useReducer(reducer, INITIAL)
  const abortRef = useRef<AbortController | null>(null)

  // Load index on mount
  useEffect(() => {
    fetchIndex()
      .then(idx => dispatch({ type: 'INDEX_LOADED', payload: idx }))
      .catch(e => dispatch({ type: 'INDEX_ERR', msg: String(e) }))
  }, [])

  // Load case when selected
  useEffect(() => {
    if (!state.selectedCase) return
    fetchCase(state.selectedCase.id)
      .then(data => dispatch({ type: 'CASE_LOADED', data }))
      .catch(e => dispatch({ type: 'CASE_ERR', msg: String(e) }))
  }, [state.selectedCase])

  // Derived: cases for selected class
  const casesForClass = state.index?.cases.filter(c => c.true_class === state.selectedClass) ?? []

  const selectClass = useCallback((cls: FaultClass) => {
    dispatch({ type: 'SELECT_CLASS', cls })
  }, [])

  const selectCase = useCallback((entry: CaseIndexEntry) => {
    dispatch({ type: 'SELECT_CASE', entry })
  }, [])

  const setModel = useCallback((id: ModelId) => {
    dispatch({ type: 'SET_MODEL', id })
  }, [])

  const toggleMask = useCallback((mod: 'D' | 'T' | 'R') => {
    dispatch({ type: 'TOGGLE_MASK', mod })
  }, [])

  const inject = useCallback(async () => {
    if (!state.selectedCase || !state.caseData || state.phase === 'injecting') return

    // Cancel previous
    abortRef.current?.abort()
    abortRef.current = new AbortController()

    dispatch({ type: 'INJECT_START' })

    const caseId   = state.selectedCase.id
    const caseData = state.caseData
    const mask     = state.mask

    // Fire all 3 models concurrently, update as each arrives
    try {
      await Promise.all(
        MODELS.map(async (mid) => {
          try {
            const dr = await diagnose(caseId, mid, mask, caseData)
            dispatch({ type: 'INJECT_RESULT', modelId: mid, diagnose: dr.data, fromCache: dr.fromCache })
            const er = await fetchEvidence(caseId, mid, mask, caseData)
            dispatch({ type: 'INJECT_EVIDENCE', modelId: mid, evidence: er.data })
          } catch (e) {
            // Individual model errors don't abort the whole run; use cache fallback
            const dr = await diagnose(caseId, mid, mask, caseData)
            dispatch({ type: 'INJECT_RESULT', modelId: mid, diagnose: dr.data, fromCache: true })
            const er = await fetchEvidence(caseId, mid, mask, caseData)
            dispatch({ type: 'INJECT_EVIDENCE', modelId: mid, evidence: er.data })
          }
        })
      )
      dispatch({ type: 'INJECT_DONE' })
    } catch (e) {
      dispatch({ type: 'INJECT_ERR', msg: String(e) })
    }
  }, [state.selectedCase, state.caseData, state.phase, state.mask])

  return {
    state,
    casesForClass,
    selectClass,
    selectCase,
    setModel,
    toggleMask,
    inject,
  }
}
