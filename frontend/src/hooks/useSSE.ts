import { useState, useCallback, useRef } from 'react'
import { Report, AgentStep, AnalysisStatus } from '../types'
import { SESSION_ID } from '../services/historyService'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const COLD_START_MS = 8000

export interface AnalysisMeta {
  requestId?: string
  createdAt?: string
}

export type LastAnalysisPayload =
  | { kind: 'text'; text: string }
  | { kind: 'file'; file: File }

function friendlyFetchError(e: unknown): string {
  const msg = String(e)
  if (msg.includes('Failed to fetch') || msg.includes('NetworkError')) {
    return `Can't reach the API at ${API_BASE}. The server may be waking up — wait up to 60 seconds, then try again.`
  }
  return msg
}

export function useSSE() {
  const [steps, setSteps] = useState<AgentStep[]>([])
  const [report, setReport] = useState<Report | null>(null)
  const [status, setStatus] = useState<AnalysisStatus>('idle')
  const [error, setError] = useState<string | null>(null)
  const [coldStartHint, setColdStartHint] = useState<string | null>(null)
  const [inputType, setInputType] = useState<'text' | 'pdf' | 'image'>('text')
  const readerRef = useRef<ReadableStreamDefaultReader<Uint8Array> | null>(null)
  const runningRef = useRef(false)
  const lastPayloadRef = useRef<LastAnalysisPayload | null>(null)

  const reset = useCallback(() => {
    readerRef.current?.cancel().catch(() => {})
    readerRef.current = null
    setSteps([])
    setReport(null)
    setStatus('idle')
    setError(null)
    setColdStartHint(null)
    runningRef.current = false
  }, [])

  const parseSSEChunk = useCallback((
    chunk: string,
    onComplete: (r: Report, meta?: AnalysisMeta) => void
  ) => {
    const lines = chunk.split('\n')
    for (const line of lines) {
      if (!line.startsWith('data:')) continue
      try {
        const data = JSON.parse(line.slice(5).trim())

        if (data.step || data.message) {
          setColdStartHint(null)
        }

        if (data.step) {
          setSteps(prev => {
            const exists = prev.find(s => s.node === data.node)
            if (exists) return prev
            return [...prev, {
              node: data.node,
              step: data.step,
              errors: data.errors ?? []
            }]
          })
        }

        if (data.input_type) {
          setInputType(data.input_type as 'text' | 'pdf' | 'image')
        }

        if (data.report) {
          setReport(data.report)
          setStatus('complete')
          setColdStartHint(null)
          runningRef.current = false
          onComplete(data.report, {
            requestId: data.request_id,
            createdAt: data.created_at,
          })
        }

        if (data.error) {
          setError(data.error)
          setStatus('error')
          setColdStartHint(null)
          runningRef.current = false
        }
      } catch {
        // skip malformed lines
      }
    }
  }, [])

  const consumeStream = useCallback(async (
    response: Response,
    onComplete: (r: Report, meta?: AnalysisMeta) => void
  ) => {
    if (!response.ok) {
      let message = `Server error (${response.status}). Please try again.`
      try {
        const body = await response.text()
        if (body) {
          try {
            const parsed = JSON.parse(body)
            message = parsed.detail ?? parsed.error ?? parsed.message ?? message
          } catch {
            if (body.length < 200) message = body
          }
        }
      } catch {
        // keep default message
      }
      setError(message)
      setStatus('error')
      setColdStartHint(null)
      runningRef.current = false
      return
    }

    if (!response.body) {
      setError('No response from server. Please try again.')
      setStatus('error')
      setColdStartHint(null)
      runningRef.current = false
      return
    }

    const reader = response.body.getReader()
    readerRef.current = reader
    const decoder = new TextDecoder()
    let buffer = ''
    let finished = false

    const onDone = (r: Report, meta?: AnalysisMeta) => {
      finished = true
      onComplete(r, meta)
    }

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''
        for (const line of lines) {
          if (line.startsWith('data:')) {
            parseSSEChunk(line + '\n', onDone)
          }
        }
      }
      if (buffer.trim()) {
        parseSSEChunk(buffer, onDone)
      }
    } finally {
      readerRef.current = null
      runningRef.current = false
      setColdStartHint(null)
      if (!finished) {
        setStatus(prev => {
          if (prev === 'running') {
            setError('Analysis ended unexpectedly. Please try again.')
            return 'error'
          }
          return prev
        })
      }
    }
  }, [parseSSEChunk])

  const startColdStartTimer = useCallback(() => {
    const timer = setTimeout(() => {
      setColdStartHint(
        'Server is starting — first request may take up to 60 seconds. Please wait...'
      )
    }, COLD_START_MS)
    return timer
  }, [])

  const prepareNewRun = useCallback(() => {
    readerRef.current?.cancel().catch(() => {})
    readerRef.current = null
    setSteps([])
    setReport(null)
    setError(null)
    setColdStartHint(null)
  }, [])

  const analyzeText = useCallback(async (
    text: string,
    onComplete: (r: Report, meta?: AnalysisMeta) => void
  ) => {
    if (runningRef.current) return
    lastPayloadRef.current = { kind: 'text', text }
    prepareNewRun()
    runningRef.current = true
    setStatus('running')
    setInputType('text')

    const coldStartTimer = startColdStartTimer()

    try {
      const response = await fetch(`${API_BASE}/analyze/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-session-id': SESSION_ID,
        },
        body: JSON.stringify({ text }),
      })

      clearTimeout(coldStartTimer)
      await consumeStream(response, onComplete)
    } catch (e) {
      clearTimeout(coldStartTimer)
      setError(friendlyFetchError(e))
      setStatus('error')
      setColdStartHint(null)
      runningRef.current = false
    }
  }, [prepareNewRun, consumeStream, startColdStartTimer])

  const analyzeFile = useCallback(async (
    file: File,
    onComplete: (r: Report, meta?: AnalysisMeta) => void
  ) => {
    if (runningRef.current) return
    lastPayloadRef.current = { kind: 'file', file }
    prepareNewRun()
    runningRef.current = true
    setStatus('running')

    const ext = file.name.split('.').pop()?.toLowerCase()
    if (ext === 'pdf') setInputType('pdf')
    else if (['jpg', 'jpeg', 'png'].includes(ext ?? '')) setInputType('image')

    const coldStartTimer = startColdStartTimer()

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(`${API_BASE}/analyze/upload/stream`, {
        method: 'POST',
        headers: { 'x-session-id': SESSION_ID },
        body: formData,
      })

      clearTimeout(coldStartTimer)
      await consumeStream(response, onComplete)
    } catch (e) {
      clearTimeout(coldStartTimer)
      setError(friendlyFetchError(e))
      setStatus('error')
      setColdStartHint(null)
      runningRef.current = false
    }
  }, [prepareNewRun, consumeStream, startColdStartTimer])

  const retryLast = useCallback((
    onComplete: (r: Report, meta?: AnalysisMeta) => void
  ) => {
    const last = lastPayloadRef.current
    if (!last) return
    if (last.kind === 'text') {
      analyzeText(last.text, onComplete)
    } else {
      analyzeFile(last.file, onComplete)
    }
  }, [analyzeText, analyzeFile])

  return {
    steps,
    report,
    status,
    error,
    coldStartHint,
    inputType,
    analyzeText,
    analyzeFile,
    retryLast,
    reset,
  }
}
