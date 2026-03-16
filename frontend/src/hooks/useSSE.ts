import { useState, useCallback, useRef } from 'react'
import { Report, AgentStep, AnalysisStatus } from '../types'

const API_BASE = 'http://localhost:8000'

export function useSSE() {
  const [steps, setSteps] = useState<AgentStep[]>([])
  const [report, setReport] = useState<Report | null>(null)
  const [status, setStatus] = useState<AnalysisStatus>('idle')
  const [error, setError] = useState<string | null>(null)
  const [inputType, setInputType] = useState<'text' | 'pdf' | 'image'>('text')
  const readerRef = useRef<ReadableStreamDefaultReader | null>(null)

  const reset = useCallback(() => {
    setSteps([])
    setReport(null)
    setStatus('idle')
    setError(null)
  }, [])

  const parseSSEChunk = useCallback((chunk: string, onComplete: (r: Report) => void) => {
    const lines = chunk.split('\n')
    for (const line of lines) {
      if (!line.startsWith('data:')) continue
      try {
        const data = JSON.parse(line.slice(5))

        if (data.step) {
          setSteps(prev => {
            const exists = prev.find(s => s.node === data.node)
            if (exists) return prev
            return [...prev, { node: data.node, step: data.step, errors: data.errors ?? [] }]
          })
        }

        if (data.input_type) {
          setInputType(data.input_type as 'text' | 'pdf' | 'image')
        }

        if (data.report) {
          setReport(data.report)
          setStatus('complete')
          onComplete(data.report)
        }

        if (data.error) {
          setError(data.error)
          setStatus('error')
        }
      } catch {
        // skip malformed lines
      }
    }
  }, [])

  const analyzeText = useCallback(async (text: string, onComplete: (r: Report) => void) => {
    reset()
    setStatus('running')
    setInputType('text')

    try {
      const response = await fetch(`${API_BASE}/analyze/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })

      const reader = response.body!.getReader()
      readerRef.current = reader
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        parseSSEChunk(decoder.decode(value), onComplete)
      }
    } catch (e) {
      setError(String(e))
      setStatus('error')
    }
  }, [reset, parseSSEChunk])

  const analyzeFile = useCallback(async (file: File, onComplete: (r: Report) => void) => {
    reset()
    setStatus('running')

    const ext = file.name.split('.').pop()?.toLowerCase()
    if (ext === 'pdf') setInputType('pdf')
    else if (['jpg', 'jpeg', 'png'].includes(ext ?? '')) setInputType('image')

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(`${API_BASE}/analyze/upload/stream`, {
        method: 'POST',
        body: formData,
      })

      const reader = response.body!.getReader()
      readerRef.current = reader
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        parseSSEChunk(decoder.decode(value), onComplete)
      }
    } catch (e) {
      setError(String(e))
      setStatus('error')
    }
  }, [reset, parseSSEChunk])

  return {
    steps, report, status, error, inputType,
    analyzeText, analyzeFile, reset
  }
}