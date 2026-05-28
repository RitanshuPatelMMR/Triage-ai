import { useState, useCallback, useEffect } from 'react'
import { HistoryEntry, Report } from '../types'
import { historyService } from '../services/historyService'
import type { AnalysisMeta } from './useSSE'

export function useHistory() {
  const [entries, setEntries] = useState<HistoryEntry[]>(() =>
    historyService.getAll()
  )
  const [historyNotice, setHistoryNotice] = useState<string | null>(null)
  const [verifyError, setVerifyError] = useState<string | null>(null)

  useEffect(() => {
    historyService.loadFromCloud().then(({ entries: cloudEntries, offline }) => {
      if (cloudEntries.length > 0) {
        setEntries(cloudEntries)
      }
      if (offline && historyService.getAll().length > 0) {
        setHistoryNotice('History synced from this device only (cloud unavailable).')
      }
    })
  }, [])

  const save = useCallback((
    report: Report,
    inputType: 'text' | 'pdf' | 'image',
    meta?: AnalysisMeta
  ) => {
    const entry = historyService.save(report, inputType, {
      requestId: meta?.requestId,
      createdAt: meta?.createdAt,
    })
    setEntries(historyService.getAll())
    return entry
  }, [])

  const remove = useCallback(async (id: string) => {
    await historyService.delete(id)
    setEntries(historyService.getAll())
  }, [])

  const clearAll = useCallback(() => {
    historyService.clearAll()
    setEntries([])
  }, [])

  const search = useCallback((query: string) => {
    if (!query.trim()) return historyService.getAll()
    return historyService.search(query)
  }, [])

  const markVerified = useCallback(async (id: string) => {
    setVerifyError(null)
    const ok = await historyService.markVerified(id)
    setEntries(historyService.getAll())
    if (!ok) {
      setVerifyError('Saved locally, but cloud verification could not be updated. Try again.')
    }
    return ok
  }, [])

  return {
    entries,
    save,
    remove,
    clearAll,
    search,
    markVerified,
    historyNotice,
    verifyError,
    clearVerifyError: () => setVerifyError(null),
  }
}
