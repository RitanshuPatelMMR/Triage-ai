import { useState, useCallback } from 'react'
import { HistoryEntry, Report } from '../types'
import { historyService } from '../services/historyService'

export function useHistory() {
  const [entries, setEntries] = useState<HistoryEntry[]>(() =>
    historyService.getAll()
  )

  const save = useCallback((report: Report, inputType: 'text' | 'pdf' | 'image') => {
    const entry = historyService.save(report, inputType)
    setEntries(historyService.getAll())
    return entry
  }, [])

  const remove = useCallback((id: string) => {
    historyService.delete(id)
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

  const markVerified = useCallback((id: string) => {
    historyService.markVerified(id)
    setEntries(historyService.getAll())
  }, [])

  return { entries, save, remove, clearAll, search, markVerified }
}