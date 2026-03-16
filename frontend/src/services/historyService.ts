import { HistoryEntry, Report } from '../types'

const STORAGE_KEY = 'triageai_history'

// ── This is the ONLY file that knows about localStorage ───────────────────
// To switch to a real API later, replace the internals of these functions.
// Nothing else in the app needs to change.

export const historyService = {

  getAll(): HistoryEntry[] {
    try {
      const raw = localStorage.getItem(STORAGE_KEY)
      if (!raw) return []
      return JSON.parse(raw) as HistoryEntry[]
    } catch {
      return []
    }
  },

  save(report: Report, inputType: 'text' | 'pdf' | 'image'): HistoryEntry {
    const entry: HistoryEntry = {
      id: crypto.randomUUID(),
      timestamp: new Date().toISOString(),
      inputType,
      report,
      humanVerified: false,
    }
    const existing = this.getAll()
    const updated = [entry, ...existing].slice(0, 100) // keep last 100
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated))
    return entry
  },

  getById(id: string): HistoryEntry | null {
    return this.getAll().find(e => e.id === id) ?? null
  },

  markVerified(id: string): void {
    const all = this.getAll()
    const updated = all.map(e =>
      e.id === id ? { ...e, humanVerified: true } : e
    )
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated))
  },

  delete(id: string): void {
    const updated = this.getAll().filter(e => e.id !== id)
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated))
  },

  clearAll(): void {
    localStorage.removeItem(STORAGE_KEY)
  },

  search(query: string): HistoryEntry[] {
    const q = query.toLowerCase()
    return this.getAll().filter(entry => {
      const r = entry.report
      const patient = r.patient_card
      const conditions = patient?.conditions_with_codes?.map(c => c.condition).join(' ') ?? ''
      const summary = r.plain_english_summary ?? ''
      return conditions.toLowerCase().includes(q) ||
             summary.toLowerCase().includes(q) ||
             entry.inputType.includes(q)
    })
  }
}