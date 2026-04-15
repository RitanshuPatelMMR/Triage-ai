import { HistoryEntry, Report } from '../types'

const STORAGE_KEY = 'triageai_history'
const SESSION_KEY = 'triageai_session_id'
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// ── Session ID (anonymous user identity) ─────────────────────────────────
// Generated once per browser, stored in localStorage forever
// This is Option A — session-based identity without auth
function getSessionId(): string {
  let sessionId = localStorage.getItem(SESSION_KEY)
  if (!sessionId) {
    sessionId = crypto.randomUUID()
    localStorage.setItem(SESSION_KEY, sessionId)
  }
  return sessionId
}

export const SESSION_ID = getSessionId()

// ── DynamoDB API calls ────────────────────────────────────────────────────
async function syncToCloud(entry: HistoryEntry): Promise<void> {
  try {
    // Reports are saved automatically by backend during analysis
    // This function is kept for future use
    console.log('Report saved to DynamoDB by backend:', entry.id)
  } catch (e) {
    console.warn('Cloud sync skipped:', e)
  }
}

async function fetchCloudHistory(): Promise<HistoryEntry[]> {
  try {
    const resp = await fetch(`${API_BASE}/history/${SESSION_ID}`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' }
    })
    if (!resp.ok) return []

    const data = await resp.json()
    const reports = data.reports || []

    // Convert DynamoDB format to HistoryEntry format
    return reports.map((item: any) => ({
      id: item.request_id,
      timestamp: item.created_at,
      inputType: item.input_type as 'text' | 'pdf' | 'image',
      report: item.report,
      humanVerified: item.human_verified || false,
    }))
  } catch (e) {
    console.warn('Cloud history fetch failed, using local:', e)
    return []
  }
}

async function verifyInCloud(created_at: string): Promise<void> {
  try {
    await fetch(`${API_BASE}/history/verify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: SESSION_ID,
        created_at
      })
    })
  } catch (e) {
    console.warn('Cloud verify failed:', e)
  }
}

async function deleteFromCloud(created_at: string): Promise<void> {
  try {
    await fetch(`${API_BASE}/history/delete`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: SESSION_ID,
        created_at
      })
    })
  } catch (e) {
    console.warn('Cloud delete failed:', e)
  }
}

// ── Repository pattern — only this file knows about storage ──────────────
// Switching to real auth DB later = change only this file
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

  // Load from DynamoDB and sync to localStorage
  async loadFromCloud(): Promise<HistoryEntry[]> {
    const cloudEntries = await fetchCloudHistory()
    if (cloudEntries.length > 0) {
      // Merge cloud entries with local (cloud takes priority)
      const local = this.getAll()
      const cloudIds = new Set(cloudEntries.map(e => e.id))
      const localOnly = local.filter(e => !cloudIds.has(e.id))
      const merged = [...cloudEntries, ...localOnly]
        .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
        .slice(0, 100)
      localStorage.setItem(STORAGE_KEY, JSON.stringify(merged))
      return merged
    }
    return this.getAll()
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
    const updated = [entry, ...existing].slice(0, 100)
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated))
    return entry
  },

  getById(id: string): HistoryEntry | null {
    return this.getAll().find(e => e.id === id) ?? null
  },

  async markVerified(id: string): Promise<void> {
    // Update localStorage
    const all = this.getAll()
    const entry = all.find(e => e.id === id)
    const updated = all.map(e =>
      e.id === id ? { ...e, humanVerified: true } : e
    )
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated))

    // Update DynamoDB
    if (entry) {
      await verifyInCloud(entry.timestamp)
    }
  },

  async delete(id: string): Promise<void> {
    const entry = this.getAll().find(e => e.id === id)
    const updated = this.getAll().filter(e => e.id !== id)
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated))

    // Delete from DynamoDB
    if (entry) {
      await deleteFromCloud(entry.timestamp)
    }
  },

  clearAll(): void {
    localStorage.removeItem(STORAGE_KEY)
    // Note: DynamoDB records remain — only local cleared
  },

  search(query: string): HistoryEntry[] {
    const q = query.toLowerCase()
    return this.getAll().filter(entry => {
      const r = entry.report
      const patient = r.patient_card
      const conditions = patient?.conditions_with_codes
        ?.map(c => c.condition).join(' ') ?? ''
      const summary = r.plain_english_summary ?? ''
      return conditions.toLowerCase().includes(q) ||
        summary.toLowerCase().includes(q) ||
        entry.inputType.includes(q)
    })
  },

  getSessionId(): string {
    return SESSION_ID
  }
}