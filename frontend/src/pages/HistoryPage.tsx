import { useState, useCallback } from 'react'
import { Search, Trash2, Clock } from 'lucide-react'
import { useHistory } from '../hooks/useHistory'
import { HistoryEntry } from '../types'
import HistoryCard from '../components/history/HistoryCard'
import HistoryModal from '../components/history/HistoryModal'
import toast from 'react-hot-toast'

export default function HistoryPage() {
  const { entries, remove, clearAll, search, markVerified } = useHistory()
  const [query, setQuery] = useState('')
  const [selected, setSelected] = useState<HistoryEntry | null>(null)

  const displayed = query.trim() ? search(query) : entries

  const handleDelete = useCallback((id: string) => {
    remove(id)
    toast.success('Entry deleted')
  }, [remove])

  const handleClearAll = () => {
    if (window.confirm('Clear all history? This cannot be undone.')) {
      clearAll()
      toast.success('History cleared')
    }
  }

  const handleVerified = () => {
    if (selected) {
      markVerified(selected.id)
      setSelected(prev => prev ? { ...prev, humanVerified: true } : null)
    }
  }

  return (
    <div className="max-w-2xl mx-auto px-4 py-8">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-lg font-semibold text-stone-900 dark:text-stone-100">
            Analysis history
          </h1>
          <p className="text-xs text-stone-400 dark:text-stone-500 mt-0.5">
            {entries.length} {entries.length === 1 ? 'analysis' : 'analyses'} saved
          </p>
        </div>
        {entries.length > 0 && (
          <button
            onClick={handleClearAll}
            className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border border-red-200 dark:border-red-800 text-red-500 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-950/20 transition-colors"
          >
            <Trash2 size={12} /> Clear all
          </button>
        )}
      </div>

      {/* Search */}
      {entries.length > 0 && (
        <div className="flex items-center gap-2 px-3 py-2 rounded-xl border border-stone-200 dark:border-stone-700 bg-white dark:bg-stone-900 mb-4">
          <Search size={14} className="text-stone-400 flex-shrink-0" />
          <input
            type="text"
            placeholder="Search by condition, date, or input type..."
            value={query}
            onChange={e => setQuery(e.target.value)}
            className="flex-1 text-sm bg-transparent text-stone-700 dark:text-stone-300 placeholder-stone-400 dark:placeholder-stone-600 focus:outline-none"
          />
        </div>
      )}

      {/* List */}
      {displayed.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <div className="w-10 h-10 rounded-xl bg-stone-100 dark:bg-stone-800 flex items-center justify-center mb-3">
            <Clock size={18} className="text-stone-400" />
          </div>
          <div className="text-sm text-stone-500 dark:text-stone-400">
            {query ? 'No results found' : 'No analyses yet'}
          </div>
          <div className="text-xs text-stone-400 dark:text-stone-500 mt-1">
            {!query && 'Your analyses will appear here after you use the analyzer'}
          </div>
        </div>
      ) : (
        <div className="space-y-2">
          {displayed.map(entry => (
            <HistoryCard
              key={entry.id}
              entry={entry}
              onClick={() => setSelected(entry)}
              onDelete={() => handleDelete(entry.id)}
            />
          ))}
        </div>
      )}

      {selected && (
        <HistoryModal
          entry={selected}
          onClose={() => setSelected(null)}
          onVerified={handleVerified}
        />
      )}
    </div>
  )
}