import { HistoryEntry } from '../../types'
import { FileText, Image, AlignLeft, CheckCircle, AlertTriangle, ChevronRight } from 'lucide-react'

interface Props {
  entry: HistoryEntry
  onClick: () => void
  onDelete: () => void
}

export default function HistoryCard({ entry, onClick, onDelete }: Props) {
  const { report, inputType, timestamp, humanVerified } = entry
  const patient = report.patient_card
  const conditions = patient?.conditions_with_codes?.map(c => c.condition) ?? []
  const warningCount = report.drug_warnings?.length ?? 0
  const flagCount = report.confidence_flags?.length ?? 0

  const date = new Date(timestamp)
  const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  const dateStr = date.toLocaleDateString([], { month: 'short', day: 'numeric' })

  const InputIcon = inputType === 'pdf' ? FileText : inputType === 'image' ? Image : AlignLeft

  return (
    <div
      onClick={onClick}
      className="group flex items-center justify-between p-4 rounded-xl border border-stone-200 dark:border-stone-700 bg-white dark:bg-stone-900 hover:border-stone-300 dark:hover:border-stone-600 hover:shadow-sm transition-all cursor-pointer"
    >
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <InputIcon size={12} className="text-stone-400 flex-shrink-0" />
          <span className="text-xs text-stone-400 dark:text-stone-500">
            {dateStr} · {timeStr}
          </span>
          {humanVerified && (
            <span className="flex items-center gap-1 text-xs text-green-600 dark:text-green-400">
              <CheckCircle size={11} /> Verified
            </span>
          )}
        </div>

        <div className="text-sm font-medium text-stone-800 dark:text-stone-200 mb-2 truncate">
          {patient?.age}
          {patient?.gender ? ` ${patient.gender}` : ''}
          {patient?.chief_complaint ? ` — ${patient.chief_complaint}` : ''}
        </div>

        <div className="flex flex-wrap gap-1.5">
          {conditions.slice(0, 3).map(c => (
            <span key={c} className="text-xs px-2 py-0.5 rounded-full bg-stone-100 dark:bg-stone-800 text-stone-600 dark:text-stone-400">
              {c}
            </span>
          ))}
          {warningCount > 0 && (
            <span className="flex items-center gap-1 text-xs px-2 py-0.5 rounded-full bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800 text-amber-700 dark:text-amber-400">
              <AlertTriangle size={10} /> {warningCount} warning{warningCount > 1 ? 's' : ''}
            </span>
          )}
          {flagCount > 0 && !humanVerified && (
            <span className="text-xs px-2 py-0.5 rounded-full bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800 text-amber-600 dark:text-amber-400">
              {flagCount} field{flagCount > 1 ? 's' : ''} need review
            </span>
          )}
        </div>
      </div>

      <div className="flex items-center gap-2 ml-3">
        <button
          onClick={e => { e.stopPropagation(); onDelete() }}
          className="opacity-0 group-hover:opacity-100 text-xs text-stone-400 hover:text-red-500 transition-all px-2 py-1 rounded-lg hover:bg-red-50 dark:hover:bg-red-950/20"
        >
          Delete
        </button>
        <ChevronRight size={16} className="text-stone-400" />
      </div>
    </div>
  )
}