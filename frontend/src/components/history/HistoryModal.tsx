import { HistoryEntry } from '../../types'
import ReportView from '../report/ReportView'
import { X } from 'lucide-react'
import { historyService } from '../../services/historyService'

interface Props {
  entry: HistoryEntry
  onClose: () => void
  onVerified: () => void
}

export default function HistoryModal({ entry, onClose, onVerified }: Props) {
  const handleVerify = () => {
    historyService.markVerified(entry.id)
    onVerified()
  }

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center bg-black/40 backdrop-blur-sm pt-10 px-4 pb-4 overflow-y-auto">
      <div className="w-full max-w-2xl bg-white dark:bg-stone-900 rounded-2xl border border-stone-200 dark:border-stone-700 shadow-xl">
        <div className="flex items-center justify-between p-4 border-b border-stone-200 dark:border-stone-700">
          <div className="text-sm font-medium text-stone-700 dark:text-stone-300">
            Analysis from {new Date(entry.timestamp).toLocaleString()}
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-stone-100 dark:hover:bg-stone-800 transition-colors"
          >
            <X size={16} className="text-stone-500" />
          </button>
        </div>
        <div className="p-5">
          <ReportView
            report={entry.report}
            onMarkVerified={handleVerify}
            isVerified={entry.humanVerified}
          />
        </div>
      </div>
    </div>
  )
}