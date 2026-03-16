import { AlertTriangle, Edit3, CheckCircle } from 'lucide-react'

interface Props {
  flags?: string[]
  onMarkVerified?: () => void
  isVerified?: boolean
}

export default function ConfidenceFlags({ flags, onMarkVerified, isVerified }: Props) {
  if (!flags?.length) return null

  return (
    <div className="animate-fade-in">
      {isVerified ? (
        <div className="flex items-center gap-2 p-3 rounded-lg bg-green-50 dark:bg-green-950/20 border border-green-200 dark:border-green-800">
          <CheckCircle size={14} className="text-green-500 flex-shrink-0" />
          <span className="text-xs text-green-700 dark:text-green-400 font-medium">
            Human verified — fields confirmed
          </span>
        </div>
      ) : (
        <div className="rounded-lg border-l-2 border-l-amber-400 border border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-950/20 p-3">
          <div className="flex items-start justify-between gap-3">
            <div className="flex items-start gap-2">
              <AlertTriangle size={14} className="text-amber-500 flex-shrink-0 mt-0.5" />
              <div>
                <div className="text-xs font-medium text-amber-700 dark:text-amber-400 mb-1">
                  Human verification needed
                </div>
                {flags.map((f, i) => (
                  <div key={i} className="text-xs text-amber-600 dark:text-amber-500">{f}</div>
                ))}
              </div>
            </div>
            {onMarkVerified && (
              <button
                onClick={onMarkVerified}
                className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-lg bg-white dark:bg-stone-800 border border-amber-200 dark:border-amber-700 text-amber-700 dark:text-amber-400 hover:bg-amber-50 dark:hover:bg-amber-900/30 transition-colors flex-shrink-0"
              >
                <Edit3 size={11} /> Mark verified
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  )
}