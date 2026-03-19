import { AlertTriangle, AlertOctagon, Info } from 'lucide-react'

// BUG #7 FIX: Accept any shape from backend — normalize here
interface RawWarning {
  drug?: string
  name?: string
  severity?: string
  warning?: string        // backend field
  interactions?: string   // backend field
  description?: string    // alternate field
  source?: string
}

interface Props { warnings?: RawWarning[] }

export default function DrugWarnings({ warnings }: Props) {
  if (!warnings?.length) return null

  const config = {
    HIGH: {
      icon: AlertOctagon,
      border: 'border-l-red-500',
      bg: 'bg-red-50 dark:bg-red-950/20',
      badge: 'bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-400 border-red-200 dark:border-red-800'
    },
    MODERATE: {
      icon: AlertTriangle,
      border: 'border-l-amber-500',
      bg: 'bg-amber-50 dark:bg-amber-950/20',
      badge: 'bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-800'
    },
    LOW: {
      icon: Info,
      border: 'border-l-blue-400',
      bg: 'bg-blue-50 dark:bg-blue-950/20',
      badge: 'bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-400 border-blue-200 dark:border-blue-800'
    },
  }

  return (
    <div className="animate-fade-in">
      <div className="text-xs font-medium text-stone-400 dark:text-stone-500 uppercase tracking-wider mb-2">
        Drug warnings
      </div>
      <div className="space-y-2">
        {warnings.map((w, i) => {
          // BUG #7 FIX: normalize all field name variants
          const drugName = typeof w === 'string'
            ? w
            : (w.drug ?? w.name ?? 'Unknown drug').replace(/_/g, ' ')

          const rawSeverity = (w.severity ?? 'LOW').toUpperCase()
          const severity = (['HIGH', 'MODERATE', 'LOW'].includes(rawSeverity)
            ? rawSeverity
            : 'LOW') as 'HIGH' | 'MODERATE' | 'LOW'

          // BUG #7 FIX: check all text field variants
          const descriptionText = w.interactions || w.warning || w.description || ''

          const c = config[severity]
          const Icon = c.icon

          return (
            <div
              key={i}
              className={`rounded-lg border border-stone-200 dark:border-stone-700 border-l-2 ${c.border} ${c.bg} p-3`}
            >
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center gap-2">
                  <Icon size={13} className="text-stone-500 flex-shrink-0" />
                  <span className="text-sm font-medium text-stone-800 dark:text-stone-200 capitalize">
                    {drugName}
                  </span>
                </div>
                <span className={`text-xs px-2 py-0.5 rounded-full border ${c.badge}`}>
                  {severity}
                </span>
              </div>

              {descriptionText && (
                <p className="text-xs text-stone-600 dark:text-stone-400 leading-relaxed line-clamp-3 mt-1">
                  {descriptionText}
                </p>
              )}

              <div className="text-xs text-stone-400 dark:text-stone-500 mt-1.5">
                Source: {w.source ?? 'FDA Drug Label Database'}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}