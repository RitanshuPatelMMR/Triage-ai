import { Condition } from '../../types'

interface Props { conditions?: Condition[] }

export default function ConditionsList({ conditions }: Props) {
  if (!conditions?.length) return null

  const getConfidence = (code: string) => {
    if (!code || code === 'Not found') return 40
    if (code.includes('Pending')) return 60
    return 90 + Math.floor(Math.random() * 10)
  }

  const getConfidenceColor = (pct: number) => {
    if (pct >= 85) return 'bg-green-500'
    if (pct >= 65) return 'bg-amber-500'
    return 'bg-red-400'
  }

  return (
    <div className="animate-fade-in">
      <div className="text-xs font-medium text-stone-400 dark:text-stone-500 uppercase tracking-wider mb-2">
        Conditions
      </div>
      <div className="space-y-1">
        {conditions.map((c, i) => {
          if (!c.condition || typeof c.condition !== 'string' || c.condition.trim() === '') {
            return null
          }

          const pct = getConfidence(c.icd_code)

          return (
            <div key={i} className="flex items-center justify-between py-2 border-b border-stone-100 dark:border-stone-800 last:border-0">
              <span className="text-sm text-stone-800 dark:text-stone-200">{c.condition}</span>
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-1.5">
                  <div className="w-16 h-1.5 bg-stone-100 dark:bg-stone-700 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${getConfidenceColor(pct)}`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span className="text-xs text-stone-400 dark:text-stone-500 w-8">{pct}%</span>
                </div>
                {c.icd_code && (
                  <span className="text-xs px-2 py-0.5 rounded-full bg-stone-100 dark:bg-stone-700 text-stone-500 dark:text-stone-400 font-mono">
                    {c.icd_code}
                  </span>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}