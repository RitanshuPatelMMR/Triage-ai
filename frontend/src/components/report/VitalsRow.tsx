import { Vitals } from '../../types'

interface Props { vitals?: Vitals }

export default function VitalsRow({ vitals }: Props) {
  if (!vitals) return null

  const items = [
    { label: 'BP', value: vitals.bp },
    { label: 'HR', value: vitals.hr },
    { label: 'RR', value: vitals.rr },
    { label: 'O₂ Sat', value: vitals.o2_sat },
    { label: 'Temp', value: vitals.temp },
  ].filter(v => v.value)

  if (!items.length) return null

  return (
    <div className="animate-fade-in">
      <div className="text-xs font-medium text-stone-400 dark:text-stone-500 uppercase tracking-wider mb-2">
        Vitals
      </div>
      <div className="grid grid-cols-5 gap-2">
        {items.map((v, i) => (
          <div key={i} className="text-center p-2 rounded-lg bg-stone-50 dark:bg-stone-800 border border-stone-200 dark:border-stone-700">
            <div className="text-xs text-stone-400 dark:text-stone-500 mb-0.5">{v.label}</div>
            <div className="text-sm font-medium text-stone-800 dark:text-stone-200">{v.value}</div>
          </div>
        ))}
      </div>
    </div>
  )
}