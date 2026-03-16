import { PatientCard as PatientCardType } from '../../types'
import { User, AlertCircle } from 'lucide-react'

interface Props { patient?: PatientCardType; urgentFlags?: string[] }

export default function PatientCard({ patient, urgentFlags }: Props) {
  if (!patient) return null

  return (
    <div className="p-4 rounded-xl bg-stone-50 dark:bg-stone-800 border border-stone-200 dark:border-stone-700 animate-fade-in">
      <div className="flex items-center gap-3 mb-3">
        <div className="w-9 h-9 rounded-full bg-brand-100 dark:bg-brand-900/40 flex items-center justify-center">
          <User size={16} className="text-brand-600 dark:text-brand-400" />
        </div>
        <div>
          <div className="text-sm font-medium text-stone-900 dark:text-stone-100">
            {patient.age ? `${patient.age}-year-old` : ''} {patient.gender ?? ''}
          </div>
          {patient.chief_complaint && (
            <div className="text-xs text-stone-500 dark:text-stone-400">
              {patient.chief_complaint}
            </div>
          )}
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        {patient.allergies?.map((a, i) => {
  // Normalize — LLM sometimes returns {allergy: "PCN"} instead of "PCN"
  const allergyText = typeof a === 'string'
    ? a
    : typeof a === 'object' && a !== null
    ? Object.values(a)[0] as string
    : String(a)

  return (
    <span key={i} className="text-xs px-2 py-0.5 rounded-full bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-800 text-red-600 dark:text-red-400">
      Allergy: {allergyText}
    </span>
  )
})}
      </div>
    </div>
  )
}