import { PatientCard as PatientCardType } from '../../types'
import { User, AlertCircle } from 'lucide-react'
import EditableField from '../common/EditableField'
import { ReportEdits } from '../../hooks/useReportEditor'

interface Props {
  patient?: PatientCardType
  urgentFlags?: string[]
  edits: ReportEdits
  confidenceFlags?: string[]
  onEdit: <K extends keyof ReportEdits>(field: K, value: ReportEdits[K]) => void
}

export default function PatientCard({ patient, urgentFlags, edits, confidenceFlags, onEdit }: Props) {
  if (!patient) return null

  // Fields flagged as inferred from OCR
  const flagged = (confidenceFlags ?? []).join(' ').toLowerCase()
  const isAgeFlagged = flagged.includes('age') || flagged.includes('inferred')
  const isGenderFlagged = flagged.includes('gender') || flagged.includes('inferred')

  return (
    <div className="p-4 rounded-xl bg-stone-50 dark:bg-stone-800 border border-stone-200 dark:border-stone-700 animate-fade-in">

      {/* Urgent flags banner */}
      {urgentFlags && urgentFlags.length > 0 && (
        <div className="flex items-center gap-2 mb-3 p-2 rounded-lg bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800">
          <AlertCircle size={13} className="text-red-500 flex-shrink-0" />
          <span className="text-xs text-red-600 dark:text-red-400">{urgentFlags.join(' · ')}</span>
        </div>
      )}

      <div className="flex items-center gap-3 mb-3">
        <div className="w-9 h-9 rounded-full bg-brand-100 dark:bg-brand-900/40 flex items-center justify-center flex-shrink-0">
          <User size={16} className="text-brand-600 dark:text-brand-400" />
        </div>
        <div>
          <div className="text-sm font-medium text-stone-900 dark:text-stone-100 flex items-center gap-1 flex-wrap">
            {/* Editable age */}
            <EditableField
              value={edits.age !== undefined ? edits.age : patient.age}
              onSave={val => onEdit('age', val)}
              placeholder="age"
              isHighlighted={isAgeFlagged}
              inputClassName="w-14"
            />
            <span className="text-stone-400 text-xs">yr</span>

            {/* Editable gender */}
            <EditableField
              value={edits.gender !== undefined ? edits.gender : patient.gender}
              onSave={val => onEdit('gender', val)}
              placeholder="gender"
              isHighlighted={isGenderFlagged}
            />
          </div>

          {/* Editable chief complaint */}
          {(patient.chief_complaint || edits.chief_complaint !== undefined) && (
            <div className="text-xs text-stone-500 dark:text-stone-400 mt-0.5">
              <EditableField
                value={edits.chief_complaint !== undefined ? edits.chief_complaint : patient.chief_complaint}
                onSave={val => onEdit('chief_complaint', val)}
                placeholder="chief complaint"
                isHighlighted={flagged.includes('complaint')}
              />
            </div>
          )}
        </div>
      </div>

      {/* Allergies */}
      {patient.allergies && patient.allergies.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {patient.allergies.map((a, i) => {
            const allergyText = typeof a === 'string'
              ? a
              : typeof a === 'object' && a !== null
              ? Object.values(a)[0] as string
              : String(a)

            const displayVal = edits.allergies?.[i] ?? allergyText

            return (
              <EditableField
                key={i}
                value={displayVal}
                onSave={val => {
                  const updated = { ...(edits.allergies ?? {}) }
                  updated[i] = val
                  onEdit('allergies', updated)
                }}
                className="text-xs px-2 py-0.5 rounded-full bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-800 text-red-600 dark:text-red-400"
                isHighlighted={flagged.includes('allerg')}
              />
            )
          })}
        </div>
      )}
    </div>
  )
}
