import { AlertTriangle } from 'lucide-react'
import EditableField from '../common/EditableField'
import { ReportEdits } from '../../hooks/useReportEditor'

interface RawMedication {
  name?: string
  medication?: string
  drug?: string
  dose?: string
  dosage?: string
  frequency?: string
  freq?: string
  [key: string]: unknown
}

interface RawWarning {
  drug?: string
  name?: string
  severity?: string
  warning?: string
  interactions?: string
  source?: string
}

interface Props {
  medications?: (RawMedication | string)[]
  warnings?: RawWarning[]
  edits: ReportEdits
  confidenceFlags?: string[]
  onEdit: <K extends keyof ReportEdits>(field: K, value: ReportEdits[K]) => void
}

function normalizeMed(raw: RawMedication | string) {
  if (typeof raw === 'string') {
    return { name: raw, dose: '', frequency: '' }
  }
  return {
    name: raw.name ?? raw.medication ?? raw.drug ?? Object.values(raw)[0] as string ?? 'Unknown',
    dose: raw.dose ?? raw.dosage ?? '',
    frequency: raw.frequency ?? raw.freq ?? '',
  }
}

export default function MedicationTable({ medications, warnings, edits, confidenceFlags, onEdit }: Props) {
  if (!medications?.length) return null

  const flagged = (confidenceFlags ?? []).join(' ').toLowerCase()
  const isMedFlagged = flagged.includes('med') || flagged.includes('inferred')

  const warnedDrugs = new Set(
    (warnings ?? []).map(w => (w.drug ?? w.name ?? '').toLowerCase())
  )

  return (
    <div className="animate-fade-in">
      <div className="text-xs font-medium text-stone-400 dark:text-stone-500 uppercase tracking-wider mb-2">
        Medications
      </div>
      <div className="rounded-lg border border-stone-200 dark:border-stone-700 overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-stone-50 dark:bg-stone-800 border-b border-stone-200 dark:border-stone-700">
              <th className="text-left px-3 py-2 text-xs font-medium text-stone-500 dark:text-stone-400">Name</th>
              <th className="text-left px-3 py-2 text-xs font-medium text-stone-500 dark:text-stone-400">Dose</th>
              <th className="text-left px-3 py-2 text-xs font-medium text-stone-500 dark:text-stone-400">Frequency</th>
              <th className="px-3 py-2"></th>
            </tr>
          </thead>
          <tbody>
            {medications.map((raw, i) => {
              const med = normalizeMed(raw as RawMedication)
              if (!med.name || med.name === 'Unknown') return null

              const editedMed = edits.medications?.[i] ?? {}
              const displayName = editedMed.name ?? med.name
              const displayDose = editedMed.dose ?? med.dose
              const displayFreq = editedMed.frequency ?? med.frequency

              const hasWarning = warnedDrugs.has(med.name.toLowerCase())

              return (
                <tr
                  key={i}
                  className={`border-b border-stone-100 dark:border-stone-800 last:border-0 ${hasWarning ? 'bg-amber-50/50 dark:bg-amber-950/20' : ''}`}
                >
                  <td className="px-3 py-2 font-medium text-stone-800 dark:text-stone-200">
                    <EditableField
                      value={displayName}
                      onSave={val => {
                        const updated = { ...(edits.medications ?? {}) }
                        updated[i] = { ...(updated[i] ?? {}), name: val }
                        onEdit('medications', updated)
                      }}
                      isHighlighted={isMedFlagged}
                    />
                  </td>
                  <td className="px-3 py-2 text-stone-600 dark:text-stone-400">
                    <EditableField
                      value={displayDose}
                      onSave={val => {
                        const updated = { ...(edits.medications ?? {}) }
                        updated[i] = { ...(updated[i] ?? {}), dose: val }
                        onEdit('medications', updated)
                      }}
                      placeholder="—"
                      isHighlighted={isMedFlagged}
                    />
                  </td>
                  <td className="px-3 py-2 text-stone-600 dark:text-stone-400">
                    <EditableField
                      value={displayFreq}
                      onSave={val => {
                        const updated = { ...(edits.medications ?? {}) }
                        updated[i] = { ...(updated[i] ?? {}), frequency: val }
                        onEdit('medications', updated)
                      }}
                      placeholder="—"
                      isHighlighted={isMedFlagged}
                    />
                  </td>
                  <td className="px-3 py-2">
                    {hasWarning && <AlertTriangle size={13} className="text-amber-500" />}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
