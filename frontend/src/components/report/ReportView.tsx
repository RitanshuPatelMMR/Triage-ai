import { Report } from '../../types'
import PatientCard from './PatientCard'
import ConditionsList from './ConditionsList'
import MedicationTable from './MedicationTable'
import DrugWarnings from './DrugWarnings'
import VitalsRow from './VitalsRow'
import ReferralText from './ReferralText'
import ConfidenceFlags from './ConfidenceFlags'
import ExportButton from './ExportButton'
import { useReportEditor } from '../../hooks/useReportEditor'
import { Pencil } from 'lucide-react'

interface Props {
  report: Report
  onMarkVerified?: () => void
  isVerified?: boolean
  // edits are managed internally — no need to pass from parent
}

export default function ReportView({ report, onMarkVerified, isVerified }: Props) {
  // hooks must be called before any early return
  const { edits, setField, hasEdits, resetEdits } = useReportEditor(report)

  if (!report || typeof report !== 'object') {
    return (
      <div className="p-4 text-sm text-stone-500 dark:text-stone-400">
        No report data available.
      </div>
    )
  }

  const { patient_card, drug_warnings, urgent_flags, plain_english_summary,
          referral_text, confidence_flags } = report

  return (
    <div className="space-y-4">

      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-medium text-stone-700 dark:text-stone-300">
            Structured report
          </h2>
          {hasEdits && (
            <span className="flex items-center gap-1 text-xs px-2 py-0.5 rounded-full bg-brand-50 dark:bg-brand-950/30 border border-brand-200 dark:border-brand-800 text-brand-600 dark:text-brand-400">
              <Pencil size={10} />
              edited
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {hasEdits && (
            <button
              onClick={resetEdits}
              className="text-xs px-2.5 py-1 rounded-lg border border-stone-200 dark:border-stone-700 text-stone-500 dark:text-stone-400 hover:bg-stone-50 dark:hover:bg-stone-800 transition-colors"
            >
              Reset edits
            </button>
          )}
          <ExportButton report={report} />
        </div>
      </div>

      {/* Patient card — editable */}
      <PatientCard
        patient={patient_card}
        urgentFlags={urgent_flags}
        edits={edits}
        confidenceFlags={confidence_flags}
        onEdit={setField}
      />

      {/* Plain English Summary */}
      {plain_english_summary && (
        <div className="animate-fade-in">
          <div className="text-xs font-medium text-stone-400 dark:text-stone-500 uppercase tracking-wider mb-2">
            Summary
          </div>
          <p className="text-sm text-stone-700 dark:text-stone-300 leading-relaxed bg-stone-50 dark:bg-stone-800 rounded-lg p-3 border border-stone-200 dark:border-stone-700">
            {plain_english_summary}
          </p>
        </div>
      )}

      {/* Conditions — editable */}
      <ConditionsList
        conditions={patient_card?.conditions_with_codes}
        edits={edits}
        confidenceFlags={confidence_flags}
        onEdit={setField}
      />

      {/* Vitals */}
      <VitalsRow vitals={patient_card?.vitals} />

      {/* Medications — editable */}
      <MedicationTable
        medications={patient_card?.medications as any}
        warnings={drug_warnings}
        edits={edits}
        confidenceFlags={confidence_flags}
        onEdit={setField}
      />

      <DrugWarnings warnings={drug_warnings} />

      <ConfidenceFlags
        flags={confidence_flags}
        onMarkVerified={onMarkVerified}
        isVerified={isVerified}
      />

      <ReferralText text={referral_text} />
    </div>
  )
}