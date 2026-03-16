import { Report } from '../../types'
import PatientCard from './PatientCard'
import ConditionsList from './ConditionsList'
import MedicationTable from './MedicationTable'
import DrugWarnings from './DrugWarnings'
import VitalsRow from './VitalsRow'
import ReferralText from './ReferralText'
import ConfidenceFlags from './ConfidenceFlags'
import ExportButton from './ExportButton'

interface Props {
  report: Report
  onMarkVerified?: () => void
  isVerified?: boolean
}

export default function ReportView({ report, onMarkVerified, isVerified }: Props) {
  // Safety check — never crash on malformed data
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
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-medium text-stone-700 dark:text-stone-300">
          Structured report
        </h2>
        <ExportButton report={report} />
      </div>

      <PatientCard patient={patient_card} urgentFlags={urgent_flags} />

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

      <ConditionsList conditions={patient_card?.conditions_with_codes} />
      <VitalsRow vitals={patient_card?.vitals} />
      <MedicationTable medications={patient_card?.medications} warnings={drug_warnings} />
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