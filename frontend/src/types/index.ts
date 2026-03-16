export interface Medication {
  name: string
  dose: string
  frequency: string
}

export interface Condition {
  condition: string
  icd_code: string
}

export interface Vitals {
  bp?: string | null
  hr?: string | null
  rr?: string | null
  o2_sat?: string | null
  temp?: string | null
  weight?: string | null
}

export interface PatientCard {
  age?: number | null
  gender?: string | null
  chief_complaint?: string
  conditions_with_codes?: Condition[]
  medications?: Medication[]
  vitals?: Vitals
  allergies?: string[]
}

export interface DrugWarning {
  drug: string
  severity: 'HIGH' | 'MODERATE' | 'LOW'
  warning: string
  interactions: string
  source: string
}

export interface Report {
  plain_english_summary?: string
  patient_card?: PatientCard
  drug_interaction_summary?: string
  referral_text?: string
  urgent_flags?: string[]
  confidence_notes?: string
  icd_codes?: Record<string, string>
  drug_warnings?: DrugWarning[]
  errors?: string[]
  confidence_flags?: string[]
}

export interface HistoryEntry {
  id: string
  timestamp: string
  inputType: 'text' | 'pdf' | 'image'
  report: Report
  humanVerified: boolean
}

export interface AgentStep {
  node: string
  step: string
  errors: string[]
}

export type AnalysisStatus = 'idle' | 'running' | 'complete' | 'error'