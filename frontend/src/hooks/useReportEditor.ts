import { useState, useCallback } from 'react'
import { Report, PatientCard } from '../types'

export interface ReportEdits {
  age?: string
  gender?: string
  chief_complaint?: string
  conditions?: Record<number, string>   // index -> edited condition name
  medications?: Record<number, { name?: string; dose?: string; frequency?: string }>
  allergies?: Record<number, string>
  vitals?: Partial<Record<string, string>>
}

export function useReportEditor(report: Report | null) {
  const [edits, setEdits] = useState<ReportEdits>({})

  const setField = useCallback(<K extends keyof ReportEdits>(
    field: K,
    value: ReportEdits[K]
  ) => {
    setEdits(prev => ({ ...prev, [field]: value }))
  }, [])

  const mergedReport = useCallback((): Report | null => {
    if (!report) return null

    const pc = { ...(report.patient_card ?? {}) } as PatientCard

    if (edits.age !== undefined) pc.age = Number(edits.age) || pc.age
    if (edits.gender !== undefined) pc.gender = edits.gender
    if (edits.chief_complaint !== undefined) pc.chief_complaint = edits.chief_complaint

    if (edits.conditions) {
      pc.conditions_with_codes = (pc.conditions_with_codes ?? []).map((c, i) => ({
        ...c,
        condition: edits.conditions?.[i] ?? c.condition,
      }))
    }

    if (edits.medications) {
      pc.medications = (pc.medications ?? []).map((m, i) => ({
        ...m,
        ...edits.medications?.[i],
      }))
    }

    if (edits.allergies) {
      pc.allergies = (pc.allergies ?? []).map((a, i) =>
        edits.allergies?.[i] ?? a
      )
    }

    if (edits.vitals) {
      pc.vitals = { ...(pc.vitals ?? {}), ...edits.vitals }
    }

    return { ...report, patient_card: pc }
  }, [report, edits])

  const hasEdits = Object.keys(edits).length > 0

  const resetEdits = useCallback(() => setEdits({}), [])

  return { edits, setField, mergedReport, hasEdits, resetEdits }
}