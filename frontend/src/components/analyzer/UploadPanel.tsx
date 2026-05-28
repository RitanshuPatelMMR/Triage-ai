import { useState, useRef, DragEvent, useEffect } from 'react'
import { Upload, FileText, Image, ChevronDown } from 'lucide-react'
import NoteQualityHint from './NoteQualityHint'
import AgentLog from './AgentLog'
import { AgentStep, AnalysisStatus } from '../../types'

const SAMPLE_NOTES = [
  {
    label: 'Emergency: Chest pain (STEMI)',
    text: 'pt 67M c/o CP x2hr rad to L arm, diaphoretic, SOB+, PMH DM2 HTN, smoker 20pk/yr, meds metformin 500 BID lisinopril 10 QD atorvastatin 40 QHS, EKG ST elev V2-V4, trop pnd, A: r/o STEMI, P: ASA 325 STAT ntg SL heparin gtt cath lab activation, BP 158/94 HR 102 RR 18 O2 94% RA, allerg PCN hives'
  },
  {
    label: 'Routine: Diabetes follow-up',
    text: 'pt 54F DM2 f/u, HbA1c 8.2 last mo, c/o fatigue wt gain, curr meds metformin 1000 BID glipizide 5 QD, BP 138/86 HR 74, no SOB no CP, diet non-compliant, A: DM2 uncontrolled, P: inc metformin to 2000 BID, nutritionist referral, recheck HbA1c 3mo'
  },
  {
    label: 'Complex: Multi-drug elderly patient',
    text: 'pt 82M nursing home transfer, PMH CHF HTN AFib CKD stage 3, meds furosemide 40 QD metoprolol 25 BID warfarin 5 QD lisinopril 5 QD, c/o worsening SOB x5d, LE edema ++, wt up 8lbs, O2 92% RA, BP 162/98 HR 88 irreg, A: CHF exacerbation, P: IV lasix 80 hold lisinopril check BMP BNP'
  },
  {
    label: 'Pediatric: Asthma attack',
    text: 'pt 8yo M c/o SOB wheeze x2hr, PMH asthma dx age 4, prev 2 hospitalizations, curr meds albuterol PRN fluticasone 44mcg BID, O2 92% RA HR 118 RR 28, accessory muscle use +, peak flow 55% predicted, A: acute asthma exacerbation moderate, P: albuterol neb q20min x3 methylprednisolone IV obs'
  },
  {
    label: 'Thyroid: Follow-up hypothyroidism',
    text: 'pt 34F f/u thyroid, tired all time wt gain 8lbs/3mo, hair falling out feels cold always, TSH last mo 8.2 high, FH mom has thyroid prob, curr meds none, O/E HR 58 BP 112/70 thyroid slightly enlarged on palp no tremor reflexes sluggish, A: hypothy hashimotos?, P: start levo 50mcg QAM empty stomach recheck TSH T4 6wk ref endocrine if no improv'
  },
]

const MIN_NOTE_LENGTH = 30
const MAX_FILE_BYTES = 10 * 1024 * 1024
const ALLOWED_EXTENSIONS = new Set(['pdf', 'jpg', 'jpeg', 'png'])

function validateFile(file: File): string | null {
  if (file.size === 0) return 'File is empty. Choose another file.'
  if (file.size > MAX_FILE_BYTES) return 'File is too large (max 10 MB).'
  const ext = file.name.split('.').pop()?.toLowerCase() ?? ''
  if (!ALLOWED_EXTENSIONS.has(ext)) {
    return 'Only PDF, JPG, and PNG files are supported.'
  }
  return null
}

interface Props {
  onAnalyzeText: (text: string) => void
  onAnalyzeFile: (file: File) => void
  steps: AgentStep[]
  status: AnalysisStatus
  coldStartHint?: string | null
  panelError?: string | null
}

export default function UploadPanel({
  onAnalyzeText,
  onAnalyzeFile,
  steps,
  status,
  coldStartHint,
  panelError,
}: Props) {
  const [text, setText] = useState('')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [fileError, setFileError] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [showSamples, setShowSamples] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const isRunning = status === 'running'

  useEffect(() => {
    if (status !== 'running') {
      setIsSubmitting(false)
    }
  }, [status])

  const selectFile = (file: File) => {
    const err = validateFile(file)
    setFileError(err)
    if (!err) {
      setSelectedFile(file)
      setText('')
    } else {
      setSelectedFile(null)
    }
  }

  const handleSubmit = () => {
    if (isRunning || isSubmitting) return
    setIsSubmitting(true)

    if (selectedFile) {
      const err = validateFile(selectedFile)
      if (err) {
        setFileError(err)
        setIsSubmitting(false)
        return
      }
      onAnalyzeFile(selectedFile)
      return
    }

    const trimmed = text.trim()
    if (trimmed.length >= MIN_NOTE_LENGTH) {
      onAnalyzeText(trimmed)
    } else {
      setIsSubmitting(false)
    }
  }

  const handleDrop = (e: DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) selectFile(file)
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) selectFile(file)
  }

  const loadSample = (sample: typeof SAMPLE_NOTES[0]) => {
    setText(sample.text)
    setSelectedFile(null)
    setFileError(null)
    setShowSamples(false)
  }

  const textTooShort = !selectedFile && text.trim().length > 0 && text.trim().length < MIN_NOTE_LENGTH
  const canSubmit =
    !isRunning &&
    !isSubmitting &&
    !textTooShort &&
    (selectedFile ? fileError === null : text.trim().length >= MIN_NOTE_LENGTH)

  return (
    <div className="flex flex-col gap-3 h-full">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-medium text-stone-700 dark:text-stone-300">
          Analyze note
        </h2>
        <div className="relative">
          <button
            onClick={() => setShowSamples(!showSamples)}
            className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-lg border border-stone-200 dark:border-stone-700 text-stone-500 dark:text-stone-400 hover:bg-stone-50 dark:hover:bg-stone-800 transition-colors"
          >
            <FileText size={12} /> Load sample <ChevronDown size={11} />
          </button>
          {showSamples && (
            <div className="absolute right-0 top-full mt-1 w-64 bg-white dark:bg-stone-900 border border-stone-200 dark:border-stone-700 rounded-xl shadow-lg z-10 overflow-hidden">
              {SAMPLE_NOTES.map((s, i) => (
                <button
                  key={i}
                  onClick={() => loadSample(s)}
                  className="w-full text-left px-3 py-2.5 text-xs text-stone-700 dark:text-stone-300 hover:bg-stone-50 dark:hover:bg-stone-800 border-b border-stone-100 dark:border-stone-800 last:border-0 transition-colors"
                >
                  {s.label}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      <textarea
        value={text}
        onChange={e => { setText(e.target.value); setSelectedFile(null); setFileError(null) }}
        placeholder="Paste clinical note here... e.g. pt 67M c/o CP x2hr, PMH DM2 HTN..."
        className="flex-1 min-h-[120px] resize-none text-sm p-3 rounded-xl border border-stone-200 dark:border-stone-700 bg-white dark:bg-stone-900 text-stone-800 dark:text-stone-200 placeholder-stone-400 dark:placeholder-stone-600 focus:outline-none focus:ring-2 focus:ring-brand-300 dark:focus:ring-brand-700 transition-all"
        disabled={isRunning}
      />

      <NoteQualityHint text={text} />

      {textTooShort && (
        <p className="text-xs text-amber-600 dark:text-amber-400">
          Add at least {MIN_NOTE_LENGTH} characters, or load a sample note.
        </p>
      )}

      <div className="flex items-center gap-3">
        <div className="flex-1 h-px bg-stone-200 dark:bg-stone-700" />
        <span className="text-xs text-stone-400 dark:text-stone-500">or upload file</span>
        <div className="flex-1 h-px bg-stone-200 dark:bg-stone-700" />
      </div>

      <div
        onDrop={handleDrop}
        onDragOver={e => { e.preventDefault(); setIsDragging(true) }}
        onDragLeave={() => setIsDragging(false)}
        onClick={() => fileInputRef.current?.click()}
        className={`p-4 rounded-xl border-2 border-dashed cursor-pointer transition-all text-center ${
          isDragging
            ? 'border-brand-400 bg-brand-50 dark:bg-brand-950/20'
            : selectedFile && !fileError
            ? 'border-green-300 dark:border-green-700 bg-green-50 dark:bg-green-950/20'
            : fileError
            ? 'border-red-300 dark:border-red-700 bg-red-50 dark:bg-red-950/20'
            : 'border-stone-200 dark:border-stone-700 hover:border-stone-300 dark:hover:border-stone-600 bg-stone-50 dark:bg-stone-800/50'
        }`}
      >
        {selectedFile && !fileError ? (
          <div className="flex items-center justify-center gap-2">
            {selectedFile.name.endsWith('.pdf')
              ? <FileText size={16} className="text-green-500" />
              : <Image size={16} className="text-green-500" />
            }
            <span className="text-xs font-medium text-green-600 dark:text-green-400">{selectedFile.name}</span>
            <button
              onClick={e => { e.stopPropagation(); setSelectedFile(null); setFileError(null) }}
              className="text-xs text-stone-400 hover:text-red-500 transition-colors ml-1"
            >✕</button>
          </div>
        ) : (
          <>
            <Upload size={16} className="text-stone-400 mx-auto mb-1" />
            <div className="text-xs text-stone-500 dark:text-stone-400">
              Drop PDF or image here
            </div>
            <div className="text-xs text-stone-400 dark:text-stone-500 mt-0.5">
              PDF · JPG · PNG · max 10 MB
            </div>
          </>
        )}
      </div>

      {fileError && (
        <p className="text-xs text-red-600 dark:text-red-400">{fileError}</p>
      )}

      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,.jpg,.jpeg,.png"
        onChange={handleFileChange}
        className="hidden"
      />

      {coldStartHint && (
        <div className="p-2.5 rounded-lg bg-amber-50 dark:bg-amber-950/20 border border-amber-200 dark:border-amber-800 text-xs text-amber-700 dark:text-amber-300">
          {coldStartHint}
        </div>
      )}

      {panelError && status === 'error' && (
        <div className="p-2.5 rounded-lg bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800 text-xs text-red-600 dark:text-red-400">
          {panelError}
        </div>
      )}

      <button
        onClick={handleSubmit}
        disabled={!canSubmit}
        className={`w-full py-2.5 rounded-xl text-sm font-medium transition-all ${
          canSubmit
            ? 'bg-brand-500 hover:bg-brand-600 text-white shadow-sm hover:shadow-md'
            : 'bg-stone-100 dark:bg-stone-800 text-stone-400 dark:text-stone-600 cursor-not-allowed'
        }`}
      >
        {isRunning ? 'Analyzing...' : status === 'error' ? 'Try again' : 'Analyze note'}
      </button>

      {status !== 'idle' && (
        <AgentLog steps={steps} status={status} />
      )}
    </div>
  )
}
