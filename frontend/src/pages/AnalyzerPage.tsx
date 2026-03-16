import { useCallback, useState } from 'react'
import { useSSE } from '../hooks/useSSE'
import { useHistory } from '../hooks/useHistory'
import UploadPanel from '../components/analyzer/UploadPanel'
import ReportView from '../components/report/ReportView'
import EmptyState from '../components/analyzer/EmptyState'
import type { Report } from '../types'
import ErrorBoundary from '../components/ErrorBoundary'


export default function AnalyzerPage() {
  const { steps, report, status, error, inputType, analyzeText, analyzeFile } = useSSE()
  const { save, markVerified, entries } = useHistory()
  const [renderError, setRenderError] = useState<string | null>(null)

  const handleComplete = useCallback((r: Report) => {
    try {
      save(r, inputType)
    } catch (e) {
      console.error('Save error:', e)
    }
  }, [save, inputType])

  const handleAnalyzeText = (text: string) => {
    setRenderError(null)
    analyzeText(text, handleComplete)
  }

  const handleAnalyzeFile = (file: File) => {
    setRenderError(null)
    analyzeFile(file, handleComplete)
  }

  const latestEntry = entries[0]
  const isVerified = latestEntry?.humanVerified ?? false

  const handleMarkVerified = () => {
    if (latestEntry) markVerified(latestEntry.id)
  }

  const renderReport = () => {
    if (!report) return null
    try {
      return (
        <ReportView
          report={report}
          onMarkVerified={handleMarkVerified}
          isVerified={isVerified}
        />
      )
    } catch (e) {
      return (
        <div className="p-4 rounded-xl bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800">
          <p className="text-sm text-red-600 dark:text-red-400">
            Report rendering error — the data was processed but could not be displayed.
          </p>
          <pre className="text-xs text-red-500 mt-2 overflow-auto">
            {JSON.stringify(report, null, 2)}
          </pre>
        </div>
      )
    }
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-6">
      <div className="grid grid-cols-[400px_1fr] gap-6 min-h-[calc(100vh-8rem)]">

        {/* Left panel */}
        <div className="bg-white dark:bg-stone-900 rounded-2xl border border-stone-200 dark:border-stone-700 p-5 h-fit sticky top-20">
          <UploadPanel
            onAnalyzeText={handleAnalyzeText}
            onAnalyzeFile={handleAnalyzeFile}
            steps={steps}
            status={status}
          />
        </div>

        {/* Right panel */}
       {/* Right panel */}
<div className="bg-white dark:bg-stone-900 rounded-2xl border border-stone-200 dark:border-stone-700 p-5">
  {(error || renderError) && (
    <div className="p-3 rounded-lg bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800 text-xs text-red-600 dark:text-red-400 mb-4">
      {error || renderError}
    </div>
  )}
  {status === 'idle' && !report && <EmptyState />}
  <ErrorBoundary>
    {renderReport()}
  </ErrorBoundary>
</div>
      </div>
    </div>
  )
}