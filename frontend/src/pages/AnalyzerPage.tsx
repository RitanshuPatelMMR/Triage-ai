import { useCallback, useState } from 'react'
import { useSSE } from '../hooks/useSSE'
import { useHistory } from '../hooks/useHistory'
import UploadPanel from '../components/analyzer/UploadPanel'
import ReportView from '../components/report/ReportView'
import EmptyState from '../components/analyzer/EmptyState'
import type { Report } from '../types'
import ErrorBoundary from '../components/ErrorBoundary'
import { Loader2 } from 'lucide-react'

function ReportSkeleton() {
  return (
    <div className="space-y-4 animate-pulse">
      <div className="h-24 rounded-xl bg-stone-100 dark:bg-stone-800" />
      <div className="h-32 rounded-xl bg-stone-100 dark:bg-stone-800" />
      <div className="h-20 rounded-xl bg-stone-100 dark:bg-stone-800" />
      <p className="text-xs text-stone-400 dark:text-stone-500 flex items-center gap-2">
        <Loader2 size={14} className="animate-spin" />
        AI agent is working — this may take up to a minute on first request...
      </p>
    </div>
  )
}

export default function AnalyzerPage() {
  const {
    steps,
    report,
    status,
    error,
    coldStartHint,
    inputType,
    analyzeText,
    analyzeFile,
    retryLast,
    reset,
  } = useSSE()

  const {
    save,
    markVerified,
    entries,
    verifyError,
    historyNotice,
    clearVerifyError,
  } = useHistory()

  const [currentEntryId, setCurrentEntryId] = useState<string | null>(null)
  const [saveWarning, setSaveWarning] = useState<string | null>(null)

  const handleComplete = useCallback((r: Report, meta?: { requestId?: string; createdAt?: string }) => {
    try {
      const entry = save(r, inputType, meta)
      setCurrentEntryId(entry.id)
      setSaveWarning(null)
    } catch (e) {
      console.error('Save error:', e)
      setSaveWarning('Report generated but could not save to history on this device.')
    }
  }, [save, inputType])

  const handleAnalyzeText = (text: string) => {
    clearVerifyError()
    setSaveWarning(null)
    analyzeText(text, handleComplete)
  }

  const handleAnalyzeFile = (file: File) => {
    clearVerifyError()
    setSaveWarning(null)
    analyzeFile(file, handleComplete)
  }

  const handleRetry = () => {
    clearVerifyError()
    setSaveWarning(null)
    retryLast(handleComplete)
  }

  const currentEntry = currentEntryId
    ? entries.find(e => e.id === currentEntryId)
    : entries[0]

  const isVerified = currentEntry?.humanVerified ?? false

  const handleMarkVerified = async () => {
    if (currentEntry) {
      await markVerified(currentEntry.id)
    }
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
    } catch {
      return (
        <div className="p-4 rounded-xl bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800">
          <p className="text-sm text-red-600 dark:text-red-400">
            Report rendering error — the data was processed but could not be displayed.
          </p>
          <pre className="text-xs text-red-500 mt-2 overflow-auto max-h-48">
            {JSON.stringify(report, null, 2)}
          </pre>
        </div>
      )
    }
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-6">
      {historyNotice && (
        <div className="mb-4 p-3 rounded-lg bg-stone-50 dark:bg-stone-800/50 border border-stone-200 dark:border-stone-700 text-xs text-stone-600 dark:text-stone-400">
          {historyNotice}
        </div>
      )}

      <div className="grid grid-cols-[400px_1fr] gap-6 min-h-[calc(100vh-8rem)]">
        <div className="bg-white dark:bg-stone-900 rounded-2xl border border-stone-200 dark:border-stone-700 p-5 h-fit sticky top-20">
          <UploadPanel
            onAnalyzeText={handleAnalyzeText}
            onAnalyzeFile={handleAnalyzeFile}
            steps={steps}
            status={status}
            coldStartHint={coldStartHint}
            panelError={error}
          />
        </div>

        <div className="bg-white dark:bg-stone-900 rounded-2xl border border-stone-200 dark:border-stone-700 p-5">
          {error && status === 'error' && (
            <div className="p-4 rounded-xl bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800 mb-4">
              <p className="text-sm font-medium text-red-700 dark:text-red-300 mb-1">
                Analysis could not be completed
              </p>
              <p className="text-xs text-red-600 dark:text-red-400 mb-3">{error}</p>
              <div className="flex gap-2">
                <button
                  onClick={handleRetry}
                  className="text-xs px-3 py-1.5 rounded-lg bg-red-600 text-white hover:bg-red-700 transition-colors"
                >
                  Retry
                </button>
                <button
                  onClick={reset}
                  className="text-xs px-3 py-1.5 rounded-lg border border-red-200 dark:border-red-800 text-red-600 dark:text-red-400 hover:bg-red-100/50 dark:hover:bg-red-950/30 transition-colors"
                >
                  Clear
                </button>
              </div>
            </div>
          )}

          {verifyError && (
            <div className="p-3 rounded-lg bg-amber-50 dark:bg-amber-950/20 border border-amber-200 dark:border-amber-800 text-xs text-amber-700 dark:text-amber-300 mb-4">
              {verifyError}
            </div>
          )}

          {saveWarning && (
            <div className="p-3 rounded-lg bg-amber-50 dark:bg-amber-950/20 border border-amber-200 dark:border-amber-800 text-xs text-amber-700 dark:text-amber-300 mb-4">
              {saveWarning}
            </div>
          )}

          {status === 'running' && !report && <ReportSkeleton />}

          {status === 'idle' && !report && <EmptyState />}

          <ErrorBoundary>
            {renderReport()}
          </ErrorBoundary>
        </div>
      </div>
    </div>
  )
}
