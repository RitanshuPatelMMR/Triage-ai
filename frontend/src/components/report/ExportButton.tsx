import { Download } from 'lucide-react'
import { Report } from '../../types'
import toast from 'react-hot-toast'

interface Props { report: Report }

export default function ExportButton({ report }: Props) {
  const handleExport = () => {
    const content = JSON.stringify(report, null, 2)
    const blob = new Blob([content], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `triageai-report-${new Date().toISOString().slice(0, 10)}.json`
    a.click()
    URL.revokeObjectURL(url)
    toast.success('Report downloaded')
  }

  return (
    <button
      onClick={handleExport}
      className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border border-stone-200 dark:border-stone-700 text-stone-500 dark:text-stone-400 hover:bg-stone-50 dark:hover:bg-stone-800 transition-colors"
    >
      <Download size={13} /> Export JSON
    </button>
  )
}