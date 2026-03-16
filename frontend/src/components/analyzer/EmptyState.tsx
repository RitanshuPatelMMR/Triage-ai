import { FileText } from 'lucide-react'

export default function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center py-16">
      <div className="w-12 h-12 rounded-xl bg-stone-100 dark:bg-stone-800 flex items-center justify-center mb-4">
        <FileText size={20} className="text-stone-400 dark:text-stone-500" />
      </div>
      <div className="text-sm font-medium text-stone-600 dark:text-stone-400 mb-1">
        Your report will appear here
      </div>
      <div className="text-xs text-stone-400 dark:text-stone-500">
        Paste a note or upload a file to get started
      </div>
    </div>
  )
}