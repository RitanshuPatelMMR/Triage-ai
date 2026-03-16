import { Activity } from 'lucide-react'

export default function Footer() {
  return (
    <footer className="border-t border-stone-200 dark:border-stone-800 py-6 mt-auto">
      <div className="max-w-6xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <div className="w-5 h-5 rounded bg-brand-500 flex items-center justify-center">
            <Activity size={11} className="text-white" />
          </div>
          <span className="text-xs text-stone-500 dark:text-stone-400">TriageAI</span>
        </div>
        <div className="text-xs text-stone-400 dark:text-stone-500 text-center">
          Built with LangGraph · Groq · FAISS · For informational use only
        </div>
        {/* <div className="text-xs text-stone-400 dark:text-stone-500">
        
        </div> */}
      </div>
    </footer>
  )
}