import { useNavigate } from 'react-router-dom'
import { ArrowRight, Brain, Pill, FileSearch, Camera, ShieldAlert } from 'lucide-react'

export default function HomePage() {
  const navigate = useNavigate()

  const features = [
    { icon: Brain, title: '5-step AI agent', desc: 'LangGraph orchestration with live streaming output', color: 'bg-purple-50 dark:bg-purple-950 text-purple-600 dark:text-purple-400' },
    { icon: Pill, title: 'FDA drug checker', desc: 'Real-time interaction warnings from FDA database', color: 'bg-orange-50 dark:bg-orange-950 text-orange-600 dark:text-orange-400' },
    { icon: FileSearch, title: 'ICD-10 mapping', desc: '82,000+ diagnosis codes via FAISS vector search', color: 'bg-blue-50 dark:bg-blue-950 text-blue-600 dark:text-blue-400' },
    { icon: Camera, title: 'Handwriting OCR', desc: 'Groq Vision reads and transcribes messy notes', color: 'bg-green-50 dark:bg-green-950 text-green-600 dark:text-green-400' },
  ]

  const steps = [
    { num: '1', title: 'Paste or upload', desc: 'Text, PDF, or handwritten image' },
    { num: '2', title: 'AI analyzes', desc: '5 autonomous agent nodes process your note' },
    { num: '3', title: 'Get report', desc: 'Structured, ICD-coded, export-ready' },
  ]

  return (
    <div className="max-w-4xl mx-auto px-4 py-16">

      {/* Hero */}
      <div className="text-center mb-20 animate-fade-in">
        <div className="inline-block px-3 py-1 rounded-full bg-brand-50 dark:bg-brand-900/30 border border-brand-200 dark:border-brand-800 text-brand-600 dark:text-brand-400 text-xs font-medium mb-6">
          Clinical AI — For informational use only
        </div>

        <h1 className="text-4xl sm:text-5xl font-semibold text-stone-900 dark:text-stone-50 leading-tight mb-4">
          Clinical notes,{' '}
          <span className="text-brand-500">structured in seconds</span>
        </h1>

        <p className="text-lg text-stone-500 dark:text-stone-400 max-w-xl mx-auto mb-8 leading-relaxed">
          Paste any messy doctor note or upload a file.
          TriageAI reads it, understands it, and returns a clean structured report.
        </p>

        <div className="flex gap-3 justify-center">
          <button
            onClick={() => navigate('/analyze')}
            className="flex items-center gap-2 px-5 py-2.5 bg-brand-500 hover:bg-brand-600 text-white rounded-xl text-sm font-medium transition-colors"
          >
            Try it now <ArrowRight size={15} />
          </button>
          <button
            onClick={() => document.getElementById('how-it-works')?.scrollIntoView({ behavior: 'smooth' })}
            className="px-5 py-2.5 border border-stone-200 dark:border-stone-700 text-stone-600 dark:text-stone-300 rounded-xl text-sm hover:bg-stone-50 dark:hover:bg-stone-800 transition-colors"
          >
            See how it works
          </button>
        </div>
      </div>

      {/* How it works */}
      <div id="how-it-works" className="mb-20">
        <h2 className="text-xs font-medium text-stone-400 dark:text-stone-500 uppercase tracking-widest text-center mb-8">
          How it works
        </h2>
        <div className="grid grid-cols-3 gap-6">
          {steps.map((step, i) => (
            <div key={i} className="text-center">
              <div className="w-8 h-8 rounded-full bg-stone-100 dark:bg-stone-800 border border-stone-200 dark:border-stone-700 flex items-center justify-center text-sm font-medium text-stone-700 dark:text-stone-300 mx-auto mb-3">
                {step.num}
              </div>
              <div className="text-sm font-medium text-stone-800 dark:text-stone-200 mb-1">
                {step.title}
              </div>
              <div className="text-xs text-stone-500 dark:text-stone-400 leading-relaxed">
                {step.desc}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Features */}
      <div className="mb-16">
        <h2 className="text-xs font-medium text-stone-400 dark:text-stone-500 uppercase tracking-widest text-center mb-8">
          Features
        </h2>
        <div className="grid grid-cols-2 gap-4">
          {features.map((f, i) => (
            <div
              key={i}
              className="p-4 rounded-xl border border-stone-200 dark:border-stone-700 bg-white dark:bg-stone-900 hover:border-stone-300 dark:hover:border-stone-600 transition-colors"
            >
              <div className={`w-8 h-8 rounded-lg ${f.color} flex items-center justify-center mb-3`}>
                <f.icon size={16} />
              </div>
              <div className="text-sm font-medium text-stone-800 dark:text-stone-200 mb-1">
                {f.title}
              </div>
              <div className="text-xs text-stone-500 dark:text-stone-400 leading-relaxed">
                {f.desc}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Disclaimer */}
      <div className="flex items-start gap-3 p-4 rounded-xl bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800">
        <ShieldAlert size={16} className="text-amber-500 flex-shrink-0 mt-0.5" />
        <p className="text-xs text-amber-700 dark:text-amber-400 leading-relaxed">
          <span className="font-medium">Medical disclaimer:</span> TriageAI is designed for informational and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.
        </p>
      </div>
    </div>
  )
}