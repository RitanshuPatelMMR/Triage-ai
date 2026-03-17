import { useNavigate } from 'react-router-dom'
import { Home, AlertCircle } from 'lucide-react'

export default function NotFoundPage() {
  const navigate = useNavigate()

  return (
    <div className="min-h-[60vh] flex flex-col items-center justify-center text-center px-4">
      <div className="w-14 h-14 rounded-2xl bg-stone-100 dark:bg-stone-800 flex items-center justify-center mb-6">
        <AlertCircle size={24} className="text-stone-400" />
      </div>
      <h1 className="text-2xl font-semibold text-stone-900 dark:text-stone-100 mb-2">
        Page not found
      </h1>
      <p className="text-stone-500 dark:text-stone-400 mb-8 max-w-sm">
        The page you're looking for doesn't exist or has been moved.
      </p>
      <button
        onClick={() => navigate('/')}
        className="flex items-center gap-2 px-5 py-2.5 bg-brand-500 hover:bg-brand-600 text-white rounded-xl text-sm font-medium transition-colors"
      >
        <Home size={15} /> Back to home
      </button>
    </div>
  )
}
