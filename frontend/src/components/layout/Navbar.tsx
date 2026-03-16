import { Link, useLocation } from 'react-router-dom'
import { Sun, Moon, Activity } from 'lucide-react'

interface NavbarProps {
  isDark: boolean
  onToggleTheme: () => void
}

export default function Navbar({ isDark, onToggleTheme }: NavbarProps) {
  const location = useLocation()

  const links = [
    { to: '/', label: 'Home' },
    { to: '/analyze', label: 'Analyzer' },
    { to: '/history', label: 'History' },
  ]

  return (
    <nav className="sticky top-0 z-50 bg-white/90 dark:bg-stone-900/90 backdrop-blur-sm border-b border-stone-200 dark:border-stone-700">
      <div className="max-w-6xl mx-auto px-4 h-14 flex items-center justify-between">

        {/* Logo */}
        <Link to="/" className="flex items-center gap-2 group">
          <div className="w-7 h-7 rounded-lg bg-brand-500 flex items-center justify-center">
            <Activity size={14} className="text-white" />
          </div>
          <span className="font-semibold text-stone-900 dark:text-stone-100 text-sm">
            TriageAI
          </span>
        </Link>

        {/* Links */}
        <div className="flex items-center gap-1">
          {links.map(link => (
            <Link
              key={link.to}
              to={link.to}
              className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
                location.pathname === link.to
                  ? 'bg-stone-100 dark:bg-stone-800 text-stone-900 dark:text-stone-100 font-medium'
                  : 'text-stone-500 dark:text-stone-400 hover:text-stone-700 dark:hover:text-stone-200 hover:bg-stone-50 dark:hover:bg-stone-800'
              }`}
            >
              {link.label}
            </Link>
          ))}

          {/* Theme toggle */}
          <button
            onClick={onToggleTheme}
            className="ml-2 p-2 rounded-lg text-stone-500 dark:text-stone-400 hover:bg-stone-100 dark:hover:bg-stone-800 transition-colors"
            aria-label="Toggle theme"
          >
            {isDark ? <Sun size={15} /> : <Moon size={15} />}
          </button>
        </div>
      </div>
    </nav>
  )
}