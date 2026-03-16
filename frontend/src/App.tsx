import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import { useTheme } from './hooks/useTheme'
import Navbar from './components/layout/Navbar'
import Footer from './components/layout/Footer'
import HomePage from './pages/HomePage'
import AnalyzerPage from './pages/AnalyzerPage'
import HistoryPage from './pages/HistoryPage'

export default function App() {
  const { isDark, toggle } = useTheme()

  return (
    <BrowserRouter>
      <div className="min-h-screen flex flex-col bg-stone-50 dark:bg-stone-950 transition-colors duration-200">
        <Navbar isDark={isDark} onToggleTheme={toggle} />
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/analyze" element={<AnalyzerPage />} />
            <Route path="/history" element={<HistoryPage />} />
          </Routes>
        </main>
        <Footer />
        <Toaster
          position="bottom-right"
          toastOptions={{
            style: {
              background: isDark ? '#1c1917' : '#fff',
              color: isDark ? '#fafaf9' : '#1c1917',
              border: '1px solid',
              borderColor: isDark ? '#44403c' : '#e7e5e4',
              borderRadius: '10px',
              fontSize: '13px',
            }
          }}
        />
      </div>
    </BrowserRouter>
  )
}