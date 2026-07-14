import { useState } from 'react'
import { Routes, Route } from 'react-router-dom'
import Sidebar from './components/Sidebar.jsx'
import Overview from './pages/Overview.jsx'
import Predictions from './pages/Predictions.jsx'
import RLPerformance from './pages/RLPerformance.jsx'
import Compare from './pages/Compare.jsx'
import TryIt from './pages/TryIt.jsx'

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false)

  return (
    <div className="flex min-h-screen w-full">
      <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      <div className="flex-1 min-w-0 flex flex-col">
        <header className="md:hidden sticky top-0 z-30 flex items-center gap-3 px-4 py-3 border-b border-[var(--color-border)] bg-[var(--color-ink)]/90 backdrop-blur-sm">
          <button
            onClick={() => setSidebarOpen(true)}
            aria-label="Open menu"
            className="p-1.5 -ml-1.5 text-[var(--color-muted)] hover:text-[var(--color-text)]"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
              <path d="M3 6 H21 M3 12 H21 M3 18 H21" />
            </svg>
          </button>
          <span className="font-semibold tracking-tight text-[15px]">SupplySense</span>
        </header>

        <main className="flex-1 min-w-0 px-4 py-6 sm:px-6 sm:py-7 lg:px-8 lg:py-8 max-w-[1400px] overflow-x-hidden">
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/predictions" element={<Predictions />} />
            <Route path="/rl-performance" element={<RLPerformance />} />
            <Route path="/compare" element={<Compare />} />
            <Route path="/try-it" element={<TryIt />} />
          </Routes>
        </main>
      </div>
    </div>
  )
}
