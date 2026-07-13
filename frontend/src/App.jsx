import { Routes, Route } from 'react-router-dom'
import Sidebar from './components/Sidebar.jsx'
import Overview from './pages/Overview.jsx'
import Predictions from './pages/Predictions.jsx'
import RLPerformance from './pages/RLPerformance.jsx'
import Compare from './pages/Compare.jsx'
import TryIt from './pages/TryIt.jsx'

export default function App() {
  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 px-8 py-8 max-w-[1400px]">
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/predictions" element={<Predictions />} />
          <Route path="/rl-performance" element={<RLPerformance />} />
          <Route path="/compare" element={<Compare />} />
          <Route path="/try-it" element={<TryIt />} />
        </Routes>
      </main>
    </div>
  )
}
