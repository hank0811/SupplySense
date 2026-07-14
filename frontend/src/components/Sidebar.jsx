import { NavLink } from 'react-router-dom'

const icons = {
  overview: (
    <path d="M3 12 L9 6 L13 10 L21 3 M21 3 L21 8 M21 3 L16 3" />
  ),
  predictions: (
    <path d="M4 20 V10 M11 20 V4 M18 20 V13" />
  ),
  rl: (
    <path d="M3 17 C7 17 7 7 11 7 C15 7 15 17 19 17 C21 17 21 17 21 17 M3 17 L21 17" />
  ),
  compare: (
    <path d="M5 21 V9 M12 21 V3 M19 21 V13 M2 21 H22" />
  ),
  tryit: (
    <path d="M12 2 L14.5 8.5 L21 9 L16 13.5 L17.5 20 L12 16.5 L6.5 20 L8 13.5 L3 9 L9.5 8.5 Z" />
  ),
}

const items = [
  { to: '/', label: 'Overview', icon: 'overview', end: true },
  { to: '/predictions', label: 'Predictions', icon: 'predictions' },
  { to: '/rl-performance', label: 'RL Performance', icon: 'rl' },
  { to: '/compare', label: 'Existing vs Proposed', icon: 'compare' },
  { to: '/try-it', label: 'Try It', icon: 'tryit' },
]

export default function Sidebar({ open, onClose }) {
  return (
    <>
      <div
        onClick={onClose}
        className={`fixed inset-0 z-40 bg-black/60 backdrop-blur-[1px] transition-opacity md:hidden ${
          open ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'
        }`}
      />

      <aside
        className={`w-64 shrink-0 h-screen fixed md:sticky top-0 z-50 border-r border-[var(--color-border)] bg-[var(--color-surface)] md:bg-[var(--color-surface)]/60 backdrop-blur-sm flex flex-col transition-transform duration-200 ease-out ${
          open ? 'translate-x-0' : '-translate-x-full'
        } md:translate-x-0`}
      >
        <div className="px-6 pt-7 pb-6 border-b border-[var(--color-border)] flex items-start justify-between">
          <div>
            <div className="flex items-center gap-2.5">
              <svg width="22" height="22" viewBox="0 0 32 32" fill="none">
                <rect width="32" height="32" rx="6" fill="#0a0d10" />
                <path d="M6 22 L12 14 L17 18 L26 8" stroke="#ffb020" strokeWidth="2.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />
                <circle cx="26" cy="8" r="2.2" fill="#2dd4bf" />
              </svg>
              <span className="font-semibold tracking-tight text-[17px]">SupplySense</span>
            </div>
            <p className="mono text-[10.5px] text-[var(--color-muted)] mt-2 tracking-wide uppercase">
              Hybrid CatBoost &middot; PPO control room
            </p>
          </div>
          <button
            onClick={onClose}
            aria-label="Close menu"
            className="md:hidden -mt-1 -mr-1 p-1.5 text-[var(--color-muted)] hover:text-[var(--color-text)]"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
              <path d="M6 6 L18 18 M18 6 L6 18" />
            </svg>
          </button>
        </div>

        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          {items.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              onClick={onClose}
              className={({ isActive }) =>
                `group flex items-center gap-3 px-3 py-2.5 rounded-md text-sm transition-colors relative ${
                  isActive
                    ? 'bg-[var(--color-surface-raised)] text-[var(--color-text)]'
                    : 'text-[var(--color-muted)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface-raised)]/60'
                }`
              }
            >
              {({ isActive }) => (
                <>
                  <span
                    className={`absolute left-0 top-1.5 bottom-1.5 w-[2.5px] rounded-full transition-opacity ${
                      isActive ? 'bg-[var(--color-amber)] opacity-100' : 'opacity-0'
                    }`}
                  />
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="shrink-0">
                    {icons[item.icon]}
                  </svg>
                  <span className="mono text-[12.5px] tracking-wide">{item.label}</span>
                </>
              )}
            </NavLink>
          ))}
        </nav>

        <div className="px-5 py-4 border-t border-[var(--color-border)]">
          <p className="mono text-[10px] text-[var(--color-muted)] leading-relaxed">
            CatBoost + PPO trained offline.<br />Served from static artifacts.
          </p>
        </div>
      </aside>
    </>
  )
}
