export function Panel({ title, subtitle, right, children, className = '' }) {
  return (
    <div className={`bg-[var(--color-surface)] border border-[var(--color-border)] rounded-xl p-5 animate-rise ${className}`}>
      {(title || right) && (
        <div className="flex items-start justify-between mb-4">
          <div>
            {title && <h3 className="text-[13px] font-semibold tracking-wide">{title}</h3>}
            {subtitle && <p className="mono text-[11px] text-[var(--color-muted)] mt-0.5">{subtitle}</p>}
          </div>
          {right}
        </div>
      )}
      {children}
    </div>
  )
}

export function Stat({ label, value, unit, accent = 'text', hint }) {
  const color = {
    text: 'text-[var(--color-text)]',
    amber: 'text-[var(--color-amber)]',
    teal: 'text-[var(--color-teal)]',
    red: 'text-[var(--color-red)]',
  }[accent]
  return (
    <div>
      <p className="mono text-[10.5px] uppercase tracking-wider text-[var(--color-muted)]">{label}</p>
      <p className={`mono text-2xl font-medium mt-1 tabular ${color}`}>
        {value}
        {unit && <span className="text-sm text-[var(--color-muted)] ml-1">{unit}</span>}
      </p>
      {hint && <p className="text-[11px] text-[var(--color-muted)] mt-1">{hint}</p>}
    </div>
  )
}

export function Loading({ label = 'Loading' }) {
  return (
    <div className="flex items-center gap-2 text-[var(--color-muted)] mono text-xs py-10 justify-center">
      <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-amber)] animate-pulse" />
      {label}&hellip;
    </div>
  )
}

export function ErrorBox({ message }) {
  return (
    <div className="border border-[var(--color-red)]/30 bg-[var(--color-red)]/10 text-[var(--color-red)] rounded-lg px-4 py-3 mono text-xs">
      {message}
    </div>
  )
}

export function Pill({ children, tone = 'default' }) {
  const styles = {
    default: 'bg-[var(--color-surface-raised)] text-[var(--color-muted)] border-[var(--color-border)]',
    amber: 'bg-[var(--color-amber)]/10 text-[var(--color-amber)] border-[var(--color-amber)]/30',
    teal: 'bg-[var(--color-teal)]/10 text-[var(--color-teal)] border-[var(--color-teal)]/30',
    red: 'bg-[var(--color-red)]/10 text-[var(--color-red)] border-[var(--color-red)]/30',
  }[tone]
  return (
    <span className={`mono text-[10px] uppercase tracking-wider px-2 py-1 rounded border ${styles}`}>
      {children}
    </span>
  )
}
