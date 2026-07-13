import { useEffect, useState } from 'react'
import { api } from '../api.js'
import { Panel, Stat, Loading, ErrorBox } from '../components/ui.jsx'

const STAGES = [
  { label: 'Historical Orders', sub: '180.5k rows · DataCo' },
  { label: 'Feature Engineering', sub: 'lead-time diff · urgency flag' },
  { label: 'CatBoost x3', sub: 'sales · delivery · profit' },
  { label: 'RL State', sub: 'predictions + inventory + demand' },
  { label: 'PPO Agent', sub: 'clipped surrogate · GAE' },
  { label: 'Reorder Decision', sub: 'optimized quantity' },
]

function Diagram() {
  return (
    <div className="overflow-x-auto">
      <svg viewBox="0 0 1180 160" className="w-full min-w-[820px]" style={{ height: 170 }}>
        <defs>
          <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="#8b96a0" />
          </marker>
        </defs>
        {STAGES.map((s, i) => {
          const x = 10 + i * 195
          const isTerminal = i === STAGES.length - 1
          const isCore = i === 2 || i === 4
          return (
            <g key={s.label}>
              <rect
                x={x} y="40" width="168" height="72" rx="10"
                fill={isCore ? 'rgba(255,176,32,0.08)' : 'var(--color-surface-raised)'}
                stroke={isCore ? '#ffb020' : '#232b31'}
                strokeWidth={isCore ? 1.4 : 1}
              />
              <text x={x + 84} y="68" textAnchor="middle" fill={isTerminal ? '#2dd4bf' : '#e8ecef'} fontSize="12.5" fontWeight="600" fontFamily="Sora, sans-serif">
                {s.label}
              </text>
              <text x={x + 84} y="88" textAnchor="middle" fill="#8b96a0" fontSize="9.5" fontFamily="IBM Plex Mono, monospace">
                {s.sub}
              </text>
              {i < STAGES.length - 1 && (
                <line x1={x + 168} y1="76" x2={x + 195 - 8} y2="76" stroke="#8b96a0" strokeWidth="1.3" markerEnd="url(#arrow)" />
              )}
            </g>
          )
        })}
      </svg>
    </div>
  )
}

export default function Overview() {
  const [metrics, setMetrics] = useState(null)
  const [compare, setCompare] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([api.allMetrics(), api.rlCompare()])
      .then(([m, c]) => { setMetrics(m); setCompare(c) })
      .catch((e) => setError(e.message))
  }, [])

  const improvementPct = compare
    ? (((compare.rl_agent.mean_reward - compare.baseline.mean_reward) / Math.abs(compare.baseline.mean_reward)) * 100)
    : null

  return (
    <div className="space-y-6">
      <div>
        <p className="mono text-[11px] uppercase tracking-widest text-[var(--color-amber)] mb-2">Overview</p>
        <h1 className="text-2xl font-semibold tracking-tight">Predictive &amp; adaptive supply chain control</h1>
        <p className="text-[var(--color-muted)] text-[14.5px] leading-relaxed mt-3 max-w-3xl">
          SupplySense pairs three CatBoost regressors — trained on 180,519 historical DataCo orders — with a PPO
          reinforcement-learning agent. CatBoost forecasts sales, delivery time, and profit for the current
          order context; those forecasts, alongside on-hand inventory and pending demand, become the state a PPO
          agent uses to decide how much to reorder. The agent is trained to maximize revenue net of holding and
          stockout costs, and its policy is benchmarked against a naive fixed reorder-point baseline.
        </p>
      </div>

      <Panel title="Architecture" subtitle="data → features → CatBoost → RL state → agent → decision">
        <Diagram />
      </Panel>

      {error && <ErrorBox message={`Could not reach the API: ${error}`} />}
      {!metrics && !error && <Loading label="Loading headline metrics" />}

      {metrics && compare && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Panel>
            <Stat label="Sales model R²" value={metrics.sales.r2.toFixed(4)} accent="amber" hint={`RMSE ${metrics.sales.rmse.toFixed(2)}`} />
          </Panel>
          <Panel>
            <Stat label="Delivery model R²" value={metrics.delivery.r2.toFixed(3)} accent="amber" hint={`MAE ${metrics.delivery.mae.toFixed(2)} days`} />
          </Panel>
          <Panel>
            <Stat label="Profit model R²" value={metrics.profit.r2.toFixed(3)} accent="amber" hint={`RMSE $${metrics.profit.rmse.toFixed(1)}`} />
          </Panel>
          <Panel>
            <Stat
              label="RL vs baseline reward"
              value={`${improvementPct >= 0 ? '+' : ''}${improvementPct.toFixed(1)}%`}
              accent={improvementPct >= 0 ? 'teal' : 'red'}
              hint={`${compare.rl_agent.mean_reward.toFixed(0)} vs ${compare.baseline.mean_reward.toFixed(0)} mean reward`}
            />
          </Panel>
        </div>
      )}
    </div>
  )
}
