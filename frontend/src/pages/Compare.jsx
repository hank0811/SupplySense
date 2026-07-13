import { useEffect, useMemo, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Cell,
} from 'recharts'
import { api } from '../api.js'
import { Panel, Loading, ErrorBox, Pill } from '../components/ui.jsx'

const AXIS_STYLE = { fill: '#8b96a0', fontSize: 10.5, fontFamily: 'IBM Plex Mono, monospace' }

function TooltipBox({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-[var(--color-surface-raised)] border border-[var(--color-border)] rounded-md px-3 py-2 mono text-[11px]">
      <p className="text-[var(--color-muted)] mb-1">{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color || p.fill }}>{p.name}: {typeof p.value === 'number' ? p.value.toFixed(1) : p.value}</p>
      ))}
    </div>
  )
}

function histogram(values, bins = 24) {
  const min = Math.min(...values)
  const max = Math.max(...values)
  const width = (max - min) / bins || 1
  const counts = new Array(bins).fill(0)
  values.forEach((v) => {
    let idx = Math.floor((v - min) / width)
    if (idx >= bins) idx = bins - 1
    if (idx < 0) idx = 0
    counts[idx] += 1
  })
  return counts.map((c, i) => ({ bin: (min + (i + 0.5) * width).toFixed(0), count: c }))
}

export default function Compare() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.rlCompare().then(setData).catch((e) => setError(e.message))
  }, [])

  const barData = useMemo(() => {
    if (!data) return []
    return [
      { metric: 'Mean Reward', Baseline: data.baseline.mean_reward, RL: data.rl_agent.mean_reward },
      { metric: 'Std Dev', Baseline: data.baseline.std_dev, RL: data.rl_agent.std_dev },
    ]
  }, [data])

  const baselineHist = data ? histogram(data.reward_series.baseline) : []
  const rlHist = data ? histogram(data.reward_series.rl_agent) : []

  const rows = data
    ? [
        ['Mean reward', data.baseline.mean_reward, data.rl_agent.mean_reward],
        ['Variance', data.baseline.variance, data.rl_agent.variance],
        ['Std deviation', data.baseline.std_dev, data.rl_agent.std_dev],
        ['Min episode reward', data.baseline.min, data.rl_agent.min],
        ['Max episode reward', data.baseline.max, data.rl_agent.max],
        ['Episodes evaluated', data.baseline.n_episodes, data.rl_agent.n_episodes],
      ]
    : []

  const improvement = data
    ? ((data.rl_agent.mean_reward - data.baseline.mean_reward) / Math.abs(data.baseline.mean_reward)) * 100
    : null

  return (
    <div className="space-y-6">
      <div>
        <p className="mono text-[11px] uppercase tracking-widest text-[var(--color-amber)] mb-2">Existing vs Proposed</p>
        <h1 className="text-2xl font-semibold tracking-tight">Fixed reorder-point vs. PPO agent</h1>
        <p className="text-[var(--color-muted)] text-[14.5px] mt-2 max-w-2xl">
          The "existing system" is a classic (s, Q) reorder-point policy sized from the historical mean order
          quantity — no learning involved. Both policies are evaluated over the same number of episodes on the
          identical environment.
        </p>
      </div>

      {error && <ErrorBox message={error} />}
      {!data && !error && <Loading label="Loading comparison" />}

      {data && (
        <>
          <Panel title="Table 7.1 — Existing vs. Proposed" subtitle={`${data.baseline.n_episodes} episodes each · ${data.episode_length} steps/episode`}>
            <table className="w-full text-[13px]">
              <thead>
                <tr className="text-left border-b border-[var(--color-border)]">
                  <th className="pb-2 font-medium text-[var(--color-muted)] mono text-[11px] uppercase tracking-wide">Metric</th>
                  <th className="pb-2 font-medium text-[var(--color-muted)] mono text-[11px] uppercase tracking-wide text-right">Existing (baseline)</th>
                  <th className="pb-2 font-medium text-[var(--color-amber)] mono text-[11px] uppercase tracking-wide text-right">Proposed (PPO)</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(([label, b, r]) => (
                  <tr key={label} className="border-b border-[var(--color-border)]/50 last:border-0">
                    <td className="py-2.5 text-[var(--color-muted)]">{label}</td>
                    <td className="py-2.5 mono tabular text-right">{b.toFixed(1)}</td>
                    <td className="py-2.5 mono tabular text-right text-[var(--color-amber)]">{r.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="mt-4">
              <Pill tone={improvement >= 0 ? 'teal' : 'red'}>
                {improvement >= 0 ? '+' : ''}{improvement.toFixed(1)}% mean-reward change vs. baseline
              </Pill>
            </div>
          </Panel>

          <div className="grid grid-cols-2 gap-4">
            <Panel title="Mean reward &amp; volatility">
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={barData} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
                  <CartesianGrid stroke="#232b31" strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="metric" tick={AXIS_STYLE} />
                  <YAxis tick={AXIS_STYLE} />
                  <Tooltip content={<TooltipBox />} cursor={{ fill: 'rgba(255,176,32,0.06)' }} />
                  <Legend wrapperStyle={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 11 }} />
                  <Bar dataKey="Baseline" fill="#8b96a0" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="RL" fill="#ffb020" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Panel>

            <Panel title="Episode reward distribution">
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={rlHist.map((h, i) => ({ ...h, baselineCount: baselineHist[i]?.count ?? 0 }))} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
                  <CartesianGrid stroke="#232b31" strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="bin" tick={{ ...AXIS_STYLE, fontSize: 9 }} interval={2} />
                  <YAxis tick={AXIS_STYLE} />
                  <Tooltip content={<TooltipBox />} cursor={{ fill: 'rgba(255,176,32,0.06)' }} />
                  <Legend wrapperStyle={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 11 }} />
                  <Bar dataKey="baselineCount" name="Baseline" fill="#8b96a0" fillOpacity={0.7} />
                  <Bar dataKey="count" name="RL agent" fill="#2dd4bf" fillOpacity={0.8} />
                </BarChart>
              </ResponsiveContainer>
            </Panel>
          </div>
        </>
      )}
    </div>
  )
}
