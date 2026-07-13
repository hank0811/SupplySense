import { useEffect, useMemo, useState } from 'react'
import {
  LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts'
import { api } from '../api.js'
import { Panel, Stat, Loading, ErrorBox } from '../components/ui.jsx'

const AXIS_STYLE = { fill: '#8b96a0', fontSize: 10.5, fontFamily: 'IBM Plex Mono, monospace' }

function TooltipBox({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-[var(--color-surface-raised)] border border-[var(--color-border)] rounded-md px-3 py-2 mono text-[11px]">
      <p className="text-[var(--color-muted)] mb-1">update {label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color }}>{p.name}: {p.value.toFixed(1)}</p>
      ))}
    </div>
  )
}

export default function RLPerformance() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.rlPerformance().then(setData).catch((e) => setError(e.message))
  }, [])

  const chartData = useMemo(() => {
    if (!data) return []
    let cumulative = 0
    return data.history.map((h) => {
      cumulative += h.mean_episode_reward
      return { ...h, cumulative }
    })
  }, [data])

  const first = data?.history[0]
  const last = data?.history[data.history.length - 1]
  const best = data ? Math.max(...data.history.map((h) => h.rolling_mean_reward)) : null

  return (
    <div className="space-y-6">
      <div>
        <p className="mono text-[11px] uppercase tracking-widest text-[var(--color-amber)] mb-2">RL Performance</p>
        <h1 className="text-2xl font-semibold tracking-tight">PPO training curve</h1>
        <p className="text-[var(--color-muted)] text-[14.5px] mt-2 max-w-2xl">
          A from-scratch NumPy PPO agent (clipped surrogate objective + GAE), trained against the inventory
          reorder environment. No Stable-Baselines3 / PyTorch — this sandbox couldn't reach a CPU-only torch
          wheel, so the same algorithm was implemented directly.
        </p>
      </div>

      {error && <ErrorBox message={error} />}
      {!data && !error && <Loading label="Loading training history" />}

      {data && (
        <>
          <div className="grid grid-cols-3 gap-4">
            <Panel><Stat label="Starting reward" value={first.mean_episode_reward.toFixed(0)} accent="red" hint="update 0" /></Panel>
            <Panel><Stat label="Best rolling mean" value={best.toFixed(0)} accent="teal" hint={`of ${data.history.length} updates`} /></Panel>
            <Panel><Stat label="Final reward" value={last.mean_episode_reward.toFixed(0)} accent="amber" hint={`rolling ${last.rolling_mean_reward.toFixed(0)}`} /></Panel>
          </div>

          <Panel title="Reward per update" subtitle={`${data.episodes_per_update} episodes/update · ${data.episode_length}-step episodes`}>
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={chartData} margin={{ top: 10, right: 20, bottom: 0, left: 0 }}>
                <CartesianGrid stroke="#232b31" strokeDasharray="3 3" />
                <XAxis dataKey="update" tick={AXIS_STYLE} />
                <YAxis tick={AXIS_STYLE} />
                <Tooltip content={<TooltipBox />} />
                <Legend wrapperStyle={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 11 }} />
                <Line type="monotone" dataKey="mean_episode_reward" name="raw reward" stroke="#8b96a0" strokeWidth={1} dot={false} opacity={0.5} />
                <Line type="monotone" dataKey="rolling_mean_reward" name="rolling mean (10)" stroke="#ffb020" strokeWidth={2.2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </Panel>

          <Panel title="Cumulative reward" subtitle="running sum of mean episode reward across training">
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={chartData} margin={{ top: 10, right: 20, bottom: 0, left: 0 }}>
                <defs>
                  <linearGradient id="cumFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#2dd4bf" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="#2dd4bf" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#232b31" strokeDasharray="3 3" />
                <XAxis dataKey="update" tick={AXIS_STYLE} />
                <YAxis tick={AXIS_STYLE} />
                <Tooltip content={<TooltipBox />} />
                <Area type="monotone" dataKey="cumulative" name="cumulative reward" stroke="#2dd4bf" strokeWidth={2} fill="url(#cumFill)" />
              </AreaChart>
            </ResponsiveContainer>
          </Panel>
        </>
      )}
    </div>
  )
}
