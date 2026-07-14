import { useEffect, useState } from 'react'
import {
  ScatterChart, Scatter, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Cell,
} from 'recharts'
import { api } from '../api.js'
import { Panel, Stat, Loading, ErrorBox, Pill } from '../components/ui.jsx'

const TARGETS = [
  { key: 'sales', label: 'Sales', unit: '$' },
  { key: 'delivery', label: 'Delivery Time', unit: 'days' },
  { key: 'profit', label: 'Profit', unit: '$' },
]

const AXIS_STYLE = { fill: '#8b96a0', fontSize: 10.5, fontFamily: 'IBM Plex Mono, monospace' }

function TooltipBox({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-[var(--color-surface-raised)] border border-[var(--color-border)] rounded-md px-3 py-2 mono text-[11px]">
      {label !== undefined && <p className="text-[var(--color-muted)] mb-1">{label}</p>}
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color || p.fill }}>{p.name}: {typeof p.value === 'number' ? p.value.toFixed(2) : p.value}</p>
      ))}
    </div>
  )
}

export default function Predictions() {
  const [target, setTarget] = useState('sales')
  const [data, setData] = useState({})
  const [error, setError] = useState(null)

  useEffect(() => {
    if (data[target]) return
    api.predictions(target)
      .then((d) => setData((prev) => ({ ...prev, [target]: d })))
      .catch((e) => setError(e.message))
  }, [target])

  const d = data[target]
  const meta = TARGETS.find((t) => t.key === target)

  const histData = d
    ? d.error_histogram.bin_edges.slice(0, -1).map((edge, i) => ({
        bin: ((edge + d.error_histogram.bin_edges[i + 1]) / 2).toFixed(1),
        count: d.error_histogram.counts[i],
      }))
    : []

  const scatterData = d
    ? d.sample.actual.map((a, i) => ({ actual: a, predicted: d.sample.predicted[i] }))
    : []

  const scatterRange = scatterData.length
    ? [Math.min(...scatterData.map((p) => p.actual), ...scatterData.map((p) => p.predicted)),
       Math.max(...scatterData.map((p) => p.actual), ...scatterData.map((p) => p.predicted))]
    : [0, 1]

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="mono text-[11px] uppercase tracking-widest text-[var(--color-amber)] mb-2">Predictions</p>
          <h1 className="text-2xl font-semibold tracking-tight">CatBoost model diagnostics</h1>
        </div>
        <div className="flex gap-1.5 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg p-1 self-start">
          {TARGETS.map((t) => (
            <button
              key={t.key}
              onClick={() => setTarget(t.key)}
              className={`mono text-[11.5px] px-3.5 py-1.5 rounded-md transition-colors ${
                target === t.key ? 'bg-[var(--color-amber)] text-[#1a1305] font-semibold' : 'text-[var(--color-muted)] hover:text-[var(--color-text)]'
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {error && <ErrorBox message={error} />}
      {!d && !error && <Loading label={`Loading ${meta.label} model`} />}

      {d && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <Panel><Stat label="RMSE" value={d.metrics.rmse.toFixed(2)} unit={meta.unit} accent="amber" /></Panel>
            <Panel><Stat label="MAE" value={d.metrics.mae.toFixed(2)} unit={meta.unit} accent="amber" /></Panel>
            <Panel>
              <Stat label="R²" value={d.metrics.r2.toFixed(4)} accent={d.metrics.r2 > 0.3 ? 'teal' : 'red'} />
            </Panel>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Panel title="Actual vs. Predicted" subtitle={`${d.n_test.toLocaleString()} held-out rows · ${scatterData.length} sampled`}>
              <ResponsiveContainer width="100%" height={280}>
                <ScatterChart margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
                  <CartesianGrid stroke="#232b31" strokeDasharray="3 3" />
                  <XAxis type="number" dataKey="actual" name="Actual" tick={AXIS_STYLE} domain={scatterRange} />
                  <YAxis type="number" dataKey="predicted" name="Predicted" tick={AXIS_STYLE} domain={scatterRange} />
                  <Tooltip content={<TooltipBox />} cursor={{ stroke: '#ffb020', strokeWidth: 1 }} />
                  <ReferenceLine segment={[{ x: scatterRange[0], y: scatterRange[0] }, { x: scatterRange[1], y: scatterRange[1] }]} stroke="#2dd4bf" strokeDasharray="4 4" />
                  <Scatter data={scatterData} fill="#ffb020" fillOpacity={0.55} />
                </ScatterChart>
              </ResponsiveContainer>
            </Panel>

            <Panel title="Prediction Error Distribution" subtitle="predicted − actual, held-out set">
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={histData} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
                  <CartesianGrid stroke="#232b31" strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="bin" tick={AXIS_STYLE} interval={Math.ceil(histData.length / 8)} />
                  <YAxis tick={AXIS_STYLE} />
                  <Tooltip content={<TooltipBox />} cursor={{ fill: 'rgba(255,176,32,0.06)' }} />
                  <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                    {histData.map((_, i) => <Cell key={i} fill="#2dd4bf" fillOpacity={0.75} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Panel>
          </div>

          <Panel title="Top Feature Importance" subtitle="CatBoost PredictionValuesChange">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={d.feature_importance} layout="vertical" margin={{ top: 10, right: 20, bottom: 0, left: 10 }}>
                <CartesianGrid stroke="#232b31" strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" tick={AXIS_STYLE} />
                <YAxis type="category" dataKey="feature" tick={{ ...AXIS_STYLE, fontSize: 10.5 }} width={170} />
                <Tooltip content={<TooltipBox />} cursor={{ fill: 'rgba(255,176,32,0.06)' }} />
                <Bar dataKey="importance" fill="#ffb020" fillOpacity={0.8} radius={[0, 3, 3, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Panel>

          {target === 'profit' && (
            <Pill tone="red">Honest result: profit per order shows near-zero R² — this dataset's profit field carries heavy unexplained noise, consistent with public analyses of this dataset.</Pill>
          )}
        </>
      )}
    </div>
  )
}
