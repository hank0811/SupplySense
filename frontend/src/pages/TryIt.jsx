import { useEffect, useState, useCallback } from 'react'
import { api } from '../api.js'
import { Panel, Stat, Loading, ErrorBox, Pill } from '../components/ui.jsx'

function Slider({ label, value, onChange, min, max, step, unit }) {
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1.5">
        <label className="mono text-[11px] uppercase tracking-wider text-[var(--color-muted)]">{label}</label>
        <span className="mono text-sm text-[var(--color-amber)] tabular">{value}{unit}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-[#ffb020] h-1.5 cursor-pointer"
      />
      <div className="flex justify-between mono text-[10px] text-[var(--color-muted)]/60 mt-1">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  )
}

export default function TryIt() {
  const [inventory, setInventory] = useState(10)
  const [demand, setDemand] = useState(4)
  const [leadTime, setLeadTime] = useState(3)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  const run = useCallback(() => {
    setLoading(true)
    setError(null)
    api.simulate({ inventory, pending_demand: demand, lead_time_days: leadTime })
      .then(setResult)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [inventory, demand, leadTime])

  useEffect(() => {
    const t = setTimeout(run, 250) // debounce slider drags
    return () => clearTimeout(t)
  }, [run])

  const netPositive = result && result.projected_outcome.net_reward >= 0

  return (
    <div className="space-y-6">
      <div>
        <p className="mono text-[11px] uppercase tracking-widest text-[var(--color-amber)] mb-2">Try It</p>
        <h1 className="text-2xl font-semibold tracking-tight">Live reorder decision</h1>
        <p className="text-[var(--color-muted)] text-[14.5px] mt-2 max-w-2xl">
          Move the sliders to describe a scenario. Each change calls the trained PPO policy directly
          (no retraining) and shows the reorder quantity it decides on, plus the projected one-step outcome.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-5">
        <Panel className="lg:col-span-2" title="Scenario inputs">
          <div className="space-y-6">
            <Slider label="Current inventory" value={inventory} onChange={setInventory} min={0} max={100} step={1} unit=" units" />
            <Slider label="Pending demand" value={demand} onChange={setDemand} min={0} max={20} step={1} unit=" units" />
            <Slider label="Scheduled lead time" value={leadTime} onChange={setLeadTime} min={0} max={15} step={1} unit=" days" />
          </div>
        </Panel>

        <Panel className="lg:col-span-3" title="Agent decision" subtitle="PPO policy mean action (deterministic)">
          {loading && !result && <Loading label="Querying policy" />}
          {error && <ErrorBox message={error} />}
          {result && (
            <div className="space-y-5">
              <div className="flex flex-wrap items-end gap-4">
                <div>
                  <p className="mono text-[10.5px] uppercase tracking-wider text-[var(--color-muted)]">Reorder quantity</p>
                  <p className="mono text-4xl sm:text-5xl font-semibold text-[var(--color-amber)] tabular mt-1">
                    {result.reorder_qty}
                    <span className="text-lg text-[var(--color-muted)] ml-2">units</span>
                  </p>
                </div>
                <Pill tone={netPositive ? 'teal' : 'red'}>
                  {netPositive ? 'net positive' : 'net negative'} projected step
                </Pill>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 pt-2 border-t border-[var(--color-border)]">
                <Stat label="Units sold" value={result.projected_outcome.units_sold} />
                <Stat label="Stockout" value={result.projected_outcome.stockout_units} accent={result.projected_outcome.stockout_units > 0 ? 'red' : 'text'} />
                <Stat label="Leftover" value={result.projected_outcome.leftover_inventory} />
                <Stat label="Revenue" value={`$${result.projected_outcome.revenue}`} accent="teal" />
                <Stat label="Holding + stockout cost" value={`$${(result.projected_outcome.holding_cost + result.projected_outcome.stockout_penalty).toFixed(2)}`} />
                <Stat label="Reorder cost" value={`$${result.projected_outcome.reorder_cost}`} />
              </div>

              <div className="pt-3 border-t border-[var(--color-border)] flex items-baseline justify-between">
                <span className="mono text-[11px] uppercase tracking-wider text-[var(--color-muted)]">Net reward (this step)</span>
                <span className={`mono text-xl font-semibold tabular ${netPositive ? 'text-[var(--color-teal)]' : 'text-[var(--color-red)]'}`}>
                  ${result.projected_outcome.net_reward}
                </span>
              </div>
              <p className="mono text-[10px] text-[var(--color-muted)]">
                assumes ${result.assumed_unit_price}/unit — the dataset-average product price
              </p>
            </div>
          )}
        </Panel>
      </div>
    </div>
  )
}
