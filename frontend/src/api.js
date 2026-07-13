const BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api'

async function request(path, options) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`${res.status} ${res.statusText}: ${text}`)
  }
  return res.json()
}

export const api = {
  health: () => request('/health'),
  predictions: (target) => request(`/predictions/${target}`),
  allMetrics: () => request('/predictions'),
  rlPerformance: () => request('/rl/performance'),
  rlCompare: () => request('/rl/compare'),
  simulate: (payload) =>
    request('/simulate', { method: 'POST', body: JSON.stringify(payload) }),
}
