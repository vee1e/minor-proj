import type {
  DatasetStats,
  Diagnostics,
  EvaluateResponse,
  LogsResponse,
  ProfilingStats,
  SolveResult,
  StatusResponse,
  StepName,
} from './types'

// Dev server proxies /api to 127.0.0.1:5001 (see vite.config.ts); production is
// served from the same origin by web_server.py.
export const API_BASE = ''

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
  get isNotFound() {
    return this.status === 404
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { Accept: 'application/json', ...(init?.headers ?? {}) },
    })
  } catch (e) {
    throw new ApiError(0, `Backend unreachable (${(e as Error).message}). Is web_server.py running on port 5001?`)
  }
  const text = await res.text()
  let body: unknown = null
  if (text) {
    try {
      body = JSON.parse(text)
    } catch {
      body = { error: text.slice(0, 300) }
    }
  }
  if (!res.ok) {
    const fromBody =
      body && typeof body === 'object' && 'error' in body ? String((body as { error: unknown }).error) : ''
    throw new ApiError(res.status, fromBody || `${res.status} ${res.statusText}`)
  }
  return body as T
}

function post<T>(path: string, payload: unknown) {
  return request<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export const api = {
  status: () => request<StatusResponse>('/api/status'),
  logs: (step: StepName) => request<LogsResponse>(`/api/logs/${step}`),
  runStep: (step: StepName, args: string[] = []) =>
    post<{ ok?: boolean; status?: string; error?: string }>('/api/run-step', { step, args }),
  profiling: () => request<ProfilingStats>('/api/profiling-stats'),
  dataset: () => request<DatasetStats>('/api/dataset-stats'),
  evaluate: () => request<EvaluateResponse>('/api/evaluate-stats'),
  diagnostics: () => request<Diagnostics>('/api/diagnostics'),
  solve: (cnf: string) => post<SolveResult>('/api/solve', { cnf }),
}
