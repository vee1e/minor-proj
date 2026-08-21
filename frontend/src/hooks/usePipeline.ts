import { useCallback, useEffect, useMemo, useState } from 'react'
import { api } from '@/lib/api'
import type { StatusResponse, StepName } from '@/lib/types'
import { useApi } from './useApi'

const EMPTY: StatusResponse = { pipeline_completed: {}, running_jobs: {} }

export function usePipeline() {
  const status = useApi(api.status, 2500)
  const data = status.data ?? EMPTY
  const runningSteps = useMemo(
    () =>
      Object.entries(data.running_jobs ?? {})
        .filter(([, j]) => j?.status === 'running')
        .map(([s]) => s as StepName),
    [data.running_jobs],
  )
  const [launching, setLaunching] = useState<StepName | null>(null)
  const [launchError, setLaunchError] = useState<string | null>(null)
  const reload = status.reload

  const run = useCallback(
    async (step: StepName, args: string[] = []) => {
      setLaunching(step)
      setLaunchError(null)
      try {
        const r = await api.runStep(step, args)
        if (r && r.error) setLaunchError(r.error)
      } catch (e) {
        setLaunchError(e instanceof Error ? e.message : String(e))
      } finally {
        setLaunching(null)
        reload()
      }
    },
    [reload],
  )

  return { ...status, data, runningSteps, run, launching, launchError }
}

/** Poll a step's log while it is running; fetch once otherwise. */
export function useStepLogs(step: StepName | null, live: boolean) {
  const [logs, setLogs] = useState('')
  const [total, setTotal] = useState(0)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    if (!step) return
    let alive = true
    const pull = () =>
      api
        .logs(step)
        .then((r) => {
          if (!alive) return
          setLogs(r.logs ?? '')
          setTotal(r.total_lines ?? 0)
          setError(null)
        })
        .catch((e: unknown) => {
          if (alive) setError(e instanceof Error ? e.message : String(e))
        })
    pull()
    const id = live ? window.setInterval(pull, 1500) : undefined
    return () => {
      alive = false
      if (id) window.clearInterval(id)
    }
  }, [step, live])
  return { logs, total, error }
}
