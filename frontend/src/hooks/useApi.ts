import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError } from '@/lib/api'

export interface ApiState<T> {
  data: T | null
  loading: boolean
  error: string | null
  notFound: boolean
  reload: () => void
}

/**
 * Fetch once (and on demand). Distinguishes 404 (artifact not produced yet)
 * from real errors so views can show a "run step X first" empty state.
 * `pollMs` re-fetches silently (no loading flicker) at the given interval.
 */
export function useApi<T>(fetcher: () => Promise<T>, pollMs = 0): ApiState<T> {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [notFound, setNotFound] = useState(false)
  const fetcherRef = useRef(fetcher)
  useEffect(() => {
    fetcherRef.current = fetcher
  })
  const aliveRef = useRef(true)

  const load = useCallback(async () => {
    try {
      const d = await fetcherRef.current()
      if (!aliveRef.current) return
      setData(d)
      setError(null)
      setNotFound(false)
    } catch (e: unknown) {
      if (!aliveRef.current) return
      if (e instanceof ApiError && e.isNotFound) {
        setNotFound(true)
        setData(null)
        setError(null)
      } else {
        setError(e instanceof Error ? e.message : String(e))
      }
    } finally {
      if (aliveRef.current) setLoading(false)
    }
  }, [])

  useEffect(() => {
    aliveRef.current = true
    void load()
    const id = pollMs > 0 ? window.setInterval(() => void load(), pollMs) : undefined
    return () => {
      aliveRef.current = false
      if (id) window.clearInterval(id)
    }
  }, [load, pollMs])

  const reload = useCallback(() => {
    setLoading(true)
    void load()
  }, [load])

  return { data, loading, error, notFound, reload }
}
