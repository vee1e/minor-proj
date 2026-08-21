import { useEffect, useMemo, useRef, useState } from 'react'
import { Check, CircleDashed, Loader2, Play, ScrollText, X, XCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Async, Callout, PageHeader, Section, Stat, StatGrid } from '@/components/Primitives'
import { usePipeline, useStepLogs } from '@/hooks/usePipeline'
import { useApi } from '@/hooks/useApi'
import { api } from '@/lib/api'
import { PIPELINE_STEPS, type StepName } from '@/lib/types'
import { fmt } from '@/lib/format'
import { cn } from '@/lib/utils'

type StepState = 'idle' | 'done' | 'running' | 'failed'

export function Overview() {
  const pipe = usePipeline()
  const [logStep, setLogStep] = useState<StepName | null>(null)
  const [smallCorpus, setSmallCorpus] = useState(true)

  const stateOf = (s: StepName): StepState => {
    const job = pipe.data.running_jobs?.[s]
    if (job?.status === 'running') return 'running'
    if (job && (job.status === 'failed' || job.status === 'error' || (job.exit_code ?? 0) !== 0)) return 'failed'
    if (pipe.data.pipeline_completed?.[s]) return 'done'
    return 'idle'
  }

  const firstIncomplete = PIPELINE_STEPS.find((s) => stateOf(s.id) !== 'done')
  const profiling = useApi(api.profiling)

  return (
    <div>
      <PageHeader
        title="Pipeline"
        lede="Run the stages in order. Each stage writes artifacts the next one reads; the Data, Results and Diagnostics pages light up as those artifacts appear."
        actions={
          <label className="flex items-center gap-2 text-sm text-muted-foreground">
            <input
              type="checkbox"
              className="size-4 accent-[hsl(var(--primary))]"
              checked={smallCorpus}
              onChange={(e) => setSmallCorpus(e.target.checked)}
            />
            pass <code className="rounded bg-muted px-1 font-mono text-xs">--small</code> to bench
          </label>
        }
      />

      {pipe.error && (
        <Callout tone="danger" title="Backend unreachable" className="mb-4">
          {pipe.error}
        </Callout>
      )}
      {pipe.launchError && (
        <Callout tone="danger" title="Could not start step" className="mb-4">
          {pipe.launchError}
        </Callout>
      )}

      <Section
        title="Stages"
        description={
          firstIncomplete ? (
            <>
              Next up: <strong className="text-foreground">{firstIncomplete.label}</strong> — {firstIncomplete.blurb}.
            </>
          ) : (
            'All stages complete.'
          )
        }
      >
        <ol className="grid gap-3 md:grid-cols-2 xl:grid-cols-7">
          {PIPELINE_STEPS.map((s, i) => {
            const st = stateOf(s.id)
            const job = pipe.data.running_jobs?.[s.id]
            const busy = st === 'running' || pipe.launching === s.id
            return (
              <li
                key={s.id}
                className={cn(
                  'flex flex-col rounded-lg border p-3',
                  st === 'done' && 'border-success/40 bg-success/5',
                  st === 'running' && 'border-primary/50 bg-accent/60',
                  st === 'failed' && 'border-destructive/40 bg-destructive/5',
                )}
              >
                <div className="flex items-center gap-2">
                  <StepIcon state={st} index={i} />
                  <span className="font-medium">{s.label}</span>
                </div>
                <p className="mt-1 flex-1 text-xs text-muted-foreground">{s.blurb}</p>
                {st === 'failed' && job?.error && (
                  <p className="mt-2 line-clamp-3 break-words font-mono text-[11px] text-destructive">{job.error}</p>
                )}
                <div className="mt-3 flex gap-1.5">
                  <Button
                    size="sm"
                    variant={st === 'done' ? 'outline' : 'default'}
                    disabled={busy || pipe.runningSteps.length > 0}
                    onClick={() => pipe.run(s.id, s.id === 'bench' && smallCorpus ? ['--small'] : [])}
                    aria-label={`Run ${s.label}`}
                  >
                    {busy ? <Loader2 className="animate-spin" aria-hidden /> : <Play aria-hidden />}
                    {st === 'done' ? 'Re-run' : 'Run'}
                  </Button>
                  <Button size="sm" variant="ghost" onClick={() => setLogStep(s.id)} aria-label={`Show ${s.label} logs`}>
                    <ScrollText aria-hidden />
                    Logs
                  </Button>
                </div>
              </li>
            )
          })}
        </ol>
      </Section>

      <div className="mt-6">
        <Section
          title="Profiling snapshot"
          description="Per-solver aggregate from the most recent profile stage. Average runtime here is wall-clock solve time only."
        >
          <Async
            state={profiling}
            emptyTitle="No profiling data yet"
            emptyHint="Run the Profile stage to measure every solver on every formula."
          >
            {(p) => (
              <>
                <StatGrid>
                  <Stat label="Formulas" value={fmt.int(p.total_formulas)} />
                  <Stat label="Solver runs" value={fmt.int(p.total_runs)} />
                  <Stat label="Families" value={fmt.int(p.families?.length ?? 0)} />
                  <Stat
                    label="Timeouts"
                    value={fmt.int((p.families ?? []).reduce((a, f) => a + (f.timeout ?? 0), 0))}
                    sub="across all families"
                  />
                </StatGrid>
                {p.solvers?.length ? (
                  <div className="mt-4 overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead className="text-left text-xs uppercase tracking-wide text-muted-foreground">
                        <tr className="border-b">
                          <th className="py-2 pr-4 font-medium">Solver</th>
                          <th className="py-2 pr-4 font-medium">Runs</th>
                          <th className="py-2 pr-4 font-medium">Solved</th>
                          <th className="py-2 pr-4 font-medium">Avg time (s)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {p.solvers.map((s) => (
                          <tr key={s.solver} className="border-b last:border-0">
                            <td className="py-2 pr-4 font-mono text-[13px]">{s.solver}</td>
                            <td className="tabular py-2 pr-4">{fmt.int(s.runs)}</td>
                            <td className="tabular py-2 pr-4">{fmt.int(s.solved)}</td>
                            <td className="tabular py-2 pr-4">{fmt.num(s.avg_time, 4)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : null}
              </>
            )}
          </Async>
        </Section>
      </div>

      {logStep && (
        <LogDrawer step={logStep} live={pipe.runningSteps.includes(logStep)} onClose={() => setLogStep(null)} />
      )}
    </div>
  )
}

function StepIcon({ state, index }: { state: StepState; index: number }) {
  const base = 'flex size-6 shrink-0 items-center justify-center rounded-full text-xs font-semibold'
  if (state === 'done')
    return (
      <span className={cn(base, 'bg-success text-success-foreground')} aria-label="complete">
        <Check className="size-3.5" />
      </span>
    )
  if (state === 'running')
    return (
      <span className={cn(base, 'bg-primary text-primary-foreground')} aria-label="running">
        <Loader2 className="size-3.5 animate-spin" />
      </span>
    )
  if (state === 'failed')
    return (
      <span className={cn(base, 'bg-destructive text-destructive-foreground')} aria-label="failed">
        <XCircle className="size-3.5" />
      </span>
    )
  return (
    <span className={cn(base, 'border bg-muted text-muted-foreground')} aria-label="not run">
      {index + 1}
    </span>
  )
}

function LogDrawer({ step, live, onClose }: { step: StepName; live: boolean; onClose: () => void }) {
  const { logs, total, error } = useStepLogs(step, live)
  const ref = useRef<HTMLPreElement>(null)
  const label = useMemo(() => PIPELINE_STEPS.find((s) => s.id === step)?.label ?? step, [step])
  useEffect(() => {
    if (live && ref.current) ref.current.scrollTop = ref.current.scrollHeight
  }, [logs, live])
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <aside
      className="fixed inset-x-0 bottom-0 z-40 flex h-[45vh] flex-col border-t bg-card shadow-2xl"
      role="dialog"
      aria-label={`${label} logs`}
    >
      <div className="flex items-center justify-between gap-3 border-b px-4 py-2">
        <div className="flex items-center gap-2 text-sm">
          <ScrollText className="size-4 text-muted-foreground" aria-hidden />
          <span className="font-medium">{label} log</span>
          {live ? <Badge variant="info">live</Badge> : <Badge variant="secondary">{fmt.int(total)} lines</Badge>}
        </div>
        <Button size="icon-sm" variant="ghost" onClick={onClose} aria-label="Close logs">
          <X />
        </Button>
      </div>
      <pre
        ref={ref}
        className="flex-1 overflow-auto whitespace-pre-wrap break-words bg-background/60 p-4 font-mono text-xs leading-relaxed"
      >
        {error ? (
          <span className="text-destructive">{error}</span>
        ) : logs ? (
          logs
        ) : (
          <span className="flex items-center gap-2 text-muted-foreground">
            <CircleDashed className="size-3.5" /> No output yet for this step.
          </span>
        )}
      </pre>
    </aside>
  )
}
