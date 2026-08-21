import { useRef, useState, type ChangeEvent } from 'react'
import { Loader2, Play, Upload } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Callout, PageHeader, Section, Stat } from '@/components/Primitives'
import { api } from '@/lib/api'
import { fmt } from '@/lib/format'
import { solverColor } from '@/lib/palette'
import type { SolveResult } from '@/lib/types'

const SAMPLES: { id: string; label: string; cnf: string }[] = [
  {
    id: 'tiny-sat',
    label: 'Tiny satisfiable (3 vars)',
    cnf: `c tiny satisfiable example
p cnf 3 3
1 -2 0
2 3 0
-1 -3 0
`,
  },
  {
    id: 'tiny-unsat',
    label: 'Tiny unsatisfiable (2 vars)',
    cnf: `c all four sign patterns on (1,2): UNSAT
p cnf 2 4
1 2 0
1 -2 0
-1 2 0
-1 -2 0
`,
  },
  {
    id: 'php-3-2',
    label: 'Pigeonhole 3 pigeons / 2 holes (UNSAT)',
    cnf: `c PHP(3,2): pigeon i in hole j is variable 2*(i-1)+j
p cnf 6 9
1 2 0
3 4 0
5 6 0
-1 -3 0
-1 -5 0
-3 -5 0
-2 -4 0
-2 -6 0
-4 -6 0
`,
  },
  {
    id: 'random-3sat',
    label: 'Random 3-SAT, 12 vars / 50 clauses',
    cnf: random3Sat(12, 50, 7),
  },
]

function random3Sat(n: number, m: number, seed: number): string {
  let s = seed >>> 0
  const rnd = () => {
    s = (s * 1664525 + 1013904223) >>> 0
    return s / 2 ** 32
  }
  const lines = [`c random 3-SAT n=${n} m=${m} seed=${seed}`, `p cnf ${n} ${m}`]
  for (let i = 0; i < m; i++) {
    const vars = new Set<number>()
    while (vars.size < 3) vars.add(1 + Math.floor(rnd() * n))
    lines.push([...vars].map((v) => (rnd() < 0.5 ? -v : v)).join(' ') + ' 0')
  }
  return lines.join('\n') + '\n'
}

export function Oracle() {
  const [cnf, setCnf] = useState(SAMPLES[0].cnf)
  const [fileName, setFileName] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<SolveResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const onFile = (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (!f) return
    f.text().then((t) => {
      setCnf(t)
      setFileName(f.name)
    })
  }

  const solve = async () => {
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      setResult(await api.solve(cnf))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const header = cnf.split('\n').find((l) => l.startsWith('p cnf'))

  return (
    <div>
      <PageHeader
        title="Oracle"
        lede="Submit a DIMACS CNF. The oracle parses it, extracts features, asks the trained selector which solver to run (or falls back to the pre-solver when its confidence is below θ), and reports where the time went."
      />
      <div className="grid gap-6 xl:grid-cols-5">
        <Section
          className="xl:col-span-3"
          title="Formula"
          description={header ? <code className="font-mono text-xs">{header}</code> : 'DIMACS CNF format: a "p cnf VARS CLAUSES" header, then clauses terminated by 0.'}
          right={
            <div className="flex items-center gap-2">
              <label className="sr-only" htmlFor="sample">
                Sample formula
              </label>
              <select
                id="sample"
                className="h-8 rounded-md border bg-card px-2 text-sm"
                defaultValue=""
                onChange={(e) => {
                  const s = SAMPLES.find((x) => x.id === e.target.value)
                  if (s) {
                    setCnf(s.cnf)
                    setFileName(null)
                    setResult(null)
                  }
                  e.target.value = ''
                }}
              >
                <option value="" disabled>
                  Load a sample…
                </option>
                {SAMPLES.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.label}
                  </option>
                ))}
              </select>
              <input ref={fileRef} type="file" accept=".cnf,.dimacs,.txt" className="hidden" onChange={onFile} />
              <Button size="sm" variant="outline" onClick={() => fileRef.current?.click()}>
                <Upload aria-hidden /> Upload
              </Button>
            </div>
          }
        >
          <label htmlFor="cnf" className="sr-only">
            CNF text
          </label>
          <textarea
            id="cnf"
            value={cnf}
            onChange={(e) => {
              setCnf(e.target.value)
              setFileName(null)
            }}
            spellCheck={false}
            className="h-80 w-full resize-y rounded-md border bg-background p-3 font-mono text-xs leading-relaxed"
          />
          <div className="mt-3 flex items-center justify-between gap-3">
            <p className="text-xs text-muted-foreground">
              {fileName ? `Loaded ${fileName} · ` : ''}
              {cnf.length.toLocaleString()} chars
            </p>
            <Button onClick={solve} disabled={busy || cnf.trim().length === 0}>
              {busy ? <Loader2 className="animate-spin" aria-hidden /> : <Play aria-hidden />}
              Solve
            </Button>
          </div>
        </Section>

        <div className="space-y-4 xl:col-span-2">
          {error && (
            <Callout tone="danger" title="Solve request failed">
              {error}
            </Callout>
          )}
          {!result && !error && (
            <Section title="Result" description="Nothing solved yet.">
              <p className="text-sm text-muted-foreground">Pick a sample or paste a formula, then press Solve.</p>
            </Section>
          )}
          {result && <ResultCard r={result} />}
        </div>
      </div>
    </div>
  )
}

function ResultCard({ r }: { r: SolveResult }) {
  const ok = r.status === 'ok' || r.status === 'solved' || r.satisfiable != null
  const verdict = r.satisfiable === true ? 'SAT' : r.satisfiable === false ? 'UNSAT' : r.status?.toUpperCase() ?? 'UNKNOWN'
  const verdictVariant = r.satisfiable === true ? 'success' : r.satisfiable === false ? 'info' : ok ? 'secondary' : 'destructive'
  const segments = [
    { key: 'parse_ms', label: 'parse', color: 'hsl(var(--chart-6))' },
    { key: 'feature_ms', label: 'features', color: 'hsl(var(--chart-5))' },
    { key: 'infer_ms', label: 'inference', color: 'hsl(var(--chart-4))' },
    { key: 'solve_ms', label: 'solve', color: 'hsl(var(--chart-1))' },
  ].map((s) => ({ ...s, value: Number(r[s.key] ?? 0) }))
  const total = segments.reduce((a, s) => a + s.value, 0) || Number(r.runtime_ms ?? 0) || 1

  return (
    <Section
      title="Result"
      description={r.fallback_used ? 'Selector confidence was below θ; fell back to the pre-solver.' : 'Selector chose the solver directly.'}
      right={<Badge variant={verdictVariant}>{verdict}</Badge>}
    >
      {r.error && (
        <Callout tone="danger" className="mb-4">
          {r.error}
        </Callout>
      )}
      <div className="grid grid-cols-2 gap-3">
        <Stat
          label="Solver used"
          value={
            <span className="flex items-center gap-2 font-mono text-base">
              <span className="inline-block size-3 rounded-sm" style={{ background: solverColor(String(r.solver_used ?? '')) }} />
              {r.solver_used ?? '—'}
            </span>
          }
          sub={r.fallback_used ? `fallback: ${r.fallback_reason ?? 'low confidence'}` : 'selected by model'}
        />
        <Stat label="Total time" value={fmt.ms(r.runtime_ms ?? total)} sub={`solve ${fmt.ms(r.solve_ms)}`} />
        <Stat label="Confidence" value={fmt.pct(r.confidence, 0)} sub={r.theta_used != null ? `θ = ${fmt.num(r.theta_used, 2)}` : undefined} tone={r.fallback_used ? 'warning' : 'default'} />
        <Stat label="Feature overhead" value={fmt.ms(r.feature_ms)} sub={`inference ${fmt.ms(r.infer_ms)}`} />
      </div>

      <div className="mt-5">
        <p className="mb-1.5 text-xs font-medium uppercase tracking-wide text-muted-foreground">Time breakdown</p>
        <div className="flex h-4 w-full overflow-hidden rounded-md bg-muted" role="img" aria-label={segments.map((s) => `${s.label} ${fmt.ms(s.value)}`).join(', ')}>
          {segments.map((s) => (
            <span key={s.key} style={{ width: `${(s.value / total) * 100}%`, background: s.color }} title={`${s.label}: ${fmt.ms(s.value)}`} />
          ))}
        </div>
        <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs">
          {segments.map((s) => (
            <li key={s.key} className="flex items-center gap-1.5">
              <span className="inline-block size-2.5 rounded-sm" style={{ background: s.color }} />
              {s.label} <span className="tabular text-muted-foreground">{fmt.ms(s.value)}</span>
            </li>
          ))}
        </ul>
      </div>

      {Array.isArray(r.assignment) && r.assignment.length > 0 && (
        <details className="mt-4">
          <summary className="cursor-pointer text-sm font-medium">Satisfying assignment ({r.assignment.length} literals)</summary>
          <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap break-all rounded-md bg-muted/60 p-3 font-mono text-xs">{r.assignment.join(' ')}</pre>
        </details>
      )}
    </Section>
  )
}
