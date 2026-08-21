import { useMemo, useState } from 'react'
import { Async, Callout, PageHeader, Section, Stat, StatGrid, WarningList } from '@/components/Primitives'
import { CategoryBars, StackedSolverBars } from '@/components/charts/Charts'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { useApi } from '@/hooks/useApi'
import { api } from '@/lib/api'
import { fmt } from '@/lib/format'
import { systemColor, SOLVERS } from '@/lib/palette'
import type { Comparison, EvaluateResults } from '@/lib/types'

export function Results() {
  const ev = useApi(api.evaluate)
  return (
    <div>
      <PageHeader
        title="Results"
        lede="Test-set performance of the learned selectors against the baselines. Lower PAR-2 is better; VBS (the oracle) is the floor and the single best solver is the bar any selector must clear."
      />
      <Async
        state={ev}
        emptyTitle="No evaluation yet"
        emptyHint="Run Train (and optionally Train GNN), then Evaluate. This page reads the evaluation results and report."
      >
        {(r) => <ResultsBody res={r.results ?? {}} report={r.report_md} />}
      </Async>
    </div>
  )
}

function isSelector(system: string) {
  const s = system.toLowerCase()
  return !(s.includes('vbs') || s.includes('oracle') || s.includes('random') || s.includes('single') || s.includes('sbs') || s.includes('best_avg') || s.includes('satzilla'))
}

function ResultsBody({ res, report }: { res: EvaluateResults; report?: string }) {
  const systems = useMemo(() => [...(res.systems ?? [])].sort((a, b) => a.par2 - b.par2), [res.systems])
  const vbs = systems.find((s) => /vbs|oracle/i.test(s.system))
  const sbs = systems.find((s) => /single|sbs|best_avg|best average/i.test(s.system))
  const chosen = res.chosen_selector ? systems.find((s) => s.system === res.chosen_selector) : systems.find((s) => isSelector(s.system))
  const comparisons = res.comparisons ?? []
  const choiceHist = res.per_system_choice_hist ?? {}
  const choiceRows = Object.entries(choiceHist).map(([system, h]) => ({ name: system, ...h }))
  const choiceSolvers = Array.from(new Set(Object.values(choiceHist).flatMap((h) => Object.keys(h))))
  choiceSolvers.sort((a, b) => (SOLVERS as readonly string[]).indexOf(a) - (SOLVERS as readonly string[]).indexOf(b))
  const perFamily = res.per_family ?? {}
  const famSystems = Array.from(new Set(Object.values(perFamily).flatMap((m) => Object.keys(m))))
  const ablation = res.feature_ablation ?? []
  const cls = res.classification ?? {}
  const [showReport, setShowReport] = useState(false)

  const randomMean = Array.isArray(res.random_mean_sd) ? res.random_mean_sd[0] : res.random_mean_sd?.mean
  const randomSd = Array.isArray(res.random_mean_sd) ? res.random_mean_sd[1] : res.random_mean_sd?.sd

  const ood = (res.ood ?? {}) as Record<string, unknown>
  const oodEntries: [string, number][] = res.ood_par2
    ? Object.entries(res.ood_par2)
    : Object.entries(ood).filter(([, v]) => typeof v === 'number').map(([k, v]) => [k, v as number])

  const collapsed = (() => {
    if (!chosen) return null
    const h = choiceHist[chosen.system]
    if (!h) return null
    const total = Object.values(h).reduce((a, b) => a + b, 0) || 1
    const [top, n] = Object.entries(h).sort((a, b) => b[1] - a[1])[0] ?? []
    return top && n / total > 0.9 ? { top, share: n / total } : null
  })()

  return (
    <div className="space-y-6">
      <WarningList warnings={res.data_quality_warnings} title="Data-quality warnings carried into evaluation" />
      {collapsed && chosen && (
        <Callout tone="warning" title={`${chosen.system} picks ${collapsed.top} on ${fmt.pct(collapsed.share, 0)} of test formulas`}>
          The selector has effectively collapsed to a single solver. See the Dataset page for the label distribution and tie rate that drive this, and the Diagnostics page for per-model predicted-class histograms.
        </Callout>
      )}

      <StatGrid>
        <Stat label="Chosen selector" value={<span className="text-lg">{chosen?.system ?? '—'}</span>} sub={res.theta != null ? `θ = ${fmt.num(res.theta, 2)}` : undefined} />
        <Stat label="Selector PAR-2" value={fmt.num(chosen?.par2)} sub={`solve rate ${fmt.pct(chosen?.solve_rate)}`} />
        <Stat label="Single best PAR-2" value={fmt.num(sbs?.par2)} sub={res.best_average_solver ?? sbs?.system ?? ''} />
        <Stat label="VBS PAR-2" value={fmt.num(vbs?.par2)} sub={`solve rate ${fmt.pct(vbs?.solve_rate)}`} tone="success" />
        <Stat
          label="Oracle gap"
          value={fmt.num(res.oracle_gap_selector)}
          sub="selector PAR-2 − VBS PAR-2"
          tone={res.oracle_gap_selector != null && sbs && vbs && res.oracle_gap_selector > sbs.par2 - vbs.par2 ? 'warning' : 'default'}
        />
        <Stat label="Random baseline" value={fmt.num(randomMean)} sub={randomSd != null ? `± ${fmt.num(randomSd)} sd` : undefined} />
      </StatGrid>

      <div className="grid gap-6 xl:grid-cols-2">
        <Section title="PAR-2 by system" description="Penalised average runtime (timeouts count 2× the time limit), test set. Dashed line is the VBS floor.">
          {systems.length === 0 ? (
            <p className="text-sm text-muted-foreground">No systems reported.</p>
          ) : (
            <CategoryBars
              data={systems}
              xKey="system"
              yKey="par2"
              yLabel="PAR-2 (s)"
              layout="vertical"
              colorOf={systemColor}
              height={40 + systems.length * 36}
              reference={vbs ? { value: vbs.par2, label: 'VBS' } : undefined}
            />
          )}
        </Section>
        <Section title="Solve rate by system" description="Fraction of test formulas solved within the time limit.">
          {systems.length === 0 ? (
            <p className="text-sm text-muted-foreground">No systems reported.</p>
          ) : (
            <CategoryBars
              data={systems}
              xKey="system"
              yKey="solve_rate"
              yLabel="solve rate"
              layout="vertical"
              colorOf={systemColor}
              height={40 + systems.length * 36}
              format={(v) => fmt.pct(v, 0)}
            />
          )}
        </Section>
      </div>

      <Section
        title="Planned comparisons"
        description="Paired per-instance PAR-2 differences (A − B): negative favours A. 95% bootstrap CI, Wilcoxon signed-rank, McNemar on solved/unsolved, Holm-corrected verdict."
      >
        {comparisons.length === 0 ? (
          <p className="text-sm text-muted-foreground">No comparisons reported.</p>
        ) : (
          <ComparisonTable rows={comparisons} />
        )}
      </Section>

      <div className="grid gap-6 xl:grid-cols-2">
        <Section title="Which solver each system picks" description="Share of test formulas routed to each solver. A healthy selector spreads choices; a collapsed one is a single colour.">
          {choiceRows.length === 0 ? (
            <p className="text-sm text-muted-foreground">No choice histogram reported.</p>
          ) : (
            <StackedSolverBars rows={choiceRows} solvers={choiceSolvers} normalize height={40 + choiceRows.length * 40} />
          )}
        </Section>
        <Section title="Classification view" description="Accuracy of predicting the per-instance fastest solver, for systems that predict one.">
          {Object.keys(cls).length === 0 ? (
            <p className="text-sm text-muted-foreground">No classification metrics reported.</p>
          ) : (
            <KeyValueTable obj={cls} />
          )}
        </Section>
      </div>

      <Section title="Per-family PAR-2" description="How each system does within each family. Families marked OOD on the Dataset page were never seen in training.">
        {Object.keys(perFamily).length === 0 ? (
          <p className="text-sm text-muted-foreground">No per-family breakdown reported.</p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Family</TableHead>
                {famSystems.map((s) => (
                  <TableHead key={s} className="text-right">
                    {s}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {Object.entries(perFamily).map(([fam, m]) => {
                const best = Math.min(...famSystems.map((s) => m[s]?.par2 ?? Infinity))
                return (
                  <TableRow key={fam}>
                    <TableCell className="font-medium">{fam}</TableCell>
                    {famSystems.map((s) => {
                      const v = m[s]
                      const isBest = v?.par2 != null && v.par2 === best
                      return (
                        <TableCell key={s} className={`tabular text-right ${isBest ? 'font-semibold text-success' : ''}`}>
                          {fmt.num(v?.par2)}
                          <span className="ml-1 text-xs text-muted-foreground">{fmt.pct(v?.solve_rate, 0)}</span>
                        </TableCell>
                      )
                    })}
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        )}
      </Section>

      <div className="grid gap-6 xl:grid-cols-3">
        <Section className="xl:col-span-2" title="Feature-cost ablation" description="Pareto trade-off between feature extraction time and selector quality.">
          {ablation.length === 0 ? (
            <p className="text-sm text-muted-foreground">No ablation reported.</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Feature set</TableHead>
                  <TableHead className="text-right">PAR-2</TableHead>
                  <TableHead className="text-right">Solve rate</TableHead>
                  <TableHead className="text-right">Accuracy</TableHead>
                  <TableHead className="text-right">Feature time</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {ablation.map((a, i) => (
                  <TableRow key={i}>
                    <TableCell className="font-medium">{String(a.config ?? a.name ?? a.feature_set ?? `#${i + 1}`)}</TableCell>
                    <TableCell className="tabular text-right">{fmt.num(a.par2)}</TableCell>
                    <TableCell className="tabular text-right">{fmt.pct(a.solve_rate)}</TableCell>
                    <TableCell className="tabular text-right">{fmt.pct(a.accuracy)}</TableCell>
                    <TableCell className="tabular text-right">{fmt.ms(a.feature_ms ?? a.feature_overhead_ms)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </Section>
        <Section title="Out-of-distribution" description="PAR-2 on held-out families the models never saw.">
          {oodEntries.length === 0 ? (
            <p className="text-sm text-muted-foreground">No OOD evaluation reported.</p>
          ) : (
            <ul className="divide-y text-sm">
              {oodEntries.map(([k, v]) => (
                <li key={k} className="flex items-center justify-between py-2">
                  <span className="font-medium">{k}</span>
                  <span className="tabular">{fmt.num(v)}</span>
                </li>
              ))}
            </ul>
          )}
        </Section>
      </div>

      {report && (
        <Section
          title="Generated report"
          description="The Markdown report written by the evaluate step."
          right={
            <Button size="sm" variant="outline" onClick={() => setShowReport((s) => !s)}>
              {showReport ? 'Hide' : 'Show'}
            </Button>
          }
        >
          {showReport ? (
            <pre className="max-h-[32rem] overflow-auto whitespace-pre-wrap rounded-md bg-muted/60 p-4 font-mono text-xs leading-relaxed">{report}</pre>
          ) : (
            <p className="text-sm text-muted-foreground">{report.split('\n').length} lines.</p>
          )}
        </Section>
      )}
    </div>
  )
}

function ComparisonTable({ rows }: { rows: Comparison[] }) {
  const span = Math.max(1e-9, ...rows.flatMap((r) => [Math.abs(r.ci_low), Math.abs(r.ci_high), Math.abs(r.mean_diff)]))
  const pos = (v: number) => `${50 + (v / span) * 48}%`
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Comparison</TableHead>
          <TableHead className="text-right">Mean Δ PAR-2</TableHead>
          <TableHead className="w-[260px]">95% CI</TableHead>
          <TableHead className="text-right">Wilcoxon p</TableHead>
          <TableHead className="text-right">McNemar p</TableHead>
          <TableHead>Holm</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {rows.map((r, i) => {
          const lo = Math.min(r.ci_low, r.ci_high)
          const hi = Math.max(r.ci_low, r.ci_high)
          const excl = r.ci_excludes_zero ?? (lo > 0 || hi < 0)
          return (
            <TableRow key={i}>
              <TableCell>
                <div className="font-medium">{r.comparison || `${r.system_a} vs ${r.system_b}`}</div>
                <div className="text-xs text-muted-foreground">
                  A = {r.system_a} · B = {r.system_b}
                </div>
              </TableCell>
              <TableCell className={`tabular text-right ${r.mean_diff < 0 ? 'text-success' : r.mean_diff > 0 ? 'text-destructive' : ''}`}>
                {r.mean_diff > 0 ? '+' : ''}
                {fmt.num(r.mean_diff)}
              </TableCell>
              <TableCell>
                <div className="relative h-6 w-full" aria-label={`CI ${fmt.num(lo)} to ${fmt.num(hi)}`}>
                  <div className="absolute inset-y-0 left-1/2 w-px bg-border" />
                  <div
                    className={`absolute top-1/2 h-1.5 -translate-y-1/2 rounded-full ${excl ? 'bg-primary' : 'bg-muted-foreground/50'}`}
                    style={{ left: pos(lo), width: `calc(${pos(hi)} - ${pos(lo)})` }}
                  />
                  <div className="absolute top-1/2 size-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-background bg-foreground" style={{ left: pos(r.mean_diff) }} />
                </div>
                <div className="tabular mt-0.5 flex justify-between text-[11px] text-muted-foreground">
                  <span>{fmt.num(lo)}</span>
                  <span>{fmt.num(hi)}</span>
                </div>
              </TableCell>
              <TableCell className="tabular text-right">{fmt.p(r.wilcoxon_p)}</TableCell>
              <TableCell className="tabular text-right">{fmt.p(r.mcnemar_p)}</TableCell>
              <TableCell>
                {r.holm_reject === true ? (
                  <Badge variant="success">significant</Badge>
                ) : r.holm_reject === false ? (
                  <Badge variant="secondary">not significant</Badge>
                ) : (
                  <Badge variant="outline">—</Badge>
                )}
              </TableCell>
            </TableRow>
          )
        })}
      </TableBody>
    </Table>
  )
}

export function KeyValueTable({ obj }: { obj: Record<string, unknown> }) {
  const entries = Object.entries(obj).filter(([, v]) => v === null || ['number', 'string', 'boolean'].includes(typeof v))
  const nested = Object.entries(obj).filter(([, v]) => v && typeof v === 'object')
  return (
    <div className="space-y-3 text-sm">
      {entries.length > 0 && (
        <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-1.5">
          {entries.map(([k, v]) => (
            <div key={k} className="contents">
              <dt className="text-muted-foreground">{k.replace(/_/g, ' ')}</dt>
              <dd className="tabular font-medium">{typeof v === 'number' ? (Math.abs(v) <= 1 && !Number.isInteger(v) ? fmt.pct(v) : fmt.num(v, Number.isInteger(v) ? 0 : 3)) : String(v)}</dd>
            </div>
          ))}
        </dl>
      )}
      {nested.map(([k, v]) => (
        <details key={k} className="rounded-md border p-2">
          <summary className="cursor-pointer text-sm font-medium">{k.replace(/_/g, ' ')}</summary>
          <pre className="mt-2 max-h-64 overflow-auto font-mono text-xs">{JSON.stringify(v, null, 2)}</pre>
        </details>
      ))}
    </div>
  )
}
