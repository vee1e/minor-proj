import { Async, Callout, PageHeader, Section, WarningList } from '@/components/Primitives'
import { CategoryBars, MultiLine } from '@/components/charts/Charts'
import { KeyValueTable } from '@/components/views/Results'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { useApi } from '@/hooks/useApi'
import { api } from '@/lib/api'
import { fmt } from '@/lib/format'
import { solverColor } from '@/lib/palette'
import type { Diagnostics as Diag, ModelCandidate } from '@/lib/types'

export function Diagnostics() {
  const dg = useApi(api.diagnostics)
  return (
    <div>
      <PageHeader
        title="Diagnostics"
        lede="Why the selector behaves the way it does: which model candidates were considered, what they predict, how the confidence threshold θ was chosen, and whether the portfolio is complementary enough to be worth selecting over."
      />
      <Async
        state={dg}
        emptyTitle="No diagnostics available"
        emptyHint="Run Train and Evaluate. The backend exposes /api/diagnostics once model_diagnostics and portfolio analysis have been written."
      >
        {(d) => <DiagBody d={d} />}
      </Async>
    </div>
  )
}

function histRows(h: Record<string, number> | undefined) {
  return Object.entries(h ?? {})
    .map(([solver, count]) => ({ solver, count: Number(count) }))
    .sort((a, b) => b.count - a.count)
}

function DiagBody({ d }: { d: Diag }) {
  const md = d.model_diagnostics ?? {}
  const candidates: ModelCandidate[] = md.candidates ?? []
  const chosenName = md.chosen ?? candidates.find((c) => c.chosen || c.selected)?.name
  const chosenModel = candidates.find((c) => (c.name ?? c.model) === chosenName) ?? candidates[0]
  const predHist = histRows(md.predicted_class_hist ?? chosenModel?.predicted_class_hist ?? chosenModel?.pred_hist)
  const trueHist = histRows(md.true_class_hist)
  const theta = d.theta ?? {}
  const curve = theta.curve ?? []
  const pf = d.portfolio ?? {}
  const solvers = pf.solvers ?? Array.from(new Set(Object.keys(pf.complementarity ?? {})))
  const comp = pf.complementarity ?? {}
  const compMax = Math.max(1e-9, ...solvers.flatMap((a) => solvers.map((b) => Number(comp[a]?.[b] ?? 0))))
  const marginal = Object.entries(pf.marginal_contribution ?? {}).map(([solver, v]) => ({ solver, value: Number(v) }))
  const unique = Object.entries(pf.unique_share ?? {}).map(([solver, v]) => ({ solver, value: Number(v) }))
  const predTop = predHist[0]
  const predTotal = predHist.reduce((a, b) => a + b.count, 0) || 1

  const extra = Object.fromEntries(
    Object.entries(md).filter(([k]) => !['candidates', 'chosen', 'predicted_class_hist', 'true_class_hist', 'warnings'].includes(k)),
  )

  return (
    <div className="space-y-6">
      <WarningList warnings={md.warnings} title="Model warnings" />
      {predTop && predTop.count / predTotal > 0.9 && (
        <Callout tone="warning" title={`Predicted class is ${predTop.solver} for ${fmt.pct(predTop.count / predTotal, 0)} of validation instances`}>
          The model has collapsed to the majority class. Compare with the true-label histogram; if they match, the dataset—not the model—is the problem.
        </Callout>
      )}

      <Section title="Model candidates" description="Every classical model trained under nested cross-validation. The chosen model minimises cross-validated PAR-2, not raw accuracy.">
        {candidates.length === 0 ? (
          <p className="text-sm text-muted-foreground">No candidates reported.</p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Model</TableHead>
                <TableHead className="text-right">CV PAR-2</TableHead>
                <TableHead className="text-right">Val PAR-2</TableHead>
                <TableHead className="text-right">Accuracy</TableHead>
                <TableHead className="text-right">Balanced acc.</TableHead>
                <TableHead>Predicted classes</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {candidates.map((c, i) => {
                const name = String(c.name ?? c.model ?? `#${i + 1}`)
                const h = histRows(c.predicted_class_hist ?? c.pred_hist)
                const tot = h.reduce((a, b) => a + b.count, 0) || 1
                return (
                  <TableRow key={name + i}>
                    <TableCell className="font-medium">
                      {name} {(c.chosen || c.selected || name === chosenName) && <Badge className="ml-1">chosen</Badge>}
                    </TableCell>
                    <TableCell className="tabular text-right">
                      {fmt.num(c.cv_par2)}
                      {c.cv_par2_sd != null && <span className="text-xs text-muted-foreground"> ± {fmt.num(c.cv_par2_sd)}</span>}
                    </TableCell>
                    <TableCell className="tabular text-right">{fmt.num(c.val_par2)}</TableCell>
                    <TableCell className="tabular text-right">{fmt.pct(c.accuracy ?? c.val_accuracy)}</TableCell>
                    <TableCell className="tabular text-right">{fmt.pct(c.balanced_accuracy)}</TableCell>
                    <TableCell>
                      {h.length === 0 ? (
                        <span className="text-muted-foreground">—</span>
                      ) : (
                        <div className="flex h-3 w-48 overflow-hidden rounded-sm" title={h.map((x) => `${x.solver}: ${x.count}`).join(', ')}>
                          {h.map((x) => (
                            <span key={x.solver} style={{ width: `${(x.count / tot) * 100}%`, background: solverColor(x.solver) }} />
                          ))}
                        </div>
                      )}
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        )}
      </Section>

      <div className="grid gap-6 xl:grid-cols-2">
        <Section title="Predicted vs true class" description="Validation-set histogram for the chosen model. Left: what it predicts. Right: the real fastest solver.">
          {predHist.length === 0 && trueHist.length === 0 ? (
            <p className="text-sm text-muted-foreground">No histograms reported.</p>
          ) : (
            <div className="grid gap-4 md:grid-cols-2">
              <div>
                <p className="mb-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">Predicted</p>
                <CategoryBars data={predHist} xKey="solver" yKey="count" yLabel="instances" layout="vertical" colorOf={solverColor} height={40 + Math.max(predHist.length, 3) * 30} format={(v) => fmt.int(v)} />
              </div>
              <div>
                <p className="mb-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">True label</p>
                {trueHist.length === 0 ? (
                  <p className="text-sm text-muted-foreground">Not reported.</p>
                ) : (
                  <CategoryBars data={trueHist} xKey="solver" yKey="count" yLabel="instances" layout="vertical" colorOf={solverColor} height={40 + Math.max(trueHist.length, 3) * 30} format={(v) => fmt.int(v)} />
                )}
              </div>
            </div>
          )}
        </Section>

        <Section
          title="Confidence threshold θ"
          description="Below θ the selector falls back to the pre-solver (single best). The curve shows validation PAR-2 and the share of instances the model actually decides on."
        >
          {curve.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              No θ sweep reported{theta.theta != null ? `; θ = ${fmt.num(theta.theta, 2)}` : ''}.
            </p>
          ) : (
            <MultiLine
              data={curve}
              xKey="theta"
              xLabel="θ"
              series={[
                { key: 'val_par2', label: 'val PAR-2' },
                { key: 'coverage', label: 'coverage', axis: 'right', format: (v) => fmt.pct(v) },
              ]}
              reference={theta.theta != null ? { x: theta.theta, label: `θ = ${fmt.num(theta.theta, 2)}` } : undefined}
            />
          )}
        </Section>
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <Section title="Complementarity" description="Row solver vs column solver: how often the row solver is strictly faster. Uniform colour means little to select between.">
          {solvers.length === 0 ? (
            <p className="text-sm text-muted-foreground">No complementarity matrix reported.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="tabular text-xs">
                <thead>
                  <tr>
                    <th className="p-1" />
                    {solvers.map((s) => (
                      <th key={s} className="p-1 text-left font-mono font-normal text-muted-foreground" style={{ writingMode: 'vertical-rl' }}>
                        {s}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {solvers.map((a) => (
                    <tr key={a}>
                      <th className="p-1 pr-2 text-right font-mono font-normal text-muted-foreground">{a}</th>
                      {solvers.map((b) => {
                        const v = Number(comp[a]?.[b] ?? 0)
                        const t = a === b ? 0 : v / compMax
                        return (
                          <td key={b} className="p-0.5">
                            <div
                              className="flex size-9 items-center justify-center rounded-sm"
                              style={{
                                background: a === b ? 'hsl(var(--muted))' : `hsl(var(--primary) / ${0.08 + 0.82 * t})`,
                                color: t > 0.55 ? 'hsl(var(--primary-foreground))' : 'hsl(var(--foreground))',
                              }}
                              title={`${a} beats ${b}: ${v <= 1 ? fmt.pct(v, 0) : fmt.num(v, 0)}`}
                            >
                              {a === b ? '' : v <= 1 ? Math.round(v * 100) : Math.round(v)}
                            </div>
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Section>

        <Section title="Portfolio contribution" description="Unique share: instances only that solver solves. Marginal contribution: VBS PAR-2 lost if the solver is removed. Pruned solvers are dominated.">
          {unique.length === 0 && marginal.length === 0 ? (
            <p className="text-sm text-muted-foreground">No portfolio analysis reported.</p>
          ) : (
            <div className="space-y-4">
              {marginal.length > 0 && (
                <div>
                  <p className="mb-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">Marginal contribution</p>
                  <CategoryBars data={marginal} xKey="solver" yKey="value" yLabel="Δ VBS PAR-2" layout="vertical" colorOf={solverColor} height={40 + marginal.length * 28} />
                </div>
              )}
              {unique.length > 0 && (
                <div>
                  <p className="mb-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">Unique share</p>
                  <CategoryBars data={unique} xKey="solver" yKey="value" yLabel="share" layout="vertical" colorOf={solverColor} height={40 + unique.length * 28} format={(v) => fmt.pct(v)} />
                </div>
              )}
              <div className="flex flex-wrap items-center gap-2 text-sm">
                <span className="text-muted-foreground">Best average solver:</span>
                <Badge variant="outline">{pf.best_average_solver ?? '—'}</Badge>
                <span className="ml-3 text-muted-foreground">Pruned:</span>
                {(pf.pruned ?? []).length === 0 ? <span>none</span> : (pf.pruned ?? []).map((s) => <Badge key={s} variant="secondary">{s}</Badge>)}
              </div>
            </div>
          )}
        </Section>
      </div>

      {Object.keys(extra).length > 0 && (
        <Section title="Other model diagnostics" description="Anything else the training step recorded.">
          <KeyValueTable obj={extra} />
        </Section>
      )}
    </div>
  )
}
