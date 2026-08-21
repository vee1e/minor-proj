import { useMemo } from 'react'
import { Async, Callout, PageHeader, Section, Stat, StatGrid, WarningList } from '@/components/Primitives'
import { CategoryBars } from '@/components/charts/Charts'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { useApi } from '@/hooks/useApi'
import { api } from '@/lib/api'
import { fmt } from '@/lib/format'
import { solverColor } from '@/lib/palette'
import type { DatasetStats } from '@/lib/types'

export function Data() {
  const ds = useApi(api.dataset)

  return (
    <div>
      <PageHeader
        title="Dataset"
        lede="What the selector actually learns from. If one solver dominates the labels, or most formulas are ties, a classifier cannot beat the single best solver — check the warnings before trusting any result."
      />
      <Async
        state={ds}
        emptyTitle="Dataset summary not built yet"
        emptyHint="Run Profile and then Build on the Pipeline page. Build writes dataset_summary.json, which this page reads."
      >
        {(d) => <DataBody d={d} />}
      </Async>
    </div>
  )
}

function DataBody({ d }: { d: DatasetStats }) {
  const labels = useMemo(() => {
    const entries = Object.entries(d.label_distribution ?? {})
    const total = entries.reduce((a, [, n]) => a + Number(n), 0) || 1
    return entries
      .map(([solver, n]) => ({ solver, count: Number(n), share: Number(n) / total }))
      .sort((a, b) => b.count - a.count)
  }, [d.label_distribution])

  const majority = d.label_majority_share ?? labels[0]?.share ?? null
  const tie = d.tie_rate ?? null
  const usable = d.n_train_usable ?? null
  const majorityTone = majority === null ? 'default' : majority > 0.7 ? 'danger' : majority > 0.5 ? 'warning' : 'success'
  const tieTone = tie === null ? 'default' : tie > 0.5 ? 'danger' : tie > 0.3 ? 'warning' : 'success'

  const derivedWarnings: string[] = []
  if (majority !== null && majority > 0.7 && labels[0])
    derivedWarnings.push(
      `${fmt.pct(majority, 0)} of labels are "${labels[0].solver}". A majority-class classifier already matches this; the selector will look like it "always picks ${labels[0].solver}".`,
    )
  if (tie !== null && tie > 0.5)
    derivedWarnings.push(
      `${fmt.pct(tie, 0)} of formulas are ties (all solvers within the tie margin). Ties are excluded from training, so few informative rows remain. Use harder instances or loosen the tie rule.`,
    )
  if (usable !== null && usable < 100)
    derivedWarnings.push(`Only ${usable} usable training rows. Expect unstable models and wide confidence intervals.`)

  const families = Object.entries(d.families ?? {}).sort(([a], [b]) => a.localeCompare(b))

  return (
    <div className="space-y-6">
      <WarningList warnings={d.warnings} title="Data-quality warnings from the build step" />
      {derivedWarnings.length > 0 && (
        <Callout tone={majorityTone === 'danger' || tieTone === 'danger' ? 'danger' : 'warning'} title="Why the selector may collapse to one solver">
          <ul className="list-disc space-y-1 pl-5">
            {derivedWarnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </Callout>
      )}

      <StatGrid>
        <Stat label="In-distribution" value={fmt.int(d.n_id)} sub={`train ${fmt.int(d.n_train)} · val ${fmt.int(d.n_val)} · test ${fmt.int(d.n_test)}`} />
        <Stat label="OOD formulas" value={fmt.int(d.n_ood)} sub="held-out families" />
        <Stat label="Deduplicated" value={fmt.int(d.n_deduped)} sub="removed as exact duplicates" />
        <Stat label="Majority label share" value={fmt.pct(majority, 0)} tone={majorityTone} sub={labels[0] ? labels[0].solver : undefined} />
        <Stat label="Tie rate" value={fmt.pct(tie, 0)} tone={tieTone} sub="no solver clearly fastest" />
        <Stat label="Usable train rows" value={fmt.int(usable)} tone={usable !== null && usable < 100 ? 'warning' : 'default'} sub={`informative ${fmt.pct(d.informative_rate, 0)}`} />
      </StatGrid>

      <div className="grid gap-6 lg:grid-cols-5">
        <Section
          className="lg:col-span-2"
          title="Label distribution"
          description="Which solver had the lowest median runtime per formula (ties excluded)."
        >
          {labels.length === 0 ? (
            <p className="text-sm text-muted-foreground">No labels recorded.</p>
          ) : (
            <>
              <CategoryBars
                data={labels}
                xKey="solver"
                yKey="count"
                yLabel="formulas"
                layout="vertical"
                colorOf={solverColor}
                height={40 + labels.length * 34}
                format={(v) => fmt.int(v)}
              />
              <ul className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
                {labels.map((l) => (
                  <li key={l.solver} className="flex items-center justify-between gap-2">
                    <span className="flex items-center gap-1.5 font-mono text-[13px]">
                      <span className="inline-block size-2.5 rounded-sm" style={{ background: solverColor(l.solver) }} />
                      {l.solver}
                    </span>
                    <span className="tabular text-muted-foreground">
                      {fmt.int(l.count)} · {fmt.pct(l.share, 0)}
                    </span>
                  </li>
                ))}
              </ul>
            </>
          )}
        </Section>

        <Section className="lg:col-span-3" title="Families" description="Instances per family and how they were split. VBS solve rate is the fraction any portfolio solver solves within the timeout.">
          {families.length === 0 ? (
            <p className="text-sm text-muted-foreground">No families recorded.</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Family</TableHead>
                  <TableHead className="text-right">n</TableHead>
                  <TableHead className="text-right">train</TableHead>
                  <TableHead className="text-right">val</TableHead>
                  <TableHead className="text-right">test</TableHead>
                  <TableHead className="text-right">VBS solve rate</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {families.map(([name, f]) => {
                  const ood = (f.n_train ?? 0) === 0 && (f.n ?? 0) > 0
                  return (
                    <TableRow key={name}>
                      <TableCell className="font-medium">
                        {name} {ood && <Badge variant="info" className="ml-1">OOD</Badge>}
                      </TableCell>
                      <TableCell className="tabular text-right">{fmt.int(f.n)}</TableCell>
                      <TableCell className="tabular text-right">{fmt.int(f.n_train)}</TableCell>
                      <TableCell className="tabular text-right">{fmt.int(f.n_val)}</TableCell>
                      <TableCell className="tabular text-right">{fmt.int(f.n_test)}</TableCell>
                      <TableCell className="tabular text-right">{fmt.pct(f.vbs_solve_rate, 0)}</TableCell>
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
          )}
        </Section>
      </div>
    </div>
  )
}
