import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { chartTheme, GRID_COLOR, seriesColor, solverColor, SUCCESS_COLOR } from '@/lib/palette'
import { fmt } from '@/lib/format'

/** Horizontal/vertical categorical bar chart with a per-bar colour function. */
export function CategoryBars({
  data,
  xKey,
  yKey,
  yLabel,
  colorOf,
  height = 260,
  reference,
  layout = 'horizontal',
  format = (v: number) => fmt.num(v),
}: {
  data: object[]
  xKey: string
  yKey: string
  yLabel: string
  colorOf: (name: string) => string
  height?: number
  reference?: { value: number; label: string }
  layout?: 'horizontal' | 'vertical'
  format?: (v: number) => string
}) {
  const vertical = layout === 'vertical'
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart
        data={data}
        layout={vertical ? 'vertical' : 'horizontal'}
        margin={{ top: 8, right: 16, left: vertical ? 8 : 0, bottom: 8 }}
      >
        <CartesianGrid stroke={GRID_COLOR} vertical={vertical} horizontal={!vertical} />
        {vertical ? (
          <>
            <XAxis type="number" tick={chartTheme.axis} tickFormatter={format} label={{ value: yLabel, position: 'insideBottom', offset: -2, fontSize: 12 }} />
            <YAxis type="category" dataKey={xKey} tick={chartTheme.axis} width={96} />
          </>
        ) : (
          <>
            <XAxis dataKey={xKey} tick={chartTheme.axis} interval={0} />
            <YAxis tick={chartTheme.axis} tickFormatter={format} label={{ value: yLabel, angle: -90, position: 'insideLeft', fontSize: 12 }} />
          </>
        )}
        <Tooltip {...chartTheme.tooltip} formatter={(v) => [format(Number(v)), yLabel]} />
        {reference && (
          <ReferenceLine
            {...(vertical ? { x: reference.value } : { y: reference.value })}
            stroke={SUCCESS_COLOR}
            strokeDasharray="4 4"
            label={{ value: reference.label, fontSize: 11, fill: SUCCESS_COLOR, position: vertical ? 'top' : 'right' }}
          />
        )}
        <Bar dataKey={yKey} radius={[3, 3, 0, 0]} isAnimationActive={false}>
          {data.map((d, i) => (
            <Cell key={i} fill={colorOf(String((d as Record<string, unknown>)[xKey]))} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

/** Stacked bars: one row per `rowKey` value, one stack segment per solver. */
export function StackedSolverBars({
  rows,
  solvers,
  rowKey = 'name',
  height = 240,
  normalize = false,
}: {
  rows: Record<string, number | string>[]
  solvers: string[]
  rowKey?: string
  height?: number
  normalize?: boolean
}) {
  const data = normalize
    ? rows.map((r) => {
        const total = solvers.reduce((a, s) => a + Number(r[s] ?? 0), 0) || 1
        const out: Record<string, number | string> = { [rowKey]: r[rowKey] }
        for (const s of solvers) out[s] = Number(r[s] ?? 0) / total
        return out
      })
    : rows
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, left: 8, bottom: 4 }}>
        <CartesianGrid stroke={GRID_COLOR} horizontal={false} />
        <XAxis
          type="number"
          tick={chartTheme.axis}
          tickFormatter={(v) => (normalize ? fmt.pct(v, 0) : String(v))}
          domain={normalize ? [0, 1] : undefined}
          label={{ value: normalize ? 'share of choices' : 'choices', position: 'insideBottom', offset: -2, fontSize: 12 }}
        />
        <YAxis type="category" dataKey={rowKey} tick={chartTheme.axis} width={120} />
        <Tooltip {...chartTheme.tooltip} formatter={(v, name) => [normalize ? fmt.pct(v as number) : String(v), String(name)]} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {solvers.map((s) => (
          <Bar key={s} dataKey={s} stackId="a" fill={solverColor(s)} isAnimationActive={false} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  )
}

export function MultiLine({
  data,
  xKey,
  xLabel,
  series,
  height = 260,
  reference,
}: {
  data: object[]
  xKey: string
  xLabel: string
  series: { key: string; label: string; axis?: 'left' | 'right'; format?: (v: number) => string }[]
  height?: number
  reference?: { x: number; label: string }
}) {
  const hasRight = series.some((s) => s.axis === 'right')
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: hasRight ? 8 : 16, left: 0, bottom: 8 }}>
        <CartesianGrid stroke={GRID_COLOR} />
        <XAxis dataKey={xKey} tick={chartTheme.axis} label={{ value: xLabel, position: 'insideBottom', offset: -2, fontSize: 12 }} />
        <YAxis yAxisId="left" tick={chartTheme.axis} />
        {hasRight && <YAxis yAxisId="right" orientation="right" tick={chartTheme.axis} tickFormatter={(v) => fmt.pct(v, 0)} domain={[0, 1]} />}
        <Tooltip
          {...chartTheme.tooltip}
          formatter={(v, name) => {
            const s = series.find((x) => x.label === name)
            return [s?.format ? s.format(Number(v)) : fmt.num(Number(v)), String(name)]
          }}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {reference && (
          <ReferenceLine yAxisId="left" x={reference.x} stroke={SUCCESS_COLOR} strokeDasharray="4 4" label={{ value: reference.label, fontSize: 11, fill: SUCCESS_COLOR }} />
        )}
        {series.map((s, i) => (
          <Line
            key={s.key}
            yAxisId={s.axis ?? 'left'}
            type="monotone"
            dataKey={s.key}
            name={s.label}
            stroke={seriesColor(i)}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}
