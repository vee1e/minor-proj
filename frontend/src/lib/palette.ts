// Single source of truth for categorical colours. Solvers get a stable colour
// regardless of the order they appear in a particular chart.

export const SOLVERS = [
  'minisat22',
  'glucose4',
  'cadical153',
  'maplesat',
  'lingeling',
  'maplecm',
  'kissat404',
] as const

const TOKENS = [
  'hsl(var(--chart-1))',
  'hsl(var(--chart-2))',
  'hsl(var(--chart-3))',
  'hsl(var(--chart-4))',
  'hsl(var(--chart-5))',
  'hsl(var(--chart-6))',
  'hsl(var(--chart-7))',
]

export const REF_COLOR = 'hsl(var(--chart-ref))'
export const GRID_COLOR = 'hsl(var(--chart-grid))'
export const PRIMARY_COLOR = 'hsl(var(--primary))'
export const SUCCESS_COLOR = 'hsl(var(--success))'
export const DESTRUCTIVE_COLOR = 'hsl(var(--destructive))'
export const WARNING_COLOR = 'hsl(var(--warning))'
export const TEXT_MUTED = 'hsl(var(--muted-foreground))'

export function solverColor(name: string): string {
  const i = (SOLVERS as readonly string[]).indexOf(name)
  if (i >= 0) return TOKENS[i]
  let h = 0
  for (const c of name) h = (h * 31 + c.charCodeAt(0)) >>> 0
  return TOKENS[h % TOKENS.length]
}

export function seriesColor(index: number): string {
  return TOKENS[index % TOKENS.length]
}

// System (selector / baseline) colours: selectors are highlighted, baselines muted.
export function systemColor(system: string): string {
  const s = system.toLowerCase()
  if (s.includes('vbs') || s.includes('oracle')) return SUCCESS_COLOR
  if (s.includes('random')) return REF_COLOR
  if (s.includes('single') || s.includes('sbs') || s.includes('best_avg') || s.includes('best average'))
    return 'hsl(var(--chart-5))'
  if (s.includes('satzilla') || s.includes('regress')) return 'hsl(var(--chart-6))'
  if (s.includes('gnn')) return 'hsl(var(--chart-4))'
  return PRIMARY_COLOR
}

/** Shared recharts styling so every chart reads as one system. */
export const chartTheme = {
  axis: { fontSize: 12, fill: TEXT_MUTED },
  tooltip: {
    contentStyle: {
      background: 'hsl(var(--popover))',
      border: '1px solid hsl(var(--border))',
      borderRadius: 8,
      color: 'hsl(var(--popover-foreground))',
      fontSize: 13,
    },
    labelStyle: { color: 'hsl(var(--foreground))', fontWeight: 600 },
    itemStyle: { color: 'hsl(var(--foreground))' },
    cursor: { fill: 'hsl(var(--muted) / 0.5)' },
  },
}
