function bad(v: unknown): boolean {
  return v === null || v === undefined || Number.isNaN(Number(v))
}

export const fmt = {
  num(v: unknown, digits = 2): string {
    return bad(v) ? '—' : Number(v).toFixed(digits)
  },
  pct(v: unknown, digits = 1): string {
    return bad(v) ? '—' : `${(Number(v) * 100).toFixed(digits)}%`
  },
  ms(v: unknown): string {
    if (bad(v)) return '—'
    const n = Number(v)
    if (n >= 1000) return `${(n / 1000).toFixed(2)} s`
    return `${n.toFixed(n < 10 ? 2 : 0)} ms`
  },
  p(v: unknown): string {
    if (bad(v)) return '—'
    const n = Number(v)
    return n < 0.001 ? '<0.001' : n.toFixed(3)
  },
  int(v: unknown): string {
    return bad(v) ? '—' : Number(v).toLocaleString()
  },
}
