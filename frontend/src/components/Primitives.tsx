import type { ReactNode } from 'react'
import { AlertTriangle, Info, Loader2, OctagonX, PackageOpen, CheckCircle2 } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import type { ApiState } from '@/hooks/useApi'

/* ---------- Page scaffolding ---------- */

export function PageHeader({ title, lede, actions }: { title: string; lede?: ReactNode; actions?: ReactNode }) {
  return (
    <header className="mb-6 flex flex-wrap items-start justify-between gap-4">
      <div className="max-w-3xl">
        <h1 className="text-2xl">{title}</h1>
        {lede && <p className="mt-1 text-[0.95rem] text-muted-foreground">{lede}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </header>
  )
}

export function Section({
  title,
  description,
  children,
  className,
  right,
}: {
  title: string
  description?: ReactNode
  children: ReactNode
  className?: string
  right?: ReactNode
}) {
  return (
    <Card className={cn('shadow-sm', className)}>
      <CardHeader className="flex-row items-start justify-between gap-4 space-y-0 p-5 pb-3">
        <div>
          <CardTitle className="text-base">{title}</CardTitle>
          {description && <CardDescription className="mt-1">{description}</CardDescription>}
        </div>
        {right}
      </CardHeader>
      <CardContent className="p-5 pt-0">{children}</CardContent>
    </Card>
  )
}

/* ---------- Callouts ---------- */

type Tone = 'info' | 'warning' | 'danger' | 'success'

const toneClass: Record<Tone, string> = {
  info: 'border-info/30 bg-info/10 text-foreground [&_svg]:text-info',
  warning: 'border-warning/40 bg-warning/10 text-foreground [&_svg]:text-warning',
  danger: 'border-destructive/40 bg-destructive/10 text-foreground [&_svg]:text-destructive',
  success: 'border-success/40 bg-success/10 text-foreground [&_svg]:text-success',
}
const toneIcon: Record<Tone, typeof Info> = {
  info: Info,
  warning: AlertTriangle,
  danger: OctagonX,
  success: CheckCircle2,
}

export function Callout({
  tone = 'info',
  title,
  children,
  className,
}: {
  tone?: Tone
  title?: ReactNode
  children?: ReactNode
  className?: string
}) {
  const Icon = toneIcon[tone]
  return (
    <div role={tone === 'danger' ? 'alert' : 'note'} className={cn('flex gap-3 rounded-lg border p-4', toneClass[tone], className)}>
      <Icon className="mt-0.5 size-5 shrink-0" aria-hidden />
      <div className="min-w-0 text-sm">
        {title && <p className="font-semibold">{title}</p>}
        {children && <div className={cn(title && 'mt-1', 'text-foreground/90')}>{children}</div>}
      </div>
    </div>
  )
}

export function WarningList({ warnings, title }: { warnings: string[] | undefined; title: string }) {
  if (!warnings || warnings.length === 0) return null
  return (
    <Callout tone="warning" title={title}>
      <ul className="list-disc space-y-1 pl-5">
        {warnings.map((w, i) => (
          <li key={i}>{w}</li>
        ))}
      </ul>
    </Callout>
  )
}

/* ---------- Async state wrappers ---------- */

export function LoadingState({ label = 'Loading…' }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 py-10 text-sm text-muted-foreground" role="status">
      <Loader2 className="size-4 animate-spin" aria-hidden />
      {label}
    </div>
  )
}

export function EmptyState({
  title,
  hint,
  action,
}: {
  title: string
  hint?: ReactNode
  action?: ReactNode
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed py-12 text-center">
      <PackageOpen className="size-7 text-muted-foreground" aria-hidden />
      <p className="font-medium">{title}</p>
      {hint && <p className="max-w-md text-sm text-muted-foreground">{hint}</p>}
      {action && <div className="mt-2">{action}</div>}
    </div>
  )
}

export function ErrorState({ message, retry }: { message: string; retry?: () => void }) {
  return (
    <Callout tone="danger" title="Could not load data">
      <p className="break-words font-mono text-xs">{message}</p>
      {retry && (
        <Button size="sm" variant="outline" className="mt-3" onClick={retry}>
          Retry
        </Button>
      )}
    </Callout>
  )
}

/**
 * Renders loading / error / 404-empty states for an ApiState, otherwise calls
 * `children` with the non-null data.
 */
export function Async<T>({
  state,
  emptyTitle,
  emptyHint,
  children,
}: {
  state: ApiState<T>
  emptyTitle: string
  emptyHint?: ReactNode
  children: (data: T) => ReactNode
}) {
  if (state.loading && state.data === null) return <LoadingState />
  if (state.error) return <ErrorState message={state.error} retry={state.reload} />
  if (state.notFound || state.data === null) return <EmptyState title={emptyTitle} hint={emptyHint} />
  return <>{children(state.data)}</>
}

/* ---------- Stats ---------- */

export function Stat({
  label,
  value,
  sub,
  tone,
}: {
  label: string
  value: ReactNode
  sub?: ReactNode
  tone?: 'default' | 'warning' | 'danger' | 'success'
}) {
  const v =
    tone === 'warning'
      ? 'text-warning'
      : tone === 'danger'
        ? 'text-destructive'
        : tone === 'success'
          ? 'text-success'
          : 'text-foreground'
  return (
    <div className="rounded-lg border bg-card p-4">
      <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{label}</p>
      <p className={cn('tabular mt-1 text-2xl font-semibold', v)}>{value}</p>
      {sub && <p className="mt-0.5 text-xs text-muted-foreground">{sub}</p>}
    </div>
  )
}

export function StatGrid({ children }: { children: ReactNode }) {
  return <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">{children}</div>
}

export function Swatch({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-sm">
      <span className="inline-block size-2.5 rounded-sm" style={{ background: color }} aria-hidden />
      {label}
    </span>
  )
}
