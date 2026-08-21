import { useEffect, useState, type ComponentType } from 'react'
import { Activity, BarChart3, Database, FlaskConical, Moon, Sun, Workflow } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useTheme } from '@/hooks/useTheme'
import { cn } from '@/lib/utils'
import { Overview } from '@/components/views/Overview'
import { Data } from '@/components/views/Data'
import { Results } from '@/components/views/Results'
import { Diagnostics } from '@/components/views/Diagnostics'
import { Oracle } from '@/components/views/Oracle'

type ViewId = 'overview' | 'data' | 'results' | 'diagnostics' | 'oracle'

const VIEWS: { id: ViewId; label: string; icon: ComponentType<{ className?: string }>; component: ComponentType }[] = [
  { id: 'overview', label: 'Pipeline', icon: Workflow, component: Overview },
  { id: 'data', label: 'Data', icon: Database, component: Data },
  { id: 'results', label: 'Results', icon: BarChart3, component: Results },
  { id: 'diagnostics', label: 'Diagnostics', icon: Activity, component: Diagnostics },
  { id: 'oracle', label: 'Oracle', icon: FlaskConical, component: Oracle },
]

function viewFromHash(): ViewId {
  const h = window.location.hash.replace('#', '') as ViewId
  return VIEWS.some((v) => v.id === h) ? h : 'overview'
}

export default function App() {
  const { theme, toggle } = useTheme()
  const [view, setView] = useState<ViewId>(viewFromHash)

  useEffect(() => {
    const onHash = () => setView(viewFromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  const go = (id: ViewId) => {
    window.location.hash = id
    setView(id)
  }

  const Active = VIEWS.find((v) => v.id === view)?.component ?? Overview

  return (
    <div className="flex min-h-screen flex-col lg:flex-row">
      <aside className="flex shrink-0 flex-col border-b bg-card lg:sticky lg:top-0 lg:h-screen lg:w-60 lg:border-b-0 lg:border-r">
        <div className="flex items-center gap-2 px-5 py-4">
          <span className="flex size-8 items-center justify-center rounded-md bg-primary font-mono text-sm font-bold text-primary-foreground" aria-hidden>
            ∨
          </span>
          <div className="leading-tight">
            <p className="font-semibold">satsel</p>
            <p className="text-xs text-muted-foreground">SAT solver selection</p>
          </div>
        </div>
        <nav aria-label="Main" className="flex gap-1 overflow-x-auto px-3 pb-3 lg:flex-col lg:pb-0">
          {VIEWS.map((v) => {
            const active = v.id === view
            const Icon = v.icon
            return (
              <button
                key={v.id}
                type="button"
                onClick={() => go(v.id)}
                aria-current={active ? 'page' : undefined}
                className={cn(
                  'flex items-center gap-2.5 whitespace-nowrap rounded-md px-3 py-2 text-sm font-medium transition-colors',
                  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
                  active ? 'bg-accent text-accent-foreground' : 'text-muted-foreground hover:bg-muted hover:text-foreground',
                )}
              >
                <Icon className="size-4" />
                {v.label}
              </button>
            )
          })}
        </nav>
        <div className="mt-auto hidden border-t p-3 lg:block">
          <Button variant="ghost" size="sm" onClick={toggle} className="w-full justify-start" aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}>
            {theme === 'dark' ? <Sun /> : <Moon />}
            {theme === 'dark' ? 'Light theme' : 'Dark theme'}
          </Button>
          <p className="mt-2 px-2 text-[11px] leading-snug text-muted-foreground">
            ML-driven solver selection over a 7-solver PySAT portfolio. MIT Manipal minor project.
          </p>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center justify-end border-b px-4 py-2 lg:hidden">
          <Button variant="ghost" size="icon-sm" onClick={toggle} aria-label="Toggle theme">
            {theme === 'dark' ? <Sun /> : <Moon />}
          </Button>
        </div>
        <main className="mx-auto w-full max-w-[1400px] flex-1 px-5 py-6 md:px-8 md:py-8">
          <Active />
        </main>
      </div>
    </div>
  )
}
