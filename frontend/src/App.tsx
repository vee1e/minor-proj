import React, { useState, useEffect, useRef } from 'react'
import { 
  Play, 
  Terminal, 
  Database, 
  Cpu, 
  Activity, 
  CheckCircle2, 
  XCircle, 
  Loader2, 
  Code,
  FastForward,
  BarChart3,
  Layers,
  Copy,
  Check
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { 
  ResponsiveContainer, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  Cell 
} from 'recharts'

const API_BASE = window.location.hostname === 'localhost' ? 'http://127.0.0.1:5001' : ''

interface PipelineStatus {
  pipeline_completed: {
    bench: boolean
    features: boolean
    profile: boolean
    build: boolean
    train: boolean
    'train-gnn': boolean
    evaluate: boolean
  }
  running_jobs: Record<string, {
    status: 'running' | 'completed' | 'failed'
    pid: number | null
    error?: string
    exit_code?: number
  }>
}

interface ProfilingStats {
  total_runs: number
  total_formulas: number
  families: Array<{ family: string; formulas: number; solved: number; timeout: number }>
  solvers: Array<{ solver: string; avg_time: number; runs: number; solved: number }>
  recent_runs: Array<{ formula_id: string; solver: string; family: string; run_idx: number; status: string; runtime_s: number }>
}

interface DatasetStats {
  n_id?: number
  n_ood?: number
  n_train?: number
  n_val?: number
  n_test?: number
  n_deduped?: number
  families?: Record<string, { n: number; n_train: number; n_val: number; n_test: number; vbs_solve_rate?: number }>
}

interface EvaluateStats {
  results?: {
    systems?: Array<{ system: string; par2: number; solve_rate: number }>
    random_mean_sd?: [number, number]
    oracle_gap_selector?: number
    chosen_selector?: string
    comparisons?: Array<{
      comparison: string
      mean_diff: number
      ci_low: number
      ci_high: number
      ci_excludes_zero: boolean
      wilcoxon_p: number
      mcnemar_p: number
      holm_reject: boolean
    }>
  }
}

function App() {
  const [status, setStatus] = useState<PipelineStatus | null>(null)
  const [activeDataView, setActiveDataView] = useState<'profile' | 'dataset' | 'evaluation'>('profile')
  
  const [profilingStats, setProfilingStats] = useState<ProfilingStats | null>(null)
  const [datasetStats, setDatasetStats] = useState<DatasetStats | null>(null)
  const [evaluateStats, setEvaluateStats] = useState<EvaluateStats | null>(null)
  
  const [logs, setLogs] = useState<string>('')
  const [activeLogStep, setActiveLogStep] = useState<string>('profile')
  const logEndRef = useRef<HTMLDivElement>(null)
  
  const [oracleInput, setOracleInput] = useState<string>(
    `c Example 3-SAT CNF formula\np cnf 3 4\n1 2 -3 0\n-1 -2 3 0\n2 3 0\n-2 -3 0`
  )
  const [oracleResult, setOracleResult] = useState<any>(null)
  const [oracleLoading, setOracleLoading] = useState(false)
  const [oracleError, setOracleError] = useState<string | null>(null)

  const fetchStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/status`)
      const data = await res.json()
      setStatus(data)
    } catch (e) {
      console.error('Failed fetching status', e)
    }
  }

  const fetchLogs = async (step: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/logs/${step}`)
      const data = await res.json()
      setLogs(data.logs || '')
    } catch (e) {
      console.error('Failed fetching logs', e)
    }
  }

  const loadViewData = async (view: 'profile' | 'dataset' | 'evaluation') => {
    try {
      const endpoint = view === 'profile' ? 'profiling-stats' : view === 'dataset' ? 'dataset-stats' : 'evaluate-stats'
      const res = await fetch(`${API_BASE}/api/${endpoint}`)
      if (res.ok) {
        const data = await res.json()
        if (view === 'profile') setProfilingStats(data)
        else if (view === 'dataset') setDatasetStats(data)
        else if (view === 'evaluation') setEvaluateStats(data)
      }
    } catch (e) {
      console.error(`Failed to load data for view ${view}`, e)
    }
  }

  const runStep = async (stepName: string, extraArgs: string[] = []) => {
    try {
      setActiveLogStep(stepName)
      await fetch(`${API_BASE}/api/run-step`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ step: stepName, args: extraArgs })
      })
      fetchStatus()
    } catch (e) {
      console.error('Failed to trigger execution', e)
    }
  }

  const solveCnf = async () => {
    setOracleLoading(true)
    setOracleError(null)
    setOracleResult(null)
    try {
      const res = await fetch(`${API_BASE}/api/solve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cnf: oracleInput })
      })
      const data = await res.json()
      if (data.error) {
        setOracleError(data.error)
      } else {
        setOracleResult(data)
      }
    } catch (e: any) {
      setOracleError(e.message || 'Network request failed')
    } finally {
      setOracleLoading(false)
    }
  }

  useEffect(() => {
    fetchStatus()
    loadViewData('profile')
    loadViewData('dataset')
    loadViewData('evaluation')
    
    const timer = setInterval(() => {
      fetchStatus()
    }, 2000)
    return () => clearInterval(timer)
  }, [])

  useEffect(() => {
    loadViewData(activeDataView)
  }, [activeDataView])

  useEffect(() => {
    fetchLogs(activeLogStep)
    if (status?.running_jobs[activeLogStep]?.status === 'running') {
      const logTimer = setInterval(() => {
        fetchLogs(activeLogStep)
      }, 1500)
      return () => clearInterval(logTimer)
    }
  }, [activeLogStep, status])

  useEffect(() => {
    if (logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [logs])

  // TokyoNight Chart Palette
  const TOKYO_COLORS = ['#7aa2f7', '#bb9af7', '#7dcfff', '#9ece6a', '#e0af68', '#ff9e64', '#f7768e']

  const activeJobStep = Object.keys(status?.running_jobs || {}).find(
    k => status?.running_jobs[k].status === 'running'
  )

  return (
    <div className="h-screen w-screen max-h-screen max-w-vw overflow-hidden bg-[#1a1b26] text-[#a9b1d6] flex flex-col font-mono text-xs select-none">
      
      {/* TokyoNight Header Bar */}
      <header className="h-10 px-3 bg-[#16161e] border-b border-[#292e42] flex justify-between items-center shrink-0">
        <div className="flex items-center gap-2">
          <Cpu className="h-4 w-4 text-[#7aa2f7]" />
          <span className="font-bold text-sm text-[#c0caf5]">satsel</span>
          <span className="text-[11px] text-[#565f89] border-l border-[#292e42] pl-2">SAT Solver Selector</span>
        </div>

        <div className="flex items-center gap-3">
          {/* RUN ALL BUTTON */}
          <button
            onClick={() => runStep('all', ['--small', '--no-gnn'])}
            disabled={!!activeJobStep}
            className="px-2.5 py-1 bg-[#7aa2f7] hover:bg-[#89b4fa] disabled:opacity-40 text-[#1a1b26] font-bold rounded flex items-center gap-1 transition-colors text-[11px]"
            title="Run complete pipeline sequentially"
          >
            <FastForward className="h-3 w-3 fill-current" />
            <span>Run All</span>
          </button>

          {/* Status Indicator */}
          {activeJobStep ? (
            <div className="flex items-center gap-1.5 text-xs text-[#e0af68] bg-[#1f2335] px-2 py-0.5 rounded border border-[#e0af68]/30">
              <Loader2 className="h-3 w-3 animate-spin text-[#e0af68]" />
              <span>Step <strong className="text-[#ff9e64]">{activeJobStep}</strong> running...</span>
            </div>
          ) : (
            <span className="text-[10px] text-[#9ece6a] bg-[#1f2335] px-2 py-0.5 rounded border border-[#9ece6a]/30 flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-[#9ece6a]"></span> Ready
            </span>
          )}
        </div>
      </header>

      {/* Main Grid Viewport - Fits 100vh perfectly */}
      <main className="flex-1 h-full min-h-0 p-2 grid grid-cols-12 gap-2 overflow-hidden">
        
        {/* LEFT PANEL: Pipeline Controls & Oracle Solver (Cols 4) */}
        <div className="col-span-4 flex flex-col gap-2 h-full min-h-0 overflow-hidden">
          
          {/* Pipeline Card */}
          <Card className="bg-[#16161e] border-[#292e42] shadow-none shrink-0">
            <CardHeader className="p-2.5 pb-1 flex flex-row justify-between items-center">
              <CardTitle className="text-xs font-bold text-[#7aa2f7] flex items-center gap-1.5">
                <Layers className="h-3.5 w-3.5" /> Pipeline Stages
              </CardTitle>
            </CardHeader>
            <CardContent className="p-2 pt-0 space-y-1">
              {[
                { name: 'bench', label: '1. Bench', desc: 'Download/gen formulas', args: ['--small'] },
                { name: 'features', label: '2. Features', desc: 'Extract vector values', args: [] },
                { name: 'profile', label: '3. Profile', desc: 'Time solver portfolio', args: ['--limit', '10'] },
                { name: 'build', label: '4. Build', desc: 'Split train/val/test', args: [] },
                { name: 'train', label: '5. Train', desc: 'Fit selector models', args: [] },
                { name: 'evaluate', label: '6. Evaluate', desc: 'Compute benchmarks', args: ['--no-gnn'] }
              ].map((step) => {
                const isCompleted = status?.pipeline_completed[step.name as keyof typeof status.pipeline_completed] || false
                const isRunning = status?.running_jobs[step.name]?.status === 'running'
                
                return (
                  <div key={step.name} className="flex items-center justify-between p-1.5 rounded bg-[#1a1b26] border border-[#292e42] text-[11px]">
                    <div className="flex items-center gap-1.5">
                      {isRunning ? (
                        <Loader2 className="h-3 w-3 text-[#e0af68] animate-spin" />
                      ) : isCompleted ? (
                        <CheckCircle2 className="h-3 w-3 text-[#9ece6a]" />
                      ) : (
                        <XCircle className="h-3 w-3 text-[#565f89]" />
                      )}
                      <span className="font-semibold text-[#c0caf5]">{step.label}</span>
                    </div>
                    
                    <button
                      onClick={() => runStep(step.name, step.args)}
                      disabled={!!activeJobStep}
                      className="px-2 py-0.5 bg-[#292e42] hover:bg-[#3b4261] disabled:opacity-30 text-[#7aa2f7] rounded text-[10px] font-bold transition-colors flex items-center gap-1"
                    >
                      <Play className="h-2.5 w-2.5" /> Run
                    </button>
                  </div>
                )
              })}
            </CardContent>
          </Card>

          {/* Interactive Oracle Solver Card */}
          <Card className="bg-[#16161e] border-[#292e42] shadow-none flex-1 min-h-0 flex flex-col overflow-hidden">
            <CardHeader className="p-2.5 pb-1 shrink-0">
              <CardTitle className="text-xs font-bold text-[#bb9af7] flex items-center gap-1.5">
                <Code className="h-3.5 w-3.5" /> Oracle Solver (DIMACS CNF)
              </CardTitle>
            </CardHeader>
            <CardContent className="p-2 pt-0 flex-1 min-h-0 flex flex-col gap-2 overflow-hidden">
              <textarea
                value={oracleInput}
                onChange={(e) => setOracleInput(e.target.value)}
                className="w-full flex-1 min-h-[70px] bg-[#15161e] border border-[#292e42] rounded p-2 font-mono text-[10px] text-[#a9b1d6] focus:outline-none focus:border-[#7aa2f7] resize-none"
                placeholder="Paste DIMACS CNF..."
              />
              <button
                onClick={solveCnf}
                disabled={oracleLoading}
                className="w-full py-1 bg-[#bb9af7] hover:bg-[#c0caf5] text-[#1a1b26] font-bold rounded text-[11px] transition-colors flex items-center justify-center gap-1 shrink-0"
              >
                {oracleLoading ? <Loader2 className="h-3 w-3 animate-spin" /> : <Play className="h-3 w-3" />}
                Run Solver Oracle
              </button>

              {oracleError && (
                <div className="p-1.5 bg-[#f7768e]/10 border border-[#f7768e]/30 text-[#f7768e] rounded text-[10px]">
                  {oracleError}
                </div>
              )}

              {oracleResult && (
                <div className="p-2 rounded bg-[#1a1b26] border border-[#292e42] space-y-1 text-[11px] shrink-0">
                  <div className="flex justify-between items-center border-b border-[#292e42] pb-1">
                    <span className="text-[#565f89]">Result Status</span>
                    <span className={`font-bold ${oracleResult.status === 'SAT' ? 'text-[#9ece6a]' : oracleResult.status === 'UNSAT' ? 'text-[#7dcfff]' : 'text-[#f7768e]'}`}>
                      {oracleResult.status}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#565f89]">Selected Solver</span>
                    <span className="font-bold text-[#e0af68]">{oracleResult.solver_used || 'None'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#565f89]">Runtime</span>
                    <span className="font-mono text-[#c0caf5]">{oracleResult.runtime_ms} ms</span>
                  </div>
                  {oracleResult.confidence !== undefined && (
                    <div className="flex justify-between">
                      <span className="text-[#565f89]">Confidence</span>
                      <span className="font-mono text-[#e0af68]">{(oracleResult.confidence * 100).toFixed(1)}%</span>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>

        </div>

        {/* RIGHT PANEL: Terminal Output & Data Analytics (Cols 8) */}
        <div className="col-span-8 flex flex-col gap-2 h-full min-h-0 overflow-hidden">
          
          {/* Terminal Console Card */}
          <Card className="bg-[#16161e] border-[#292e42] shadow-none flex-1 min-h-0 flex flex-col overflow-hidden">
            <CardHeader className="p-2.5 pb-1 flex flex-row justify-between items-center shrink-0 border-b border-[#292e42]">
              <CardTitle className="text-xs font-bold text-[#7dcfff] flex items-center gap-1.5">
                <Terminal className="h-3.5 w-3.5" /> Output Logs: <span className="text-[#e0af68]">{activeLogStep}</span>
              </CardTitle>
              <div className="flex gap-1">
                {['bench', 'features', 'profile', 'build', 'train', 'evaluate'].map((st) => (
                  <button
                    key={st}
                    onClick={() => setActiveLogStep(st)}
                    className={`text-[9px] px-1.5 py-0.5 rounded transition-colors ${
                      activeLogStep === st ? 'bg-[#7aa2f7] text-[#1a1b26] font-bold' : 'bg-[#292e42] hover:bg-[#3b4261] text-[#a9b1d6]'
                    }`}
                  >
                    {st}
                  </button>
                ))}
              </div>
            </CardHeader>
            <CardContent className="p-0 flex-1 min-h-0 overflow-hidden relative">
              <div className="absolute inset-0 bg-[#15161e] p-2.5 font-mono text-[10px] leading-tight overflow-y-auto text-[#a9b1d6]">
                <pre className="whitespace-pre-wrap">{logs || 'No log messages.'}</pre>
                <div ref={logEndRef} />
              </div>
            </CardContent>
          </Card>

          {/* Stats Analytics Card */}
          <Card className="bg-[#16161e] border-[#292e42] shadow-none flex-1 min-h-0 flex flex-col overflow-hidden">
            <CardHeader className="p-2.5 pb-1 flex flex-row justify-between items-center shrink-0 border-b border-[#292e42]">
              <CardTitle className="text-xs font-bold text-[#9ece6a] flex items-center gap-1.5">
                <Activity className="h-3.5 w-3.5" /> Stats & Analytics Explorer
              </CardTitle>
              <div className="flex gap-1">
                {[
                  { id: 'profile', label: 'Profiling' },
                  { id: 'dataset', label: 'Dataset' },
                  { id: 'evaluation', label: 'Evaluation' }
                ].map((v) => (
                  <button
                    key={v.id}
                    onClick={() => setActiveDataView(v.id as any)}
                    className={`text-[9px] px-2 py-0.5 rounded transition-colors ${
                      activeDataView === v.id ? 'bg-[#292e42] text-[#9ece6a] font-bold border border-[#3b4261]' : 'text-[#565f89] hover:text-[#a9b1d6]'
                    }`}
                  >
                    {v.label}
                  </button>
                ))}
              </div>
            </CardHeader>
            
            <CardContent className="p-2 flex-1 min-h-0 overflow-y-auto">
              
              {/* Profiling View */}
              {activeDataView === 'profile' && (
                <div className="space-y-2 text-[11px]">
                  {!profilingStats ? (
                    <p className="text-center py-4 text-[#565f89]">No profiling stats. Run profiling stage.</p>
                  ) : (
                    <>
                      <div className="grid grid-cols-3 gap-2 text-center">
                        <div className="bg-[#1a1b26] p-1.5 rounded border border-[#292e42]">
                          <span className="block text-[9px] text-[#565f89]">Formulas</span>
                          <span className="font-bold text-[#7aa2f7]">{profilingStats.total_formulas}</span>
                        </div>
                        <div className="bg-[#1a1b26] p-1.5 rounded border border-[#292e42]">
                          <span className="block text-[9px] text-[#565f89]">Total Runs</span>
                          <span className="font-bold text-[#bb9af7]">{profilingStats.total_runs}</span>
                        </div>
                        <div className="bg-[#1a1b26] p-1.5 rounded border border-[#292e42]">
                          <span className="block text-[9px] text-[#565f89]">Solved</span>
                          <span className="font-bold text-[#9ece6a]">
                            {profilingStats.solvers.reduce((acc, s) => acc + s.solved, 0)}
                          </span>
                        </div>
                      </div>

                      <div className="h-24">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={profilingStats.solvers} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
                            <XAxis dataKey="solver" stroke="#565f89" tick={{ fontSize: 8 }} />
                            <YAxis stroke="#565f89" tick={{ fontSize: 8 }} />
                            <Tooltip contentStyle={{ backgroundColor: '#16161e', borderColor: '#292e42', color: '#a9b1d6', fontSize: 10 }} />
                            <Bar dataKey="avg_time" radius={[2, 2, 0, 0]}>
                              {profilingStats.solvers.map((entry, idx) => (
                                <Cell key={`c-${idx}`} fill={TOKYO_COLORS[idx % TOKYO_COLORS.length]} />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>

                      <Table className="text-[10px]">
                        <TableHeader>
                          <TableRow className="border-[#292e42] hover:bg-transparent">
                            <TableHead className="py-0.5 text-[#565f89]">Solver</TableHead>
                            <TableHead className="py-0.5 text-right text-[#565f89]">Avg Time</TableHead>
                            <TableHead className="py-0.5 text-right text-[#565f89]">Solved Runs</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {profilingStats.solvers.map((s) => (
                            <TableRow key={s.solver} className="border-[#292e42] hover:bg-[#1a1b26]">
                              <TableCell className="py-1 font-semibold text-[#c0caf5]">{s.solver}</TableCell>
                              <TableCell className="py-1 text-right font-mono text-[#7aa2f7]">{s.avg_time.toFixed(3)}s</TableCell>
                              <TableCell className="py-1 text-right text-[#9ece6a]">{s.solved}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </>
                  )}
                </div>
              )}

              {/* Dataset View */}
              {activeDataView === 'dataset' && (
                <div className="space-y-2 text-[11px]">
                  {!datasetStats ? (
                    <p className="text-center py-4 text-[#565f89]">No dataset summary. Run build stage.</p>
                  ) : (
                    <>
                      <div className="grid grid-cols-4 gap-1.5 text-center">
                        <div className="bg-[#1a1b26] p-1 rounded border border-[#292e42]">
                          <span className="block text-[8px] text-[#565f89]">Train</span>
                          <span className="font-bold text-[#9ece6a]">{datasetStats.n_train}</span>
                        </div>
                        <div className="bg-[#1a1b26] p-1 rounded border border-[#292e42]">
                          <span className="block text-[8px] text-[#565f89]">Val</span>
                          <span className="font-bold text-[#e0af68]">{datasetStats.n_val}</span>
                        </div>
                        <div className="bg-[#1a1b26] p-1 rounded border border-[#292e42]">
                          <span className="block text-[8px] text-[#565f89]">Test</span>
                          <span className="font-bold text-[#ff9e64]">{datasetStats.n_test}</span>
                        </div>
                        <div className="bg-[#1a1b26] p-1 rounded border border-[#292e42]">
                          <span className="block text-[8px] text-[#565f89]">OOD</span>
                          <span className="font-bold text-[#7dcfff]">{datasetStats.n_ood}</span>
                        </div>
                      </div>

                      <Table className="text-[10px]">
                        <TableHeader>
                          <TableRow className="border-[#292e42] hover:bg-transparent">
                            <TableHead className="py-0.5 text-[#565f89]">Family</TableHead>
                            <TableHead className="py-0.5 text-right text-[#565f89]">Total (n)</TableHead>
                            <TableHead className="py-0.5 text-right text-[#565f89]">Train</TableHead>
                            <TableHead className="py-0.5 text-right text-[#565f89]">Val / Test</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {datasetStats.families && Object.entries(datasetStats.families).map(([fam, d]) => (
                            <TableRow key={fam} className="border-[#292e42] hover:bg-[#1a1b26]">
                              <TableCell className="py-1 font-semibold text-[#c0caf5]">{fam}</TableCell>
                              <TableCell className="py-1 text-right font-mono">{d.n}</TableCell>
                              <TableCell className="py-1 text-right font-mono text-[#9ece6a]">{d.n_train}</TableCell>
                              <TableCell className="py-1 text-right font-mono">{d.n_val} / {d.n_test}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </>
                  )}
                </div>
              )}

              {/* Evaluation View */}
              {activeDataView === 'evaluation' && (
                <div className="space-y-2 text-[11px]">
                  {!evaluateStats || !evaluateStats.results ? (
                    <p className="text-center py-4 text-[#565f89]">No evaluation stats. Run evaluate stage.</p>
                  ) : (
                    <>
                      <div className="flex justify-between items-center bg-[#1a1b26] p-1.5 rounded border border-[#292e42]">
                        <div>
                          <span className="text-[9px] text-[#565f89]">Chosen Selector</span>
                          <div className="font-bold text-[#e0af68]">{evaluateStats.results.chosen_selector || 'None'}</div>
                        </div>
                        <div className="text-right">
                          <span className="text-[9px] text-[#565f89]">Oracle Gap</span>
                          <div className="font-bold text-[#bb9af7]">
                            {evaluateStats.results.oracle_gap_selector ? evaluateStats.results.oracle_gap_selector.toFixed(4) : '0'}
                          </div>
                        </div>
                      </div>

                      <Table className="text-[10px]">
                        <TableHeader>
                          <TableRow className="border-[#292e42] hover:bg-transparent">
                            <TableHead className="py-0.5 text-[#565f89]">System</TableHead>
                            <TableHead className="py-0.5 text-right text-[#565f89]">PAR-2 Score</TableHead>
                            <TableHead className="py-0.5 text-right text-[#565f89]">Solve Rate</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {evaluateStats.results.systems && evaluateStats.results.systems.map((sys) => (
                            <TableRow key={sys.system} className="border-[#292e42] hover:bg-[#1a1b26]">
                              <TableCell className="py-1 font-semibold text-[#c0caf5]">{sys.system}</TableCell>
                              <TableCell className="py-1 text-right font-mono text-[#bb9af7]">{sys.par2.toFixed(3)}</TableCell>
                              <TableCell className="py-1 text-right font-mono text-[#9ece6a]">{(sys.solve_rate * 100).toFixed(1)}%</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </>
                  )}
                </div>
              )}

            </CardContent>
          </Card>

        </div>

      </main>
    </div>
  )
}

export default App
