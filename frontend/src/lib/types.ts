// Shapes returned by web_server.py. All fields are treated as optional on the
// wire; the API client normalises missing values so views can stay simple.

export type StepName =
  | 'bench'
  | 'features'
  | 'profile'
  | 'build'
  | 'train'
  | 'train-gnn'
  | 'evaluate'

export const PIPELINE_STEPS: { id: StepName; label: string; blurb: string }[] = [
  { id: 'bench', label: 'Bench', blurb: 'Generate the CNF corpus per family' },
  { id: 'features', label: 'Features', blurb: 'Syntactic, graph and probe features under budget' },
  { id: 'profile', label: 'Profile', blurb: 'Run all 7 solvers, 3 repeats each, median runtime' },
  { id: 'build', label: 'Build', blurb: 'Assemble labels, ties, splits, dataset summary' },
  { id: 'train', label: 'Train', blurb: 'Classical RF / XGB / SVM with nested CV + calibration' },
  { id: 'train-gnn', label: 'Train GNN', blurb: 'GraSS-style tripartite GNN selector' },
  { id: 'evaluate', label: 'Evaluate', blurb: 'PAR-2, solve rate, oracle gap, bootstrap CI, Wilcoxon' },
]

export interface JobInfo {
  status?: 'running' | 'done' | 'failed' | 'error' | string
  pid?: number
  error?: string | null
  exit_code?: number | null
}

export interface StatusResponse {
  pipeline_completed: Partial<Record<StepName, boolean>>
  running_jobs: Partial<Record<StepName, JobInfo>>
}

export interface LogsResponse {
  logs: string
  total_lines: number
}

export interface ProfilingStats {
  total_runs: number
  total_formulas: number
  families: { family: string; formulas: number; solved: number; timeout: number }[]
  solvers: { solver: string; avg_time: number; runs: number; solved: number }[]
  recent_runs: Record<string, unknown>[]
}

export interface FamilyStats {
  n?: number
  n_train?: number
  n_val?: number
  n_test?: number
  vbs_solve_rate?: number
}

export interface DatasetStats {
  n_id?: number
  n_ood?: number
  n_train?: number
  n_val?: number
  n_test?: number
  n_deduped?: number
  families?: Record<string, FamilyStats>
  label_distribution?: Record<string, number>
  label_majority_share?: number
  tie_rate?: number
  informative_rate?: number
  n_train_usable?: number
  warnings?: string[]
}

export interface SystemRow {
  system: string
  par2: number
  solve_rate: number
  n?: number
}

export interface Comparison {
  comparison: string
  system_a: string
  system_b: string
  mean_diff: number
  ci_low: number
  ci_high: number
  ci_excludes_zero?: boolean
  wilcoxon_p?: number | null
  mcnemar_p?: number | null
  bootstrap_p_approx?: number | null
  holm_reject?: boolean
}

export interface FeatureAblationRow {
  config?: string
  name?: string
  feature_set?: string
  par2?: number
  solve_rate?: number
  feature_ms?: number
  feature_overhead_ms?: number
  accuracy?: number
  [k: string]: unknown
}

export interface EvaluateResults {
  systems?: SystemRow[]
  random_mean_sd?: { mean?: number; sd?: number } | [number, number] | null
  oracle_gap_selector?: number | null
  chosen_selector?: string | null
  best_average_solver?: string | null
  theta?: number | null
  comparisons?: Comparison[]
  per_system_choice_hist?: Record<string, Record<string, number>>
  classification?: Record<string, unknown>
  per_family?: Record<string, Record<string, { par2?: number; solve_rate?: number }>>
  ood?: Record<string, unknown>
  ood_par2?: Record<string, number>
  label_stats?: Record<string, unknown>
  feature_ablation?: FeatureAblationRow[]
  data_quality_warnings?: string[]
  [k: string]: unknown
}

export interface EvaluateResponse {
  results: EvaluateResults
  report_md?: string
}

export interface ModelCandidate {
  name?: string
  model?: string
  chosen?: boolean
  selected?: boolean
  val_par2?: number
  cv_par2?: number
  cv_par2_sd?: number
  accuracy?: number
  val_accuracy?: number
  balanced_accuracy?: number
  predicted_class_hist?: Record<string, number>
  pred_hist?: Record<string, number>
  [k: string]: unknown
}

export interface Diagnostics {
  model_diagnostics?: {
    candidates?: ModelCandidate[]
    chosen?: string
    predicted_class_hist?: Record<string, number>
    true_class_hist?: Record<string, number>
    warnings?: string[]
    [k: string]: unknown
  }
  portfolio?: {
    solvers?: string[]
    best_average_solver?: string
    unique_share?: Record<string, number>
    pruned?: string[]
    complementarity?: Record<string, Record<string, number>>
    marginal_contribution?: Record<string, number>
  }
  theta?: {
    theta?: number
    curve?: { theta: number; val_par2: number; coverage: number }[]
  }
}

export interface SolveResult {
  status: string
  satisfiable?: boolean | null
  solver_used?: string | null
  runtime_ms?: number
  parse_ms?: number
  feature_ms?: number
  infer_ms?: number
  solve_ms?: number
  confidence?: number | null
  theta_used?: number | null
  fallback_used?: boolean
  fallback_reason?: string | null
  error?: string | null
  assignment?: number[] | null
  [k: string]: unknown
}
