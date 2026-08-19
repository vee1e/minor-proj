from __future__ import annotations

import os
import sys
import json
import sqlite3
import subprocess
import threading
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Dynamically locate repo root and config
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from satsel.core.config import load_config
from satsel.oracle.oracle import SolverOracle

CFG = load_config()

# Thread-safe tracker for background jobs
BACKGROUND_JOBS = {}
JOBS_LOCK = threading.Lock()
LOGS_DIR = REPO_ROOT / "artifacts" / "web_logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

def run_job_thread(step: str, command: list[str], log_path: Path):
    try:
        with open(log_path, "w") as log_file:
            process = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(REPO_ROOT),
                text=True
            )
            
            with JOBS_LOCK:
                BACKGROUND_JOBS[step] = {
                    "status": "running",
                    "pid": process.pid,
                    "log_path": str(log_path)
                }
            
            # Wait for execution to finish
            process.wait()
            
            with JOBS_LOCK:
                status = "completed" if process.returncode == 0 else "failed"
                BACKGROUND_JOBS[step] = {
                    "status": status,
                    "exit_code": process.returncode,
                    "log_path": str(log_path)
                }
    except Exception as e:
        with JOBS_LOCK:
            BACKGROUND_JOBS[step] = {
                "status": "failed",
                "error": str(e),
                "log_path": str(log_path)
            }

@app.route("/api/status", methods=["GET"])
def get_status():
    manifest_path = Path(CFG["paths"]["manifests"]) / "formulas.parquet"
    features_dir = Path(CFG["paths"]["features"])
    db_path = Path(CFG["paths"]["db"])
    datasets_dir = Path(CFG["paths"]["datasets"])
    models_dir = Path(CFG["paths"]["models"])
    reports_dir = Path(CFG["paths"]["reports"])
    
    # Check what is completed
    status = {
        "bench": manifest_path.exists(),
        "features": features_dir.exists() and any(features_dir.iterdir()) if features_dir.exists() else False,
        "profile": db_path.exists(),
        "build": (datasets_dir / "dataset_summary.json").exists(),
        "train": (models_dir / "portfolio.json").exists(),
        "train-gnn": (models_dir / "gnn.pt").exists(),
        "evaluate": (reports_dir / "results.json").exists()
    }
    
    with JOBS_LOCK:
        running_jobs = {
            step: {
                "status": job["status"],
                "pid": job.get("pid"),
                "error": job.get("error"),
                "exit_code": job.get("exit_code")
            } for step, job in BACKGROUND_JOBS.items()
        }
        
    return jsonify({
        "pipeline_completed": status,
        "running_jobs": running_jobs
    })

@app.route("/api/run-step", methods=["POST"])
def run_step():
    data = request.json or {}
    step = data.get("step")
    args = data.get("args", [])
    
    if not step:
        return jsonify({"error": "No step specified"}), 400
        
    allowed_steps = ["bench", "features", "profile", "build", "train", "train-gnn", "evaluate", "all"]
    if step not in allowed_steps:
        return jsonify({"error": f"Invalid step: {step}"}), 400
        
    with JOBS_LOCK:
        if step in BACKGROUND_JOBS and BACKGROUND_JOBS[step]["status"] == "running":
            return jsonify({"error": f"Step {step} is already running"}), 400
            
    # Assemble CLI command
    command = ["uv", "run", "satsel", step] + args
    log_path = LOGS_DIR / f"{step}.log"
    
    # Start process in a background thread
    t = threading.Thread(target=run_job_thread, args=(step, command, log_path))
    t.daemon = True
    t.start()
    
    return jsonify({"message": f"Started step {step} in the background", "log_file": str(log_path)})

@app.route("/api/logs/<step>", methods=["GET"])
def get_logs(step):
    log_path = LOGS_DIR / f"{step}.log"
    if not log_path.exists():
        return jsonify({"logs": "(No logs generated yet)"})
        
    # Read last 500 lines of logs to prevent huge response sizes
    try:
        with open(log_path, "r", errors="replace") as f:
            lines = f.readlines()
            last_lines = lines[-500:]
            return jsonify({"logs": "".join(last_lines), "total_lines": len(lines)})
    except Exception as e:
        return jsonify({"error": f"Failed to read logs: {str(e)}"}), 500

@app.route("/api/profiling-stats", methods=["GET"])
def get_profiling_stats():
    db_path = Path(CFG["paths"]["db"])
    if not db_path.exists():
        return jsonify({"error": "Profiling database does not exist yet. Run the profile stage first."}), 404
        
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Total runs
        total_runs = cursor.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        
        # Unique formulas profiled
        total_formulas = cursor.execute("SELECT COUNT(DISTINCT formula_id) FROM runs").fetchone()[0]
        
        # Family details
        family_rows = cursor.execute(
            "SELECT family, COUNT(DISTINCT formula_id) as num_formulas, "
            "SUM(status = 'solved') as solved, SUM(status = 'timeout') as timeout "
            "FROM runs GROUP BY family ORDER BY num_formulas DESC"
        ).fetchall()
        
        families = []
        for r in family_rows:
            families.append({
                "family": r[0],
                "formulas": r[1],
                "solved": r[2] or 0,
                "timeout": r[3] or 0
            })
            
        # Solver statistics (mean runtime and count of wins on solved instances)
        # To make it simple, let's query average runtime per solver per family
        solver_avg = cursor.execute(
            "SELECT solver, AVG(runtime_s) as avg_time, COUNT(*) as runs_count, "
            "SUM(status = 'solved') as solved_count "
            "FROM runs GROUP BY solver ORDER BY avg_time ASC"
        ).fetchall()
        
        solvers = [{
            "solver": r[0],
            "avg_time": r[1],
            "runs": r[2],
            "solved": r[3] or 0
        } for r in solver_avg]
        
        # Recent runs
        recent_rows = cursor.execute(
            "SELECT formula_id, solver, family, run_idx, status, runtime_s "
            "FROM runs ORDER BY rowid DESC LIMIT 50"
        ).fetchall()
        
        recent_runs = [{
            "formula_id": r[0],
            "solver": r[1],
            "family": r[2],
            "run_idx": r[3],
            "status": r[4],
            "runtime_s": r[5]
        } for r in recent_rows]
        
        conn.close()
        return jsonify({
            "total_runs": total_runs,
            "total_formulas": total_formulas,
            "families": families,
            "solvers": solvers,
            "recent_runs": recent_runs
        })
    except Exception as e:
        return jsonify({"error": f"Failed to query profiling database: {str(e)}"}), 500

@app.route("/api/dataset-stats", methods=["GET"])
def get_dataset_stats():
    datasets_dir = Path(CFG["paths"]["datasets"])
    summary_path = datasets_dir / "dataset_summary.json"
    if not summary_path.exists():
        return jsonify({"error": "Dataset summary not found. Run dataset build step first."}), 404
        
    try:
        summary = json.loads(summary_path.read_text())
        return jsonify(summary)
    except Exception as e:
        return jsonify({"error": f"Failed to read dataset summary: {str(e)}"}), 500

@app.route("/api/evaluate-stats", methods=["GET"])
def get_evaluate_stats():
    reports_dir = Path(CFG["paths"]["reports"])
    results_path = reports_dir / "results.json"
    report_path = reports_dir / "report.md"
    
    res = {}
    if results_path.exists():
        try:
            res["results"] = json.loads(results_path.read_text())
        except Exception as e:
            res["results_error"] = str(e)
            
    if report_path.exists():
        try:
            res["report_md"] = report_path.read_text()
        except Exception as e:
            res["report_error"] = str(e)
            
    if not res:
        return jsonify({"error": "Evaluation report not found. Run evaluate step first."}), 404
        
    return jsonify(res)

@app.route("/api/solve", methods=["POST"])
def solve():
    # Attempt to load SolverOracle
    try:
        oracle = SolverOracle(CFG)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize SolverOracle: {str(e)}"}), 500
        
    cnf_content = request.json.get("cnf") if request.is_json else None
    if not cnf_content:
        # Check files
        if "file" in request.files:
            cnf_content = request.files["file"].read().decode("utf-8", errors="replace")
        else:
            return jsonify({"error": "No CNF content provided"}), 400
            
    try:
        cnf_bytes = cnf_content.encode("utf-8")
        result = oracle.solve(cnf_bytes)
        
        # Convert SolverOracleResult to dict
        res_dict = result.to_dict()
        # Some values like tuple/bytes need cleanup for JSON serialization
        if res_dict.get("assignment") is not None:
            res_dict["assignment"] = list(res_dict["assignment"])
        if res_dict.get("features") is not None:
            res_dict["features"] = list(res_dict["features"])
            
        return jsonify(res_dict)
    except Exception as e:
        return jsonify({"error": f"Solver execution failed: {str(e)}"}), 500

# Serve React static assets
FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path != "" and os.path.exists(FRONTEND_DIST / path):
        return send_from_directory(FRONTEND_DIST, path)
    else:
        if os.path.exists(FRONTEND_DIST / "index.html"):
            return send_from_directory(FRONTEND_DIST, "index.html")
        else:
            return (
                "<html><body style='background-color:#1e1e2e;color:#cdd6f4;font-family:sans-serif;text-align:center;padding:50px;'>"
                "<h1>satsel Web Server</h1>"
                "<p>The frontend React application is not compiled yet.</p>"
                "<p>Please build the React frontend using: <code>cd frontend && bun run build</code></p>"
                "</body></html>"
            )

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Start the satsel web interface")
    parser.add_argument("--port", type=int, default=5001, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    args = parser.parse_args()
    
    print(f"Starting satsel web server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=True)

if __name__ == "__main__":
    main()
