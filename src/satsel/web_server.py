"""Flask web server for satsel.

Serves the compiled React frontend from ``frontend/dist`` and exposes a small
JSON API over the pipeline artifacts. Pipeline steps are launched as
``python -m satsel.cli <step> ...`` child processes using the *same*
interpreter that runs the server, so no ``uv`` binary needs to be on PATH.

Endpoints
---------
GET  /api/status            pipeline completion flags + background jobs
POST /api/run-step          {step, args[]} -> start a pipeline step
GET  /api/logs/<step>       tail of a step's log
GET  /api/profiling-stats   aggregates from artifacts/runs/profiling.db
GET  /api/dataset-stats     artifacts/datasets/dataset_summary.json
GET  /api/evaluate-stats    {results, report_md}
GET  /api/diagnostics       {model_diagnostics, portfolio, theta}
GET  /api/report            {report_md}
GET  /api/config            loaded config (paths stringified)
POST /api/solve             {cnf} JSON or multipart 'file' -> oracle result
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from satsel.core.config import REPO_ROOT, load_config

app = Flask(__name__)
CORS(app)

CFG = load_config()
# Reject oversized bodies before Flask buffers them, not after.
app.config["MAX_CONTENT_LENGTH"] = int(
    CFG.get("oracle", {}).get("max_bytes", 512 * 1024 * 1024))


@app.errorhandler(413)
def _too_large(_e):
    return jsonify({"error": "request body exceeds oracle.max_bytes"}), 413

# Thread-safe tracker for background jobs
BACKGROUND_JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
LOGS_DIR = REPO_ROOT / "artifacts" / "web_logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_STEPS = ["bench", "features", "profile", "build", "train", "train-gnn",
                 "evaluate", "all", "diagnose"]
ARG_RE = re.compile(r"^[-A-Za-z0-9_=.]+$")
FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _read_json(path: Path) -> Any | None:
    """Return parsed JSON or None if the file is missing/unreadable."""
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _jsonable(obj: Any) -> Any:
    """Recursively convert Paths and other non-JSON values to strings."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# Background jobs
# ---------------------------------------------------------------------------

def run_job_thread(step: str, command: list[str], log_path: Path) -> None:
    started = _now_iso()
    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"$ {' '.join(command)}\n")
            log_file.flush()
            process = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(REPO_ROOT),
                text=True,
            )
            with JOBS_LOCK:
                BACKGROUND_JOBS[step] = {
                    "status": "running",
                    "pid": process.pid,
                    "command": command,
                    "log_path": str(log_path),
                    "started_at": started,
                    "finished_at": None,
                }
            process.wait()
            with JOBS_LOCK:
                status = "completed" if process.returncode == 0 else "failed"
                BACKGROUND_JOBS[step] = {
                    "status": status,
                    "pid": process.pid,
                    "exit_code": process.returncode,
                    "command": command,
                    "log_path": str(log_path),
                    "started_at": started,
                    "finished_at": _now_iso(),
                }
    except Exception as e:  # noqa: BLE001
        with JOBS_LOCK:
            BACKGROUND_JOBS[step] = {
                "status": "failed",
                "error": str(e),
                "command": command,
                "log_path": str(log_path),
                "started_at": started,
                "finished_at": _now_iso(),
            }


@app.route("/api/status", methods=["GET"])
def get_status():
    manifest_path = Path(CFG["paths"]["manifests"]) / "formulas.parquet"
    features_dir = Path(CFG["paths"]["features"])
    db_path = Path(CFG["paths"]["db"])
    datasets_dir = Path(CFG["paths"]["datasets"])
    models_dir = Path(CFG["paths"]["models"])
    reports_dir = Path(CFG["paths"]["reports"])

    status = {
        "bench": manifest_path.exists(),
        "features": features_dir.exists() and any(features_dir.iterdir()),
        "profile": db_path.exists(),
        "build": (datasets_dir / "dataset_summary.json").exists(),
        "train": (models_dir / "portfolio.json").exists(),
        "train-gnn": (models_dir / "gnn.pt").exists(),
        "evaluate": (reports_dir / "results.json").exists(),
    }

    with JOBS_LOCK:
        running_jobs = {
            step: {
                "status": job["status"],
                "pid": job.get("pid"),
                "error": job.get("error"),
                "exit_code": job.get("exit_code"),
                "started_at": job.get("started_at"),
                "finished_at": job.get("finished_at"),
            }
            for step, job in BACKGROUND_JOBS.items()
        }

    return jsonify({
        "pipeline_completed": status,
        "running_jobs": running_jobs,
        "frontend_built": (FRONTEND_DIST / "index.html").exists(),
    })


@app.route("/api/run-step", methods=["POST"])
def run_step():
    data = request.get_json(silent=True) or {}
    step = data.get("step")
    args = data.get("args", [])

    if not step or not isinstance(step, str):
        return jsonify({"error": "No step specified"}), 400
    if step not in ALLOWED_STEPS:
        return jsonify({"error": f"Invalid step: {step}",
                        "allowed": ALLOWED_STEPS}), 400
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        return jsonify({"error": "args must be a list of strings"}), 400
    bad = [a for a in args if not ARG_RE.match(a)]
    if bad:
        return jsonify({"error": f"Invalid argument(s): {bad}. "
                                 "Only [-A-Za-z0-9_=.] are allowed."}), 400

    with JOBS_LOCK:
        job = BACKGROUND_JOBS.get(step)
        if job and job["status"] == "running":
            return jsonify({"error": f"Step {step} is already running"}), 409

    command = [sys.executable, "-u", "-m", "satsel.cli", step, *args]
    log_path = LOGS_DIR / f"{step}.log"

    t = threading.Thread(target=run_job_thread, args=(step, command, log_path),
                         daemon=True)
    t.start()

    return jsonify({
        "message": f"Started step {step} in the background",
        "command": command,
        "log_file": str(log_path),
    })


@app.route("/api/logs/<step>", methods=["GET"])
def get_logs(step: str):
    if step not in ALLOWED_STEPS:
        return jsonify({"error": f"Invalid step: {step}"}), 400
    log_path = LOGS_DIR / f"{step}.log"
    if not log_path.exists():
        return jsonify({"logs": "", "total_lines": 0, "exists": False})
    try:
        tail = int(request.args.get("tail", 500))
    except ValueError:
        tail = 500
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return jsonify({"logs": "".join(lines[-tail:]), "total_lines": len(lines),
                        "exists": True})
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"Failed to read logs: {e}"}), 500


# ---------------------------------------------------------------------------
# Read-only artifact endpoints
# ---------------------------------------------------------------------------

@app.route("/api/profiling-stats", methods=["GET"])
def get_profiling_stats():
    db_path = Path(CFG["paths"]["db"])
    if not db_path.exists():
        return jsonify({"error": "Profiling database does not exist yet. "
                                 "Run the profile stage first."}), 404
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cur = conn.cursor()
        total_runs = cur.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        total_formulas = cur.execute(
            "SELECT COUNT(DISTINCT formula_id) FROM runs").fetchone()[0]

        families = [
            {"family": r[0], "formulas": r[1], "solved": r[2] or 0, "timeout": r[3] or 0}
            for r in cur.execute(
                "SELECT family, COUNT(DISTINCT formula_id), "
                "SUM(status = 'solved'), SUM(status = 'timeout') "
                "FROM runs GROUP BY family ORDER BY 2 DESC").fetchall()
        ]

        solvers = [
            {"solver": r[0], "avg_time": r[1], "runs": r[2], "solved": r[3] or 0}
            for r in cur.execute(
                "SELECT solver, AVG(runtime_s), COUNT(*), SUM(status = 'solved') "
                "FROM runs GROUP BY solver ORDER BY 2 ASC").fetchall()
        ]

        # Per-formula winner counts (which solver had the fastest solved run).
        win_rows = cur.execute(
            "SELECT solver, COUNT(*) FROM ("
            "  SELECT formula_id, solver, MIN(runtime_s) AS t FROM runs "
            "  WHERE status = 'solved' GROUP BY formula_id, solver"
            ") a WHERE t = (SELECT MIN(t) FROM ("
            "  SELECT formula_id, solver, MIN(runtime_s) AS t FROM runs "
            "  WHERE status = 'solved' GROUP BY formula_id, solver) b "
            "  WHERE b.formula_id = a.formula_id) GROUP BY solver ORDER BY 2 DESC"
        ).fetchall()
        fastest_counts = {r[0]: r[1] for r in win_rows}

        recent_runs = [
            {"formula_id": r[0], "solver": r[1], "family": r[2], "run_idx": r[3],
             "status": r[4], "runtime_s": r[5]}
            for r in cur.execute(
                "SELECT formula_id, solver, family, run_idx, status, runtime_s "
                "FROM runs ORDER BY rowid DESC LIMIT 50").fetchall()
        ]
        conn.close()
        return jsonify({
            "total_runs": total_runs,
            "total_formulas": total_formulas,
            "families": families,
            "solvers": solvers,
            "fastest_counts": fastest_counts,
            "recent_runs": recent_runs,
        })
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"Failed to query profiling database: {e}"}), 500


@app.route("/api/dataset-stats", methods=["GET"])
def get_dataset_stats():
    summary_path = Path(CFG["paths"]["datasets"]) / "dataset_summary.json"
    if not summary_path.exists():
        return jsonify({"error": "Dataset summary not found. "
                                 "Run the build step first."}), 404
    summary = _read_json(summary_path)
    if summary is None:
        return jsonify({"error": "Failed to read dataset summary"}), 500
    return jsonify(summary)


@app.route("/api/evaluate-stats", methods=["GET"])
def get_evaluate_stats():
    reports_dir = Path(CFG["paths"]["reports"])
    results_path = reports_dir / "results.json"
    report_path = reports_dir / "report.md"

    res: dict[str, Any] = {}
    if results_path.exists():
        results = _read_json(results_path)
        if results is None:
            res["results_error"] = "results.json is not valid JSON"
        else:
            res["results"] = results
    if report_path.exists():
        try:
            res["report_md"] = report_path.read_text(encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            res["report_error"] = str(e)
    if not res:
        return jsonify({"error": "Evaluation report not found. "
                                 "Run the evaluate step first."}), 404
    return jsonify(res)


@app.route("/api/diagnostics", methods=["GET"])
def get_diagnostics():
    """Model-side diagnostics: predicted-class distribution, CV scores,
    portfolio complementarity, theta calibration curve. Any missing file is
    reported as null so the frontend can render an empty state."""
    models_dir = Path(CFG["paths"]["models"])
    return jsonify({
        "model_diagnostics": _read_json(models_dir / "model_diagnostics.json"),
        "portfolio": _read_json(models_dir / "portfolio.json"),
        "theta": _read_json(models_dir / "theta.json"),
    })


@app.route("/api/report", methods=["GET"])
def get_report():
    report_path = Path(CFG["paths"]["reports"]) / "report.md"
    if not report_path.exists():
        return jsonify({"report_md": None,
                        "error": "Report not found. Run the evaluate step first."}), 404
    try:
        return jsonify({"report_md": report_path.read_text(encoding="utf-8")})
    except Exception as e:  # noqa: BLE001
        return jsonify({"report_md": None, "error": str(e)}), 500


@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify(_jsonable(CFG))


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------

@app.route("/api/solve", methods=["POST"])
def solve():
    max_bytes = int(CFG.get("oracle", {}).get("max_bytes", 512 * 1024 * 1024))

    cnf_bytes: bytes | None = None
    if request.is_json:
        body = request.get_json(silent=True) or {}
        cnf = body.get("cnf")
        if isinstance(cnf, str):
            cnf_bytes = cnf.encode("utf-8")
    if cnf_bytes is None and "file" in request.files:
        cnf_bytes = request.files["file"].read()
    if cnf_bytes is None and not request.is_json and request.mimetype in (
            "text/plain", "application/octet-stream"):
        cnf_bytes = request.get_data()

    if not cnf_bytes or not cnf_bytes.strip():
        return jsonify({"error": "No CNF content provided. Send JSON {\"cnf\": ...} "
                                 "or a multipart 'file' field."}), 400
    if len(cnf_bytes) > max_bytes:
        return jsonify({"error": f"CNF too large ({len(cnf_bytes)} bytes > "
                                 f"oracle.max_bytes={max_bytes})"}), 413

    raw_timeout = request.args.get("timeout")
    if raw_timeout is None and request.is_json:
        raw_timeout = (request.get_json(silent=True) or {}).get("timeout")
    hard_cap = float(CFG.get("max_timeout", 300))
    timeout_s = hard_cap
    if raw_timeout is not None:
        try:
            parsed = float(raw_timeout)
        except (TypeError, ValueError):
            return jsonify({"error": "timeout must be a number"}), 400
        # min(nan, cap) returns nan, and a negative timeout would flow straight
        # into the oracle, so both are rejected explicitly.
        if not math.isfinite(parsed) or parsed <= 0:
            return jsonify({"error": "timeout must be a positive finite number"}), 400
        timeout_s = min(parsed, hard_cap)

    try:
        from satsel.oracle.oracle import SolverOracle
        oracle = SolverOracle(CFG)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"Failed to initialize SolverOracle: {e}"}), 500

    try:
        result = oracle.solve(cnf_bytes, timeout_s=timeout_s)
        res_dict = _jsonable(result.to_dict())
        if res_dict.get("assignment") is not None:
            res_dict["assignment"] = list(res_dict["assignment"])
        if res_dict.get("features") is not None:
            res_dict["features"] = list(res_dict["features"])
        return jsonify(res_dict)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"Solver execution failed: {e}"}), 500


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path: str):
    if path.startswith("api/"):
        return jsonify({"error": "Unknown API endpoint"}), 404
    if path and (FRONTEND_DIST / path).is_file():
        return send_from_directory(FRONTEND_DIST, path)
    if (FRONTEND_DIST / "index.html").exists():
        return send_from_directory(FRONTEND_DIST, "index.html")
    return (
        "<html><body style='font-family:sans-serif;max-width:40em;margin:4em auto;'>"
        "<h1>satsel web server</h1>"
        "<p>The API is running, but the frontend has not been built.</p>"
        "<p>Build it with:</p>"
        "<pre>cd frontend\nnpm install\nnpm run build</pre>"
        "<p>then reload this page. Or start the dev server with "
        "<code>npm run dev</code>.</p>"
        "</body></html>"
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Start the satsel web interface")
    parser.add_argument("--port", type=int, default=5001, help="Port (default 5001)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--debug", action="store_true",
                        help="enable Flask debug mode + reloader (off by default)")
    args = parser.parse_args(argv)

    print(f"satsel web server: http://{args.host}:{args.port}")
    print(f"frontend build present: {(FRONTEND_DIST / 'index.html').exists()}")
    print(f"pipeline steps run as: {sys.executable} -m satsel.cli <step>")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True,
            use_reloader=args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
