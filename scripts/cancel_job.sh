#!/usr/bin/env bash
# Emergency cleanup: archive status/logs of ONE Job, delete it, wait until the project has no live Jobs.
# Usage: bash scripts/cancel_job.sh <aijob-id>
# Runs on this machine from the repository root, or on CONTROL_HOST in REMOTE_DIR (see run_gen.sh).
set -euo pipefail
JOB=${1:?aijob id}
HERE=$(cd "$(dirname "$0")/.." && pwd)
read -r -a SSH_EXTRA <<< "${SSH_OPTS:-}"
run_py() {
  if [ -n "${CONTROL_HOST:-}" ]; then
    ssh ${SSH_EXTRA[@]+"${SSH_EXTRA[@]}"} "$CONTROL_HOST" \
      "cd ${REMOTE_DIR:?set REMOTE_DIR} && export PATH=\$HOME/.local/bin:\$PATH JOB=$JOB${NEBIUS_CLI:+ NEBIUS_CLI=$NEBIUS_CLI} && python3 -u -"
  else
    (cd "$HERE" && JOB=$JOB python3 -u -)
  fi
}
run_py <<'PY'
import json, os, shutil, subprocess, time
from datetime import datetime, timezone
job = os.environ["JOB"]
CLI = os.environ.get("NEBIUS_CLI") or shutil.which("nebius") or os.path.expanduser("~/.local/bin/nebius")
for d in ("config", "metrics"):
    os.makedirs(d, exist_ok=True)
pid = json.load(open("config/resources.json"))["project_id"]
def cli(*a, timeout=60):
    try:
        return subprocess.run([CLI, *a], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(a, 124, "", f"timed out after {timeout}s")
print("Checking", job, flush=True)
r = cli("ai", "job", "get", job, "--format", "json")
if r.returncode:
    print("Job not found (already deleted?):", r.stderr.strip()[:200])
    info = None
else:
    info = json.loads(r.stdout)
if info:
    name = info["metadata"]["name"]
    if info["metadata"].get("parent_id") not in (None, pid):
        raise SystemExit("Job is not in this project; refusing")
    open(f"metrics/{name}-cancel-status.json", "w").write(r.stdout)
    print("Archiving logs (max 30 s; a never-started Job may have none)...", flush=True)
    logs = cli("ai", "job", "logs", job, timeout=30)
    open(f"metrics/{name}-cancel.log", "w").write(logs.stdout + logs.stderr)
    print(f"Archived status/logs of {name} (state {info['status'].get('state')}) to metrics/")
    d = cli("ai", "job", "delete", job, "--async", "--format", "json")
    open(f"config/{name}-cancel-delete.txt", "w").write(d.stdout + d.stderr)
    print("Delete requested" if not d.returncode else "Delete error: " + d.stderr[:300])
TERMINAL = {"COMPLETED", "FAILED", "ERROR", "CANCELLED", "CANCELED", "STOPPED", "DELETED"}
for _ in range(24):
    items = json.loads(cli("ai", "job", "list", "--parent-id", pid, "--format", "json").stdout or "{}").get("items", [])
    live = [(i["metadata"]["name"], i["status"].get("state")) for i in items if i.get("status", {}).get("state") not in TERMINAL]
    if not live:
        print(datetime.now(timezone.utc).isoformat(), "No live Jobs in", pid)
        break
    print("Still live:", live)
    time.sleep(10)
else:
    raise SystemExit("Job still present after 4 minutes - check the console")
PY
