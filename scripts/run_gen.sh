#!/usr/bin/env bash
# One text-to-image batch on a preemptible Nebius GPU, as one Serverless AI Job.
#
#   bash scripts/run_gen.sh <sana|qwen|zimage> <l40s|h100> <spec file in prompts/> <empty local destination>
#   DRYRUN_ONLY=yes bash scripts/run_gen.sh ...   # validate + Nebius dry-run only: no GPU, no cost
#   CONFIRM=yes   bash scripts/run_gen.sh ...     # do not ask before submitting the paid Job
#
# Where the Nebius CLI runs (the "control machine"):
#   default            - this machine, from the repository root
#   CONTROL_HOST=user@host REMOTE_DIR=/path  - a remote Linux machine over ssh; scripts/, prompts/
#                        and pyproject.toml are copied to REMOTE_DIR, which must hold config/
#   SSH_OPTS="-4 ..."  - extra ssh/scp options
# Other variables passed through: S3_PROFILE, NEBIUS_CLI.
#
# Steps: validate -> dry-run -> confirm -> submit ONE Job -> watch -> fetch from the bucket and verify
# hashes -> contact sheet + run-summary.json -> archive logs, delete the Job -> inventory check -> copy
# results to the destination. The Job is deleted on any exit after submission.
set -euo pipefail
export MODEL_KIND=${1:?model kind: sana|qwen|zimage} GPU_PROFILE=${2:?gpu profile: l40s|h100}
SPEC_NAME=${3:?spec file name in prompts/}
DEST=${4:?local destination folder}
HERE=$(cd "$(dirname "$0")/.." && pwd)
SPEC=prompts/$SPEC_NAME
CONTROL_HOST=${CONTROL_HOST:-}
read -r -a SSH_EXTRA <<< "${SSH_OPTS:-}"
PASS="MODEL_KIND=$MODEL_KIND GPU_PROFILE=$GPU_PROFILE S3_PROFILE=${S3_PROFILE:-default}${NEBIUS_CLI:+ NEBIUS_CLI=$NEBIUS_CLI}"
log() { echo "[$(date -u +%H:%M:%S)Z] $*"; }
if [ -n "$CONTROL_HOST" ]; then
  REMOTE_DIR=${REMOTE_DIR:?set REMOTE_DIR when CONTROL_HOST is used}
  r() { ssh ${SSH_EXTRA[@]+"${SSH_EXTRA[@]}"} -o ServerAliveInterval=30 -o ServerAliveCountMax=10 "$CONTROL_HOST" \
          "cd $REMOTE_DIR && export PATH=\$HOME/.local/bin:\$PATH $PASS && $*"; }
  pull() { scp ${SSH_EXTRA[@]+"${SSH_EXTRA[@]}"} -r "$CONTROL_HOST:$1" "$2"; }
else
  REMOTE_DIR=$HERE
  r() { (cd "$HERE" && env $PASS bash -c "$*"); }
  pull() { cp -R "$1" "$2"; }
fi
[ -f "$HERE/$SPEC" ] || { echo "No spec $HERE/$SPEC"; exit 1; }
if [ "${DRYRUN_ONLY:-}" != yes ] && [ -e "$DEST" ] && [ -n "$(ls -A "$DEST" 2>/dev/null)" ]; then
  echo "Destination $DEST is not empty; refusing to overwrite"; exit 1
fi

if [ -n "$CONTROL_HOST" ]; then
  log "1/9 Copy scripts and prompts to $CONTROL_HOST:$REMOTE_DIR"
  COPYFILE_DISABLE=1 tar --no-xattrs -C "$HERE" -cf - scripts prompts pyproject.toml \
    | ssh ${SSH_EXTRA[@]+"${SSH_EXTRA[@]}"} "$CONTROL_HOST" "mkdir -p $REMOTE_DIR && cd $REMOTE_DIR && tar -xf -"
else
  log "1/9 Local control machine: $HERE"
fi

log "2/9 Validate and dry-run ($MODEL_KIND on $GPU_PROFILE; no GPU)"
r "uv run python scripts/gen_job.py check --spec $SPEC"

if [ "${DRYRUN_ONLY:-}" = yes ]; then log "Dry-run only: nothing submitted."; exit 0; fi
if [ "${CONFIRM:-}" != yes ]; then
  read -r -p "Dry-run accepted. Submit ONE paid preemptible $GPU_PROFILE Job for $MODEL_KIND? [yes/N] " ans
  [ "$ans" = yes ] || { log "Not submitted."; exit 0; }
fi

log "3/9 Submit"
r "uv run python scripts/gen_job.py run --spec $SPEC"
cleanup() {
  log "8/9 Archive logs, delete Job, verify inventory"
  r "uv run python scripts/gen_job.py cleanup" || log "WARNING: cleanup failed - check the Nebius console"
  r "uv run python scripts/gen_job.py inventory" || log "WARNING: inventory failed - check the Nebius console"
}
trap cleanup EXIT

log "4/9 Watch"
r "uv run python scripts/gen_job.py watch" || log "Watch ended with an error; fetching whatever reached the bucket"
log "5/9 Fetch from the bucket and verify hashes"
r "uv run python scripts/gen_job.py fetch" || log "Fetch reported a problem"
STATE=$(r "uv run python scripts/gen_job.py state" | tail -1)
log "6/9 Contact sheet and metrics"
r "STATE_FILE=$STATE uv run python scripts/review_keyframe_batch.py" || log "Review needs at least one image"

log "7/9 Copy to $DEST"
RUN_DIR=$(r "uv run python scripts/gen_job.py path" | tail -1)
mkdir -p "$DEST"
pull "$RUN_DIR/." "$DEST/"
python3 - "$DEST" <<'PY'
import hashlib, json, sys, pathlib
d = pathlib.Path(sys.argv[1]); m = json.loads((d / "metrics.json").read_text())
bad = [e["id"] for e in m.get("assets", []) if hashlib.sha256((d / (e["id"] + ".png")).read_bytes()).hexdigest() != e["sha256"]]
print(json.dumps({"local_dir": str(d), "status": m["status"], "images": len(m.get("assets", [])), "hash_mismatches": bad}))
PY
cleanup; trap - EXIT
log "9/9 Copy Job record, logs and inventory"
RUN_ID=$(basename "$RUN_DIR")
for f in "$RUN_ID-job.json" "$RUN_ID.log" "$RUN_ID-inventory.json" "$RUN_ID-observations.jsonl"; do
  pull "$REMOTE_DIR/metrics/$f" "$DEST/" || true
done
log "Done: $DEST"
