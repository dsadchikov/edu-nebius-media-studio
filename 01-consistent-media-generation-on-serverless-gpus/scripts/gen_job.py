"""Nebius Serverless AI Job lifecycle for one text-to-image batch.

Actions: check (validate + `ai job create --dry-run`), run, watch, status, fetch,
cleanup, inventory, path, state.
Environment: MODEL_KIND = sana | qwen | zimage ; GPU_PROFILE = l40s | h100 ;
NEBIUS_CLI (default: `nebius` on PATH, then ~/.local/bin/nebius) ;
S3_PROFILE (AWS-style profile with the bucket's access key, default "default").
Local files (not in Git): config/resources.json, config/subnets.json,
config/base-image.json; state, Job records and outputs go to config/, metrics/, outputs/.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
CLI = os.environ.get("NEBIUS_CLI") or shutil.which("nebius") or str(Path.home() / ".local/bin/nebius")
TERMINAL = {"COMPLETED", "FAILED", "ERROR", "CANCELLED", "CANCELED", "STOPPED", "DELETED"}
KIND = os.environ.get("MODEL_KIND", "qwen")
PROFILE = os.environ.get("GPU_PROFILE", "l40s")
KINDS = {"sana": "sana_worker.py", "qwen": "qwen_worker.py", "zimage": "zimage_worker.py"}
PROFILES = {
    "l40s": {"platform": "gpu-l40s-a", "preset": "1gpu-8vcpu-32gb", "disk": "200Gi", "timeout": "75m",
             "rates": {"preemptible_fixed": 0.65 + 8 * 0.006 + 32 * 0.0016}},
    "h100": {"platform": "gpu-h100-sxm", "preset": "1gpu-16vcpu-200gb", "disk": "200Gi", "timeout": "60m",
             "rates": {"spot_minimum": 0.79, "on_demand_ceiling": 4.50}},
}
CFG = PROFILES[PROFILE]
WORKER = ROOT / "scripts" / KINDS[KIND]
LAST = ROOT / f"config/last-gen-{KIND}-{PROFILE}.json"
for _d in ("config", "metrics", "outputs"):
    (ROOT / _d).mkdir(exist_ok=True)


def call(*args):
    r = subprocess.run([CLI, *args, "--format", "json"], capture_output=True, text=True, timeout=120)
    if r.returncode:
        raise RuntimeError(r.stderr)
    return json.loads(r.stdout) if r.stdout.strip() else {}


def get_run():
    return json.loads(LAST.read_text())


def status(run):
    x = call("ai", "job", "get", run["job_id"])
    (ROOT / "metrics" / (run["run_id"] + "-job.json")).write_text(json.dumps(x, indent=2))
    return x


def s3client():
    import boto3
    return boto3.Session(profile_name=os.environ.get("S3_PROFILE", "default")).client(
        "s3", endpoint_url="https://storage.eu-north1.nebius.cloud", region_name="eu-north1")


def submit(real, spec_path):
    spec_mod = importlib.util.spec_from_file_location("worker", WORKER)
    worker = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(worker)
    spec = json.loads(spec_path.read_text())
    worker.validate(spec)
    resources = json.loads((ROOT / "config/resources.json").read_text())
    active = call("ai", "job", "list", "--parent-id", resources["project_id"])
    if any(x.get("status", {}).get("state") not in TERMINAL for x in active.get("items", [])):
        raise RuntimeError("An active Job exists in this project; refusing to create another")
    for item in spec.get("inputs", []):
        if hashlib.sha256((ROOT / item["server_path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Input hash mismatch on server: " + item["server_path"])
    subnet = json.loads((ROOT / "config/subnets.json").read_text())["items"][0]["metadata"]["id"]
    digest = json.loads((ROOT / "config/base-image.json").read_text())["digest"]
    run_id = f"{KIND}-{PROFILE}-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    env = ["--env", "RUN_ID=" + run_id]
    if spec.get("inputs"):
        env += ["--env", "INPUTS_DIR=/data/runs/inputs/" + run_id]
    args = [CLI, "ai", "job", "create", "--name", run_id, "--parent-id", resources["project_id"],
            "--image", "pytorch/pytorch@" + digest, "--platform", CFG["platform"], "--preset", CFG["preset"],
            "--subnet-id", subnet, "--disk-size", CFG["disk"], "--preemptible", "--follows-spot-price",
            "--timeout", CFG["timeout"], "--restart-policy", "never",
            "--volume", resources["bucket_id"] + ":/data:rw",
            "--inject-file", str(WORKER) + ":/app/worker.py",
            "--inject-file", str(ROOT / "scripts/gen_bootstrap.sh") + ":/app/bootstrap.sh",
            "--inject-file", str(spec_path) + ":/app/keyframe-spec.json",
            *env, "--container-command", "bash", "--args", "/app/bootstrap.sh",
            "--format", "json", "--async" if real else "--dry-run"]
    if real and spec.get("inputs"):
        s3 = s3client()
        for item in spec["inputs"]:
            s3.upload_file(str(ROOT / item["server_path"]), resources["bucket_name"],
                           f"runs/inputs/{run_id}/{item['file']}")
    r = subprocess.run(args, capture_output=True, text=True, timeout=120)
    (ROOT / "config" / (run_id + ("-operation.json" if real else "-dry-run.json"))).write_text(r.stdout + r.stderr)
    if r.returncode:
        raise RuntimeError(r.stderr)
    if not real:
        print(f"{KIND} x{spec['count']} on {PROFILE} ({CFG['platform']}/{CFG['preset']}) dry-run accepted; no GPU allocated")
        return
    op = json.loads(r.stdout)
    run = {"run_id": run_id, "job_id": op["resource_id"], "operation_id": op["id"], "model_kind": KIND,
           "gpu_profile": PROFILE, "project_id": resources["project_id"], "bucket_name": resources["bucket_name"],
           "spec_file": str(spec_path.relative_to(ROOT)), "model": spec["model"], "model_revision": spec["revision"],
           "platform": CFG["platform"], "preset": CFG["preset"], "billing": "preemptible", "disk": CFG["disk"],
           "requested_count": spec["count"], "hourly_compute_usd": min(CFG["rates"].values()),
           "rate_scenarios_usd_per_hour": CFG["rates"], "timeout": CFG["timeout"],
           "output_prefix": "runs/keyframes/" + run_id + "/",
           "price_source": "https://docs.nebius.com/compute/resources/pricing", "price_checked_on": "2026-11-21",
           "submitted_by": "scripts/run_gen.sh"}
    LAST.write_text(json.dumps(run, indent=2))
    print(json.dumps(run, indent=2), flush=True)


def watch():
    run = get_run()
    previous, history = None, ROOT / "metrics" / (run["run_id"] + "-observations.jsonl")
    for _ in range(360):
        try:
            x = status(run)
        except Exception as exc:
            print(json.dumps({"watch_error": str(exc)[:300]}), flush=True)
            time.sleep(15)
            continue
        s = x.get("status", {})
        with history.open("a") as f:
            f.write(json.dumps({"observed_at": datetime.now(timezone.utc).isoformat(), "status": s}) + "\n")
        if s.get("state") != previous:
            print(json.dumps({"observed_at": datetime.now(timezone.utc).isoformat(), "state": s.get("state")}), flush=True)
            previous = s.get("state")
        if s.get("state") in TERMINAL:
            return
        time.sleep(15)
    raise RuntimeError("Monitoring bound reached; cleanup will delete the Job")


def run_dir():
    return ROOT / "outputs/keyframes" / get_run()["run_id"]


def fetch():
    run = get_run()
    s3 = s3client()
    target = run_dir()
    target.mkdir(parents=True, exist_ok=True)
    for obj in s3.list_objects_v2(Bucket=run["bucket_name"], Prefix=run["output_prefix"]).get("Contents", []):
        name = obj["Key"].removeprefix(run["output_prefix"])
        if name and "/" not in name and name not in (".", ".."):
            s3.download_file(run["bucket_name"], obj["Key"], str(target / name))
    m = json.loads((target / "metrics.json").read_text())
    assert hashlib.sha256((target / "generation-spec.json").read_bytes()).hexdigest() == m["spec_sha256"]
    for e in m.get("assets", []):
        assert hashlib.sha256((target / (e["id"] + ".png")).read_bytes()).hexdigest() == e["sha256"]
    print(json.dumps({"status": m["status"], "verified_images": len(m.get("assets", [])), "error": m.get("error")}, indent=2))


def cleanup():
    run = get_run()
    try:
        status(run)
    except Exception as exc:
        print("status before cleanup failed:", exc)
    r = subprocess.run([CLI, "ai", "job", "logs", run["job_id"]], capture_output=True, text=True, timeout=180)
    (ROOT / "metrics" / (run["run_id"] + ".log")).write_text(r.stdout + r.stderr)
    r = subprocess.run([CLI, "ai", "job", "delete", run["job_id"], "--async", "--format", "json"],
                       capture_output=True, text=True, timeout=120)
    (ROOT / "config" / (run["run_id"] + "-delete.txt")).write_text(r.stdout + r.stderr)
    print("Logs archived; Job deletion requested" if not r.returncode else "Delete returned: " + r.stderr[:300])


def inventory():
    run = get_run()
    report = {"checked_at": datetime.now(timezone.utc).isoformat()}
    for label, args in {"jobs": ("ai", "job", "list"), "endpoints": ("ai", "endpoint", "list"),
                        "instances": ("compute", "instance", "list"), "disks": ("compute", "disk", "list")}.items():
        for _ in range(10):
            try:
                items = call(*args, "--parent-id", run["project_id"]).get("items", [])
            except Exception as exc:
                report[label] = "check failed: " + str(exc)[:200]
                break
            live = [i.get("metadata", {}).get("name") for i in items if i.get("status", {}).get("state") not in TERMINAL]
            report[label] = live
            if not live:
                break
            time.sleep(30)
    (ROOT / "metrics" / (run["run_id"] + "-inventory.json")).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["check", "run", "watch", "status", "fetch", "cleanup", "inventory", "path", "state"])
    p.add_argument("--spec", type=Path)
    a = p.parse_args()
    if a.action in ("check", "run"):
        submit(a.action == "run", a.spec.resolve())
    elif a.action == "path":
        print(run_dir())
    elif a.action == "state":
        print(LAST)
    else:
        {"watch": watch, "fetch": fetch, "cleanup": cleanup, "inventory": inventory,
         "status": lambda: print(json.dumps(status(get_run())["status"], indent=2))}[a.action]()
