"""Verify a downloaded keyframe batch, build uncropped contact sheets and a cost summary.

Model-agnostic successor of review_sana_variants.py (which hard-codes Sana field
names). Fixed cost = queue/startup + environment setup + every one-off worker
phase; marginal cost c = median generation time x hourly rate.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
ONE_OFF = ["import_s", "tokenizer_check_s", "model_download_s", "text_encoder_load_s",
           "prompt_encode_s", "transformer_load_s", "pipeline_load_s"]


def main():
    p = argparse.ArgumentParser()
    import os
    default_state = ROOT / f"config/last-gen-{os.environ.get('MODEL_KIND', 'qwen')}-{os.environ.get('GPU_PROFILE', 'l40s')}.json"
    p.add_argument("--state", type=Path, default=Path(os.environ["STATE_FILE"]) if os.environ.get("STATE_FILE")
                   else default_state)
    a = p.parse_args()
    run = json.loads(a.state.read_text())
    folder = ROOT / "outputs/keyframes" / run["run_id"]
    spec = json.loads((folder / "generation-spec.json").read_text())
    m = json.loads((folder / "metrics.json").read_text())
    assets = m.get("assets", [])
    if not assets:
        raise ValueError(f"No images to review; worker status={m['status']} error={m.get('error')}")
    if hashlib.sha256((folder / "generation-spec.json").read_bytes()).hexdigest() != m["spec_sha256"]:
        raise ValueError("Generation spec digest mismatch")
    for e in assets:
        with Image.open(folder / (e["id"] + ".png")) as img:
            if "width" in spec and img.size != (spec["width"], spec["height"]):
                raise ValueError("Unexpected image dimensions")
    with Image.open(folder / (assets[0]["id"] + ".png")) as img:
        img_w, img_h = img.size
    face = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")

    def sheet(entries, columns, width, name):
        height = round(width * img_h / img_w)
        gap, label = 16, 44
        rows = math.ceil(len(entries) / columns)
        canvas = Image.new("RGB", (gap + columns * (width + gap), gap + rows * (height + label + gap)), "#F5F4F0")
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.truetype(str(face), 22)
        for n, e in enumerate(entries):
            x, y = gap + (n % columns) * (width + gap), gap + (n // columns) * (height + label + gap)
            with Image.open(folder / (e["id"] + ".png")) as raw:
                canvas.paste(ImageOps.contain(raw.convert("RGB"), (width, height)), (x, y))
            label_text = f"{e['id'].removeprefix('variant-')} | seed {e['seed']}" + (f" | {e['source']}" if "source" in e else "")
            draw.text((x, y + height + 8), label_text, font=font, fill="#203648")
        canvas.save(folder / name)

    sheet(assets, 5, 400, "contact-sheet-all.png")
    for s in range(0, len(assets), 4):
        sheet(assets[s:s + 4], 2, 832, f"contact-sheet-{s+1:02}-{min(s+4, len(assets)):02}.png")

    job = json.loads((ROOT / "metrics" / (run["run_id"] + "-job.json")).read_text())
    created = datetime.fromisoformat(job["metadata"]["created_at"].replace("Z", "+00:00"))
    started = job.get("status", {}).get("started_at")
    startup = (datetime.fromisoformat(started.replace("Z", "+00:00")) - created).total_seconds() if started else None
    rate = run["hourly_compute_usd"] / 3600
    times = [e["generation_s"] for e in assets]
    one_off = sum(m.get(k, 0) for k in ONE_OFF)
    fixed_s = (startup or 0) + m["environment_setup_s"] + one_off
    median = statistics.median(times)
    c = median * rate
    report = {
        "run_id": run["run_id"], "job_id": run["job_id"], "model": spec["model"], "revision": spec["revision"],
        "gpu": m.get("gpu"), "platform": run["platform"], "preset": run["preset"], "billing": run["billing"],
        "transformer_precision": m.get("transformer_precision"), "worker_status": m["status"],
        "requested_count": spec["count"], "verified_images": len(assets),
        "distinct_image_hashes": len({e["sha256"] for e in assets}), "size": [img_w, img_h],
        "steps": spec["num_inference_steps"], "true_cfg_scale": spec.get("true_cfg_scale"),
        "guidance_scale": spec.get("guidance_scale"), "model_revision_resolved": m.get("revision"),
        "prompt_tokens": m.get("prompt_tokens"), "negative_prompt_tokens": m.get("negative_prompt_tokens"),
        "startup_s": startup, "environment_setup_s": m["environment_setup_s"],
        **{k: m.get(k) for k in ONE_OFF}, "model_load_s": m.get("model_load_s"),
        "gpu_free_after_load_gib": m.get("gpu_free_after_load_gib"), "peak_vram_gib": m.get("peak_vram_gib"),
        "first_generation_s": times[0], "median_generation_s": median,
        "min_generation_s": min(times), "max_generation_s": max(times), "total_generation_s": sum(times),
        "hourly_compute_usd": run["hourly_compute_usd"],
        "fixed_seconds_per_run": fixed_s, "fixed_compute_usd_per_run": fixed_s * rate,
        "marginal_compute_usd_per_asset_c": c,
        "projected_usd_per_asset_at_N1000": (fixed_s * rate + 1000 * c) / 1000,
        "projected_entire_run_compute_usd": ((startup or 0) + m["environment_setup_s"] + m["worker_elapsed_s"]) * rate,
        "excluded": ["disk", "storage", "egress", "teardown", "tax", "failed/retried runs"],
        "billing_usage_reconciled": False, "quality_status": "pending_visual_review",
    }
    scenarios = {}
    rates = dict(run.get("rate_scenarios_usd_per_hour", {"published": run["hourly_compute_usd"]}))
    if os.environ.get("PAID_HOURLY_USD"):  # the price the run actually paid (Billing -> Usage), per GPU-hour preset
        rates["paid"] = float(os.environ["PAID_HOURLY_USD"])
    for name, hourly in rates.items():
        rr = hourly / 3600
        scenarios[name] = {"hourly_usd": hourly, "fixed_usd_per_run": fixed_s * rr,
                           "marginal_usd_per_asset_c": median * rr,
                           "usd_per_asset_at_N1000": (fixed_s * rr + 1000 * median * rr) / 1000,
                           "entire_run_usd": ((startup or 0) + m["environment_setup_s"] + m["worker_elapsed_s"]) * rr}
    report["cost_scenarios"] = scenarios
    (folder / "run-summary.json").write_text(json.dumps(report, indent=2))
    (folder / "prompt-en.txt").write_text(spec.get("prompt", spec.get("instruction", "")) + "\n\nNegative prompt:\n" + spec["negative_prompt"] + "\n")
    (folder / "seed-index.json").write_text(json.dumps(assets, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
