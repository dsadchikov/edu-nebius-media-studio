"""Z-Image-Turbo seed batch with one model load (Claude, 21 Nov 2026).

Same metrics format as qwen_worker.py so review_keyframe_batch.py works.
Z-Image-Turbo is a distilled few-step model: guidance_scale 0.0, so the negative
prompt is NOT used (recorded in metrics). 6B transformer + Qwen3 text encoder fit
an L40S in BF16 without offloading. The revision is resolved to a commit hash at
runtime and recorded in metrics.json.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time


def check_identity(spec):
    """Specs declare required_phrase (character identity stays explicit) and optional forbidden_terms."""
    texts = [spec.get("prompt", ""), spec.get("negative_prompt", "")]
    if "required_phrase" not in spec:
        raise ValueError("Spec must declare required_phrase")
    if spec["required_phrase"] not in spec["prompt"]:
        raise ValueError("Prompt lacks required phrase: " + spec["required_phrase"])
    for term in spec.get("forbidden_terms", []):
        if any(term.lower() in t.lower() for t in texts):
            raise ValueError("Forbidden term in prompt: " + term)


def validate(spec):
    if spec["model"] != "Tongyi-MAI/Z-Image-Turbo":
        raise ValueError("This worker runs Tongyi-MAI/Z-Image-Turbo only")
    if not 1 <= spec["count"] <= 30 or len(spec["seeds"]) != spec["count"] or len(set(spec["seeds"])) != spec["count"]:
        raise ValueError("Expected 1 to 30 distinct seeds matching count")
    if spec["width"] % 16 or spec["height"] % 16:
        raise ValueError("Bad dimensions")
    check_identity(spec)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--validate-only", action="store_true")
    a = p.parse_args()
    raw = Path(a.spec).read_bytes()
    spec = json.loads(raw)
    validate(spec)
    if a.validate_only:
        print(f"Validated {spec['count']} Z-Image-Turbo variants; no GPU imports or allocation")
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    metrics = {"status": "running", "spec_sha256": hashlib.sha256(raw).hexdigest(), "model": spec["model"],
               "revision_requested": spec["revision"], "worker_started_unix": started,
               "environment_setup_s": max(0, started - float(os.environ.get("BOOT_UNIX", started))),
               "python": platform.python_version(), "requested_count": spec["count"],
               "settings": {k: spec[k] for k in ["width", "height", "num_inference_steps", "guidance_scale", "max_sequence_length"]},
               "negative_prompt_used": spec["guidance_scale"] > 0, "assets": []}

    def save():
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    def event(name, **kw):
        print(json.dumps({"event": name, **kw}), flush=True)

    (out / "generation-spec.json").write_bytes(raw)
    save()
    try:
        t0 = time.perf_counter()
        import torch
        import diffusers
        import transformers
        from diffusers import ZImagePipeline
        from huggingface_hub import HfApi, snapshot_download
        from transformers import AutoTokenizer
        metrics["import_s"] = time.perf_counter() - t0
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required")
        metrics.update(gpu=torch.cuda.get_device_name(), torch=torch.__version__, diffusers=diffusers.__version__,
                       transformers=transformers.__version__, cuda=torch.version.cuda,
                       gpu_total_gib=torch.cuda.get_device_properties(0).total_memory / 2**30)
        rev = HfApi().model_info(spec["model"], revision=spec["revision"]).sha
        metrics["revision"] = rev
        save()

        t0 = time.perf_counter()
        tok_dir = snapshot_download(spec["model"], revision=rev, allow_patterns=["tokenizer/*"])
        tok = AutoTokenizer.from_pretrained(tok_dir, subfolder="tokenizer")
        templated = tok.apply_chat_template([{"role": "user", "content": spec["prompt"]}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=True)
        metrics["prompt_tokens"] = len(tok(templated).input_ids)
        metrics["tokenizer_check_s"] = time.perf_counter() - t0
        save()
        event("tokens", prompt=metrics["prompt_tokens"])
        if metrics["prompt_tokens"] > spec["max_sequence_length"]:
            raise ValueError("Prompt exceeds max_sequence_length; refusing silent truncation")

        t0 = time.perf_counter()
        local = snapshot_download(spec["model"], revision=rev)
        metrics["model_download_s"] = time.perf_counter() - t0
        save()
        event("downloaded", seconds=metrics["model_download_s"])

        t0 = time.perf_counter()
        pipe = ZImagePipeline.from_pretrained(local, torch_dtype=torch.bfloat16).to("cuda")
        pipe.set_progress_bar_config(disable=True)
        torch.cuda.synchronize()
        metrics["pipeline_load_s"] = metrics["model_load_s"] = time.perf_counter() - t0
        metrics["gpu_free_after_load_gib"] = torch.cuda.mem_get_info()[0] / 2**30
        metrics["transformer_precision"] = "bf16"
        save()
        event("model_loaded", load_s=metrics["model_load_s"], free_gib=metrics["gpu_free_after_load_gib"])

        t0 = time.perf_counter()
        with torch.inference_mode():
            pe, _ = pipe.encode_prompt(prompt=spec["prompt"], device="cuda", do_classifier_free_guidance=False,
                                       max_sequence_length=spec["max_sequence_length"])
        torch.cuda.synchronize()
        metrics["prompt_encode_s"] = time.perf_counter() - t0

        gen_started = time.time()
        for index, seed in enumerate(spec["seeds"], start=1):
            if index > 1 and time.time() - gen_started > spec["generation_wall_budget_s"]:
                metrics["stopped_by_budget"] = True
                break
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                image = pipe(prompt_embeds=pe, width=spec["width"], height=spec["height"],
                             num_inference_steps=spec["num_inference_steps"], guidance_scale=spec["guidance_scale"],
                             max_sequence_length=spec["max_sequence_length"],
                             generator=torch.Generator("cuda").manual_seed(seed)).images[0]
            torch.cuda.synchronize()
            seconds = time.perf_counter() - t0
            path = out / f"variant-{index:02}.png"
            t0 = time.perf_counter()
            image.save(path)
            entry = {"id": f"variant-{index:02}", "seed": seed, "generation_s": seconds,
                     "save_s": time.perf_counter() - t0, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            metrics["assets"].append(entry)
            metrics["peak_vram_gib"] = torch.cuda.max_memory_allocated() / 2**30
            save()
            event("generated", **entry)
        metrics["status"] = "completed" if len(metrics["assets"]) == spec["count"] else "partial"
    except BaseException as exc:
        metrics["status"] = "failed"
        metrics["error_type"] = type(exc).__name__
        metrics["error"] = str(exc)[:2000]
        raise
    finally:
        metrics["worker_elapsed_s"] = time.time() - started
        save()


if __name__ == "__main__":
    main()
