"""Sana 1.6B seed batch (Claude, 21 Nov 2026), derived from an earlier internal Sana worker by Codex.

Same loading recipe as the earlier Sana runs (fp16 variant, text encoder and VAE in bf16,
diffusers default complex-human-instruction) so results stay comparable; added:
generic identity/forbidden-term checks, exact token check BEFORE download,
download timed separately from loading, metrics format shared with the other
workers (pipeline_load_s, model_download_s) for review_keyframe_batch.py.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

MODEL = "Efficient-Large-Model/Sana_1600M_1024px_diffusers"


def check_identity(spec):
    texts = [spec.get("prompt", ""), spec.get("negative_prompt", "")]
    if "required_phrase" in spec and spec["required_phrase"] not in spec["prompt"]:
        raise ValueError("Prompt lacks required phrase: " + spec["required_phrase"])
    for term in spec.get("forbidden_terms", []):
        if any(term.lower() in t.lower() for t in texts):
            raise ValueError("Forbidden term in prompt: " + term)


def validate(spec):
    if spec["model"] != MODEL or len(spec["revision"]) != 40:
        raise ValueError("This worker runs pinned Sana 1600M only")
    if not 1 <= spec["count"] <= 30 or len(spec["seeds"]) != spec["count"] or len(set(spec["seeds"])) != spec["count"]:
        raise ValueError("Expected 1 to 30 distinct seeds matching count")
    if spec["width"] % 32 or spec["height"] % 32:
        raise ValueError("Dimensions must be divisible by 32")
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
        print(f"Validated {spec['count']} Sana variants; no GPU imports or allocation")
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    metrics = {"status": "running", "spec_sha256": hashlib.sha256(raw).hexdigest(), "model": spec["model"],
               "revision": spec["revision"], "worker_started_unix": started,
               "environment_setup_s": max(0, started - float(os.environ.get("BOOT_UNIX", started))),
               "python": platform.python_version(), "requested_count": spec["count"],
               "settings": {k: spec[k] for k in ["width", "height", "num_inference_steps", "guidance_scale", "max_sequence_length"]},
               "transformer_precision": "fp16 (text encoder and VAE bf16)", "assets": []}

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
        from diffusers import SanaPipeline
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        metrics["import_s"] = time.perf_counter() - t0
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required")
        metrics.update(gpu=torch.cuda.get_device_name(), torch=torch.__version__, diffusers=diffusers.__version__,
                       transformers=transformers.__version__, cuda=torch.version.cuda,
                       gpu_total_gib=torch.cuda.get_device_properties(0).total_memory / 2**30)
        save()

        t0 = time.perf_counter()
        tok_dir = snapshot_download(spec["model"], revision=spec["revision"], allow_patterns=["tokenizer/*"])
        tok = AutoTokenizer.from_pretrained(tok_dir, subfolder="tokenizer")
        metrics["prompt_tokens"] = len(tok.encode(spec["prompt"]))
        metrics["tokenizer_check_s"] = time.perf_counter() - t0
        save()
        event("tokens", prompt=metrics["prompt_tokens"])
        if metrics["prompt_tokens"] > spec["max_sequence_length"]:
            raise ValueError("Prompt exceeds max_sequence_length; refusing silent truncation")

        t0 = time.perf_counter()
        local = SanaPipeline.download(spec["model"], revision=spec["revision"], variant=spec["variant"])
        metrics["model_download_s"] = time.perf_counter() - t0
        save()
        event("downloaded", seconds=metrics["model_download_s"])

        t0 = time.perf_counter()
        pipe = SanaPipeline.from_pretrained(local, variant=spec["variant"], torch_dtype=torch.float16).to("cuda")
        pipe.text_encoder.to(torch.bfloat16)
        pipe.vae.to(torch.bfloat16)
        pipe.set_progress_bar_config(disable=True)
        torch.cuda.synchronize()
        metrics["pipeline_load_s"] = metrics["model_load_s"] = time.perf_counter() - t0
        metrics["gpu_free_after_load_gib"] = torch.cuda.mem_get_info()[0] / 2**30
        save()
        event("model_loaded", load_s=metrics["model_load_s"])

        gen_started = time.time()
        for index, seed in enumerate(spec["seeds"], start=1):
            if index > 1 and time.time() - gen_started > spec["generation_wall_budget_s"]:
                metrics["stopped_by_budget"] = True
                break
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            image = pipe(prompt=spec["prompt"], negative_prompt=spec["negative_prompt"],
                         width=spec["width"], height=spec["height"],
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
