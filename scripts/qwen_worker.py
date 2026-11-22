"""Qwen-Image seed batch with a single model load (fits a 48 GB L40S; used unchanged on H100 80 GB).

Memory plan (authored by Claude, 2026-11-21):
  * BF16 transformer (~38.05 GiB) + text encoder (~15.45 GiB) cannot share 48 GB VRAM.
  * The prompt is identical for every seed, so text embeddings are computed ONCE,
    the text encoder is released, and only transformer + VAE stay on the GPU.
  * The transformer is loaded straight to CUDA (device_map) so the 32 GB host RAM
    preset is never asked to hold it.
  * VAE decode is tiled to bound the decode memory spike.
  * Optional, recorded fallback: FP8 *storage* (BF16 compute) for the transformer
    only if free VRAM after loading is below the spec threshold.
Everything measured is written to metrics.json after every step, so a failed or
pre-empted Job still leaves evidence in S3.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import time

TEMPLATE = ("<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, "
            "text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}"
            "<|im_end|>\n<|im_start|>assistant\n")
DROP_IDX = 34  # diffusers 0.39.0 QwenImagePipeline.prompt_template_encode_start_idx


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
    if spec["model"] != "Qwen/Qwen-Image" or len(spec["revision"]) != 40:
        raise ValueError("Need pinned Qwen/Qwen-Image revision")
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
        print(f"Validated {spec['count']} Qwen-Image variants; no GPU imports or allocation")
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    keys = ["width", "height", "num_inference_steps", "true_cfg_scale", "max_sequence_length"]
    metrics = {"status": "running", "spec_sha256": hashlib.sha256(raw).hexdigest(),
               "model": spec["model"], "revision": spec["revision"], "worker_started_unix": started,
               "environment_setup_s": max(0, started - float(os.environ.get("BOOT_UNIX", started))),
               "python": platform.python_version(), "requested_count": spec["count"],
               "settings": {k: spec[k] for k in keys}, "memory_strategy": spec["memory_strategy"],
               "transformer_precision": "bf16", "assets": []}

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
        from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        metrics["import_s"] = time.perf_counter() - t0
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required")
        metrics.update(gpu=torch.cuda.get_device_name(), torch=torch.__version__, diffusers=diffusers.__version__,
                       transformers=transformers.__version__, cuda=torch.version.cuda,
                       gpu_total_gib=torch.cuda.get_device_properties(0).total_memory / 2**30)
        save()

        # 1. Cheap fail-fast: exact token count before downloading ~57 GB of weights.
        t0 = time.perf_counter()
        tok_dir = snapshot_download(spec["model"], revision=spec["revision"], allow_patterns=["tokenizer/*"])
        tok = AutoTokenizer.from_pretrained(tok_dir, subfolder="tokenizer")
        count = lambda text: len(tok(TEMPLATE.format(text)).input_ids) - DROP_IDX
        metrics["prompt_tokens"] = count(spec["prompt"])
        metrics["negative_prompt_tokens"] = count(spec["negative_prompt"])
        metrics["tokenizer_check_s"] = time.perf_counter() - t0
        save()
        event("tokens", prompt=metrics["prompt_tokens"], negative=metrics["negative_prompt_tokens"])
        if max(metrics["prompt_tokens"], metrics["negative_prompt_tokens"]) > spec["max_sequence_length"]:
            raise ValueError("Prompt exceeds max_sequence_length; refusing silent truncation")

        # 2. Download measured separately from loading.
        t0 = time.perf_counter()
        local = snapshot_download(spec["model"], revision=spec["revision"])
        metrics["model_download_s"] = time.perf_counter() - t0
        save()
        event("downloaded", seconds=metrics["model_download_s"])

        # 3. Text encoder phase: encode once, then release it.
        bf16 = torch.bfloat16
        t0 = time.perf_counter()
        pipe = QwenImagePipeline.from_pretrained(local, transformer=None, torch_dtype=bf16)
        pipe.text_encoder.to("cuda")
        pipe.vae.to("cuda")
        torch.cuda.synchronize()
        metrics["text_encoder_load_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        with torch.inference_mode():
            pe, pm = pipe.encode_prompt(prompt=spec["prompt"], device="cuda",
                                        max_sequence_length=spec["max_sequence_length"])
            ne, nm = pipe.encode_prompt(prompt=spec["negative_prompt"], device="cuda",
                                        max_sequence_length=spec["max_sequence_length"])
        torch.cuda.synchronize()
        metrics["prompt_encode_s"] = time.perf_counter() - t0
        pipe.register_modules(text_encoder=None)
        gc.collect()
        torch.cuda.empty_cache()
        save()
        event("prompt_encoded", seconds=metrics["prompt_encode_s"])

        # 4. Transformer straight to the GPU.
        t0 = time.perf_counter()
        transformer = QwenImageTransformer2DModel.from_pretrained(
            local, subfolder="transformer", torch_dtype=bf16, device_map="cuda")
        pipe.register_modules(transformer=transformer)
        torch.cuda.synchronize()
        metrics["transformer_load_s"] = time.perf_counter() - t0
        metrics["model_load_s"] = metrics["text_encoder_load_s"] + metrics["transformer_load_s"]
        free, _ = torch.cuda.mem_get_info()
        metrics["gpu_free_after_load_gib"] = free / 2**30
        if free / 2**30 < spec["fp8_fallback_if_free_gib_below"]:
            transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=bf16)
            gc.collect()
            torch.cuda.empty_cache()
            metrics["transformer_precision"] = "fp8-storage/bf16-compute (fallback)"
            metrics["gpu_free_after_fallback_gib"] = torch.cuda.mem_get_info()[0] / 2**30
        pipe.vae.enable_tiling()
        pipe.set_progress_bar_config(disable=True)
        save()
        event("model_loaded", load_s=metrics["model_load_s"], free_gib=metrics["gpu_free_after_load_gib"],
              precision=metrics["transformer_precision"])

        # 5. Sequential seeds with a wall-clock budget below the Job timeout.
        gen_started = time.time()
        for index, seed in enumerate(spec["seeds"], start=1):
            if index > 1 and time.time() - gen_started > spec["generation_wall_budget_s"]:
                metrics["stopped_by_budget"] = True
                break
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                image = pipe(prompt_embeds=pe, prompt_embeds_mask=pm,
                             negative_prompt_embeds=ne, negative_prompt_embeds_mask=nm,
                             true_cfg_scale=spec["true_cfg_scale"], width=spec["width"], height=spec["height"],
                             num_inference_steps=spec["num_inference_steps"],
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
