"""Merge LoRA adapters into the base and export deployment artifacts.

For each variant, produces whatever formats are listed in
``variants.yaml :: <variant>.export.formats``:

    * safetensors-bf16   -> HF-ready ``<out>/hf_bf16/``
    * safetensors-int8   -> HF-ready bnb-int8 weights
    * gguf-q4_k_m        -> llama.cpp GGUF Q4_K_M (requires llama.cpp checkout)
    * gguf-q5_k_m        -> llama.cpp GGUF Q5_K_M

GGUF export shells out to llama.cpp's ``convert-hf-to-gguf.py`` and
``llama-quantize``; pass ``--llama-cpp /path/to/llama.cpp`` if it's not
on PATH.

Usage:
    python training/export.py \
        --variant local \
        --adapters out/solace_local_dpo/adapters \
        --out dist/solace-local
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_variant(name: str) -> dict:
    return yaml.safe_load((REPO_ROOT / "training" / "variants.yaml").read_text())[name]


def merge_lora(variant: dict, adapters: Path, dtype: torch.dtype) -> tuple:
    base = variant["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=dtype, attn_implementation="sdpa"
    )
    model = PeftModel.from_pretrained(model, str(adapters))
    model = model.merge_and_unload()
    return model, tokenizer


def export_bf16(model, tokenizer, out: Path) -> Path:
    target = out / "hf_bf16"
    target.mkdir(parents=True, exist_ok=True)
    model.to(torch.bfloat16).save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)
    print(f"[export] bf16 safetensors -> {target}")
    return target


def export_int8(model, tokenizer, out: Path) -> Path:
    from transformers import BitsAndBytesConfig
    target = out / "hf_int8"
    target.mkdir(parents=True, exist_ok=True)
    # bnb-int8 is loaded-as rather than saved-as; we ship bf16 + a loader hint.
    model.to(torch.bfloat16).save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)
    (target / "loader_hint.txt").write_text(
        "Load with:\n"
        "  AutoModelForCausalLM.from_pretrained(\n"
        "      path, quantization_config=BitsAndBytesConfig(load_in_8bit=True))\n"
    )
    print(f"[export] int8-ready bf16 + loader hint -> {target}")
    return target


def export_gguf(bf16_dir: Path, out: Path, quant: str, llama_cpp: Path | None) -> Path:
    if llama_cpp is None:
        llama_cpp = Path(shutil.which("llama-quantize") or "").parent.parent
    convert = llama_cpp / "convert_hf_to_gguf.py"
    quantize = llama_cpp / "build" / "bin" / "llama-quantize"
    if not convert.exists() or not quantize.exists():
        raise SystemExit(
            f"llama.cpp not found (looked in {llama_cpp}). Pass --llama-cpp /path/to/llama.cpp."
        )

    f16_gguf = out / f"{out.name}-f16.gguf"
    quant_gguf = out / f"{out.name}-{quant.lower()}.gguf"
    out.mkdir(parents=True, exist_ok=True)

    print(f"[export] convert -> {f16_gguf}")
    subprocess.run(
        ["python", str(convert), str(bf16_dir), "--outfile", str(f16_gguf), "--outtype", "f16"],
        check=True,
    )
    print(f"[export] quantize {quant} -> {quant_gguf}")
    subprocess.run([str(quantize), str(f16_gguf), str(quant_gguf), quant], check=True)
    f16_gguf.unlink(missing_ok=True)
    return quant_gguf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["local", "mid", "hf"])
    ap.add_argument("--adapters", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--llama-cpp", type=Path, default=None,
                    help="Path to llama.cpp checkout (for GGUF export).")
    args = ap.parse_args()

    variant = load_variant(args.variant)
    dtype = getattr(torch, variant.get("dtype", "bfloat16"))
    args.out.mkdir(parents=True, exist_ok=True)

    model, tokenizer = merge_lora(variant, args.adapters, dtype)
    formats = variant["export"]["formats"]

    bf16_dir = None
    if "safetensors-bf16" in formats:
        bf16_dir = export_bf16(model, tokenizer, args.out)
    if "safetensors-int8" in formats:
        export_int8(model, tokenizer, args.out)

    gguf_formats = [f for f in formats if f.startswith("gguf-")]
    if gguf_formats:
        if bf16_dir is None:
            bf16_dir = export_bf16(model, tokenizer, args.out)
        for f in gguf_formats:
            quant = f.replace("gguf-", "").upper()  # e.g. Q4_K_M
            export_gguf(bf16_dir, args.out, quant, args.llama_cpp)

    print(f"[export] all artifacts -> {args.out}")


if __name__ == "__main__":
    main()
