"""Direct Preference Optimization for a Claude Solace variant.

Runs after SFT. Loads the SFT adapters as the policy, freezes the base in
4-bit for QLoRA variants (keeps VRAM low), and optimizes on the DPO mix
produced by ``prepare_data.py --stage dpo``.

Usage:
    accelerate launch training/train_dpo.py \
        --variant local \
        --sft out/solace_local_sft/adapters \
        --data data/dpo_mix \
        --out out/solace_local_dpo \
        --epochs 1 \
        --beta 0.1
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_variant(name: str) -> dict:
    return yaml.safe_load((REPO_ROOT / "training" / "variants.yaml").read_text())[name]


def build_policy(variant: dict, sft_adapters: Path):
    base = variant["base_model"]
    peft_cfg = variant["peft"]
    dtype = getattr(torch, variant.get("dtype", "bfloat16"))

    load_kwargs: dict = {"torch_dtype": dtype, "attn_implementation": "sdpa"}
    if peft_cfg["method"] == "qlora":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(base, **load_kwargs)
    base_model.config.use_cache = False
    # Wrap with SFT adapters, then add a new DPO LoRA on top.
    policy = PeftModel.from_pretrained(base_model, str(sft_adapters), is_trainable=True)
    return policy, tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["local", "mid", "hf"])
    ap.add_argument("--sft", type=Path, required=True,
                    help="Path to SFT-trained LoRA adapters.")
    ap.add_argument("--data", type=Path, required=True,
                    help="Output of prepare_data.py --stage dpo.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-6)
    args = ap.parse_args()

    variant = load_variant(args.variant)
    ds = load_from_disk(str(args.data))

    policy, tokenizer = build_policy(variant, args.sft)

    dpo_cfg = DPOConfig(
        output_dir=str(args.out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_length=args.seq_len,
        max_prompt_length=args.seq_len // 2,
        beta=args.beta,
        loss_type="sigmoid",               # Vanilla DPO. Try "ipo" for clipped.
        logging_steps=20,
        save_steps=500,
        save_total_limit=2,
        bf16=variant.get("dtype", "bfloat16") == "bfloat16",
        fp16=variant.get("dtype") == "float16",
        optim="paged_adamw_8bit" if variant["peft"]["method"] == "qlora" else "adamw_torch",
        report_to=os.environ.get("SOLACE_REPORT_TO", "none"),
        seed=42,
    )

    trainer = DPOTrainer(
        model=policy,
        ref_model=None,                    # PEFT: reference = adapter-disabled base
        args=dpo_cfg,
        train_dataset=ds,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.model.save_pretrained(args.out / "adapters")
    tokenizer.save_pretrained(args.out / "adapters")
    print(f"[dpo] done -> {args.out}/adapters")


if __name__ == "__main__":
    main()
