"""Supervised fine-tune a pre-trained base into a Claude Solace variant.

Single-entrypoint trainer used by all three variants. The variant tag picks
its row out of ``training/variants.yaml`` (base model, LoRA config, dtype,
export format). QLoRA kicks in automatically when ``peft.method == qlora``,
which is the default for the ``local`` variant so it fits on a free Colab T4.

Usage:
    accelerate launch training/train_sft.py \
        --variant local \
        --data data/sft_mix \
        --out out/solace_local_sft \
        --epochs 2 \
        --seq-len 2048 \
        --batch-size 1 \
        --grad-accum 16
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_variant(name: str) -> dict:
    spec = yaml.safe_load((REPO_ROOT / "training" / "variants.yaml").read_text())
    if name not in spec:
        raise SystemExit(f"unknown variant {name!r}; available: {list(spec)}")
    return spec[name]


def build_model_and_tokenizer(variant: dict):
    base = variant["base_model"]
    peft_cfg = variant["peft"]
    dtype = getattr(torch, variant.get("dtype", "bfloat16"))

    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = {
        "torch_dtype": dtype,
        "attn_implementation": "sdpa",  # "flash_attention_2" if installed
    }

    if peft_cfg["method"] == "qlora":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(base, **load_kwargs)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    if peft_cfg["method"] == "qlora":
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=peft_cfg["r"],
        lora_alpha=peft_cfg["alpha"],
        lora_dropout=peft_cfg.get("dropout", 0.0),
        target_modules=peft_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["local", "mid", "hf", "high"])
    ap.add_argument("--data", type=Path, required=True, help="output of prepare_data.py --stage sft")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--eval-split", type=float, default=0.005,
                    help="Fraction of the SFT mix held out for eval loss.")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    variant = load_variant(args.variant)

    ds = load_from_disk(str(args.data))
    if args.eval_split > 0:
        split = ds.train_test_split(test_size=args.eval_split, seed=17)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = ds, None

    model, tokenizer = build_model_and_tokenizer(variant)

    sft_cfg = SFTConfig(
        output_dir=str(args.out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.0,
        max_seq_length=args.seq_len,
        packing=True,                      # Pack short samples to fill seq_len
        dataset_text_field="text",
        logging_steps=20,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.save_steps if eval_ds is not None else None,
        bf16=variant.get("dtype", "bfloat16") == "bfloat16",
        fp16=variant.get("dtype") == "float16",
        optim="paged_adamw_8bit" if variant["peft"]["method"] == "qlora" else "adamw_torch",
        report_to=os.environ.get("SOLACE_REPORT_TO", "none"),
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )
    trainer.train(resume_from_checkpoint=args.resume)

    # Save LoRA adapters only here; ``export.py`` does the merge.
    trainer.model.save_pretrained(args.out / "adapters")
    tokenizer.save_pretrained(args.out / "adapters")
    print(f"[sft] done -> {args.out}/adapters")


if __name__ == "__main__":
    main()
