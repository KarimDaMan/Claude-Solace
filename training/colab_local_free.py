"""Single-file QLoRA recipe for the Claude-Solace-Local variant on a Colab
free-tier T4 (16 GB VRAM). Paste into a Colab notebook cell and run.

What it does, end-to-end, in one ~4-6 hr session:
  1. Installs the post-training stack (transformers / peft / trl / bnb / datasets).
  2. Pulls a capped SFT mix from the 2024-2025 sources in ``prepare_data.py``.
  3. QLoRA-SFT Gemma 3 4B on ~50k examples at seq_len=1024.
  4. Optionally QLoRA-DPO on ~10k preference pairs.
  5. Merges adapters, saves BF16 safetensors, and writes a Q4_K_M GGUF if
     llama.cpp is cloned locally.

Trade-offs vs. the full ``train_sft.py`` path:
  - Sequence length capped at 1024 (T4 has 16 GB VRAM).
  - Sample counts dialed down to fit a Colab session.
  - Packing on, grad-accum high to keep effective batch size sane.

Run from repo root as a plain script:
    python training/colab_local_free.py --stage sft
    python training/colab_local_free.py --stage dpo
    python training/colab_local_free.py --stage export
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_SFT = REPO / "data" / "sft_mix_small"
DATA_DPO = REPO / "data" / "dpo_mix_small"
OUT_SFT = REPO / "out" / "solace_local_sft"
OUT_DPO = REPO / "out" / "solace_local_dpo"
DIST = REPO / "dist" / "solace-local"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def install() -> None:
    # Matches training/requirements.txt but pinned narrower for the T4 path.
    pkgs = [
        "torch", "transformers>=4.46", "accelerate>=1.0", "datasets>=3.0",
        "peft>=0.13", "trl>=0.12", "bitsandbytes>=0.44",
        "sentencepiece", "safetensors", "pyyaml",
    ]
    _run([sys.executable, "-m", "pip", "install", "-q", "-U", *pkgs])


def prep_data(stage: str) -> None:
    out = DATA_SFT if stage == "sft" else DATA_DPO
    out.parent.mkdir(parents=True, exist_ok=True)
    # Aggressive per-source caps so the whole set fits in Colab disk (~100 GB)
    # and processes in under an hour.
    _run([
        sys.executable, str(REPO / "training" / "prepare_data.py"),
        "--stage", stage,
        "--out", str(out),
        "--max-per-source", "25000" if stage == "sft" else "8000",
    ])


def train_sft() -> None:
    OUT_SFT.mkdir(parents=True, exist_ok=True)
    _run([
        sys.executable, str(REPO / "training" / "train_sft.py"),
        "--variant", "local",
        "--data", str(DATA_SFT),
        "--out", str(OUT_SFT),
        "--epochs", "2",
        "--seq-len", "1024",        # T4-friendly
        "--batch-size", "1",
        "--grad-accum", "16",
        "--lr", "2e-4",
        "--save-steps", "250",
    ])


def train_dpo() -> None:
    OUT_DPO.mkdir(parents=True, exist_ok=True)
    _run([
        sys.executable, str(REPO / "training" / "train_dpo.py"),
        "--variant", "local",
        "--sft", str(OUT_SFT / "adapters"),
        "--data", str(DATA_DPO),
        "--out", str(OUT_DPO),
        "--epochs", "1",
        "--seq-len", "1024",
        "--batch-size", "1",
        "--grad-accum", "16",
        "--lr", "5e-6",
        "--beta", "0.1",
    ])


def export(llama_cpp: str | None) -> None:
    DIST.mkdir(parents=True, exist_ok=True)
    adapters = OUT_DPO / "adapters"
    if not adapters.exists():
        adapters = OUT_SFT / "adapters"
    cmd = [
        sys.executable, str(REPO / "training" / "export.py"),
        "--variant", "local",
        "--adapters", str(adapters),
        "--out", str(DIST),
    ]
    if llama_cpp:
        cmd += ["--llama-cpp", llama_cpp]
    _run(cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["install", "prep-sft", "prep-dpo", "sft", "dpo", "export", "all"],
                    default="all")
    ap.add_argument("--llama-cpp", default=os.environ.get("LLAMA_CPP"),
                    help="Path to llama.cpp checkout for GGUF export.")
    ap.add_argument("--skip-dpo", action="store_true",
                    help="SFT-only run for an even shorter Colab session.")
    args = ap.parse_args()

    steps = {
        "install":  lambda: install(),
        "prep-sft": lambda: prep_data("sft"),
        "prep-dpo": lambda: prep_data("dpo"),
        "sft":      lambda: train_sft(),
        "dpo":      lambda: train_dpo(),
        "export":   lambda: export(args.llama_cpp),
    }

    if args.stage == "all":
        order = ["install", "prep-sft", "sft"]
        if not args.skip_dpo:
            order += ["prep-dpo", "dpo"]
        order += ["export"]
        for s in order:
            steps[s]()
    else:
        steps[args.stage]()


if __name__ == "__main__":
    main()
