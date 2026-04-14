"""Run lm-eval-harness over a trained Solace variant.

Benchmarks picked to be fast enough for a laptop/Colab sanity check while
still covering the four buckets we care about: general knowledge (MMLU
subset), math (GSM8K), code (HumanEval via bigcode-eval if installed),
and instruction following (IFEval).

Usage:
    python training/evaluate.py \
        --model dist/solace-local/hf_bf16 \
        --variant local \
        --out evals/solace_local.json

Pass ``--tasks quick`` for the <15 min sanity set, or ``--tasks full`` for
the full suite.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

TASK_SETS = {
    "quick": [
        "mmlu_abstract_algebra",
        "mmlu_elementary_mathematics",
        "mmlu_world_religions",
        "gsm8k_cot",
        "ifeval",
    ],
    "full": [
        "mmlu",
        "gsm8k_cot",
        "ifeval",
        "arc_challenge",
        "hellaswag",
        "truthfulqa_mc2",
        "winogrande",
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True,
                    help="HF-format model dir (output of export.py).")
    ap.add_argument("--variant", required=True, choices=["local", "mid", "hf"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks", choices=list(TASK_SETS), default="quick")
    ap.add_argument("--batch-size", default="auto")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tasks = ",".join(TASK_SETS[args.tasks])

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={args.model},dtype=bfloat16,trust_remote_code=true",
        "--tasks", tasks,
        "--batch_size", str(args.batch_size),
        "--device", args.device,
        "--output_path", str(args.out),
    ]
    print("[eval]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Surface a compact summary alongside the full JSON.
    result_files = list(args.out.glob("*.json"))
    if result_files:
        data = json.loads(result_files[0].read_text())
        summary = {
            "variant": args.variant,
            "model": str(args.model),
            "task_set": args.tasks,
            "results": {
                task: {k: v for k, v in metrics.items() if k != "alias"}
                for task, metrics in data.get("results", {}).items()
            },
        }
        (args.out.parent / f"{args.out.stem}_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        print(f"[eval] summary -> {args.out.parent}/{args.out.stem}_summary.json")


if __name__ == "__main__":
    main()
