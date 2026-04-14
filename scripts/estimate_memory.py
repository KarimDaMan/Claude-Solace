"""Verify that each Claude Solace variant fits inside its declared RAM cap.

Run:
    python -m scripts.estimate_memory

Prints a table with parameter count, weight bytes at the recommended
quantization, KV-cache bytes at the native context length, and total
runtime memory for each variant. Exits non-zero if any variant blows
its declared budget.

This script has *no* heavy dependencies beyond ``transformers``; it does
not allocate the model weights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow `python scripts/estimate_memory.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solace.configuration_solace import SolaceConfig  # noqa: E402


# Bytes per parameter for the quantizations we ship.
QUANT_BYTES_PER_PARAM = {
    "bf16": 2.0,
    "fp16": 2.0,
    "int8": 1.0,
    "Q5_K_M": 5.5 / 8,
    "Q4_K_M": 4.5 / 8,
}


def gib(b: int) -> float:
    return b / (1024 ** 3)


def load_variant(name: str) -> tuple[SolaceConfig, dict]:
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / f"solace_{name}.json"
    raw = json.loads(cfg_path.read_text())
    deployment = raw.pop("deployment", {})
    raw.pop("description", None)
    raw.pop("variant", None)
    return SolaceConfig(**raw), deployment


def main() -> int:
    rows = []
    failed = False

    for name in ("local", "mid", "hf"):
        config, deploy = load_variant(name)
        quant = deploy.get("recommended_quantization", "bf16")
        weight_bytes_per_param = QUANT_BYTES_PER_PARAM.get(quant, 2.0)

        runtime = config.estimate_runtime_bytes(
            context_length=config.max_position_embeddings,
            weight_dtype_bytes=weight_bytes_per_param,
            kv_dtype_bytes=2.0,  # KV cache stays in fp16/bf16 for accuracy
        )

        cap = deploy.get("target_ram_gb", 0)
        cap_bytes = cap * (1024 ** 3)
        ok = runtime["total_bytes"] <= cap_bytes
        if not ok:
            failed = True

        rows.append({
            "variant": f"claude-solace-{name}",
            "params_b": config.approx_parameters / 1e9,
            "quant": quant,
            "weights_gib": gib(runtime["weights_bytes"]),
            "kv_gib": gib(runtime["kv_cache_bytes"]),
            "total_gib": gib(runtime["total_bytes"]),
            "cap_gib": float(cap),
            "ctx": runtime["context_length"],
            "fits": ok,
        })

    headers = ["variant", "params (B)", "quant", "weights GiB", "KV GiB", "total GiB", "cap GiB", "ctx", "fits?"]
    widths = [22, 11, 8, 12, 8, 10, 8, 8, 6]

    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        cells = [
            r["variant"],
            f"{r['params_b']:.2f}",
            r["quant"],
            f"{r['weights_gib']:.2f}",
            f"{r['kv_gib']:.2f}",
            f"{r['total_gib']:.2f}",
            f"{r['cap_gib']:.0f}",
            str(r["ctx"]),
            "yes" if r["fits"] else "NO",
        ]
        print("  ".join(c.ljust(w) for c, w in zip(cells, widths)))

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
