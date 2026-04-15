"""Pure-arithmetic memory estimator. No PyTorch / transformers required.

Mirrors ``SolaceConfig.approx_parameters`` and
``SolaceConfig.estimate_runtime_bytes``. Used in CI to assert each
variant's declared RAM cap.

Run:
    python scripts/estimate_memory_standalone.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

QUANT_BYTES_PER_PARAM = {
    "bf16": 2.0,
    "fp16": 2.0,
    "int8": 1.0,
    "Q5_K_M": 5.5 / 8,
    "Q4_K_M": 4.5 / 8,
}


def gib(b: float) -> float:
    return b / (1024 ** 3)


def approx_parameters(c: dict) -> int:
    v, d = c["vocab_size"], c["hidden_size"]
    n, f = c["num_hidden_layers"], c["intermediate_size"]
    head_dim = c.get("head_dim") or (d // c["num_attention_heads"])
    kv_dim = c["num_key_value_heads"] * head_dim

    embed = v * d
    lm_head = 0 if c.get("tie_word_embeddings", True) else v * d
    attn = 2 * d * d + 2 * d * kv_dim
    mlp = 3 * d * f
    sandwich = 4 if c.get("use_pre_post_norm_sandwich", True) else 2
    per_block_norms = sandwich * d
    qk_norm = 2 * head_dim if c.get("use_qk_norm", True) else 0
    per_layer = attn + mlp + per_block_norms + qk_norm
    final_norm = d
    return embed + lm_head + n * per_layer + final_norm


def runtime_bytes(c: dict, ctx: int, weight_bpp: float, kv_bpp: float = 2.0,
                  activation_overhead: int = 1_500_000_000) -> dict:
    head_dim = c.get("head_dim") or (c["hidden_size"] // c["num_attention_heads"])
    kv_dim = c["num_key_value_heads"] * head_dim

    pattern = c.get("attention_pattern") or (["local"] * 5 + ["global"])
    n_layers = c["num_hidden_layers"]
    layer_kinds = [pattern[i % len(pattern)] for i in range(n_layers)]
    n_global = sum(1 for k in layer_kinds if k == "global")
    n_local = n_layers - n_global

    sliding = c.get("sliding_window", 4096)
    sinks = c.get("attention_sinks", 0)
    local_tokens = min(ctx, sliding) + sinks

    per_token_kv = 2 * kv_dim * kv_bpp
    kv = int(n_global * ctx * per_token_kv + n_local * local_tokens * per_token_kv)

    weights = int(approx_parameters(c) * weight_bpp)
    return {
        "weights": weights,
        "kv": kv,
        "activations": activation_overhead,
        "total": weights + kv + activation_overhead,
        "n_global": n_global,
        "n_local": n_local,
    }


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    rows = []
    failed = False

    for name in ("local", "mid", "hf", "high"):
        cfg = json.loads((repo / "configs" / f"solace_{name}.json").read_text())
        deploy = cfg.get("deployment", {})
        quant = deploy.get("recommended_quantization", "bf16")
        bpp = QUANT_BYTES_PER_PARAM.get(quant, 2.0)
        ctx = cfg["max_position_embeddings"]
        rt = runtime_bytes(cfg, ctx=ctx, weight_bpp=bpp)
        cap_bytes = deploy.get("target_ram_gb", 0) * (1024 ** 3)
        ok = rt["total"] <= cap_bytes
        failed = failed or not ok
        rows.append({
            "name": f"claude-solace-{name}",
            "params_b": approx_parameters(cfg) / 1e9,
            "quant": quant,
            "weights_gib": gib(rt["weights"]),
            "kv_gib": gib(rt["kv"]),
            "act_gib": gib(rt["activations"]),
            "total_gib": gib(rt["total"]),
            "cap_gib": deploy.get("target_ram_gb", 0),
            "ctx": ctx,
            "global": rt["n_global"],
            "local": rt["n_local"],
            "ok": ok,
        })

    headers = ["variant", "params(B)", "quant", "wts GiB", "KV GiB", "act GiB",
               "total GiB", "cap GiB", "ctx", "G/L", "fits"]
    widths = [22, 9, 7, 8, 7, 8, 10, 8, 8, 8, 5]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        cells = [
            r["name"], f"{r['params_b']:.2f}", r["quant"],
            f"{r['weights_gib']:.2f}", f"{r['kv_gib']:.2f}", f"{r['act_gib']:.2f}",
            f"{r['total_gib']:.2f}", f"{r['cap_gib']:.0f}",
            str(r["ctx"]), f"{r['global']}/{r['local']}",
            "yes" if r["ok"] else "NO",
        ]
        print("  ".join(c.ljust(w) for c, w in zip(cells, widths)))

    if failed:
        print("\nFAIL: at least one variant exceeds its declared RAM cap.")
        return 1
    print("\nOK: all variants fit their declared caps at native context length.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
