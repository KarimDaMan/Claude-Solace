"""Build SFT and DPO datasets for Claude Solace post-training.

Both stages pull from free, openly-licensed HF datasets whose curation
windows land in 2024-2025 (which is what gives Solace its recency even
though the base model was frozen earlier). Each example is normalised
to the Solace chat template before it hits the trainer.

Usage:
    python training/prepare_data.py --stage sft --out data/sft_mix
    python training/prepare_data.py --stage dpo --out data/dpo_mix
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from datasets import Dataset, concatenate_datasets, load_dataset

# ---------------------------------------------------------------------------
# Source recipes. Each tuple is (HF id, split, sample cap, 'date_tag').
# Date tags are the curation-window labels we surface in the model card.
# All sources are free + permissively licensed.
# ---------------------------------------------------------------------------

SFT_SOURCES: list[tuple[str, str, int, str]] = [
    # Allen AI's Tulu 3 SFT mix (Nov 2024): instructions + math + code + safety.
    ("allenai/tulu-3-sft-mixture",            "train", 300_000, "2024-11"),
    # Curated 2024 conversational SFT.
    ("teknium/OpenHermes-2.5",                "train", 200_000, "2024-05"),
    # Math reasoning, released mid-2024.
    ("meta-math/MetaMathQA",                  "train",  80_000, "2024-04"),
    # Code SFT, late-2024 curation.
    ("ise-uiuc/Magicoder-Evol-Instruct-110K", "train",  80_000, "2024-09"),
    # 2024-2025 multi-turn chat with tool calls.
    ("Salesforce/xlam-function-calling-60k",  "train",  40_000, "2025-02"),
]

DPO_SOURCES: list[tuple[str, str, int, str]] = [
    ("allenai/llama-3.1-tulu-3-8b-preference-mixture", "train", 150_000, "2024-11"),
    ("argilla/ultrafeedback-binarized-preferences-cleaned", "train", 60_000, "2024-06"),
    ("HuggingFaceH4/orca_dpo_pairs",                   "train",  40_000, "2024-03"),
]

# ---------------------------------------------------------------------------
# Chat template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are Claude Solace, a careful and helpful assistant. "
    "Answer honestly, admit uncertainty, and refuse harmful requests."
)


def render_chat(messages: list[dict]) -> str:
    """Solace chat template. Mirrors Gemma-3's turn format but with Claude-
    style <|user|>/<|assistant|>/<|system|> role tags so it round-trips
    through llama.cpp and vLLM."""
    parts = []
    for m in messages:
        role = m["role"]
        parts.append(f"<|{role}|>\n{m['content'].strip()}\n<|end|>\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Normalisers: each source has its own schema; we collapse them to a single
# (messages, date_tag) shape for SFT or (chosen, rejected, date_tag) for DPO.
# ---------------------------------------------------------------------------

def _sft_normalise(example: dict, source: str, date_tag: str) -> dict | None:
    msgs: list[dict] = []
    if "messages" in example and example["messages"]:
        msgs = [{"role": m["role"], "content": m["content"]} for m in example["messages"]]
    elif "conversations" in example and example["conversations"]:
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        for turn in example["conversations"]:
            r = role_map.get(turn.get("from", ""), turn.get("from", "user"))
            msgs.append({"role": r, "content": turn.get("value", "")})
    elif "instruction" in example and "output" in example:
        if example.get("input"):
            msgs = [
                {"role": "user", "content": f"{example['instruction']}\n\n{example['input']}"},
                {"role": "assistant", "content": example["output"]},
            ]
        else:
            msgs = [
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": example["output"]},
            ]
    elif "query" in example and "answers" in example:
        msgs = [
            {"role": "user", "content": example["query"]},
            {"role": "assistant", "content": json.dumps(example["answers"])},
        ]
    else:
        return None

    if not msgs or not any(m["role"] == "assistant" for m in msgs):
        return None
    if msgs[0]["role"] != "system":
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, *msgs]

    return {
        "text": render_chat(msgs) + msgs[-1]["content"] + "<|end|>",
        "messages": msgs,
        "source": source,
        "date_tag": date_tag,
    }


def _dpo_normalise(example: dict, source: str, date_tag: str) -> dict | None:
    prompt = example.get("prompt") or example.get("question") or example.get("input")
    chosen = example.get("chosen") or example.get("chosen_response")
    rejected = example.get("rejected") or example.get("rejected_response")
    if prompt is None or chosen is None or rejected is None:
        return None
    # Some sources store list-of-messages under chosen/rejected.
    if isinstance(chosen, list):
        chosen = chosen[-1].get("content", "")
    if isinstance(rejected, list):
        rejected = rejected[-1].get("content", "")
    rendered_prompt = render_chat(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    )
    return {
        "prompt": rendered_prompt,
        "chosen": chosen + "<|end|>",
        "rejected": rejected + "<|end|>",
        "source": source,
        "date_tag": date_tag,
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

@dataclass
class BuildArgs:
    stage: str
    out: Path
    max_per_source: int | None
    seed: int


def _iter_normalised(sources, normaliser, max_per_source) -> Iterable[Dataset]:
    for hf_id, split, cap, date_tag in sources:
        cap = min(cap, max_per_source) if max_per_source else cap
        print(f"[load] {hf_id} split={split} cap={cap} tag={date_tag}")
        try:
            ds = load_dataset(hf_id, split=split, streaming=False)
        except Exception as e:
            print(f"  ! skip {hf_id}: {e}")
            continue
        if cap and len(ds) > cap:
            ds = ds.shuffle(seed=17).select(range(cap))

        def _map(ex, _hf_id=hf_id, _tag=date_tag):
            return normaliser(ex, source=_hf_id, date_tag=_tag) or {"__drop__": True}

        ds = ds.map(_map, remove_columns=ds.column_names, desc=f"norm:{hf_id}")
        ds = ds.filter(lambda ex: "__drop__" not in ex)
        yield ds


def build(args: BuildArgs) -> None:
    if args.stage == "sft":
        parts = list(_iter_normalised(SFT_SOURCES, _sft_normalise, args.max_per_source))
    elif args.stage == "dpo":
        parts = list(_iter_normalised(DPO_SOURCES, _dpo_normalise, args.max_per_source))
    else:
        raise ValueError(f"unknown stage {args.stage!r}")

    if not parts:
        raise SystemExit("no sources produced data; check network / HF auth")

    merged = concatenate_datasets(parts).shuffle(seed=args.seed)
    print(f"[build] {args.stage} total rows = {len(merged):,}")

    args.out.mkdir(parents=True, exist_ok=True)
    merged.save_to_disk(str(args.out))

    # Small JSONL preview for eyeballing.
    preview_path = args.out / "preview.jsonl"
    with preview_path.open("w") as f:
        for row in merged.select(range(min(100, len(merged)))):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[build] wrote {args.out} (+ preview.jsonl)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sft", "dpo"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-per-source", type=int, default=None,
                    help="Cap per source (useful for dry runs on Colab free).")
    ap.add_argument("--seed", type=int, default=42)
    build(BuildArgs(**vars(ap.parse_args())))


if __name__ == "__main__":
    main()
