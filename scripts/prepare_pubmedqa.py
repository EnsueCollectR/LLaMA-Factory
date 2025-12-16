"""Convert the PubMedQA dataset to the Alpaca SFT format used by LLaMA-Factory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping

from datasets import load_dataset


def to_alpaca(example: Mapping[str, str]) -> MutableMapping[str, str]:
    """Map the PubMedQA record to Alpaca keys."""
    return {
        "instruction": example.get("instruction", "").strip(),
        "input": example.get("input", "").strip(),
        "output": example.get("output", "").strip(),
    }


def write_jsonl(examples: Iterable[Mapping[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in examples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PubMedQA in Alpaca format.")
    parser.add_argument(
        "--split",
        default="train",
        help="Split to export (train|test|validation if present). Default: train.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/pubmedqa_alpaca.jsonl"),
        help="Destination JSONL path. Default: data/pubmedqa_alpaca.jsonl",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of samples (useful for smoke tests).",
    )
    args = parser.parse_args()

    dataset = load_dataset("llamafactory/PubMedQA", split=args.split)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    alpaca_rows = (to_alpaca(example) for example in dataset)
    write_jsonl(alpaca_rows, args.output)

    print(f"Wrote {len(dataset)} samples to {args.output}")


if __name__ == "__main__":
    main()

