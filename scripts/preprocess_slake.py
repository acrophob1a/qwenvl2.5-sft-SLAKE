#!/usr/bin/env python3
"""Validate SLAKE samples and optionally export an English-only manifest."""

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default="/root/autodl-tmp/qwenvl-sft/datasets/SLAKE",
    )
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--q-lang", default="en", choices=["en", "zh", "all"])
    parser.add_argument("--output", default=None, help="Optional manifest JSON output path")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    ann_file = root / f"{args.split}.json"
    samples = json.load(open(ann_file))

    if args.q_lang != "all":
        samples = [s for s in samples if s.get("q_lang") == args.q_lang]

    valid, missing = [], 0
    for sample in samples:
        img_path = root / "imgs" / sample["img_name"]
        if img_path.exists():
            valid.append(sample)
        else:
            missing += 1

    print(f"split={args.split} q_lang={args.q_lang}")
    print(f"total={len(samples)} valid={len(valid)} missing_images={missing}")
    print("answer_type:", Counter(s["answer_type"] for s in valid))
    print("sample:", valid[0]["question"], "->", valid[0]["answer"])

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(valid, open(out, "w"), ensure_ascii=False, indent=2)
        print(f"manifest saved to {out}")


if __name__ == "__main__":
    main()
