#!/usr/bin/env python3
"""Evaluate SLAKE validation set: base model vs SFT checkpoint."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from tqdm import tqdm


def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[.\,!?\;:\"'\(\)\[\]]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_correct(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def load_samples(dataset_root: Path, split: str, q_lang: str) -> list[dict]:
    samples = json.load(open(dataset_root / f"{split}.json"))
    if q_lang != "all":
        samples = [s for s in samples if s.get("q_lang") == q_lang]
    valid = []
    for s in samples:
        img = dataset_root / "imgs" / s["img_name"]
        if img.exists():
            valid.append(s)
    return valid


def load_model(model_path: str, processor_path: str | None = None, device: str = "cuda"):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(processor_path or model_path)
    model.eval()
    return model, processor


@torch.inference_mode()
def predict(model, processor, image_path: Path, question: str, max_new_tokens: int = 64) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, output_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def run_inference(
    model_name: str,
    samples: list[dict],
    image_root: Path,
    output_jsonl: Path,
    max_new_tokens: int,
    processor_path: str | None = None,
) -> None:
    model, processor = load_model(model_name, processor_path=processor_path)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for sample in tqdm(samples, desc=f"infer:{Path(model_name).name}"):
            pred = predict(
                model,
                processor,
                image_root / sample["img_name"],
                sample["question"],
                max_new_tokens=max_new_tokens,
            )
            row = {
                "qid": sample.get("qid"),
                "img_name": sample["img_name"],
                "question": sample["question"],
                "answer": sample["answer"],
                "answer_type": sample["answer_type"],
                "prediction": pred,
                "correct": is_correct(pred, sample["answer"]),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    del model
    torch.cuda.empty_cache()


def load_predictions(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def compute_metrics(rows: list[dict]) -> dict:
    buckets = defaultdict(lambda: {"correct": 0, "total": 0})
    for row in rows:
        key = row["answer_type"]
        buckets[key]["total"] += 1
        buckets["ALL"]["total"] += 1
        if row["correct"]:
            buckets[key]["correct"] += 1
            buckets["ALL"]["correct"] += 1
    metrics = {}
    for k, v in buckets.items():
        metrics[k] = {
            "accuracy": v["correct"] / v["total"] if v["total"] else 0.0,
            "correct": v["correct"],
            "total": v["total"],
        }
    return metrics


def plot_accuracy_comparison(base_metrics: dict, sft_metrics: dict, out_png: Path) -> None:
    groups = ["ALL", "CLOSED", "OPEN"]
    labels = ["Overall", "Closed", "Open"]

    def get_acc(metrics: dict, key: str) -> float:
        return metrics.get(key, {"accuracy": 0.0})["accuracy"] * 100

    base_vals = [get_acc(base_metrics, g) for g in groups]
    sft_vals = [get_acc(sft_metrics, g) for g in groups]

    x = range(len(groups))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], base_vals, width, label="Base", color="#94a3b8")
    ax.bar([i + width / 2 for i in x], sft_vals, width, label="SFT", color="#2563eb")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("SLAKE Validation (EN): Base vs SFT")
    ax.set_ylim(0, 100)
    for i, (b, s) in enumerate(zip(base_vals, sft_vals)):
        ax.text(i - width / 2, b + 1, f"{b:.1f}%", ha="center", fontsize=9)
        ax.text(i + width / 2, s + 1, f"{s:.1f}%", ha="center", fontsize=9)
        delta = s - b
        ax.text(i, max(b, s) + 8, f"Δ{delta:+.1f}%", ha="center", fontsize=9, color="#059669")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def pick_examples(base_rows: list[dict], sft_rows: list[dict], n: int = 4) -> list[tuple[dict, dict, dict]]:
    by_qid_base = {r["qid"]: r for r in base_rows}
    examples = []
    for sft in sft_rows:
        base = by_qid_base[sft["qid"]]
        if (not base["correct"]) and sft["correct"]:
            examples.append((base, sft, sft))
        if len(examples) >= n:
            break
    if len(examples) < n:
        for sft in sft_rows:
            base = by_qid_base[sft["qid"]]
            pair = (base, sft, sft)
            if pair not in examples:
                examples.append(pair)
            if len(examples) >= n:
                break
    return examples[:n]


def plot_qualitative_examples(
    examples: list[tuple[dict, dict, dict]],
    image_root: Path,
    out_png: Path,
) -> None:
    n = len(examples)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    if n == 1:
        axes = [axes]
    for ax_row, (base, sft, meta) in zip(axes, examples):
        img = Image.open(image_root / meta["img_name"]).convert("RGB")
        for ax, tag, row in zip(ax_row, ["Base", "SFT"], [base, sft]):
            ax.imshow(img)
            ax.axis("off")
            status = "✓" if row["correct"] else "✗"
            text = (
                f"[{tag}] {status}\n"
                f"Q: {row['question']}\n"
                f"GT: {row['answer']}\n"
                f"Pred: {row['prediction']}"
            )
            ax.set_title(text, fontsize=9, loc="left")
    fig.suptitle("Qualitative Comparison (Validation Samples)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/root/autodl-tmp/qwenvl-sft/datasets/SLAKE")
    parser.add_argument("--base-model", default="/root/autodl-tmp/qwenvl-sft/pretrained/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--sft-model", default="/root/autodl-tmp/qwenvl-sft/work_dirs/slake-vqa-sft-v1/checkpoint-1845")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--q-lang", default="en")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-dir", default="/root/autodl-tmp/qwenvl-sft/records/results/exp-008")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    image_root = dataset_root / "imgs"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(dataset_root, args.split, args.q_lang)
    if args.max_samples:
        samples = samples[: args.max_samples]

    base_jsonl = out_dir / "pred_base.jsonl"
    sft_jsonl = out_dir / "pred_sft.jsonl"

    t0 = time.time()
    if not base_jsonl.exists():
        run_inference(args.base_model, samples, image_root, base_jsonl, args.max_new_tokens)
    if not sft_jsonl.exists():
        run_inference(
            args.sft_model,
            samples,
            image_root,
            sft_jsonl,
            args.max_new_tokens,
            processor_path=args.base_model,
        )

    base_rows = load_predictions(base_jsonl)
    sft_rows = load_predictions(sft_jsonl)
    base_metrics = compute_metrics(base_rows)
    sft_metrics = compute_metrics(sft_rows)

    metrics = {
        "split": args.split,
        "q_lang": args.q_lang,
        "num_samples": len(samples),
        "base_model": args.base_model,
        "sft_model": args.sft_model,
        "base": base_metrics,
        "sft": sft_metrics,
        "runtime_sec": round(time.time() - t0, 1),
    }
    json.dump(metrics, open(out_dir / "metrics.json", "w"), indent=2, ensure_ascii=False)

    plot_accuracy_comparison(
        base_metrics,
        sft_metrics,
        out_dir / "accuracy_comparison.png",
    )
    examples = pick_examples(base_rows, sft_rows, n=4)
    plot_qualitative_examples(examples, image_root, out_dir / "qualitative_examples.png")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
