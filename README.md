# Qwen2.5-VL SFT on SLAKE

Fine-tuning [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) for medical Visual Question Answering (VQA) on the [SLAKE](https://huggingface.co/datasets/BoKelvin/SLAKE) dataset — a radiology image QA benchmark covering CT, MRI, and X-Ray modalities.

## Results

After 3-epoch full-parameter SFT (vision encoder frozen, MLP + LLM trainable), the model achieves dramatic improvement on SLAKE English validation set:

![Validation Accuracy: Base vs SFT](records/results/exp-008/accuracy_comparison.png)

| Metric | Base Model | After SFT | Improvement |
|--------|-----------|-----------|-------------|
| **Overall** | 0.1% | **81.3%** | +81.2% |
| **Closed (Yes/No)** | 0.0% | **87.0%** | +87.0% |
| **Open** | 0.2% | **77.5%** | +77.3% |

The base model tends to produce long descriptive paragraphs (e.g., *"The image appears to be an MRI scan..."*), which fail exact-match against SLAKE's short ground-truth answers (`MRI`, `Yes`, `Abdomen`, etc.). SFT teaches the model to output concise answers matching the expected format.

![Qualitative Examples](records/results/exp-008/qualitative_examples.png)

Training loss converged from ~2.6 to **0.082** over 1845 steps (3 epochs):

| Training Detail | Value |
|----------------|-------|
| Base Model | Qwen2.5-VL-3B-Instruct (3.75B params) |
| Trainable Params | 3.12B (Vision encoder frozen) |
| Training Data | 4,919 English QA pairs from SLAKE train set |
| Hardware | Single NVIDIA A800 80GB |
| Effective Batch Size | 8 (per_device=2 × grad_accum=4) |
| Learning Rate | 2e-5, cosine schedule |
| Precision | bf16 + Flash Attention 2 + gradient checkpointing |
| Training Time | ~70 minutes (wall time, including one disk-full interruption and resume) |
| Final Train Loss | 0.082 |

## Project Structure

```
qwenvl2.5-sft-SLAKE/
├── finetuning/
│   ├── configs/sft_slake.py           # SFT training configuration
│   ├── dataset/
│   │   ├── slake_vqa_dataset.py       # SLAKE VQA Dataset (JSON + image loading)
│   │   └── collator.py                # Data collator (padding, multimodal batching)
│   ├── scripts/sft_slake.sh           # Training launch script
│   └── train.py                       # Training entry point
├── scripts/
│   ├── preprocess_slake.py            # Data preprocessing & validation
│   └── eval_slake_val.py              # Validation set evaluation pipeline
├── records/
│   ├── experiments.md                 # Complete experiment log (exp-001 ~ exp-008)
│   ├── data/CHANGELOG.md              # Dataset version history
│   └── results/exp-008/               # Accuracy charts & metrics
├── configs/snapshots/                 # Config snapshots for reproducibility
└── SFT_DATAFLOW.md                    # Detailed SFT data flow analysis (tensor shapes)
```

## Reproducing

### 1. Environment Setup

```bash
pip install torch==2.6.0 torchvision==0.21.0
pip install -e . --no-deps
pip install -e finetuning --no-deps
pip install -r requirements.txt
MAX_JOBS=8 pip install flash-attn==2.7.4.post1 --no-build-isolation
```

### 2. Data Preparation

Download SLAKE from HuggingFace:

```bash
export HF_ENDPOINT=https://hf-mirror.com  # if huggingface.co is unreachable
hf download BoKelvin/SLAKE --repo-type dataset --local-dir datasets/SLAKE
cd datasets/SLAKE && unzip -q imgs.zip && unzip -q KG.zip
```

Preprocess English training split:

```bash
python scripts/preprocess_slake.py --split train --q-lang en \
  --output datasets/SLAKE/manifests/train_en.json
```

### 3. Training

Smoke test first (64 samples, 10 steps):

```bash
bash finetuning/scripts/sft_slake_smoke.sh
```

Full training (3 epochs):

```bash
bash finetuning/scripts/sft_slake.sh
```

> **Note**: Checkpoints with optimizer states are ~19GB each. Use `save_only_model=True` in the training config to reduce checkpoint size to ~7GB each and avoid disk overflow.

### 4. Evaluation

```bash
python scripts/eval_slake_val.py --output-dir records/results/eval --max-new-tokens 32
```

## Experiment Log

All experiments from environment setup to final evaluation are documented in [records/experiments.md](records/experiments.md), including:

- exp-001: Environment & base model initialization
- exp-003: Disk space management (system disk cleanup)
- exp-004: SLAKE dataset download & verification
- exp-005: Training framework restoration & data preprocessing
- exp-006: Smoke test (loss 2.88 → 0.10 in 10 steps)
- exp-007: Full SFT training (3 epochs, loss 2.6 → 0.082)
- exp-008: Validation evaluation & visualization (81.3% accuracy)

Config snapshots for each experiment are saved in [configs/snapshots/](configs/snapshots/).

## Technical Details

- **Data Format**: Conversations follow Qwen2.5-VL chat template with `<image>` placeholder; the image processor uses mRoPE (multimodal Rotary Position Embedding) for vision tokens.
- **Loss Masking**: Only assistant response tokens contribute to the loss; system prompts, user questions, vision tokens, and padding are masked with `IGNORE_INDEX=-100`.
- **Multimodal Fusion**: Vision features are inserted into the text embedding sequence via `masked_scatter` at positions corresponding to `<|image_pad|>` tokens (not concatenation or cross-attention).

A detailed tensor-shape-level walkthrough of the entire SFT data flow (from raw JSON to loss scalar) is available in [SFT_DATAFLOW.md](SFT_DATAFLOW.md).

## License

This project is for learning and research purposes.
