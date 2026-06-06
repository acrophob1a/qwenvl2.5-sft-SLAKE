# Qwen2.5-VL SFT on SLAKE

基于 [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) 的医学 VQA 监督微调项目，目标是在 **SLAKE 数据集** 上进行 SFT。

> 原教程自带的 Visual Grounding demo 代码已删除（任务方向不符），仅保留基座模型。训练代码待 SLAKE 数据到位后搭建。

## 项目状态

| 阶段 | 状态 |
|------|------|
| 环境配置 + 基座模型下载 | ✅ exp-001 |
| 清理 Grounding demo | ✅ exp-003 |
| SLAKE 数据下载 | ✅ exp-004 |
| SLAKE 数据预处理 + VQA 训练框架 | ✅ exp-005 |
| SFT 冒烟测试 | ✅ exp-006 |
| 正式 SFT 微调（3 epoch） | ✅ exp-007 |
| 验证集评估 + 推理 | ⏳ 待做 |

## 目录结构

```
/root/autodl-tmp/qwenvl-sft/     # 项目根（数据盘，关机不丢）
├── pretrained/Qwen2.5-VL-3B-Instruct/   # 基座模型 7.1G
├── finetuning/                          # SFT 训练代码
│   ├── configs/sft_slake.py             # SLAKE VQA 配置
│   ├── dataset/slake_vqa_dataset.py       # SLAKE dataloader
│   └── scripts/sft_slake_smoke.sh       # 冒烟测试脚本
├── scripts/preprocess_slake.py          # 数据预处理验证
├── records/                               # 实验留痕
├── configs/snapshots/                     # 配置快照
└── datasets/                              # SLAKE 数据（不进 Git）
```

## 存储说明

| 位置 | 挂载 | 说明 |
|------|------|------|
| `/root/autodl-tmp/` | 数据盘 50G | **大文件放这里**（模型、数据集、pip 缓存） |
| `/` | 系统盘 30G | conda 环境，保持精简 |

## 实验留痕

见 [`records/experiments.md`](records/experiments.md)，规范见 [`EXPERIMENT_TRACE.md`](EXPERIMENT_TRACE.md)。

## 环境

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export PIP_CACHE_DIR=/root/autodl-tmp/.pip-cache

# 基座模型路径
MODEL_PATH=/root/autodl-tmp/qwenvl-sft/pretrained/Qwen2.5-VL-3B-Instruct
```
