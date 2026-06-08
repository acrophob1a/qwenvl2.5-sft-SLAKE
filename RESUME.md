# Qwen2.5-VL 医学视觉问答（VQA）微调

> 简历项目描述 · 仓库：https://github.com/acrophob1a/qwenvl2.5-sft-SLAKE

---

## 一句话介绍

基于 **Qwen2.5-VL-3B** 多模态大模型，在 **SLAKE 医学 VQA 数据集**上完成监督微调（SFT），实现放射影像问答，验证集准确率从 **0.1% 提升至 81.3%**。

---

## 简历写法（可直接复制）

**Qwen2.5-VL 医学视觉问答微调** | 个人项目 | 2026.06

- 基于 Qwen2.5-VL-3B-Instruct，在 SLAKE 医学 VQA 数据集（642 张 CT/MRI/X-Ray 影像，14K QA 对）上完成多模态 SFT
- 设计 SLAKE 专用 DataLoader，将 JSON 标注转为 Qwen 对话格式（`<image>` + question → answer），支持 OPEN/CLOSED 题型
- 单卡 A800 80G 完成 3 epoch 全参数微调（MLP + LLM，vision 冻结），训练 loss 从 2.6 降至 0.08
- 搭建验证集评估 pipeline，对比微调前后效果：Overall **81.3%** / Closed **87.0%** / Open **77.5%**（exact match）
- 建立完整实验留痕体系（8 次实验记录、配置快照、前后对比可视化），代码与记录可复现

**技术栈**：PyTorch · Transformers · Flash Attention · Qwen2.5-VL · SLAKE · SFT · AutoDL

---

## 项目背景

SLAKE 是面向放射影像的医学 VQA  benchmark，涵盖 CT、MRI、X-Ray 三种模态，问答类型包括开放式（器官名称、模态识别）和封闭式（Yes/No 异常判断）。

本项目目标：让通用视觉语言模型 Qwen2.5-VL-3B 学会 SLAKE 的**短答案输出格式**，用于医学影像问答场景。

---

## 核心工作

### 1. 数据工程

- 下载并解压 SLAKE 1.0（HuggingFace: BoKelvin/SLAKE）
- 编写 `SlakeVQADataset`，直接从 JSON + 本地图片加载，无需离线转 TSV
- 英文训练集 4919 条 / 验证集 1053 条，图片路径校验通过率 100%

### 2. 模型微调

| 配置项 | 值 |
|--------|-----|
| 基座模型 | Qwen2.5-VL-3B-Instruct（3.75B 参数） |
| 微调范围 | Vision 冻结，MLP + LLM 可训练（3.12B 参数） |
| 训练数据 | 4919 条英文 QA |
| Epoch / Steps | 3 epoch / 1845 steps |
| 有效 Batch Size | 8（per_device=2 × grad_accum=4） |
| 学习率 | 2e-5，cosine schedule |
| 精度 | bf16 + Flash Attention 2 + gradient checkpointing |
| 最终 train_loss | 0.082 |

### 3. 评估与可视化

- 编写 `scripts/eval_slake_val.py`，对基座模型与 SFT 权重在验证集上批量推理
- 指标：归一化 exact match accuracy
- 产出前后对比柱状图与定性样例图（见 `records/results/exp-008/`）

---

## 关键结果

![验证集准确率对比](records/results/exp-008/accuracy_comparison.png)

| 指标 | 微调前（Base） | 微调后（SFT） | 提升 |
|------|----------------|---------------|------|
| Overall | 0.1% | **81.3%** | +81.2% |
| Closed（Yes/No） | 0.0% | **87.0%** | +87.0% |
| Open（开放式） | 0.2% | **77.5%** | +77.3% |

> Base 模型倾向输出长段落，无法匹配 SLAKE 短答案格式；SFT 后学会输出 `MRI`、`Abdomen`、`Yes/No` 等简洁回答。

---

## 面试可能追问

**Q：为什么 Base 准确率这么低？**

A：Qwen2.5-VL-3B 是通用 VLM，未针对 SLAKE 短答案格式训练。Base 推理时输出长文本（如 *"The image appears to be an MRI scan..."*），与 ground truth（如 *"MRI"*）无法 exact match。SFT 的核心价值是让模型学会 SLAKE 的答案风格。

**Q：为什么冻结 Vision Tower？**

A：3B 模型 + 4919 样本，全量微调容易过拟合且显存压力大。冻结 vision、微调 MLP + LLM 是常见做法，实验表明效果已足够（81.3%）。

**Q：Closed 为什么比 Open 高？**

A：Closed 题答案空间小（Yes/No），Open 题答案多样（器官名、模态、位置等），exact match 更严格。87% vs 77.5% 符合预期。

**Q：遇到过什么工程问题？**

A：训练 step 1000 时数据盘被含 optimizer 的 checkpoint（~19GB/个）写满。解决：清理旧 checkpoint，改用 `save_only_model=True`（~7GB/个），从 checkpoint-500 续训完成。

**Q：如何复现？**

A：见仓库 `records/experiments.md`（exp-001 ~ exp-008 完整记录），配置快照在 `configs/snapshots/`，评估脚本 `scripts/eval_slake_val.py`。

---

## 仓库结构（面试展示用）

```
qwenvl2.5-sft-SLAKE/
├── finetuning/dataset/slake_vqa_dataset.py   # 核心 DataLoader
├── finetuning/configs/sft_slake.py           # 训练配置
├── scripts/eval_slake_val.py                 # 评估脚本
├── records/experiments.md                    # 8 次实验完整留痕
├── records/results/exp-008/                  # 对比图 + metrics
└── configs/snapshots/                        # 可复现配置快照
```

---

## 可补充的后续方向（简历加分项）

- [ ] 测试集（2094 条）最终评估
- [ ] 中英文双语训练对比
- [ ] LoRA 等参数高效微调对比
- [ ] Gradio 推理 Demo 部署
