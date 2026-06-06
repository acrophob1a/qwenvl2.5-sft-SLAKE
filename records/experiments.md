# 实验主日志

> 规范见 `EXPERIMENT_TRACE.md`

---

## exp-001 — 环境与基座模型初始化 — 2026-06-06

**状态**：已完成

**动机**：按 `qwenvl-sft.md` 完成 Qwen2.5-VL SFT 项目的代码下载、依赖安装与基座模型下载。

**数据版本**：data-v0（尚未开始数据处理）

**配置**：
| 项 | 值 |
|----|-----|
| 项目路径 | `/root/autodl-tmp/qwenvl-sft` |
| GPU | NVIDIA A800 80GB PCIe |
| torch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| transformers | 4.51.3 |
| 基座模型 | `pretrained/Qwen2.5-VL-3B-Instruct`（7.1G） |
| 下载镜像 | `HF_ENDPOINT=https://hf-mirror.com`（huggingface.co 不可达） |
| pip 镜像 | `https://pypi.tuna.tsinghua.edu.cn/simple` |

**命令**：
```bash
# 代码下载
export HF_ENDPOINT=https://hf-mirror.com
hf download Brilliant-B/awesome-demos demo1.tar.gz --local-dir /root/autodl-tmp/downloads
tar -xf /root/autodl-tmp/downloads/demo1.tar.gz -C /root/autodl-tmp/qwenvl-sft

# 模型下载
hf download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir /root/autodl-tmp/qwenvl-sft/pretrained/Qwen2.5-VL-3B-Instruct

# 环境安装
pip install torch==2.6.0 torchvision==0.21.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r /tmp/req_core.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e /root/autodl-tmp/qwenvl-sft --no-deps
pip install -e /root/autodl-tmp/qwenvl-sft/finetuning --no-deps
pip install -r /tmp/req_finetune.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
MAX_JOBS=8 pip install flash-attn==2.7.4.post1 --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**结果**：
- 代码仓库：91MB tar 包，6.7s 下载完成
- 基座模型：7.1G，约 130s 通过 hf-mirror 下载完成
- 模型加载验证：3.75B 参数，bf16 + device_map=auto 加载成功
- flash-attn：2.7.4.post1 安装成功（复用前次编译缓存，约 18s）
- vllm / ray / mpi4py：暂未安装（推理/分布式阶段再装）

**观察与结论**：
- `huggingface.co` 直连不可用，改用 `hf-mirror.com` 后下载速度正常（~55MB/s 量级）
- 首次 `pip install torchvision==0.21.0` 无镜像、会连带下载 torch 大包，耗时 >8min 被中断；改用清华 PyPI 镜像后 torch 2.6.0 安装约 103s
- AutoDL 预装 torch 2.8.0 与 torchvision 0.21.0 不兼容，已降级 torch 至 2.6.0 符合教程要求

**下一步**：关联 GitHub 仓库，规划 SLAKE 数据预处理

---

## exp-002 — GitHub 仓库关联与任务规划 — 2026-06-06

**状态**：已完成

**动机**：将项目关联到 [acrophob1a/qwenvl2.5-sft-SLAKE](https://github.com/acrophob1a/qwenvl2.5-sft-SLAKE)，建立留痕体系，明确 Demo 任务 vs SLAKE 计划任务，用于简历展示。

**数据版本**：data-v0

**配置**：
| 项 | 值 |
|----|-----|
| GitHub 远程 | `https://github.com/acrophob1a/qwenvl2.5-sft-SLAKE.git` |
| 本地路径 | `/root/autodl-tmp/qwenvl-sft` |
| Demo 微调任务 | Visual Grounding（phrase → bbox） |
| 计划微调任务 | SLAKE 医学 VQA（image + question → answer） |

**命令**：
```bash
cd /root/autodl-tmp/qwenvl-sft
git init
git remote add origin https://github.com/acrophob1a/qwenvl2.5-sft-SLAKE.git
```

**结果**：
- 已创建 `README.md`、`records/`、`configs/snapshots/`
- Git 已初始化并关联远程（尚未 push，远程仓库当前为空）

**观察与结论**：
- 教程 Demo 任务是 **Visual Grounding**，数据集为 `Grounding-ToyData`，训练入口 `finetuning/scripts/sft_gnd.sh`
- 本仓库名含 SLAKE，计划任务为 **医学 VQA**，需后续改造 dataloader/collator
- 简历叙事建议：Demo 跑通 grounding → 迁移到 SLAKE VQA（体现数据工程 + 任务适配能力）

**下一步**：SLAKE 数据下载与预处理（data-v1）

---

## exp-003 — 清理 Grounding demo + 释放系统盘 — 2026-06-06

**状态**：已完成

**动机**：系统盘（30G）使用率达 85%；Grounding demo 代码方向错误（应为 SLAKE 医学 VQA），需删除并释放空间。

**数据版本**：data-v0 → 已删除 Grounding-ToyData

**配置**：
| 项 | 值 |
|----|-----|
| 系统盘清理前 | 26G / 30G（85%） |
| 系统盘清理后 | 7.2G / 30G（24%） |
| 保留 | `pretrained/Qwen2.5-VL-3B-Instruct`（7.1G，数据盘） |
| 删除 | finetuning/、inference/、datasets/、visualizer/、demo1.tar.gz |

**命令**：
```bash
# 释放系统盘
pip cache purge                          # 清 pip 缓存 ~11G
rm -rf /tmp/pip-* /tmp/pip-unpack-*      # 清安装残留 ~8G

# 删除 Grounding demo，保留模型
cd /root/autodl-tmp/qwenvl-sft
rm -rf finetuning inference datasets visualizer setup.py requirements.txt
rm -rf /root/autodl-tmp/downloads

# 后续 pip/hf 缓存指向数据盘（已写入 ~/.bashrc）
export PIP_CACHE_DIR=/root/autodl-tmp/.pip-cache
export HF_HOME=/root/autodl-tmp/.cache/huggingface
```

**结果**：
- 系统盘释放约 **19G**
- 基座模型完好，路径不变
- Grounding 相关代码与 toy 数据已删除

**观察与结论**：
- 系统盘爆满主因：`/root/.cache/pip`（11G）+ `/tmp/pip-unpack-*`（8G），均为安装依赖时的缓存/临时文件
- 模型本身一直在数据盘 `/root/autodl-tmp`，不在系统盘
- conda 环境（~7.7G）必须在系统盘，无法迁移

**下一步**：用户下载 SLAKE 数据 → 搭建 VQA 训练 pipeline

---

## exp-004 — SLAKE 数据集下载 — 2026-06-06

**状态**：已完成

**动机**：获取 SLAKE 医学 VQA 训练数据，为后续 SFT 做准备。

**数据版本**：data-v1

**配置**：
| 项 | 值 |
|----|-----|
| 数据源 | [BoKelvin/SLAKE](https://huggingface.co/datasets/BoKelvin/SLAKE) |
| 本地路径 | `datasets/SLAKE/` |
| 下载体积 | 207 MB（zip） |
| 解压后 | 470 MB |
| 影像数 | 642 |
| QA 对数 | 14,028（train/val/test = 9835/2099/2094） |

**命令**：
```bash
export HF_ENDPOINT=https://hf-mirror.com
hf download BoKelvin/SLAKE --repo-type dataset \
  --local-dir /root/autodl-tmp/qwenvl-sft/datasets/SLAKE
cd /root/autodl-tmp/qwenvl-sft/datasets/SLAKE
unzip -q imgs.zip && unzip -q KG.zip
```

**结果**：
- 下载耗时约 15s（hf-mirror）
- 目录含 `imgs/`、`KG/`、`train.json`、`validation.json`、`test.json`
- 数据仅存数据盘，已加入 `.gitignore`

**观察与结论**：
- `hf download` 必须加 `--repo-type dataset`，否则会 401
- 数据集体积小（<500MB），对数据盘压力可忽略

**下一步**：数据预处理 → SFT 训练代码搭建 → exp-005

---

## exp-005 — 恢复训练框架 + SLAKE VQA 数据预处理 — 2026-06-06

**状态**：已完成

**动机**：从 demo1 恢复 finetuning 代码，编写 SLAKE 医学 VQA 专用 dataloader 与预处理脚本。

**数据版本**：data-v2

**配置**：
| 项 | 值 |
|----|-----|
| 代码来源 | `Brilliant-B/awesome-demos` demo1.tar.gz |
| 新增 Dataset | `finetuning/dataset/slake_vqa_dataset.py` |
| 预处理脚本 | `scripts/preprocess_slake.py` |
| 训练配置 | `finetuning/configs/sft_slake.py` |
| 语言过滤 | 英文（`q_lang=en`，4919 条） |
| 对话格式 | `<image>\n{question}` → `{answer}` |

**命令**：
```bash
cd /root/autodl-tmp/qwenvl-sft
hf download Brilliant-B/awesome-demos demo1.tar.gz --local-dir /root/autodl-tmp/downloads
tar -xf /root/autodl-tmp/downloads/demo1.tar.gz -C _demo_extract
cp -r _demo_extract/finetuning . && cp _demo_extract/setup.py .
pip install -e . --no-deps && pip install -e finetuning --no-deps
python3 scripts/preprocess_slake.py --split train --q-lang en \
  --output datasets/SLAKE/manifests/train_en.json
```

**结果**：
- finetuning 框架恢复（124MB 临时解压目录 `_demo_extract/`，不进 Git）
- 英文训练集 4919 条，图片路径全部有效（missing=0）
- `SlakeVQADataset` 可直接读取 JSON + 本地图片，无需转 TSV

**观察与结论**：
- SLAKE 原生 JSON 格式足够简单，直接写 Dataset 比转 TSV 更轻量
- OPEN/CLOSED 问答混合（2976/1943），冒烟阶段不区分

**下一步**：冒烟测试训练 → exp-006

---

## exp-006 — SLAKE VQA SFT 冒烟测试 — 2026-06-06

**状态**：已完成

**动机**：验证 SLAKE VQA 训练 pipeline 端到端可跑（数据加载 → 前向 → loss 下降 → 无 OOM）。

**数据版本**：data-v2

**配置**：
| 项 | 值 |
|----|-----|
| 脚本 | `finetuning/scripts/sft_slake_smoke.sh` |
| 配置快照 | `configs/snapshots/exp-006-slake-smoke.py` |
| 样本数 | 64（随机子集） |
| max_steps | 10 |
| batch_size | 1 × grad_accum 2 |
| 微调模块 | MLP + LLM（vision 冻结） |
| lr | 2e-5 |
| 输出 | `work_dirs/slake-vqa-smoke/` |

**命令**：
```bash
cd /root/autodl-tmp/qwenvl-sft
bash finetuning/scripts/sft_slake_smoke.sh
```

**结果**：
- 训练 10 steps，耗时 8.75s，无 OOM
- loss：2.88 → 0.10（step 9），平均 train_loss=1.27
- 可训练参数：3.12B（vision 冻结后）

**观察与结论**：
- A800 80G 单卡 batch_size=1 运行稳定
- flash-attn + gradient checkpointing 正常工作
- 可进入正式训练（全量 4919 英文样本，3–5 epoch）

**下一步**：正式 SFT 微调 + 验证集评估

---

## exp-007 — SLAKE VQA 正式 SFT 训练 — 2026-06-06

**状态**：已完成

**动机**：在全量 SLAKE 英文 VQA 数据上完成 3 epoch SFT，得到可用于推理的微调权重。

**数据版本**：data-v2

**配置**：
| 项 | 值 |
|----|-----|
| 脚本 | `finetuning/scripts/sft_slake.sh` |
| 配置快照 | `configs/snapshots/exp-007-slake-sft-v1.py` |
| 样本数 | 4919（`q_lang=en`） |
| epoch | 3（1845 steps） |
| batch | 2 × grad_accum 4（有效 batch=8） |
| 微调模块 | MLP + LLM（vision 冻结） |
| lr | 2e-5 |
| save | steps=500, save_only_model=True |
| 输出 | `work_dirs/slake-vqa-sft-v1/checkpoint-1845` |

**命令**：
```bash
cd /root/autodl-tmp/qwenvl-sft
bash finetuning/scripts/sft_slake.sh
# step 1000 因磁盘满中断，清理后加 --save_only_model True 从 checkpoint-500 续训
```

**结果**：
- 3 epoch 全部完成，train_loss=**0.082**
- 最终 checkpoint：`checkpoint-1845`（~7.6GB，仅模型权重）
- 中间 checkpoint：`checkpoint-1000`、`checkpoint-1500`
- 续训段耗时 2398s（~40 min），总 wall time ~70 min（含首次中断）

**观察与结论**：
- 含 optimizer 的 checkpoint 约 19GB/个，50G 数据盘会在 step 1000 写满；改用 `save_only_model=True` 后每个 checkpoint ~7GB
- loss 从 ~2.6（epoch 0）降至 ~0.08（epoch 3），收敛正常
- 权重路径：`/root/autodl-tmp/qwenvl-sft/work_dirs/slake-vqa-sft-v1/checkpoint-1845`

**下一步**：验证集推理评估 + 样例可视化

---

## exp-008 — SLAKE 验证集推理评估 — 2026-06-06

**状态**：已完成

**动机**：对比基座模型与 SFT 权重在 SLAKE 英文验证集上的 VQA 准确率，生成前后对比图。

**数据版本**：data-v2

**配置**：
| 项 | 值 |
|----|-----|
| 验证集 | `validation.json`，`q_lang=en`，1053 条 |
| 基座模型 | `pretrained/Qwen2.5-VL-3B-Instruct` |
| SFT 模型 | `work_dirs/slake-vqa-sft-v1/checkpoint-1845` |
| 评估脚本 | `scripts/eval_slake_val.py` |
| 匹配方式 | 归一化 exact match |
| max_new_tokens | 32 |

**命令**：
```bash
cd /root/autodl-tmp/qwenvl-sft
python3 scripts/eval_slake_val.py \
  --output-dir records/results/exp-008 \
  --max-new-tokens 32
```

**结果**：

| 指标 | Base | SFT | Δ |
|------|------|-----|---|
| Overall | 0.1% (1/1053) | **81.3%** (856/1053) | +81.2% |
| Closed | 0.0% (0/422) | **87.0%** (367/422) | +87.0% |
| Open | 0.2% (1/631) | **77.5%** (489/631) | +77.3% |

**产出物**：
- `records/results/exp-008/accuracy_comparison.png` — 前后对比柱状图
- `records/results/exp-008/qualitative_examples.png` — 样例可视化
- `records/results/exp-008/metrics.json` — 完整指标

**观察与结论**：
- 基座模型倾向生成长文本，几乎无法 exact match SLAKE 短答案格式
- SFT 后模型学会输出简短答案（如 `MRI`、`Yes/No`），Closed 题提升最明显
- 评估耗时 ~21 min（1053 × 2 模型）

**下一步**：测试集评估 / 推理 demo

---
