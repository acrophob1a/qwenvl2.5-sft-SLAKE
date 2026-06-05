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
