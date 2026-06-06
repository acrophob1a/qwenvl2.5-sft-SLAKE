# 数据集变更日志

## data-v0 — 2026-06-06

**变更类型**：删除

**摘要**：删除教程自带的 Grounding-ToyData（任务方向不符，应为 SLAKE 医学 VQA）

**详情**：
- 原路径：`datasets/Grounding-ToyData/`（已删除）
- 原因：demo 为 Visual Grounding，与本项目 SLAKE VQA 目标不符

**关联实验**：exp-003

**验证命令**：
```bash
ls /root/autodl-tmp/qwenvl-sft/datasets/  # 待用户下载 SLAKE 后重建
```

---

## data-v1 — 2026-06-06

**变更类型**：新增

**摘要**：下载并解压 SLAKE 1.0 医学 VQA 数据集（BoKelvin/SLAKE）

**详情**：
- 来源/路径：`/root/autodl-tmp/qwenvl-sft/datasets/SLAKE/`
- 下载源：https://huggingface.co/datasets/BoKelvin/SLAKE（hf-mirror）
- 规模：642 张影像目录，14,028 条 QA（train 9835 / val 2099 / test 2094）
- 解压后体积：约 470 MB
- 模态：CT / MRI / X-Ray（放射影像，五部位）

**关联实验**：exp-004

**验证命令**：
```bash
ls /root/autodl-tmp/qwenvl-sft/datasets/SLAKE/imgs | wc -l   # 642
python3 -c "import json; b='/root/autodl-tmp/qwenvl-sft/datasets/SLAKE';
print({s: len(json.load(open(f'{b}/{s}.json'))) for s in ['train','validation','test']})"
# {'train': 9835, 'validation': 2099, 'test': 2094}
du -sh /root/autodl-tmp/qwenvl-sft/datasets/SLAKE
```

---

## data-v2 — 2026-06-06

**变更类型**：清洗

**摘要**：SLAKE 英文 VQA 预处理验证，生成英文 manifest，对接 SlakeVQADataset

**详情**：
- 输入：`datasets/SLAKE/train.json`（9835 条，含中英双语）
- 过滤：`q_lang=en` → 4919 条，图片缺失 0
- 问答类型：OPEN 2976 / CLOSED 1943
- 对话格式：User `<image>\n{question}` / Assistant `{answer}`
- manifest 路径：`datasets/SLAKE/manifests/train_en.json`（1.9MB，不进 Git）

**关联实验**：exp-005, exp-006

**验证命令**：
```bash
python3 scripts/preprocess_slake.py --split train --q-lang en
# total=4919 valid=4919 missing_images=0
```
