# SLAKE VQA · SFT 阶段完整数据流拆解

> 对应要求：`codelearning_prompt.md`  
> 范围：**SFT 监督微调**（本项目未实现 Pretrain / DPO）  
> 基座：Qwen2.5-VL-3B-Instruct

---

## 最小测试输入样本（贯穿全文）

以下样本来自 `datasets/SLAKE/validation.json` 第 1 条英文 QA，已在 AutoDL 环境实测所有 shape。

```python
raw_sample = {
    "img_name": "xmlab0/source.jpg",          # 相对 datasets/SLAKE/imgs/
    "question": "What modality is used to take this image?",
    "answer": "MRI",
    "q_lang": "en",
    "answer_type": "OPEN",
}
# 对应影像：256×256 RGB，路径 datasets/SLAKE/imgs/xmlab0/source.jpg
```

**符号约定**

| 符号 | 含义 |
|------|------|
| B | batch_size |
| T | 文本序列长度（token 数） |
| T_max | batch 内 padding 后的最大序列长度 |
| V | 词表大小 = 151936 |
| H | LLM hidden_size = 2048 |
| P | 视觉 patch 数（进 Vision Tower 前）= 324 |
| N_img | LLM 中 `<\|image_pad\|>` placeholder 个数 = 81 |
| C_patch | 每个 patch 展平后的通道维 = 1176 |

---

# 阶段 A：数据预处理（CPU）

> 入口：`SlakeVQADataset.__getitem__`  
> 出口：`DataCollatorForSupervisedDataset.__call__` 输出的 batch dict

---

## A1. 读取原始 JSON 样本

**代码**（`finetuning/dataset/slake_vqa_dataset.py` 90–93 行）：

```90:93:finetuning/dataset/slake_vqa_dataset.py
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            sample = self.samples[i]
            image_path = self.image_root / sample["img_name"]
```

| | 内容 |
|---|---|
| **输入** | Python `dict`，无 tensor shape |
| **输入示例** | `{"img_name":"xmlab0/source.jpg", "question":"What modality...", "answer":"MRI", ...}` |
| **输出** | 同上 + `image_path` = `Path("datasets/SLAKE/imgs/xmlab0/source.jpg")` |
| **目的** | 从 SLAKE JSON 取出一条 QA，拼接图片绝对路径 |

---

## A2. 加载并 RGB 化影像（PIL）

**代码**（`slake_vqa_dataset.py` 74–76 行）：

```74:76:finetuning/dataset/slake_vqa_dataset.py
    def process_image(self, image_file: Path):
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")
```

| | 内容 |
|---|---|
| **输入** | 磁盘上的 JPEG 文件 |
| **输入 shape** | PIL `Image`，**(W, H) = (256, 256)**，3 通道 |
| **输出** | PIL `Image`，**(256, 256, 3)**，uint8，RGB |
| **目的** | 统一为 RGB，供 Qwen ImageProcessor 使用 |

---

## A3. ImageProcessor 预处理（resize + normalize + patchify）

**代码**（`slake_vqa_dataset.py` 77–81 行）：

```77:81:finetuning/dataset/slake_vqa_dataset.py
        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
```

| | 内容 |
|---|---|
| **输入** | PIL (256, 256, 3) |
| **配置** | `min_pixels=12544`, `max_pixels=1003520`（`sft_slake.py` 9–10 行） |
| **输出 pixel_values** | `float32`, shape **(P, C_patch) = (324, 1176)** |
| **输出 grid_thw** | `int64`, shape **(3,) = [T_grid, H_grid, W_grid] = [1, 18, 18]** |
| **维度含义** | P=324=18×18 个视觉 patch；1176=每个 patch 展平后的像素特征维 |
| **目的** | 将任意尺寸图 resize 到合法范围，切 patch，供 Vision Tower 消费 |

---

## A4. 构造对话文本

**代码**（`slake_vqa_dataset.py` 84–88 行）：

```84:88:finetuning/dataset/slake_vqa_dataset.py
    def build_conversations(self, sample: dict) -> List[dict]:
        return [
            {"from": "human", "value": f"<image>\n{sample['question']}"},
            {"from": "gpt", "value": sample["answer"]},
        ]
```

| | 内容 |
|---|---|
| **输入** | `sample["question"]`, `sample["answer"]` 字符串 |
| **输出** | `List[dict]`，2 轮对话 |
| **输出示例** | `[{"from":"human","value":"<image>\nWhat modality is used to take this image?"}, {"from":"gpt","value":"MRI"}]` |
| **目的** | 转为 Qwen SFT 所需的 human/gpt 对话格式 |

---

## A5. 计算 image_pad token 数量

**代码**（`slake_vqa_dataset.py` 103 行）：

```103:103:finetuning/dataset/slake_vqa_dataset.py
        grid_thw_merged = [grid_thw.prod() // self.data_args.image_processor.merge_size**2]
```

| | 内容 |
|---|---|
| **输入** | `grid_thw = [1, 18, 18]`，`merge_size = 2` |
| **计算** | `1×18×18 // 2² = 324 // 4 = **81**` |
| **输出** | `grid_thw_merged = [81]` |
| **目的** | 决定文本序列中插入多少个 `<\|image_pad\|>` placeholder，必须与 Vision Merger 输出 token 数一致 |

---

## A6. Chat Template + Tokenize + 视觉占位符替换

**代码**（`finetuning/dataset/tsv_dataset.py` 20–98 行，`preprocess_qwen_2_visual`）：

**6a. System 消息 tokenize**（47–50 行）：

```47:50:finetuning/dataset/tsv_dataset.py
        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)
```

| | 内容 |
|---|---|
| **输入** | `"You are a helpful medical assistant."` |
| **输出 input_ids 片段** | 长度 11 的 token 列表，首 token `151644` = `<\|im_start\|>` |
| **输出 labels 片段** | 11 个 `-100`（IGNORE_INDEX，不参与 loss） |
| **目的** | 添加 system prompt，SFT 时不学习这部分 |

**6b. User 消息：`<image>` → vision placeholder**（61–77 行）：

```61:77:finetuning/dataset/tsv_dataset.py
            if role == "user":
                visual_tag = f"<{visual_type}>"
                if visual_tag in content:
                    parts = content.split(visual_tag)
                    ...
                        replacement = (
                            "<|vision_start|>"
                            + f"<|{visual_type}_pad|>"
                            * grid_thw[visual_replicate_index]
                            + "<|vision_end|>"
                        )
```

| | 内容 |
|---|---|
| **输入文本** | `"<image>\nWhat modality is used to take this image?"` |
| **替换后文本** | `"<|vision_start|>" + "<|image_pad|>"×81 + "<|vision_end|>\nWhat modality..."` |
| **目的** | 在文本序列中预留 81 个视觉 token 槽位，后续由 Vision Tower 输出嵌入填充 |

**6c. Assistant 消息 labels 掩码**（84–87 行）：

```84:87:finetuning/dataset/tsv_dataset.py
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask
```

| | 内容 |
|---|---|
| **输入** | `"MRI"` → token ids `[78670, 151645, 198]`（`MRI`, `<\|im_end\|>`, `\n`） |
| **labels** | `[-100, -100, -100, 78670, 151645, 198]` 中前 3 个 mask 对应 `<\|im_start\>assistant\n` |
| **实际 supervised token** | **3 个**：`MRI`, `<\|im_end\|>`, `\n` |
| **目的** | 只对 **答案内容** 算 loss，不学习 assistant 角色标记 |

**6d. 汇总为 tensor**（93–98 行）：

| | 内容 |
|---|---|
| **输出 input_ids** | `LongTensor`, shape **(1, T) = (1, 117)** |
| **输出 labels** | `LongTensor`, shape **(1, 117)**，其中 114 个为 -100，3 个为真实 token id |
| **T=117 结构** | `[0:11]` system · `[11:16]` user 头 · `[16:97]` 81×image_pad · `[97:117]` 问题文本+assistant 答案 |

---

## A7. 计算 3D RoPE position_ids

**代码**（`slake_vqa_dataset.py` 112–116 行 + `utils/rope2d.py` 5–65 行）：

```112:116:finetuning/dataset/slake_vqa_dataset.py
        position_ids, _ = self.get_rope_index(
            self.data_args.image_processor.merge_size,
            data_dict["input_ids"],
            torch.stack([grid_thw], dim=0),
        )
```

| | 内容 |
|---|---|
| **输入** | `input_ids (1,117)`, `image_grid_thw (1,3)=[1,18,18]` |
| **输出 position_ids** | `LongTensor`, shape **(3, 1, 117)** |
| **维度含义** | 3 = (temporal, height, width) 三路 RoPE；视觉区用 3D 坐标，文本区三路相同递增 |
| **目的** | Qwen2.5-VL 使用 mRoPE，视觉 token 需要 3D 位置编码 |

---

## A8. Dataset 单样本输出

**代码**（`slake_vqa_dataset.py` 118–124 行）：

```118:124:finetuning/dataset/slake_vqa_dataset.py
        data_dict = dict(
            input_ids=data_dict["input_ids"][0],
            labels=data_dict["labels"][0],
            position_ids=position_ids,
            pixel_values=image_tensor,
            image_grid_thw=[grid_thw],
        )
```

| 键 | shape | 示例/说明 |
|----|-------|-----------|
| `input_ids` | **(117,)** | token 序列 |
| `labels` | **(117,)** | 114×(-100) + 3×监督 |
| `position_ids` | **(3, 1, 117)** | mRoPE |
| `pixel_values` | **(324, 1176)** | 视觉 patch 序列 |
| `image_grid_thw` | list[`(3,)`] | `[1,18,18]` |

---

## A9. Collator：Batch Padding & 拼接

**代码**（`finetuning/dataset/collator.py` 38–117 行）：

**9a. 文本 padding**（43–48 行）：

```43:48:finetuning/dataset/collator.py
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
```

| | B=2 实测 |
|---|---|
| **输入** | 2 条样本，T₁=117, T₂=120 |
| **output input_ids** | **(2, 120)**，短序列末尾 pad `pad_token_id` |
| **output labels** | **(2, 120)**，短序列末尾 pad `-100` |
| **目的** | 对齐 batch 内序列长度 |

**9b. attention_mask**（60 行）：

```60:60:finetuning/dataset/collator.py
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
```

| | 内容 |
|---|---|
| **输出** | **(B, T_max)**，`bool`，真实 token=1，pad=0 |
| **目的** | 注意力层忽略 padding 位置 |

**9c. 视觉 batch 拼接**（62–91 行）：

| | B=1 实测 | B=2 实测 |
|---|---|---|
| **pixel_values** | **(381024,)** = flatten(324×1176) | **(762048,)** = 2 图拼接 |
| **image_grid_thw** | **(1, 3)** | **(2, 3)** |
| **目的** | 多图沿 patch 维 concat，供 Vision Tower 一次处理 |

**9d. position_ids padding**（11–22, 49 行 `pad_and_cat`）：

| | B=2 实测 |
|---|---|
| **输出 position_ids** | **(3, 2, 120)**，较短序列在 dim=2 上 pad 常数 1 |

---

# 阶段 B：模型前向传播（GPU）

> 入口：`Qwen2_5_VLForConditionalGeneration.forward`（transformers `modeling_qwen2_5_vl.py` 1688 行）  
> 训练调用链：`Trainer` → `model(**batch)` → `loss`

以下以 **B=1** 最小样本为例。

---

## B1. 文本嵌入 lookup

**代码**（`modeling_qwen2_5_vl.py` 1753–1754 行）：

```1753:1754:miniconda3/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
```

| | 内容 |
|---|---|
| **输入 input_ids** | **(1, 117)**，`LongTensor` |
| **输出 inputs_embeds** | **(1, 117, 2048)** = **(B, T, H)** |
| **目的** | 每个 token id 映射为 H 维向量；此时 81 个 image_pad 位置仍是「占位」嵌入 |

---

## B2. 视觉模态独立编码（Vision Tower）

**代码**（`modeling_qwen2_5_vl.py` 1756–1757 行 → `Qwen2_5_VisionTransformerPretrainedModel.forward` 495 行）：

```1756:1757:miniconda3/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
```

| 步骤 | shape 变化 | 说明 |
|------|-----------|------|
| 输入 pixel_values | **(381024,)** → reshape 为 **(324, 1176)** | 324 个 patch |
| patch_embed | **(324, 1280)** | Conv3d 投影到 vision hidden |
| 32×VisionBlock | **(324, 1280)** | 视觉自注意力 |
| merger | **(81, 2048)** = **(N_img, H)** | spatial merge 2×2，324→81 |
| **输出 image_embeds** | **(81, 2048)** | 与 LLM hidden 对齐 |

**目的**：将像素 patch 编码为 81 个视觉 token 嵌入，供后续填入文本序列。

---

## B3. 多模态融合（masked_scatter）

**代码**（`modeling_qwen2_5_vl.py` 1758–1771 行）——**融合发生在此**：

```1758:1771:miniconda3/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(...)
                mask = input_ids == self.config.image_token_id
                ...
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
```

| | 内容 |
|---|---|
| **融合方式** | **按位替换（masked_scatter）**，非拼接、非 cross-attention |
| **融合前 inputs_embeds** | **(1, 117, 2048)**，81 个位置是 image_pad 的 embed |
| **融合后 inputs_embeds** | **(1, 117, 2048)**，shape 不变 |
| **替换规则** | `input_ids == 151655`（image_token_id）的 81 个位置 ← `image_embeds (81,2048)` |
| **硬约束** | `n_image_tokens == n_image_features`，否则 ValueError |
| **目的** | 把视觉特征嵌入到文本序列的视觉槽位，形成统一多模态序列 |

---

## B4. LLM Backbone（36 层 Decoder）

**代码**（`modeling_qwen2_5_vl.py` 1825–1836 行）：

```1825:1836:miniconda3/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            ...
            inputs_embeds=inputs_embeds,
```

| | 内容 |
|---|---|
| **输入** | `inputs_embeds (1,117,2048)`, `position_ids (3,1,117)`, `attention_mask (1,117)` |
| **每层** | Self-Attention（Flash Attn 2，因果掩码）+ MLP |
| **输出 hidden_states** | **(1, 117, 2048)** = **(B, T, H)** |
| **目的** | 多模态序列联合建模，视觉+文本联合 self-attention |

---

## B5. LM Head → Logits

**代码**（`modeling_qwen2_5_vl.py` 1838–1839 行）：

```1838:1839:miniconda3/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
```

| | 内容 |
|---|---|
| **输入** | **(1, 117, 2048)** |
| **输出 logits** | **(1, 117, 151936)** = **(B, T, V)** |
| **目的** | 每个位置预测下一个 token 的概率分布 |

---

## B6. Loss 计算（因果 LM + label 掩码）

**代码**（`modeling_qwen2_5_vl.py` 1841–1854 行）：

```1845:1854:miniconda3/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ...
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            loss = loss_fct(shift_logits, shift_labels)
```

| 步骤 | shape | 说明 |
|------|-------|------|
| logits | (1, 117, V) | 原始输出 |
| shift_logits | **(1, 116, V)** | 位置 t 预测 token t+1 |
| shift_labels | **(1, 116)** | labels 右移 1 |
| flatten | **(116, V)** vs **(116,)** | CrossEntropyLoss |
| 有效 loss 位置 | **仅 3 个** | labels≠-100 的位置（`MRI`, `<\|im_end\|>`, `\n`） |
| **输出 loss** | **标量**，实测 base 模型 ≈ **1.72** | 对有效 token 平均 CE |

**labels=-100 的处理**：`CrossEntropyLoss` 默认 `ignore_index=-100`，system/user/视觉/pad 位置均不参与 loss。

---

# 完整 Shape 变化流程图（纯文本）

```
【原始数据】
  JSON dict + PIL Image(256,256,3)
       │
       ▼ A3 ImageProcessor
  pixel_values (324,1176)    grid_thw [1,18,18]
       │                          │
       │                    N_img = 81 image_pad slots
       ▼ A6 preprocess_qwen_2_visual
  input_ids (117,)           labels (117,)  [114×mask + 3×supervised]
  position_ids (3,1,117)
       │
       ▼ A9 Collator [B=1]
  input_ids (1,117)          labels (1,117)       attention_mask (1,117)
  pixel_values (381024,)     image_grid_thw (1,3) position_ids (3,1,117)
       │
       ▼ B1 embed_tokens
  inputs_embeds (1,117,2048)
       │
       ├─ B2 visual(pixel_values) ──► image_embeds (81,2048)
       │
       ▼ B3 masked_scatter 融合
  inputs_embeds (1,117,2048)   ← 81个视觉槽位已替换
       │
       ▼ B4 LLM ×36层 (mRoPE + causal attn)
  hidden_states (1,117,2048)
       │
       ▼ B5 lm_head
  logits (1,117,151936)
       │
       ▼ B6 shift + CE, ignore_index=-100
  loss (scalar)
```

---

# 潜在 Shape 不匹配风险（本项目代码）

| # | 位置 | 风险 | 后果 |
|---|------|------|------|
| 1 | A5 `grid_thw_merged` vs A3 `grid_thw` | 若 `merge_size` 与 processor 不一致，81 个 placeholder 可能对不上 | B3 抛 `Image features and image tokens do not match` |
| 2 | `collator.py` 62–81 行 | `itertools.chain(*pixel_values)` 对 2D tensor 按行迭代，再 `cat` 变 1D | 当前能跑通，但 shape 隐式依赖 flatten；改 tensor 维度易 silently break |
| 3 | `slake_vqa_dataset.py` 126 行 | `input_ids.size(0) > max_length` 静默 skip 换下一条 | 长问题样本被跳过，训练分布偏移 |
| 4 | `collator.py` 50–56 行 | 超长序列只 print 警告，不 truncate | OOM 或 Flash Attn 报错 |
| 5 | B3 `n_image_tokens != n_image_features` | 训练/推理时 resize 策略不同 → grid 变化 | 推理崩溃 |
| 6 | `position_ids` pad 值=1（collator 17 行） | padding 位置的 mRoPE 坐标非零 | 通常被 attention_mask 屏蔽，影响较小 |

---

# 附：本项目 SFT 训练配置入口

| 文件 | 作用 |
|------|------|
| `finetuning/configs/sft_slake.py` | Dataset + Collator 配置 |
| `finetuning/train.py` 185–211 行 | BUILDER 构建 Dataset/Collator，交给 HF Trainer |
| `finetuning/scripts/sft_slake.sh` | 启动 3 epoch 正式训练 |

**未覆盖阶段**：Pretrain（无）、DPO/RL（`finetuning/verl/` 目录存在但未在本项目 SLAKE 流程中启用）。
