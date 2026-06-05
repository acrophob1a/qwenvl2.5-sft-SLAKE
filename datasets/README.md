# 数据集目录

大文件不进 Git，见 `.gitignore`。

## SLAKE

- **来源**：[BoKelvin/SLAKE](https://huggingface.co/datasets/BoKelvin/SLAKE)
- **本地路径**：`datasets/SLAKE/`
- **下载命令**：

```bash
export HF_ENDPOINT=https://hf-mirror.com
hf download BoKelvin/SLAKE --repo-type dataset \
  --local-dir datasets/SLAKE
cd datasets/SLAKE && unzip -q imgs.zip && unzip -q KG.zip
```
