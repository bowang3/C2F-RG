# C2F-RG 复现代码

本目录从 **ClinFusion-8B** 完整权重开始，依次训练单序列 MRI 对齐（A1、A2）、患者级多序列平面融合、文本解码器 LoRA。三个入口均使用 **2 张 GPU、Accelerate DDP、BF16**；不会自动运行训练。模型结构保留与 ClinFusion 检查点兼容的类和参数名。

## 准备

在仓库根目录运行，建议 Python 3.11、PyTorch 2.8、CUDA 12.8。安装 `requirements.txt` 后，还需安装与本机 PyTorch/CUDA 匹配的 **flash-attn 2**。训练使用 Qwen3-VL 的 `flash_attention_2`；至少准备两张 48 GB GPU 和足够的 CPU 内存、磁盘空间保存完整模型。

```bash
git clone https://github.com/bowang3/C2F-RG.git
cd C2F-RG
python -m pip install -r requirements.txt
```

将以下完整模型放到同一个目录，并将 `MODEL_ROOT` 指向它：

```text
models/
  ClinFusion-8B/
  Qwen3-VL-8B-Instruct/
  dinov2-large/
  CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/
```

权重和病例数据不包含在本目录。训练 JSONL 每行一例。A1/A2 每例恰好一条 NIfTI 序列；后两阶段每例为 1–5 条序列，文件名须以 `-T1CA`、`-T2A`、`-ADC`、`-T1CS` 或 `-T2S` 结尾（后接 `.nii` 或 `.nii.gz`）。路径可为绝对路径，也可相对 JSONL 所在目录。

```json
{"messages":{"prompt":"Describe the imaging findings.","nifti":["images/case01-T2A.nii.gz"]},"GT":"Single-sequence description."}
{"messages":{"prompt":"Write a cervical MRI report.","nifti":["images/case01-T2A.nii.gz","images/case01-T2S.nii.gz"]},"GT":"Full report."}
```

第一行示例用于对齐数据，第二行示例用于多序列数据。论文使用从训练报告提取的 9,053 条单序列描述，以及 2,399 条患者级训练记录；125 条测试记录不参与训练。公开代码无法代替未公开的 MRI、报告、单序列描述和 ClinFusion 权重；要复现论文数值，须使用相同数据划分及权重。

## 训练

以下命令按顺序执行。`GPU_IDS` 默认 `0,1`；`PYTHON_BIN` 可指定已安装依赖的 Python。训练产物均为完整模型 checkpoint，供下一阶段直接加载。

```bash
export MODEL_ROOT=/absolute/path/to/models
export OUTPUT_ROOT=/absolute/path/to/output
export PYTHON_BIN=python
export GPU_IDS=0,1

export TRAIN_DATA=/absolute/path/to/alignment.jsonl
bash train_alignment/run.sh

export TRAIN_DATA=/absolute/path/to/train.jsonl
bash train_multi/run.sh
bash train_multi_llm/run.sh
```

| 阶段 | 训练参数 | epoch | 有效全局 batch | 学习率 | 输出 |
| --- | --- | ---: | ---: | --- | --- |
| A1 | 3D projector、3D CaSL | 1 | 16 | 1e-5 | `alignment_a1/final` |
| A2 | A1 参数、PE-3D | 3 | 16 | 5e-6；PE-3D 1e-6 | `alignment_a2/final` |
| 多序列 | Axial/Sagittal 四分支融合器 | 5 | 4 | 1e-4 | `multi/final` |
| 多序列 + LLM | 融合器、文本注意力 Q/K/V/O LoRA | 3 | 16 | 融合器 1e-5；LoRA 3e-5 | `multi_llm/final` |

每卡 batch 默认 1，脚本根据两卡和目标全局 batch 计算梯度累积。每阶段只保存最后一个完整模型；最后阶段将 LoRA 合并进模型权重。`output` 下的 `training.log` 与 `final/training_metadata.json` 记录运行参数。若训练中断，当前精简脚本需要从该阶段重新开始。
