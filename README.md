# 项目结构说明

本项目按数据与脚本分层组织，便于可复现的数据预处理与模型训练。

## 目录结构

- **data/**：存放数据与缓存
  - 大数据集不纳入版本库，仅 .gitkeep 占位
  - 预处理后的数据集存储于此：`lunara_final_dataset_rev/`

- **notebooks/**：存放 .ipynb 实验代码与探索性笔记（检查点已忽略）

- **scripts/**：按阶段组织的脚本
  - **preprocessing/** 阶段1 - 数据预处理脚本
    - `data_import.py`：从 HuggingFace 导入数据集
    - `data_check.py`：数据质量检查与统计分析
    - `data_correct.py`：数据标签纠错
    - `merge_labels.py`：合并修正后的标签为本地数据包
    - `report_render.py`：生成报告图表
  
  - **model_verification/** 阶段2 - Stable Diffusion 前向传播验证
    - `test_forward.py`：最小化前向推理示例
    - `forward_components.py`：深入理解 SD 各组件（Text Encoder、UNet、VAE、Scheduler）
  
  - **fine_tuning/**：阶段3 - LoRA 微调训练（筹备中）
    - `dataset_adapter.py`：数据适配层 - 图像/文本预处理及 DataLoader 工厂
    - `lora_config.py`：LoRA 配置（占位）
    - `train_lora.py`：LoRA 微调训练脚本（占位）
    - `eval.py`：模型评估脚本（占位）
    - `checkpoint/`：LoRA 微调权重保存目录

- **output/**：模型推理输出文件

- **report_figs/**：报告图表输出目录

## 项目进度

### ✅ 已完成
- [x] 数据预处理流程（preprocessing/）
- [x] Stable Diffusion v1.5 前向传播验证（model_verification/）
  - 验证环境配置和模型加载
  - 理解模型的核心组件和前向流程
- [x] 微调用数据适配框架（fine_tuning/dataset_adapter.py）
  - 图像预处理：resize、normalize、Tensorization
  - 文本处理：tokenization、padding
  - 条件 Dropout：提高无条件指导能力
  - DataLoader 工厂：便捷创建训练/验证数据加载器

### 🚧 进行中 / 待完成
- [ ] LoRA 微调训练脚本（fine_tuning/train_lora.py）
- [ ] 训练配置模块（fine_tuning/lora_config.py）
- [ ] 模型评估脚本（fine_tuning/eval.py）

## 使用建议
- 将体积较大的数据、敏感信息放在 data/ 并通过 .gitignore 忽略，保留占位文件 .gitkeep 以便提交目录结构
- notebooks 中的检查点目录已忽略，建议定期导出关键结果到 README 或文档
- 预处理与训练脚本分类清晰：请从 scripts/preprocessing 或 scripts/training 调用

## 环境与依赖
- 建议使用虚拟环境：.venv/ 或 venv/（已在 .gitignore 中忽略）
- 常见的临时文件与日志已默认忽略，可按需扩展 .gitignore

## 使用流程
### 阶段1: 环境配置与数据预处理
1. 安装依赖
   
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. 运行数据导入示例（确认环境与缓存路径）
   
   ```bash
   .venv/bin/python scripts/preprocessing/data_import.py
   ```
3. 生成数据检查报告（可选）
   
   ```bash
   .venv/bin/python scripts/preprocessing/data_check.py --split 'train[:100]' --sample-size 100 --cache-dir ./data --out data/lunara_check_report.json
   ```
4. 生成订正后的标签（纠错导出 CSV 与摘要）
   
   ```bash
   .venv/bin/python scripts/preprocessing/data_correct.py --split train --cache-dir ./data --out-csv data/lunara_corrected_labels.csv --summary data/lunara_correction_summary.json
   ```
5. 合并修正后的标签为本地数据包
   
   复用缓存（固定版本，必要时联网拉取并缓存）：
   ```bash
   .venv/bin/python scripts/preprocessing/merge_labels.py \
     --csv-path "data/lunara_corrected_labels.csv" \
     --cache-dir "data/moonworks___lunara-aesthetic" \
     --output-dir "data/lunara_final_dataset_rev" \
     --revision "851305085843a2b2d96ea0d44904bc54a670c5f4"
   ```
   严格离线（仅读取本地缓存分片 .parquet/.arrow）：
   ```bash
   .venv/bin/python scripts/preprocessing/merge_labels.py \
     --csv-path "data/lunara_corrected_labels.csv" \
     --cache-dir "data/moonworks___lunara-aesthetic" \
     --output-dir "data/lunara_final_dataset_offline" \
     --revision "851305085843a2b2d96ea0d44904bc54a670c5f4" \
     --offline
   ```
   后续训练模型加载数据集（本地）：
   ```python
   from datasets import load_from_disk
   dataset = load_from_disk("data/lunara_final_dataset_rev")
   ```

### 阶段2: Stable Diffusion前向传播验证

本阶段验证环境配置，确保 SD v1.5 能正常加载和推理。

#### 运行推理测试

   ```bash
   .venv/bin/python scripts/training/test_forward.py
   ```

   该脚本将：
   根据设置的提示词生成图像，保存到output/test_output.png
   
   首次运行需要下载模型，需要一些时间；推理也需要一些时间。需要耐心等待。

#### 理解模型组件

   ```bash
   python scripts/model_verification/forward_components.py
   ```

   该脚本详细讲解：
  
   - Text Encoder: 将文本提示词转换为嵌入向量
   - UNet: 扩散模型的核心，预测噪声
   - VAE Decoder: 将潜在空间向量解码为图像
   - Scheduler: 控制去噪步骤和时间步

### 阶段3: 模型微调

   运行数据适配测试

   ```bash
   python scripts/fine_tuning/dataset_adapter.py
   ```
