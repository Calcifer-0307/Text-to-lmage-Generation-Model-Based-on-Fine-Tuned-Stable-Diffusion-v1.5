# 项目结构说明

本项目按数据与脚本分层组织，便于可复现的数据预处理与模型训练。

## 目录
- data/：存放数据与缓存（大数据集不纳入版本库，仅 .gitkeep 占位）
- notebooks/：存放 .ipynb 实验代码与探索性笔记（检查点已忽略）
- scripts/
  - preprocessing/：数据预处理脚本（数据检查、纠错、合并等）
  - training/：模型训练脚本（占位）
- report_figs/：报告图输出目录

## 使用建议
- 将体积较大的数据、敏感信息放在 data/ 并通过 .gitignore 忽略，保留占位文件 .gitkeep 以便提交目录结构
- notebooks 中的检查点目录已忽略，建议定期导出关键结果到 README 或文档
- 预处理与训练脚本分类清晰：请从 scripts/preprocessing 或 scripts/training 调用

## 环境与依赖
- 建议使用虚拟环境：.venv/ 或 venv/（已在 .gitignore 中忽略）
- 常见的临时文件与日志已默认忽略，可按需扩展 .gitignore

## 使用流程（复现数据改动）
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
