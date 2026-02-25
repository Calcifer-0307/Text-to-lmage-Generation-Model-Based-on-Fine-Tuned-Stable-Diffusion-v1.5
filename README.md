# 项目结构说明

本项目按数据与代码分层组织，便于实验、服务与处理脚本协同开发。

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

## 快速开始
- 安装依赖：
  
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
  
- 直接在代码中使用：
  
  ```python
  from datasets import load_dataset
  ds = load_dataset("moonworks/lunara-aesthetic")
  ```

## 数据检查与报告
- 生成数据检查报告（JSON）：
  
  ```bash
  .venv/bin/python scripts/preprocessing/data_check.py --split 'train[:100]' --sample-size 100 --cache-dir ./data --out data/lunara_check_report.json
  ```

## 永久拼写订正
- 生成并保存全量订正后的标签 CSV 与摘要：
  
  ```bash
  .venv/bin/python scripts/preprocessing/data_correct.py --split train --cache-dir ./data --out-csv data/lunara_corrected_labels.csv --summary data/lunara_correction_summary.json
  ```
  
- 订正逻辑：
  - 自动识别高相似且高频的标签作为建议项（如 digitial -> digital）
  - 输出订正映射、订正后分布与订正明细（correction_stats）
 
## 合并修正标签到数据集（离线版本）
  
- 严格离线（只读本地缓存分片 .parquet/.arrow）：
  ```bash
  .venv/bin/python scripts/preprocessing/merge_labels.py \
    --csv-path "data/lunara_corrected_labels.csv" \
    --cache-dir "data/moonworks___lunara-aesthetic" \
    --output-dir "data/lunara_final_dataset_offline" \
    --revision "851305085843a2b2d96ea0d44904bc54a670c5f4" \
    --offline
  ```
  
- 训练加载（本地）：
  ```python
  from datasets import load_from_disk
  dataset = load_from_disk("/Users/eason/TRAE/Text-to-lmage-Generation-Model-Based-on-Fine-Tuned-Stable-Diffusion-v1.5/data/lunara_final_dataset_rev")
  ```
  
## 团队复现建议
- 固定原始数据 revision，确保拉取同一快照
- 将修正 CSV 与合并脚本纳入仓库，按统一命令在本地重建数据包
- 大体量数据与缓存不入库；如需共享，可选择私有 Hub 或对象存储分发
