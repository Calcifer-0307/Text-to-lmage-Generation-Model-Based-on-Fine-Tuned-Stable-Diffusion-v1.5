# 项目结构说明

本项目按数据与代码分层组织，便于实验、服务与处理脚本协同开发。

## 目录
- data/：存放数据脚本或少量示例数据（大数据集不纳入版本库）
- notebooks/：存放 .ipynb 实验代码与探索性笔记
- src/：源代码
  - api/：FastAPI 相关代码
  - processing/：数据处理脚本

## 使用建议
- 将体积较大的数据、敏感信息放在 data/ 并通过 .gitignore 忽略，保留占位文件 .gitkeep 以便提交目录结构
- notebooks 中的检查点目录已忽略，建议定期导出关键结果到 README 或文档
- 在 src 中分别维护服务与处理逻辑，保持模块边界清晰

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
  
- 加载 Lunara Aesthetic 并导出示例：
  
  ```bash
  python src/processing/lunara_loader.py --sample-size 100 --sample-out data/lunara_sample.jsonl
  ```
  
- 直接在代码中使用：
  
  ```python
  from datasets import load_dataset
  ds = load_dataset("moonworks/lunara-aesthetic")
  ```

## 数据检查与报告
- 生成数据检查报告（JSON）：
  
  ```bash
  .venv/bin/python scripts/data_check.py --split 'train[:100]' --sample-size 100 --cache-dir ./data --out data/lunara_check_report.json
  ```
  
- 将 JSON 报告渲染为 Markdown，并生成频次可视化图：
  
  ```bash
  .venv/bin/python scripts/report_render.py --input data/lunara_check_report.json --out data/lunara_check_report.md --fig-dir data/report_figs
  ```
  
- 直接运行渲染脚本（无参数）：若 JSON 不存在，将自动执行一次数据检查并生成报告与图表
  
  ```bash
  .venv/bin/python scripts/report_render.py
  ```
  
- 输出说明：
  - 报告文件：data/lunara_check_report.md
  - 图像目录：data/report_figs/
  - 报告与图表采用英文显示，便于跨平台阅读

## 永久拼写订正
- 生成并保存全量订正后的标签 CSV 与摘要：
  
  ```bash
  .venv/bin/python scripts/data_correct.py --split train --cache-dir ./data --out-csv data/lunara_corrected_labels.csv --summary data/lunara_correction_summary.json
  ```
  
- 订正逻辑：
  - 自动识别高相似且高频的标签作为建议项（如 digitial -> digital）
  - 输出订正映射、订正后分布与订正明细（correction_stats）
  
- 渲染仅包含“修正后”报告与图表：
  
  ```bash
.venv/bin/python scripts/report_render.py --input data/lunara_check_report.json --out data/lunara_check_report.md --fig-dir report_figs
  ```

- 输出说明：
  - 报告文件：data/lunara_check_report.md
  - 图像目录：report_figs/
  - 报告内图片链接为绝对 file:// 路径，跨查看器可用
# Text-to-lmage-Generation-Model-Based-on-Fine---Tuned-Stable-Diffusion-v1.5

# Shealyn到此一游