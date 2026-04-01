<br />

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
  - **model\_verification/** 阶段2 - Stable Diffusion 前向传播验证
    - `test_forward.py`：最小化前向推理示例
    - `forward_components.py`：深入理解 SD 各组件（Text Encoder、UNet、VAE、Scheduler）
  - **fine\_tuning/**：阶段3 - LoRA 微调训练（筹备中）
    - `dataset_adapter.py`：数据适配层 - 图像/文本预处理及 DataLoader 工厂
    - `train_overfit.py`：LoRA 过拟合测试
    - `train_lora.py`：LoRA 微调训练脚本
    - `fix_configs.py`：关键脚本，用于修复 LoRA 配置中的命名空间冲突。
    - `eval.py`：模型评估脚本
    - `inference_engine.py`：高性能推理引擎，支持多线程去噪与实时回调
    - `checkpoint/`：LoRA 微调权重保存目录
- **output/**：模型推理输出文件

## 项目进度

### ✅ 已完成

- [x] 数据预处理流程（preprocessing/）
- [x] Stable Diffusion v1.5 前向传播验证（model\_verification/）
  - 验证环境配置和模型加载
  - 理解模型的核心组件和前向流程
- [x] 微调用数据适配框架（fine\_tuning/dataset\_adapter.py）
  - 图像预处理：resize、normalize、Tensorization
  - 文本处理：tokenization、padding
  - 条件 Dropout：提高无条件指导能力
  - DataLoader 工厂：便捷创建训练/验证数据加载器
- [x] LoRA 微调训练脚本（fine\_tuning/train\_lora.py）
- [x] 模型评估脚本（fine\_tuning/eval.py）
- [x] 实时生成后端实现 (main.py)

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
python scripts/model_verification/test_forward.py
```

该脚本将：
根据设置的提示词生成图像，保存到output/test\_output.png

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

### 阶段3: 模型微调 (Model Fine-Tuning)

在此阶段，我们利用 LoRA (Low-Rank Adaptation) 技术对 Stable Diffusion v1.5 进行风格化微调。本项目提供了两种微调策略：仅微调 UNet 以及 UNet + Text Encoder 联合微调。

#### 3.1 预训练验证 (Pre-training Validation)

在正式开启大规模训练前，建议依次运行以下脚本以验证数据流水线与 LoRA 注入逻辑：

*   **数据适配器测试**：检查图像 Resize/Normalize 及文本 Tokenization 是否符合模型预期。
    ```bash
    python scripts/fine_tuning/dataset_adapter.py
    ```
*   **单图过拟合微调测试**：确保训练代码、损失函数及权重更新逻辑正常。若模型能完美复现单张训练图，则证明微调环境已就绪。
    ```bash
    python scripts/fine_tuning/train_overfit.py
    ```

#### 3.2 执行微调训练 (Training Execution)

微调产生的 Checkpoints 将自动保存在 `scripts/fine_tuning/checkpoint/` 路径下。

*   **策略 A：UNet LoRA 微调**
    重点学习视觉布局与画面整体风格。
    ```bash
    python scripts/fine_tuning/train_lora.py
    ```
*   **策略 B：UNet + Text Encoder 联合微调**
    深度学习语义关联，使模型生成的图像在细腻质感和 Prompt 匹配度上表现更佳。
    ```bash
    python scripts/fine_tuning/train_lora2.py
    ```

#### 3.3 修复权重配置文件 (Critical Fix)

**注意（必须执行）**：由于训练包装器在保存时会产生命名空间前缀（如 `base_model.model`），直接加载会导致层名不匹配。推理前必须运行修复脚本以移除多余前缀：
```bash
python scripts/fine_tuning/fix_configs.py
```

#### 3.4 模型推理评估 (Evaluation)

使用微调后的模型进行生成测试，建议通过 `--step` 参数对比不同训练步数的生成质量：

*   **测试 UNet 微调模型效果**：
    推理结果保存于 `output/eval_results/`。
    ```bash
    python scripts/fine_tuning/eval.py --step 25000 --prompt "Coastal cliffside cottage, golden hour, soft waves, painterly style, wide lens, serene mood."
    ```
*   **测试联合微调模型效果**：
    推理结果保存于 `output/eval_results_v2/`。
    ```bash
    python scripts/fine_tuning/eval2.py --step 25000 --prompt "Coastal cliffside cottage, golden hour, soft waves, painterly style, wide lens, serene mood."
    ```

### 阶段4: 部署实时生成服务 (Imaginary AI)

本项目提供了一个支持“边画边看”的实时预览接口，用户可以看到扩散模型从噪声中逐渐合成图像的过程。前后端采用了 FastAPI (WebSocket) + React (Vite) 架构。

**最新特性：**
- **多图生成与画廊轮播：** 支持一次请求生成多张图像，并在前端通过 Carousel 优雅展示。
- **IndexedDB 历史持久化：** 使用 `localforage` 彻底解决 localStorage 5MB 限制，支持海量历史图片（Base64）和参数的本地保存、单条删除与一键清空。
- **UI 体验升级：** 类似主流 AI 对话产品的精美布局，包含启动主页、高级参数调整（步数、引导词权重、生成数量）及 Mock Demo 模式。

#### 1. 启动后端服务

```bash
# 激活环境
source .venv/bin/activate

# 启动 FastAPI WebSocket 服务
python scripts/backend/app.py
```

服务将运行在 `http://0.0.0.0:8000`

#### 2. 启动前端演示

```bash
# 进入前端目录
cd scripts/frontend

# 安装前端依赖 (仅首次需要)
npm install

# 启动 Vite 预览
npm run dev
```

打开浏览器访问 `http://localhost:5173`。

1. 首先你会看到 **Imaginary AI** 的炫酷主页。
2. 点击 "Start Creating" 进入创作工作区。
3. 点击输入框右侧的设置按钮，可调节 `Inference Steps`, `Guidance Scale` 以及 `Number of Images`。
4. 输入提示词（Prompt），点击右下角的闪亮生成按钮。你将实时看到进度条跳动，以及图像每隔几步去噪后的模糊预览逐渐清晰。
5. 生成完毕后，如果选择了多张图片，可以在画廊中左右翻页。左侧边栏会实时保存你的生成历史，点击历史记录可瞬间切换回看之前的作品，且数据永久保存在浏览器数据库中。

### 阶段5: 在 HPC 集群上运行 (SLURM) 及 SSH 隧道直连

若在学校高性能计算集群上运行，请执行以下命令：

#### 1. 训练与评估

```bash
# 1. 登录集群并进入项目根目录
cd Text-to-lmage-Generation-Model-Based-on-Fine-Tuned-Stable-Diffusion-v1.5

# 2. 提交训练任务
sbatch submit_train.sh

# 3. 提交评估任务
sbatch submit_loraeval.sh

# 监控与管理
squeue -u $USER                # 查看任务状态
scancel <JOBID>                # 取消运行的程序
tail -f logs/train_<JOBID>.out # 查看运行进度
```

#### 2. 使用 HPC 算力进行前端实时生成 (SSH Tunnel)

如果你觉得本地 CPU 生成过慢，可以申请 HPC 的 GPU 节点，并通过本地前端直接连接：

1. **在 HPC 上启动后端：**
   ```bash
   sbatch submit_backend.sh
   ```
   查看 `logs/backend_*.out` 日志文件，它会告诉你分配的计算节点以及你需要复制的 SSH 隧道命令。
2. 当在HPC上启用后端服务后，可在本地建立SSH隧道
   ```
   ssh -L 8000:hpcgpu107:8000 username@hpcsrv001.xxx
   ```

