import os
import torch
import torch.nn.functional as F
import itertools
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig, get_peft_model
from transformers import CLIPTokenizer, CLIPTextModel
from datasets import load_from_disk
from dataset_adapter import DataLoaderFactory # 复用你的工厂类

def train():
    # --- 1. 参数配置 ---
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model_id = "runwayml/stable-diffusion-v1-5"
    output_dir = "scripts/fine_tuning/checkpoint_v2" # 建议换个新目录区分
    os.makedirs(output_dir, exist_ok=True)
    
    # 关键超参数
    batch_size = 4           
    num_train_epochs = 50    
    unet_learning_rate = 1e-4
    text_encoder_learning_rate = 5e-5 # Text Encoder 学习率通常设为 UNet 的一半或更低
    max_grad_norm = 1.0      
    
    # --- 2. 加载模型 ---
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
    vae = StableDiffusionPipeline.from_pretrained(model_id).vae.to(device)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    # 冻结基础模型的所有参数
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # --- 3. 注入 LoRA (UNet + Text Encoder) ---
    print("正在为 UNet 和 Text Encoder 注入 LoRA 模块...")
    
    # 3.1 UNet LoRA 配置
    unet_lora_config = LoraConfig(
        r=16, lora_alpha=16,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.05, bias="none"
    )
    unet = get_peft_model(unet, unet_lora_config)
    unet.train()

    # 3.2 Text Encoder (CLIP) LoRA 配置
    # 注意：CLIP 的 attention 投影层名称与 UNet 不同
    text_lora_config = LoraConfig(
        r=16, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"], 
        lora_dropout=0.05, bias="none"
    )
    text_encoder = get_peft_model(text_encoder, text_lora_config)
    text_encoder.train()

    # --- 4. 准备全量数据 ---
    raw_dataset = load_from_disk("data/lunara_final_dataset_rev")
    train_dataloader = DataLoaderFactory.create_train_dataloader(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        image_size=512
    )

    # --- 5. 配置优化器 (合并参数) ---
    # 为 UNet 和 Text Encoder 设置不同的学习率
    optimizer_grouped_parameters = [
        {"params": unet.parameters(), "lr": unet_learning_rate},
        {"params": text_encoder.parameters(), "lr": text_encoder_learning_rate},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    
    # 用于梯度裁剪的参数迭代器
    params_to_clip = itertools.chain(unet.parameters(), text_encoder.parameters())

    # --- 6. 正式训练循环 ---
    global_step = 0
    print(f"🚀 开始双路 LoRA 微调，总样本数: {len(raw_dataset)}")
    
    for epoch in range(num_train_epochs):
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        for batch in progress_bar:
            # 数据准备
            pixel_values = batch["pixel_values"].to(device) # [B, 3, 512, 512]
            input_ids = batch["input_ids"].to(device)       # [B, 77]

            # VAE 编码 (无需梯度)
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
            
            # Text Encoder 编码 (此时需要计算梯度，所以不用 no_grad)
            encoder_hidden_states = text_encoder(input_ids)[0]

            # 扩散加噪
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 预测与 Loss
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
            # 优化
            loss.backward()
            # 对 UNet 和 Text Encoder 一起做梯度裁剪
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_grad_norm) 
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            progress_bar.set_description(f"Loss: {loss.item():.4f}")

            # 每 500 步保存一次权重
            if global_step % 500 == 0:
                step_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                os.makedirs(step_dir, exist_ok=True)
                
                # 分别保存 UNet 和 Text Encoder 的 LoRA 权重
                unet.save_pretrained(os.path.join(step_dir, "unet"))
                text_encoder.save_pretrained(os.path.join(step_dir, "text_encoder"))
                print(f"💾 已保存 Checkpoint 到 {step_dir}")

    # --- 7. 最终保存 ---
    final_dir = os.path.join(output_dir, "lora_final")
    os.makedirs(final_dir, exist_ok=True)
    unet.save_pretrained(os.path.join(final_dir, "unet"))
    text_encoder.save_pretrained(os.path.join(final_dir, "text_encoder"))
    print("✅ 训练完成！")

if __name__ == "__main__":
    train()