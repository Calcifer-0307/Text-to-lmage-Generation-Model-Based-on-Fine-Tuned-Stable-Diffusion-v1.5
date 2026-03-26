import os
import torch
import torch.nn.functional as F
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
    output_dir = "scripts/fine_tuning/checkpoint"
    os.makedirs(output_dir, exist_ok=True)
    
    # 关键超参数
    batch_size = 4           # 根据显存调整
    num_train_epochs = 50    # 全量数据通常跑 50-100 epoch
    learning_rate = 1e-4
    max_grad_norm = 1.0      # 梯度裁剪，防止崩坏
    
    # --- 2. 加载模型 ---
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
    vae = StableDiffusionPipeline.from_pretrained(model_id).vae.to(device)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # --- 3. 注入 LoRA (仅 UNet) ---
    lora_config = LoraConfig(
        r=16, lora_alpha=16,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.05, bias="none"
    )
    unet = get_peft_model(unet, lora_config)
    unet.train()

    # --- 4. 准备全量数据 ---
    raw_dataset = load_from_disk("data/lunara_final_dataset_rev")
    train_dataloader = DataLoaderFactory.create_train_dataloader(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        image_size=512
    )

    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)
    
    # --- 5. 正式训练循环 ---
    global_step = 0
    print(f"🚀 开始正式微调，总样本数: {len(raw_dataset)}")
    
    for epoch in range(num_train_epochs):
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        for batch in progress_bar:
            # 数据准备
            pixel_values = batch["pixel_values"].to(device) # [B, 3, 512, 512]
            input_ids = batch["input_ids"].to(device)       # [B, 77]

            # 编码
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
                encoder_hidden_states = text_encoder(input_ids)[0]

            # 扩散
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 预测与 Loss
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
            # 优化
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm) # 梯度裁剪
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            progress_bar.set_description(f"Loss: {loss.item():.4f}")

            # 每 500 步保存一次权重
            if global_step % 500 == 0:
                save_path = os.path.join(output_dir, f"lora_step_{global_step}.safetensors")
                # 使用 peft 的保存方法，只保存几十MB的 LoRA 权重
                unet.save_pretrained(os.path.join(output_dir, f"checkpoint-{global_step}"))
                print(f"💾 已保存 Checkpoint 到 {save_path}")

    # --- 6. 最终保存 ---
    unet.save_pretrained(os.path.join(output_dir, "lora_final"))
    print("✅ 训练完成！")

if __name__ == "__main__":
    train()
    