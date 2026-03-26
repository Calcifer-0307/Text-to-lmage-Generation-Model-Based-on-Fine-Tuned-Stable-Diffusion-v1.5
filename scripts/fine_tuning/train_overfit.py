import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig, get_peft_model
from transformers import CLIPTokenizer, CLIPTextModel

from datasets import load_from_disk

from dataset_adapter import StableDiffusionDataset

def train_overfit():
    # --- 1. 配置与设备 ---
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model_id = "runwayml/stable-diffusion-v1-5"
    num_epochs = 500
    learning_rate = 1e-4
    
    # --- 2. 加载模型组件 ---
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
    vae = StableDiffusionPipeline.from_pretrained(model_id).vae.to(device)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    # 固定 VAE 和 Text Encoder，不训练它们
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # --- 3. 注入 LoRA ---
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.05,
        bias="none"
    )
    unet = get_peft_model(unet, lora_config)
    unet.train()
    print(f"✅ LoRA 注入完成，仅训练 LoRA 参数。")

    # --- 4. 准备单个样本数据 ---
    raw_dataset = load_from_disk("data/lunara_final_dataset_rev")
    # 只取第 0 张图进行过拟合测试
    single_item_dataset = raw_dataset.select(range(1)) 
    dataset = StableDiffusionDataset(
        dataset=single_item_dataset,
        tokenizer=tokenizer,
        image_size=512,
        conditioning_dropout_prob=0.0 # 测试时不使用 dropout
    )
    
    batch = dataset[0]
    pixel_values = batch["pixel_values"].unsqueeze(0).to(device) # [1, 3, 512, 512]
    input_ids = batch["input_ids"].unsqueeze(0).to(device)       # [1, 77]
    prompt = batch["prompt"]

    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)

    # --- 5. 训练循环 ---
    print(f"🚀 开始过拟合测试 (Prompt: {prompt})...")
    progress_bar = tqdm(range(num_epochs))
    
    for epoch in progress_bar:
        # A. 将图像编码到潜空间
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * 0.18215 # SD 标准缩放系数

        # B. 采样随机噪声
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device).long()
        
        # C. 前向加噪 (Forward Diffusion)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # D. 获取文本 Embedding
        encoder_hidden_states = text_encoder(input_ids)[0]

        # E. 模型预测噪声
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # F. 计算 Loss (MSE)
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        
        # G. 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        progress_bar.set_description(f"Loss: {loss.item():.4f}")

    # --- 6. 验证推理 ---
    print("\n⚖️ 训练完成，正在验证生成效果...")
    unet.eval()
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        unet=unet, # 使用我们训练好的带有 LoRA 的 unet
        text_encoder=text_encoder,
        vae=vae,
        torch_dtype=torch.float32 # 验证用 float32 保证稳定
    ).to(device)
    
    with torch.no_grad():
        output_image = pipeline(
            prompt, 
            num_inference_steps=50, # 增加步数让画面更稳
            guidance_scale=1.0      # 关键：降低引导强度
        ).images[0]
        output_image.save("output/debug_overfit_result.png")

    # 将训练图像pixel_values也保存为png格式的图片
    with torch.no_grad():
        # 解码潜空间到图像
        decoded_image = vae.decode(latents / 0.18215).sample
        decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)
        decoded_image = decoded_image.cpu().permute(0, 2, 3, 1).numpy()
        decoded_image = (decoded_image[0] * 255).astype("uint8")
        pil_decoded_image = Image.fromarray(decoded_image)
        pil_decoded_image.save("output/debug_overfit_input.png")
    
    print("✨ 测试图像已保存为 'output/debug_overfit_result.png'。")
    print("💡 请检查该图片是否与你的原始训练图一致。如果一致，环境与代码即为完美！")

if __name__ == "__main__":
    train_overfit()