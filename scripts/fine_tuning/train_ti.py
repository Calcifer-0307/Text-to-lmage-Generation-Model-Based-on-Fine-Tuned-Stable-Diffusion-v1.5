import os
import argparse
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from accelerate import Accelerator
from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

# 1. 定义数据集类
class TextualInversionDataset(Dataset):
    def __init__(self, data_root, tokenizer, placeholder_token, size=512, interpolation=Image.BILINEAR, flip_p=0.5):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.placeholder_token = placeholder_token
        self.size = size
        self.flip_p = flip_p
        
        self.image_paths = [os.path.join(data_root, file) for file in os.listdir(data_root) if file.endswith(('.png', '.jpg', '.jpeg'))]
        self.num_images = len(self.image_paths)
        
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=interpolation),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(p=flip_p),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        # 训练模板：让模型理解这个新词可以出现在不同语境中
        self.templates = ["a photo of a {}", "a rendering of a {}", "a cropped photo of the {}", "the photo of a {}"]

    def __len__(self):
        return self.num_images

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i]).convert("RGB")
        example["pixel_values"] = self.transform(image)
        
        import random
        prompt = random.choice(self.templates).format(self.placeholder_token)
        example["input_ids"] = self.tokenizer(
            prompt, padding="max_length", truncation=True, max_length=self.tokenizer.model_max_length, return_tensors="pt"
        ).input_ids[0]
        return example

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--placeholder_token", type=str, required=True, help="例如 <my-cat>")
    parser.add_argument("--initializer_token", type=str, required=True, help="例如 cat")
    parser.add_argument("--output_dir", type=str, default="./output/ti_model")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision="fp16")

    # 2. 加载模型组件
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")

    # 3. 添加 Placeholder Token
    num_added_tokens = tokenizer.add_tokens(args.placeholder_token)
    placeholder_token_id = tokenizer.convert_tokens_to_ids(args.placeholder_token)
    
    # 调整 Embedding 层大小并用 Initializer 初始化
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    initial_id = tokenizer.encode(args.initializer_token, add_special_tokens=False)[0]
    token_embeds[placeholder_token_id] = token_embeds[initial_id].clone()

    # 4. 冻结参数（只训练 Embedding 层）
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.patch_embedding.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    optimizer = torch.optim.AdamW(text_encoder.get_input_embeddings().parameters(), lr=args.lr)

    # 5. 数据准备
    dataset = TextualInversionDataset(args.data_dir, tokenizer, args.placeholder_token)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    text_encoder, optimizer, train_dataloader = accelerator.prepare(text_encoder, optimizer, train_dataloader)
    vae.to(accelerator.device)
    unet.to(accelerator.device)

    # 6. 训练循环
    global_step = 0
    progress_bar = tqdm(range(args.steps), desc="Training Steps")
    
    while global_step < args.steps:
        text_encoder.train()
        for batch in train_dataloader:
            with accelerator.accumulate(text_encoder):
                # 将图像编码为 Latents
                latents = vae.encode(batch["pixel_values"].to(dtype=torch.float16)).latent_dist.sample().detach()
                latents = latents * 0.18215

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)

                # 【核心】：梯度置零，只更新我们新加的 Token
                grads = text_encoder.get_input_embeddings().weight.grad
                index_grads_to_zero = torch.arange(len(tokenizer)) != placeholder_token_id
                grads.data[index_grads_to_zero, :] = 0

                optimizer.step()
                optimizer.zero_grad()
            
            global_step += 1
            progress_bar.update(1)
            if global_step >= args.steps: break

    # 7. 保存结果
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
        learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}
        torch.save(learned_embeds_dict, os.path.join(args.output_dir, "learned_embeds.bin"))
        print(f"训练完成！文件已保存至 {args.output_dir}/learned_embeds.bin")

if __name__ == "__main__":
    main()