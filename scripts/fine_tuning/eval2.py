import os
import gc
import torch
import argparse
from peft import PeftModel
from diffusers import StableDiffusionPipeline

def evaluate():
    parser = argparse.ArgumentParser()
    # 默认步数可以根据你的训练情况调整
    parser.add_argument("--step", type=int, default=25000, help="Checkpoint step. Use 0 for base model. Use step number for LoRA model.")
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="Coastal cliffside cottage, golden hour, soft waves", 
        help="Text prompt"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    args = parser.parse_args()

    model_id = "runwayml/stable-diffusion-v1-5"
    # 建议将输出文件夹也区分开，避免和单路 LoRA 的结果混淆
    output_dir = "output/eval_results_v2" 
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    print(f"Load base model: {model_id}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)

    try:
        if args.step == 0:
            print("🚀 Step 0: Use base model (No LoRA).")
        else:
            # 读取我们在 train_lora2.py 中使用的新路径 checkpoint_v2
            ckpt_path = f"scripts/fine_tuning/checkpoint_v2/checkpoint-{args.step}"
            print(f"🔗 Injecting Dual LoRA from: {ckpt_path}")
            
            # 分别构建 UNet 和 Text Encoder 的路径
            unet_path = os.path.join(ckpt_path, "unet")
            text_encoder_path = os.path.join(ckpt_path, "text_encoder")
            
            if not os.path.exists(unet_path) or not os.path.exists(text_encoder_path):
                raise FileNotFoundError(f"Checkpoint paths not found in: {ckpt_path}")
            
            # 核心修改点：分别注入两个模型的 LoRA 权重
            print("   -> Injecting UNet LoRA...")
            pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_path)
            
            print("   -> Injecting Text Encoder LoRA...")
            pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, text_encoder_path)

        # Set random seed for reproducibility
        generator = torch.Generator(device=device).manual_seed(args.seed)
        
        with torch.inference_mode():
            image = pipe(
                prompt=args.prompt,
                num_inference_steps=25,
                guidance_scale=7.5,
                generator=generator
            ).images[0]
        
        # Save the generated image
        file_name = f"comparison_v2_step_{args.step}.png"
        save_path = os.path.join(output_dir, file_name)
        image.save(save_path)
        print(f"✅ Successfully saved: {save_path}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    evaluate()