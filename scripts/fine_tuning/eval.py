import os
import gc
import torch
import argparse
from peft import PeftModel
from diffusers import StableDiffusionPipeline

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=25000, help="Checkpoint step. Use 0 for base model. Use 25000 for LoRA model.")
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="Coastal cliffside cottage, golden hour, soft waves", 
        help="Text prompt"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    args = parser.parse_args()

    model_id = "runwayml/stable-diffusion-v1-5"
    output_dir = "output/eval_results"
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
            print("🚀 Step 0: Use base model (No LoRA)。")
        else:
            ckpt_path = f"scripts/fine_tuning/checkpoint/checkpoint-{args.step}"
            print(f"🔗 Injecting LoRA: {ckpt_path}")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f" Checkpoint path not found: {ckpt_path}")
            pipe.unet = PeftModel.from_pretrained(pipe.unet, ckpt_path)

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
        file_name = f"comparison_step_{args.step}.png"
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
        torch.cuda.empty_cache()

if __name__ == "__main__":
    evaluate()