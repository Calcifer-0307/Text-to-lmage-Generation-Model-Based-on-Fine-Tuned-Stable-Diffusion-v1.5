import torch
import os
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from PIL import Image

def generate_showcase_images():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    output_dir = "scripts/frontend/public/showcase"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the 3 specific prompts provided by the user
    prompts = [
        "A young girl, dark hair, golden light, coastal backdrop, painterly style, 8k resolution.",
        "Alpine village at twilight, snow-dusted rooftops, glowing amber windows, soft-focus telephoto lens, tranquil winter mood.",
        "Sunset over bamboo forest, golden light filtering through leaves, serene path, 50mm lens."
    ]
    
    # Shared parameters for fair comparison
    seed = 42
    num_steps = 30
    guidance_scale = 7.5
    
    # 1. Load Base Model (No LoRA)
    print("Loading Base Model (SD 1.5)...")
    base_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float16 if "cpu" not in device else torch.float32,
        safety_checker=None # Disable to avoid false positives during bulk generation
    ).to(device)
    
    # Generate Base Images
    for i, prompt in enumerate(prompts):
        print(f"\nGenerating Base Image {i+1}/3: {prompt[:50]}...")
        generator = torch.Generator(device=device).manual_seed(seed)
        image = base_pipe(
            prompt, 
            num_inference_steps=num_steps, 
            guidance_scale=guidance_scale, 
            generator=generator
        ).images[0]
        
        save_path = os.path.join(output_dir, f"base_{i+1}.png")
        image.save(save_path)
        print(f"Saved: {save_path}")
        
    # Free memory before loading LoRA
    del base_pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # 2. Load Model with LoRA
    print("\nLoading Model with LoRA...")
    lora_path = "scripts/fine_tuning/checkpoint/checkpoint-25000"
    if not os.path.exists(lora_path):
        print(f"WARNING: LoRA checkpoint not found at {lora_path}!")
        print("Please run this script on the HPC or environment where the checkpoint exists.")
        return
        
    lora_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float16 if "cpu" not in device else torch.float32,
        safety_checker=None
    ).to(device)
    
    lora_pipe.unet = PeftModel.from_pretrained(lora_pipe.unet, lora_path)
    
    # Generate LoRA Images
    for i, prompt in enumerate(prompts):
        print(f"\nGenerating LoRA Image {i+1}/3: {prompt[:50]}...")
        generator = torch.Generator(device=device).manual_seed(seed)
        image = lora_pipe(
            prompt, 
            num_inference_steps=num_steps, 
            guidance_scale=guidance_scale, 
            generator=generator
        ).images[0]
        
        save_path = os.path.join(output_dir, f"lora_{i+1}.png")
        image.save(save_path)
        print(f"Saved: {save_path}")
        
    print("\nAll showcase images generated successfully!")

if __name__ == "__main__":
    generate_showcase_images()