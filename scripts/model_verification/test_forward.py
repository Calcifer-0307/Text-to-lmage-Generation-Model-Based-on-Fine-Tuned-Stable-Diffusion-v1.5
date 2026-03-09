import time
import torch
import argparse
import os
from diffusers import StableDiffusionPipeline

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate images using Stable Diffusion v1.5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python test_forward.py --prompt "A photo of a cat"
        python test_forward.py --prompt "A beautiful landscape" --num_inference_steps 50 --guidance_scale 10.0
        python test_forward.py --prompt "A dog playing" --num_images_per_prompt 2 --output_dir ./my_output
                """
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default="A photo of a cat.",
        help="Text prompt for image generation (default: 'A photo of a cat.')"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of inference steps (default: 20)"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for classifier-free guidance (default: 7.5)"
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate per prompt (default: 1)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory to save generated images (default: 'output')"
    )
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # use cuda or mps if available
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the pre-trained Stable Diffusion model
    model_id = "runwayml/stable-diffusion-v1-5"
    print(f"Loading model: {model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    pipe = pipe.to(device)

    # Generate images with the provided prompt
    print(f"Prompt: {args.prompt}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Number of images: {args.num_images_per_prompt}")
    print("Generating images...")
    
    with torch.no_grad():
        images = pipe(
            args.prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=args.num_images_per_prompt
        ).images

    # Save the generated images
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    for i, image in enumerate(images):
        filename = f"test_output_{timestamp}"
        if args.num_images_per_prompt > 1:
            filename += f"_{i+1}"
        filepath = os.path.join(args.output_dir, f"{filename}.png")
        image.save(filepath)
        print(f"Image saved: {filepath}")

if __name__ == "__main__":
    main()
