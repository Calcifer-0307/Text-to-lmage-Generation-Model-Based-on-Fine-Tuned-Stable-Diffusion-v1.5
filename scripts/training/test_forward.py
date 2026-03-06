import torch
from diffusers import StableDiffusionPipeline

# use cuda or tps if available
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Load the pre-trained Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe = pipe.to(device)

# Test the forward pass with a simple prompt
prompt = "a photo of a cat"
with torch.no_grad():
    image = pipe(
        prompt,
        num_inference_steps=20,
        guidance_scale=7.5
    ).images[0]

image.save("output/test_output.png")
print("Test image saved as test_output.png")
