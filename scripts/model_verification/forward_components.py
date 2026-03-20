from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os

# use cuda or tps if available
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)

# Visit each component in the pipeline
text_encoder = pipe.text_encoder
tokenizer = pipe.tokenizer
unet = pipe.unet
vae = pipe.vae
scheduler = pipe.scheduler

scheduler.set_timesteps(20) # set inference steps

# Manually execute the forward propagation process
prompt = "a beautiful landscape"
text_inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
print("text_inputs: ", text_inputs)
exit()

with torch.no_grad():
    text_embeddings = text_encoder(text_inputs.input_ids)[0]
    print("text_embeddings: ", text_embeddings)

#     # Initialize random noise
#     latents = torch.randn((1, 4, 64, 64), device=device, dtype=text_embeddings.dtype)

#     # Iterate through the scheduler timesteps and perform the forward pass
#     for t in scheduler.timesteps:
#         latent_model_input = scheduler.scale_model_input(latents, t)
#         noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
#         latents = scheduler.step(noise_pred, t, latents).prev_sample

#     image = vae.decode(latents / vae.config.scaling_factor).sample

# # Decode to image
# image = (image / 2 + 0.5).clamp(0, 1)
# image = image.cpu().permute(0, 2, 3, 1).numpy()
# image = (image[0] * 255).astype("uint8")
# pil_image = Image.fromarray(image)

# os.makedirs("output", exist_ok=True)
# pil_image.save("output/forward_test_output.png")
# print("✅ Image saved successfully!")