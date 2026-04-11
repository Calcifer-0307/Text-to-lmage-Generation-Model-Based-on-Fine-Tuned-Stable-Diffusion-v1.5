import time
from matplotlib.pyplot import step
import torch
import io
import base64
import asyncio # 💡 导入 asyncio
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from PIL import Image

class LunaraInferenceEngine:
    def __init__(self, model_id, lora_path, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
            
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if "cpu" not in self.device else torch.float32,
            low_cpu_mem_usage=True,
            cache_dir="data/huggingface_cache"
        ).to(self.device)
        
        if lora_path:
            self.pipe.unet = PeftModel.from_pretrained(self.pipe.unet, lora_path)
        print("Model Ready on", self.device)

    def _tensor_to_base64(self, latents):
        with torch.no_grad():
            latents = 1 / 0.18215 * latents
            image = self.pipe.vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = (image[0] * 255).astype("uint8")
            pil_img = Image.fromarray(image)
            pil_img.thumbnail((256, 256)) 
            # 将图片保存到output/real_time_preview路径下
            pil_img.save(f"output/real_time_preview/preview_{time.time():f}.jpg")
            buffered = io.BytesIO()
            pil_img.save(buffered, format="JPEG", quality=60)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    # 💡 注意：这里改成了普通函数 def，不再是 async def
    def generate(self, prompt, num_steps=25, seed=42, num_images=1, progress_callback=None, loop=None):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        def latents_callback(pipe, step, timestep, callback_kwargs):
            if progress_callback and loop:
                latents = callback_kwargs["latents"]
                percent = int((step / num_steps) * 100)
                
                preview_base64 = None
                # 每 2 步解码一次预览图
                if step % 2 == 0 or step == num_steps - 1:
                    preview_base64 = self._tensor_to_base64(latents)
                
                # 💡 关键：使用 loop.call_soon_threadsafe 将任务推回主线程发送
                loop.call_soon_threadsafe(
                    lambda: asyncio.ensure_future(progress_callback(percent, preview_base64))
                )
                
            return callback_kwargs

        result = self.pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            num_images_per_prompt=num_images,
            generator=generator,
            callback_on_step_end=latents_callback,
            callback_on_step_end_tensor_inputs=["latents"]
        )
        print(f"DEBUG: Pipeline returning {len(result.images)} images")
        
        # 修改为返回完整的图片数组，以支持前端多图画廊
        return result.images
