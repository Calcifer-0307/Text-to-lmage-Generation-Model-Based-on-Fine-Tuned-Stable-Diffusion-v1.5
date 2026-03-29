import base64
import io
import time
import asyncio
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from PIL import Image

# Import the Inference Engine developed by the team
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from scripts.fine_tuning.inference_engine import LunaraInferenceEngine

app = FastAPI(title="Text-to-Image API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the engine
engine = None

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    num_images_per_prompt: int = 1
    seed: int = -1
    mock: bool = False # Enable mock mode for frontend testing

@app.on_event("startup")
async def startup_event():
    global engine
    print("Initializing Lunara Inference Engine with LoRA...")
    try:
        engine = LunaraInferenceEngine(
            model_id="runwayml/stable-diffusion-v1-5",
            lora_path="scripts/fine_tuning/checkpoint/checkpoint-25000"
        )
        print("Engine initialized successfully!")
    except Exception as e:
        print(f"Error loading model/engine: {e}")

@app.get("/health")
def health_check():
    return {"status": "ok", "engine_loaded": engine is not None}

@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_event_loop()
    
    try:
        data = await websocket.receive_json()
        prompt = data.get("prompt", "")
        seed = data.get("seed", 42)
        num_steps = data.get("num_inference_steps", 25)
        mock = data.get("mock", False)

        async def send_progress(percent, preview_b64):
            payload = {"type": "progress", "value": percent}
            if preview_b64:
                payload["preview"] = f"data:image/jpeg;base64,{preview_b64}"
            try:
                await websocket.send_json(payload)
            except: pass

        if mock:
            print(f"Mock generation via WS for prompt: {prompt}")
            for i in range(1, 6):
                await asyncio.sleep(0.5)
                await send_progress(i * 20, None)
                
            from PIL import ImageDraw
            img = Image.new('RGB', (512, 512), color=(73, 109, 137))
            d = ImageDraw.Draw(img)
            d.text((200, 250), "MOCK MODE", fill=(255, 255, 255))
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            final_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            await websocket.send_json({
                "type": "final",
                "image": f"data:image/png;base64,{final_b64}"
            })
            return

        if engine is None:
            await websocket.send_json({"type": "error", "message": "Model not loaded"})
            return

        print(f"Generating image via WS for prompt: {prompt}")
        
        # We currently only generate and return one image back to the UI per request,
        # but the engine can be updated to handle `num_images_per_prompt` if we modify the return type.
        # For now, we will generate the requested number of images and just return the first one
        # to match the frontend's current single-image view.
        num_images = data.get("num_images_per_prompt", 1)
        
        final_images = await run_in_threadpool(
            engine.generate, 
            prompt=prompt, 
            num_steps=num_steps,
            seed=seed, 
            num_images=num_images,
            progress_callback=send_progress,
            loop=loop
        )

        # 修复：确保 final_images 始终是一个列表，即使底层引擎返回了单张图片对象
        if not isinstance(final_images, list):
            final_images = [final_images]

        final_b64_images = []
        for img in final_images:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            final_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            final_b64_images.append(f"data:image/png;base64,{final_b64}")
        
        await websocket.send_json({
            "type": "final",
            "images": final_b64_images
        })

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Generation error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except: pass
    finally:
        try: await websocket.close()
        except: pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
