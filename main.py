from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool # 💡 导入这个工具
from scripts.fine_tuning.inference_engine import LunaraInferenceEngine
import asyncio
import io
import base64

app = FastAPI()

engine = LunaraInferenceEngine(
    model_id="runwayml/stable-diffusion-v1-5",
    lora_path="scripts/fine_tuning/checkpoint/checkpoint-25000"
)

@app.websocket("/ws/generate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # 💡 获取当前的主线程事件循环
    loop = asyncio.get_event_loop()
    
    try:
        data = await websocket.receive_json()
        prompt = data.get("prompt", "")
        seed = data.get("seed", 42)

        async def send_progress(percent, preview_b64):
            # 这个函数会在主线程执行，不会被卡住
            payload = {"type": "progress", "value": percent}
            if preview_b64:
                payload["preview"] = f"data:image/jpeg;base64,{preview_b64}"
            try:
                await websocket.send_json(payload)
            except: pass

        # 💡 核心改动：在线程池中运行同步的 generate 函数
        final_image = await run_in_threadpool(
            engine.generate, 
            prompt=prompt, 
            seed=seed, 
            progress_callback=send_progress,
            loop=loop # 传入 loop 供回调使用
        )

        # 发送最终图像
        buffered = io.BytesIO()
        final_image.save(buffered, format="PNG")
        final_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        await websocket.send_json({
            "type": "final",
            "image": f"data:image/png;base64,{final_b64}"
        })

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try: await websocket.close()
        except: pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)