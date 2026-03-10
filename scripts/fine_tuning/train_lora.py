import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer

# 1. 加载你项目中已经在用的基础模型
model_id = "runwayml/stable-diffusion-v1-5"
tokenizer = CLIPTokenizer.from_pretrained(model_id)
text_encoder = CLIPTextModel.from_pretrained(model_id)
unet = UNet2DConditionModel.from_pretrained(model_id)

# 2. 配置 LoRA (这是微调的核心参数)
# r=8 是矩阵秩，越大学得越像但文件越大；target_modules 指定微调哪些层
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["to_k", "to_q", "to_v", "to_out.0"], 
    lora_dropout=0.1,
    bias="none",
)

# 3. 将 LoRA 注入到 UNet 模型中
unet = get_peft_model(unet, lora_config)
unet.print_trainable_parameters() # 你会看到只有很小比例的参数在参加训练

print("LoRA 训练环境准备就绪，可以开始读取你项目 data 目录下的图片进行训练。")