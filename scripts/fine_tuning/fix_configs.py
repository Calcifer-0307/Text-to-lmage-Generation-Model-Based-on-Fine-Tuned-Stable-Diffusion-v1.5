import json
import os
import glob

def fix_json(file_path):
    with open(file_path, 'r') as f:
        config = json.load(f)
    
    if "target_modules" in config:
        new_targets = []
        for target in config["target_modules"]:
            # 💡 核心逻辑：去掉 base_model.model. 前缀
            # 如果名字里包含 .unet. 也要去掉
            clean_target = target.replace("base_model.model.unet.", "").replace("base_model.model.", "")
            new_targets.append(clean_target)
        
        config["target_modules"] = new_targets
        
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ 已修复: {file_path}")

# 遍历所有 checkpoint 文件夹
config_files = glob.glob("scripts/fine_tuning/checkpoint/checkpoint-*/adapter_config.json")
for cfg in config_files:
    fix_json(cfg)