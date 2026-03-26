import os
import subprocess

print("打包项目文件 (排除 .venv, __pycache__, huggingface_cache)...")
tar_cmd = "tar --exclude='.venv' --exclude='.git' --exclude='data/huggingface_cache' --exclude='__pycache__' -czvf hpc_deploy.tar.gz ."
subprocess.run(tar_cmd, shell=True)

print("\n打包完成！准备在远程服务器创建目录...")
print("⚠️ 提示：稍后如果提示输入密码，请输入: L2PKo54ow")
ssh_mkdir_cmd = "ssh ruilingyuan@luhpc.ln.edu.hk \"mkdir -p ~/prj_ruilingyuan/Text-to-lmage-Generation-Model-Based-on-Fine-Tuned-Stable-Diffusion-v1.5-main\""
subprocess.run(ssh_mkdir_cmd, shell=True)

print("\n正在上传到 HPC...")
print("⚠️ 提示：如果提示输入密码，请输入: L2PKo54ow")
scp_cmd = "scp hpc_deploy.tar.gz ruilingyuan@luhpc.ln.edu.hk:~/prj_ruilingyuan/Text-to-lmage-Generation-Model-Based-on-Fine-Tuned-Stable-Diffusion-v1.5-main/"
subprocess.run(scp_cmd, shell=True)

print("\n上传完成！正在解压并清理临时文件...")
print("⚠️ 提示：如果提示输入密码，请输入: L2PKo54ow")
ssh_extract_cmd = "ssh ruilingyuan@luhpc.ln.edu.hk \"cd ~/prj_ruilingyuan/Text-to-lmage-Generation-Model-Based-on-Fine-Tuned-Stable-Diffusion-v1.5-main && tar -xzvf hpc_deploy.tar.gz && rm hpc_deploy.tar.gz\""
subprocess.run(ssh_extract_cmd, shell=True)

# 清理本地压缩包
if os.path.exists("hpc_deploy.tar.gz"):
    os.remove("hpc_deploy.tar.gz")

print("\n✅ 部署脚本执行完毕！")