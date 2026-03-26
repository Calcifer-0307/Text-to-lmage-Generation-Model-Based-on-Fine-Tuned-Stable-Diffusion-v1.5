import subprocess
import time
import os
import signal
import sys
import threading
import paramiko
import subprocess
import time

def main():
    hostname = "luhpc.ln.edu.hk"
    port = 22
    username = "ruilingyuan"
    password = "L2PKo54ow"
    
    node_name = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        print(f"Connecting to {hostname}...")
        client.connect(hostname, port, username, password)
        print("Connected successfully!")
        
        # 尝试启动远程后端
        print("Starting remote Uvicorn server...")
        remote_command = (
            "cd ~/prj_ruilingyuan/Text-to-lmage-Generation-Model-Based-on-Fine-Tuned-Stable-Diffusion-v1.5-main && "
            "source venv_hpc/bin/activate && "
            "nohup uvicorn scripts.backend.app:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &"
        )
        if node_name != "localhost":
             remote_command = f"ssh -o StrictHostKeyChecking=no {node_name} \"{remote_command}\""
             
        client.exec_command(remote_command)
        print("Remote server start command sent.")
        
        # 保持隧道开启
        # forward_tunnel(8001, node_name, 8000, client.get_transport())
        
        # 为了实现免密隧道，我们直接使用 paramiko 的 invoke_shell 并让主线程等待
        # 真正的数据转发使用 Windows 的自带 ssh 解决免密问题很难，
        # 我们这里折中一下：使用 Paramiko 启动远程后端后，
        # 在前端提示用户：后端已启动，请在新终端执行 SSH 隧道命令
        print("\n" + "="*60)
        print("✅ 后端服务已在 HPC 启动！")
        print("👉 请在你的电脑上打开一个新的命令行终端 (CMD / PowerShell)")
        print(f"👉 复制并运行以下命令来建立隧道：")
        print(f"   ssh -L 8001:{node_name}:8000 {username}@{hostname}")
        print("👉 当提示输入密码时，请输入: " + password)
        print("="*60 + "\n")
        
        # 保持脚本运行，不退出，直到用户手动停止
        while True:
            time.sleep(1)
        
    except KeyboardInterrupt:
        print("Tunnel closed.")
    except Exception as e:
        print(f"Failed: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    main()