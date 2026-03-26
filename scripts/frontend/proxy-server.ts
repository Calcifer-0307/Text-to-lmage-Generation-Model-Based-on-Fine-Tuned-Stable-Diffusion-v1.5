import express from 'express';
import cors from 'cors';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

// Node.js ES Modules 中没有 __dirname，需要手动获取
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json());

let tunnelProcess: ReturnType<typeof spawn> | null = null;

// Replace this with your actual ssh key path or specify password auth strategy
// If you use password, you might need sshpass or manual entry.
// For automated node.js SSH, using a key is highly recommended.
// Here we assume standard ssh command can execute (maybe you have configured ssh keys)
// Alternatively, if you need password, you can use a library like 'node-ssh'.

// Since you mentioned password L2PKo54ow, we will use a basic python script or powershell script to handle the connection if sshpass is not available on Windows.
// A simpler way on Windows is to use the `ssh` command if the user has already accepted the host key, but password input is tricky.
// We'll write a simple python script to establish the tunnel using `paramiko` or just execute a shell command and hope the user has set up ssh keys or an agent.

app.post('/api/tunnel/start', (req, res) => {
  if (tunnelProcess) {
    return res.json({ status: 'already_running' });
  }

  // 前端如果传了节点名字，就用传的，否则用 localhost
  const nodeName = req.body.nodeName || 'localhost';
  console.log(`Starting SSH tunnel to HPC (Node: ${nodeName})...`);
  
  const pyScriptPath = path.join(__dirname, '../../scripts/tunnel.py');
  
  // 在 Windows 环境下使用虚拟环境的 python
  const venvPython = path.join(__dirname, '../../.venv/Scripts/python.exe');
  
  tunnelProcess = spawn(venvPython, [pyScriptPath, nodeName], {
    stdio: 'inherit' // 这会让密码输入提示直接显示在当前 proxy server 的终端里
  });

  tunnelProcess.on('close', (code) => {
    console.log(`Tunnel process exited with code ${code}`);
    tunnelProcess = null;
  });

  // Give it a second to start
  setTimeout(() => {
    res.json({ status: 'started' });
  }, 2000);
});

app.post('/api/tunnel/stop', (req, res) => {
  if (tunnelProcess) {
    console.log("Stopping SSH tunnel...");
    // On Windows, killing the child process might not kill the ssh process spawned by python.
    // So we can send a request to a special endpoint or just kill python processes running tunnel.py
    spawn('taskkill', ['/F', '/IM', 'ssh.exe']); // Force kill ssh.exe (might be aggressive, but works)
    tunnelProcess.kill();
    tunnelProcess = null;
    res.json({ status: 'stopped' });
  } else {
    res.json({ status: 'not_running' });
  }
});

const PORT = 3001;
app.listen(PORT, () => {
  console.log(`Local Proxy Server running on port ${PORT}`);
  console.log(`Ready to manage SSH tunnels for HPC connection.`);
});