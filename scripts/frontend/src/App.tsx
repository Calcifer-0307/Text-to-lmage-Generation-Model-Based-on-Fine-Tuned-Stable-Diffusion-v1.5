import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Loader2, Download, Image as ImageIcon, Sparkles, Settings2, XCircle, MessageSquarePlus, MessageSquare, Menu, PanelLeftClose, PanelLeftOpen, ChevronLeft, ChevronRight, Trash2 } from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import localforage from 'localforage';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// 配置 localforage
localforage.config({
  name: 'ImaginaryAI',
  storeName: 'generations'
});

interface GenerateResponse {
  images: string[]; // 改为数组以支持多图
  prompt: string;
  params: any;
}

interface HistoryItem {
  id: string;
  prompt: string;
  images: string[]; // 改为数组
  timestamp: number;
}

function App() {
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [steps, setSteps] = useState(20);
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [numImages, setNumImages] = useState(1);
  const [seed, setSeed] = useState(-1);
  const [useMock, setUseMock] = useState(false);
  const [useHPC, setUseHPC] = useState(false);
  const [hpcNodeName, setHpcNodeName] = useState('localhost');
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [result, setResult] = useState<GenerateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [tunnelStatus, setTunnelStatus] = useState<'idle' | 'connecting' | 'connected' | 'error'>('idle');

  // 历史记录和布局状态
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const [showHome, setShowHome] = useState(true);

  // 多图轮播状态
  const [currentImageIndex, setCurrentImageIndex] = useState(0);

  // 标记是否已经完成首次从 IndexedDB 的加载
  const isHistoryLoaded = useRef(false);

  // 初始化时读取本地存储的历史记录
  useEffect(() => {
    const loadHistory = async () => {
      try {
        const saved = await localforage.getItem<HistoryItem[]>('generation_history');
        if (saved) {
          setHistory(saved);
        }
      } catch (e) {
        console.error('Failed to load history from IndexedDB', e);
        
        // 回退机制：尝试从 localStorage 读取旧数据并迁移
        const legacySaved = localStorage.getItem('generation_history');
        if (legacySaved) {
          try {
            const parsed = JSON.parse(legacySaved);
            setHistory(parsed);
            await localforage.setItem('generation_history', parsed);
            localStorage.removeItem('generation_history'); // 迁移后清理
          } catch (legacyErr) {
            console.error('Failed to parse legacy history', legacyErr);
          }
        }
      } finally {
        isHistoryLoaded.current = true;
      }
    };
    
    loadHistory();
  }, []);

  // 历史记录更新时保存到本地
  useEffect(() => {
    // 只有在完成首次加载后，才允许将当前组件的 history 状态写入 IndexedDB
    // 否则 React 初始的空数组会把 DB 里的数据清空！
    if (isHistoryLoaded.current) {
      localforage.setItem('generation_history', history).catch(err => {
        console.error('Failed to save history to IndexedDB', err);
      });
    }
  }, [history]);

  // 清除所有历史记录
  const clearHistory = async (e: React.MouseEvent) => {
    // 阻止事件冒泡，防止触发其他可能影响渲染的事件
    e.stopPropagation();
    
    if (window.confirm('Are you sure you want to clear all history? This cannot be undone.')) {
      try {
        await localforage.removeItem('generation_history');
        setHistory([]);
        if (currentId) {
          startNew();
        }
      } catch (e) {
        console.error('Failed to clear history', e);
      }
    }
  };

  // 删除单条历史记录
  const deleteHistoryItem = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    const newHistory = history.filter(item => item.id !== id);
    setHistory(newHistory);
    try {
      await localforage.setItem('generation_history', newHistory);
      if (currentId === id) {
        startNew();
      }
    } catch (err) {
      console.error('Failed to delete history item', err);
    }
  };

  const startNew = () => {
    setCurrentId(null);
    setPrompt('');
    setPreviewImage(null);
    setResult(null);
    setProgress(0);
    setError(null);
    setCurrentImageIndex(0);
  };

  const loadHistory = (item: HistoryItem) => {
    if (loading) return;
    setCurrentId(item.id);
    setPrompt(item.prompt);
    // 兼容旧的历史记录数据结构 (单图)
    const itemImages = item.images || ((item as any).image ? [(item as any).image] : []);
    setResult({ images: itemImages, prompt: item.prompt, params: {} });
    setPreviewImage(null);
    setProgress(0);
    setError(null);
    setCurrentImageIndex(0);
  };

  // 当勾选 HPC 时，请求本地的一个接口来自动建立隧道
  const handleHPCChange = async (checked: boolean) => {
    setUseHPC(checked);
    if (checked) {
      // 既然用户已经手动建立好了隧道并且唤醒了后端，
      // 前端这里其实不需要再调用代理去执行一次 python 脚本了。
      // 我们直接把状态标记为 connected，这样 WebSocket 就会自动连到 8001 端口。
      setTunnelStatus('connected');
    } else {
      setTunnelStatus('idle');
    }
  };

  const handleGenerate = async () => {
    if (!prompt.trim() || loading) return;

    setLoading(true);
    setError(null);
    setPreviewImage(null);
    setProgress(0);
    setResult(null);
    
    // 生成新的 ID
    const newId = Date.now().toString();
    setCurrentId(newId);

    // If mock backend is enabled, simulate generation locally
    if (useMock) {
      // 模拟多张图片（前端UI目前只展示第一张，后续可以扩展为画廊）
      // ... 保持原有逻辑不变
      let currentProgress = 0;
      const interval = setInterval(() => {
        currentProgress += 5;
        setProgress(currentProgress);
        if (currentProgress >= 100) {
          clearInterval(interval);
          // 这里动态生成 mock 图片数组，长度为当前滑块的值 numImages
          const mockImages = Array.from({ length: numImages }).map((_, i) => 
            `https://images.unsplash.com/photo-1682687220742-aba13b6e50ba?q=80&w=1000&auto=format&fit=crop&sig=${Date.now()}_${i}` // 加上时间戳让 React 认为是不同的图
          );
          
          const newResult = {
            images: mockImages,
            prompt: prompt,
            params: {}
          };
          setResult(newResult);
          setCurrentImageIndex(0);
          
          // 保存到历史记录
          const newItem: HistoryItem = {
            id: newId,
            prompt: prompt,
            images: newResult.images,
            timestamp: Date.now()
          };
          setHistory(prev => [newItem, ...prev]);
          
          setLoading(false);
        }
      }, 100);
      return;
    }

    try {
      const targetPort = useHPC ? '8001' : '8000';
      const ws = new WebSocket(`ws://localhost:${targetPort}/ws/generate`);
      
      ws.onopen = () => {
        // 将 numImages 作为参数传给后端
        ws.send(JSON.stringify({
          prompt,
          negative_prompt: negativePrompt,
          num_inference_steps: steps,
          guidance_scale: guidanceScale,
          num_images_per_prompt: numImages, // 确保发送给后端的是当前的 numImages
          seed,
          mock: useMock
        }));
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'progress') {
          setProgress(data.value);
          if (data.preview) {
            setPreviewImage(data.preview);
          }
        } else if (data.type === 'final') {
          console.log("Received final data:", data); // Debug log
          // 处理后端返回的多图或者单图
          const finalImages = Array.isArray(data.images) && data.images.length > 0 
            ? data.images 
            : (data.image ? [data.image] : []);
          
          console.log("Processed finalImages length:", finalImages.length); // Debug log
          
          const newResult = {
            images: finalImages,
            prompt: prompt,
            params: {}
          };
          setResult(newResult);
          setCurrentImageIndex(0);
          
          // 保存到历史记录
          const newItem: HistoryItem = {
            id: newId,
            prompt: prompt,
            images: finalImages,
            timestamp: Date.now()
          };
          
          setHistory(prev => {
            const exists = prev.find(p => p.id === newItem.id);
            if (exists) {
              return prev.map(p => p.id === newItem.id ? newItem : p);
            }
            return [newItem, ...prev];
          });

          setLoading(false);
          ws.close();
        } else if (data.type === 'error') {
          setError(data.error || data.message || 'Failed to generate image.');
          setLoading(false);
          ws.close();
        }
      };

      ws.onerror = (err) => {
        console.error('WebSocket error:', err);
        setError("WebSocket connection failed. Is the backend running?");
        setLoading(false);
      };

      ws.onclose = () => {
        if (loading) {
          setError("Connection closed unexpectedly.");
          setLoading(false);
        }
      };
    } catch (err) {
      setError("Failed to start generation.");
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (!result || !result.images || result.images.length === 0) return;
    const link = document.createElement('a');
    link.href = result.images[currentImageIndex];
    link.download = `imaginary-${Date.now()}-${currentImageIndex + 1}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <>
      {showHome ? (
        <div className="min-h-screen bg-black text-white flex flex-col items-center justify-center relative overflow-hidden font-sans">
          {/* Background effects */}
          <div className="absolute inset-0 z-0">
            <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-purple-600/30 rounded-full blur-[128px] animate-pulse"></div>
            <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-indigo-600/30 rounded-full blur-[128px] animate-pulse" style={{ animationDelay: '2s' }}></div>
            <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 mix-blend-overlay"></div>
          </div>

          <div className="z-10 text-center space-y-8 px-4 flex flex-col items-center max-w-4xl">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 backdrop-blur-md mb-4 animate-in fade-in slide-in-from-bottom-4 duration-700">
              <Sparkles className="w-4 h-4 text-purple-400" />
              <span className="text-sm font-medium tracking-wide text-zinc-300">Powered by Stable Diffusion v1.5</span>
            </div>
            
            <h1 className="text-6xl md:text-8xl font-bold tracking-tight animate-in fade-in slide-in-from-bottom-8 duration-700 delay-150">
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400">
                Imaginary
              </span> AI
            </h1>
            
            <p className="text-xl md:text-2xl text-zinc-400 max-w-2xl mx-auto leading-relaxed animate-in fade-in slide-in-from-bottom-8 duration-700 delay-300">
              Transform your thoughts into stunning visual art in real-time. Experience the magic of AI-driven creation.
            </p>

            <div className="pt-8 animate-in fade-in zoom-in-95 duration-700 delay-500">
              <button
                onClick={() => setShowHome(false)}
                className="group relative inline-flex items-center justify-center gap-3 px-8 py-4 text-lg font-semibold text-white bg-white/10 border border-white/20 rounded-full overflow-hidden transition-all hover:scale-105 hover:bg-white/20 active:scale-95"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                <span className="relative z-10">Start Creating</span>
                <Sparkles className="w-5 h-5 relative z-10 group-hover:rotate-12 transition-transform" />
              </button>
            </div>
          </div>
        </div>
      ) : (
    <div className="flex h-screen bg-white text-zinc-900 font-sans overflow-hidden">
      {/* Sidebar */}
      <div 
        className={cn(
          "bg-zinc-50 border-r border-zinc-200 flex flex-col transition-all duration-300 z-20 shrink-0",
          isSidebarOpen ? "w-64" : "w-0 border-r-0 opacity-0 overflow-hidden"
        )}
      >
        <div className="p-4 flex flex-col gap-2 h-full">
          <button 
            onClick={startNew}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-white border border-zinc-200 rounded-xl shadow-sm hover:bg-zinc-50 hover:border-zinc-300 text-sm font-medium text-zinc-700 transition-all duration-200 active:scale-95"
          >
            <MessageSquarePlus className="w-4 h-4" />
            New Generation
          </button>

          <div className="flex-1 overflow-y-auto mt-4 -mx-2 px-2 space-y-1">
            <div className="flex items-center justify-between px-3 py-2 mb-1">
              <span className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">
                History
              </span>
              {history.length > 0 && (
                <button
                  onClick={clearHistory}
                  className="text-xs text-zinc-400 hover:text-red-500 transition-colors flex items-center gap-1"
                  title="Clear all history"
                >
                  <Trash2 className="w-3 h-3" />
                  Clear
                </button>
              )}
            </div>
            {history.length === 0 && (
              <div className="text-center text-xs text-zinc-400 mt-4">
                No generations yet
              </div>
            )}
            {history.map(item => (
              <div
                key={item.id}
                className={cn(
                  "relative w-full flex items-center group rounded-lg transition-colors",
                  currentId === item.id 
                    ? "bg-zinc-200" 
                    : "hover:bg-zinc-100"
                )}
              >
                <button
                  onClick={() => loadHistory(item)}
                  className={cn(
                    "flex-1 flex items-center gap-3 text-left px-3 py-2.5 text-sm",
                    currentId === item.id 
                      ? "text-zinc-900 font-medium" 
                      : "text-zinc-600"
                  )}
                >
                  <MessageSquare className={cn("w-4 h-4 shrink-0", currentId === item.id ? "text-zinc-700" : "text-zinc-400 group-hover:text-zinc-600")} />
                  <span className="truncate">{item.prompt}</span>
                </button>
                <button
                  onClick={(e) => deleteHistoryItem(e, item.id)}
                  className="p-2 opacity-0 group-hover:opacity-100 text-zinc-400 hover:text-red-500 transition-all absolute right-0"
                  title="Delete item"
                >
                  <XCircle className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col relative min-w-0 bg-white">
        {/* Toggle Sidebar Button */}
        <div className="absolute top-4 left-4 z-10 flex items-center gap-4">
          <button
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            className="p-2 rounded-lg bg-white/80 backdrop-blur border border-zinc-200 text-zinc-600 hover:bg-zinc-50 transition-colors shadow-sm"
            title={isSidebarOpen ? "Close sidebar" : "Open sidebar"}
          >
            {isSidebarOpen ? <PanelLeftClose className="w-5 h-5" /> : <PanelLeftOpen className="w-5 h-5" />}
          </button>
          
          {/* Header Title */}
          <div 
            className="font-semibold text-lg text-zinc-800 flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity"
            onClick={() => setShowHome(true)}
          >
            <div className="w-8 h-8 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg flex items-center justify-center shadow-md">
              <Sparkles className="w-4 h-4 text-white" />
            </div>
            Imaginary AI
          </div>
        </div>

        <main className="flex-1 w-full max-w-4xl mx-auto flex flex-col items-center justify-center p-4 pb-32 pt-20">
          {/* Result Area */}
          {(result || loading) ? (
                <div className="w-full aspect-square md:aspect-video max-h-[65vh] flex items-center justify-center bg-zinc-50/50 border border-zinc-200/80 rounded-2xl relative overflow-hidden group animate-in fade-in zoom-in-95 duration-300 shadow-sm">
                  {result ? (
                    <div className="relative w-full h-full flex flex-col items-center justify-center p-4">
                      <img
                        src={result.images[currentImageIndex]}
                        alt={result.prompt}
                        className="max-w-full max-h-full object-contain rounded-lg shadow-xl transition-all duration-300"
                      />
                      
                      {/* 多图切换控制 */}
                      {result.images.length > 1 && (
                        <>
                          <button
                            onClick={() => setCurrentImageIndex(prev => Math.max(0, prev - 1))}
                            disabled={currentImageIndex === 0}
                            className="absolute left-4 top-1/2 -translate-y-1/2 p-2 rounded-full bg-white/80 hover:bg-white text-zinc-800 shadow-md border border-zinc-200 transition-all disabled:opacity-30 disabled:cursor-not-allowed opacity-0 group-hover:opacity-100"
                          >
                            <ChevronLeft className="w-6 h-6" />
                          </button>
                          <button
                            onClick={() => setCurrentImageIndex(prev => Math.min(result.images.length - 1, prev + 1))}
                            disabled={currentImageIndex === result.images.length - 1}
                            className="absolute right-4 top-1/2 -translate-y-1/2 p-2 rounded-full bg-white/80 hover:bg-white text-zinc-800 shadow-md border border-zinc-200 transition-all disabled:opacity-30 disabled:cursor-not-allowed opacity-0 group-hover:opacity-100"
                          >
                            <ChevronRight className="w-6 h-6" />
                          </button>
                          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-black/50 backdrop-blur-md">
                            {result.images.map((_, idx) => (
                              <button
                                key={idx}
                                onClick={() => setCurrentImageIndex(idx)}
                                className={cn(
                                  "w-2 h-2 rounded-full transition-all",
                                  currentImageIndex === idx ? "bg-white scale-125" : "bg-white/40 hover:bg-white/60"
                                )}
                              />
                            ))}
                          </div>
                        </>
                      )}

                      <div className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity">
                        <button
                          onClick={handleDownload}
                          className="bg-white/80 hover:bg-white text-zinc-800 p-2 rounded-lg backdrop-blur-sm transition-colors shadow-md border border-zinc-200"
                          title="Download Current Image"
                        >
                          <Download className="w-5 h-5" />
                        </button>
                      </div>
                    </div>
                  ) : (
              <div className="relative w-full h-full flex flex-col items-center justify-center p-4">
                {previewImage ? (
                  <div className="relative w-full h-full flex flex-col items-center justify-center">
                    <img
                      src={previewImage}
                      alt="Generating..."
                      className="max-w-full max-h-full object-contain rounded-lg transition-all duration-300 shadow-xl"
                    />
                    <div className="absolute bottom-4 left-4 right-4 flex flex-col items-center gap-2">
                       <div className="w-full max-w-md bg-white/90 backdrop-blur-md p-3 rounded-xl shadow-lg border border-zinc-200/50 flex flex-col gap-2">
                         <div className="flex justify-between items-center px-1">
                           <span className="text-xs font-medium text-indigo-600 flex items-center gap-2">
                             <Loader2 className="w-3 h-3 animate-spin" />
                             Generating
                           </span>
                           <span className="text-xs font-bold text-zinc-700">{progress}%</span>
                         </div>
                         <div className="w-full h-1.5 bg-zinc-200 rounded-full overflow-hidden">
                           <div 
                             className="h-full bg-indigo-600 transition-all duration-300 ease-out"
                             style={{ width: `${progress}%` }}
                           />
                         </div>
                       </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center text-zinc-500">
                    <div className="flex flex-col items-center gap-4">
                      <Loader2 className="w-10 h-10 animate-spin text-indigo-600" />
                      <p className="text-lg font-medium animate-pulse text-indigo-500">Dreaming up your image...</p>
                      {progress > 0 && (
                        <div className="w-48 mt-2">
                           <div className="flex justify-between items-center px-1 mb-1">
                             <span className="text-xs font-medium text-indigo-600">Initializing</span>
                             <span className="text-xs font-bold text-zinc-700">{progress}%</span>
                           </div>
                           <div className="w-full h-1.5 bg-zinc-200 rounded-full overflow-hidden">
                             <div 
                               className="h-full bg-indigo-600 transition-all duration-300 ease-out"
                               style={{ width: `${progress}%` }}
                             />
                           </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-center space-y-4 opacity-40">
             <div className="w-20 h-20 bg-zinc-100 rounded-3xl flex items-center justify-center mb-4">
                <Sparkles className="w-10 h-10 text-zinc-400" />
             </div>
             <h2 className="text-2xl font-bold text-zinc-400">Ready to create</h2>
          </div>
        )}
      </main>

      {/* Input Section - Fixed at bottom */}
      <div className="fixed bottom-0 left-0 right-0 bg-gradient-to-t from-white via-white to-transparent pt-20 pb-8 px-4 z-10" style={{ paddingLeft: isSidebarOpen ? '17rem' : '1rem', transition: 'padding 300ms ease-in-out' }}>
        <div className="w-full max-w-3xl mx-auto space-y-4">
          
          {/* Advanced Settings Panel (Pop-up style) */}
          {showAdvanced && (
            <div className="mb-4 p-6 bg-white/90 backdrop-blur-xl border border-zinc-200 rounded-2xl shadow-xl animate-in slide-in-from-bottom-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-6">
                  <label className="block">
                    <div className="flex justify-between mb-2">
                      <span className="text-sm font-medium text-zinc-600">Inference Steps</span>
                      <input
                        type="number"
                        min="20"
                        max="50"
                        value={steps}
                        onChange={(e) => setSteps(Math.max(20, Math.min(50, parseInt(e.target.value) || 20)))}
                        className="w-16 bg-zinc-50 border border-zinc-300 rounded px-2 py-1 text-sm text-center focus:outline-none focus:border-indigo-500"
                      />
                    </div>
                    <input
                      type="range"
                      min="20"
                      max="50"
                      step="1"
                      value={steps}
                      onChange={(e) => setSteps(parseInt(e.target.value))}
                      className="w-full accent-indigo-600 h-1 bg-zinc-200 rounded-lg appearance-none cursor-pointer"
                    />
                  </label>
                  <label className="block">
                    <div className="flex justify-between mb-2">
                      <span className="text-sm font-medium text-zinc-600">Number of Images</span>
                      <input
                        type="number"
                        min="1"
                        max="4"
                        value={numImages}
                        onChange={(e) => setNumImages(Math.max(1, Math.min(4, parseInt(e.target.value) || 1)))}
                        className="w-16 bg-zinc-50 border border-zinc-300 rounded px-2 py-1 text-sm text-center focus:outline-none focus:border-indigo-500"
                      />
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="4"
                      step="1"
                      value={numImages}
                      onChange={(e) => setNumImages(parseInt(e.target.value))}
                      className="w-full accent-indigo-600 h-1 bg-zinc-200 rounded-lg appearance-none cursor-pointer"
                    />
                  </label>
                </div>
                <div className="space-y-6">
                  <label className="block">
                    <div className="flex justify-between mb-2">
                      <span className="text-sm font-medium text-zinc-600">Guidance Scale</span>
                      <input
                        type="number"
                        min="1"
                        max="20"
                        step="0.5"
                        value={guidanceScale}
                        onChange={(e) => setGuidanceScale(Math.max(1, Math.min(20, parseFloat(e.target.value) || 7.5)))}
                        className="w-16 bg-zinc-50 border border-zinc-300 rounded px-2 py-1 text-sm text-center focus:outline-none focus:border-indigo-500"
                      />
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="20"
                      step="0.5"
                      value={guidanceScale}
                      onChange={(e) => setGuidanceScale(parseFloat(e.target.value))}
                      className="w-full accent-indigo-600 h-1 bg-zinc-200 rounded-lg appearance-none cursor-pointer"
                    />
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer mt-8 p-2 rounded hover:bg-zinc-100 transition-colors">
                    <input
                      type="checkbox"
                      checked={useMock}
                      onChange={(e) => setUseMock(e.target.checked)}
                      className="rounded bg-zinc-50 border-zinc-300 text-indigo-600 focus:ring-indigo-500"
                    />
                    <span className="text-sm text-zinc-600">Mock Backend (Demo Mode)</span>
                  </label>
                  <label className="flex flex-col gap-2 mt-2 p-3 bg-zinc-50 rounded-lg border border-zinc-200 transition-colors hover:bg-zinc-100/50">
                    <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2 cursor-pointer">
                              <input
                                type="checkbox"
                                checked={useHPC}
                                onChange={(e) => handleHPCChange(e.target.checked)}
                                className="rounded bg-zinc-50 border-zinc-300 text-indigo-600 focus:ring-indigo-500 disabled:opacity-50"
                              />
                              <span className="text-sm font-medium text-zinc-700">Use HPC (Local SSH Tunnel)</span>
                            </div>
                            {tunnelStatus === 'connected' && <span className="text-xs text-green-600 font-medium px-2 py-1 bg-green-100 rounded">Connected</span>}
                          </div>
                          {useHPC && (
                            <div className="mt-2 pl-6">
                              <p className="text-[10px] text-zinc-500 mt-1">
                                Please ensure you have run the ssh tunnel command in your terminal:
                                <br/>
                                <code className="bg-zinc-200 px-1 rounded">ssh -L 8001:hpcgpuXXX:8000 ...</code>
                              </p>
                            </div>
                          )}
                  </label>
                </div>
              </div>
            </div>
          )}

          {/* Input Bar */}
          <div className="relative group">
            <div className="absolute -inset-0.5 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-2xl opacity-20 group-hover:opacity-40 transition duration-500 blur"></div>
            <div className="relative bg-white border border-zinc-200 rounded-2xl flex items-end p-2 shadow-xl shadow-indigo-500/5">
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Describe the image you want to create..."
                className="w-full bg-transparent text-lg px-4 py-3 min-h-[60px] max-h-[200px] resize-none focus:outline-none placeholder:text-zinc-400 text-zinc-800 rounded-xl"
                disabled={loading}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleGenerate();
                  }
                }}
              />
              <div className="flex items-center gap-2 pb-2 pr-2">
                <button
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className={cn(
                    "p-2 rounded-lg transition-colors hover:bg-zinc-100",
                    showAdvanced ? "text-indigo-600 bg-zinc-100" : "text-zinc-400 hover:text-zinc-600"
                  )}
                  title="Settings"
                >
                  <Settings2 className="w-5 h-5" />
                </button>
                <button
                  onClick={handleGenerate}
                  disabled={!prompt.trim() || loading}
                  className={cn(
                    "p-2 rounded-xl transition-all duration-200",
                    prompt.trim() && !loading
                      ? "bg-indigo-600 text-white hover:bg-indigo-700 shadow-lg shadow-indigo-500/30"
                      : "bg-zinc-100 text-zinc-400 cursor-not-allowed"
                  )}
                >
                  {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Sparkles className="w-5 h-5" />}
                </button>
              </div>
            </div>
          </div>
          
          <div className="text-center">
             <p className="text-xs text-zinc-400">Powered by Stable Diffusion v1.5 • Fine-tuned Model</p>
          </div>
        </div>
      </div>

        {/* Error Toast */}
        {error && (
          <div className="fixed top-6 right-6 z-50 bg-red-500/10 border border-red-500/20 text-red-400 px-4 py-3 rounded-xl flex items-center gap-3 backdrop-blur-md animate-in slide-in-from-top-2 shadow-lg">
            <XCircle className="w-5 h-5 shrink-0" />
            <p className="text-sm">{error}</p>
            <button onClick={() => setError(null)} className="ml-auto hover:text-red-300"><XCircle className="w-4 h-4" /></button>
          </div>
        )}
      </div>
    </div>
      )}
    </>
  );
}

export default App;
