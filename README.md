# 训练实时监控面板 (Train Monitor)

一个轻量级的桌面 GUI 工具，实时显示 GPU 利用率、显存、温度、功耗、训练进程、Checkpoint 输出等信息，**无需安装任何第三方依赖**。

![windows](https://img.shields.io/badge/Windows-10%2F11-blue)
![macos](https://img.shields.io/badge/macOS-11%2B-green)
![python](https://img.shields.io/badge/Python-3.8+-yellow)
![nvidia](https://img.shields.io/badge/NVIDIA-GPU-green)

## 功能特性

- 🌡️ **GPU 实时状态**：利用率 / 显存 / 温度 / 功耗（通过 nvidia-smi）
- 📈 **利用率趋势图**：在界面内绘制实时折线，直观观察波动
- 🧠 **训练进程监测**：自动识别 Python / weclone 等训练进程
- 📁 **Checkpoint 自动发现**：扫描项目目录，自动列出最近生成的 checkpoint / safetensors / adapter 文件
- 🖼️ **纯 GUI 窗口**：暗色主题 + Tkinter 原生控件，无需安装任何 pip 包
- ⚠️ **预警提示**：显存占用过高 / 温度过高时高亮提醒

## 快速开始 — 推荐：免安装版

### 方式一：下载可执行文件（无需装 Python）

去 [Releases 页面](https://github.com/aiyangdie/train-monitor/releases) 下载对应系统的可执行文件，**双击直接运行**。

| 系统 | 下载文件 |
|------|---------|
| Windows 10/11 (x64) | `train-monitor-windows-x86_64.exe` |
| macOS (x64 / Apple Silicon) | `train-monitor-macos-x86_64` |

> macOS 首次打开若提示"无法打开"，请在"系统设置 → 隐私与安全性"中点击"仍要打开"，或在终端运行：
> ```bash
> xattr -d com.apple.quarantine train-monitor-macos-x86_64
> ```

### 方式二：源码运行（开发者）

```bash
# Windows
python monitor_gui.py

# macOS
python3 monitor_gui.py
```

## 环境要求

- Windows 10 / 11 + NVIDIA GPU + 已安装驱动（`nvidia-smi` 命令可用）
- 或 macOS 11+（需能执行 `nvidia-smi`，或仅用进程/文件监测功能）
- Python 3.8+（Tkinter 随 Python 一起安装，无需额外 pip）

### 自定义目录扫描

默认扫描项目所在目录，可在 `monitor_gui.py` 中修改：

```python
SEARCH_DIRS = [
    Path(r"你的项目路径"),
    Path.home() / ".cache" / "huggingface",
]
```

## 自行打包 / 从源码构建可执行文件

### Windows

```powershell
pip install pyinstaller
python -m PyInstaller --onefile --windowed --name train-monitor monitor_gui.py
# 产物: dist\train-monitor.exe
```

或直接运行构建脚本：

```powershell
powershell -ExecutionPolicy Bypass -File build-windows.ps1
```

### macOS

```bash
python3 -m pip install pyinstaller
python3 -m PyInstaller --onefile --windowed --name train-monitor monitor_gui.py
# 产物: dist/train-monitor
```

或直接运行构建脚本：

```bash
chmod +x build-macos.sh && ./build-macos.sh
```

## 工作原理

```
monitor_gui.py
  ├── get_gpu_info()             # 调用 nvidia-smi 查询 GPU
  ├── list_training_processes()  # 用 Windows API (Toolhelp32) / macOS (ps)
  ├── find_ckpt_files()          # 扫描 checkpoint / 模型文件
  └── MonitorApp (Tkinter)       # 每 3 秒刷新界面
```

零第三方依赖，所有 Windows API 调用走标准库 `ctypes`。

## GitHub Actions 自动构建

本仓库已配置 CI：推送以 `v` 开头的 tag 时，会自动在 **Windows 和 macOS** 两个平台构建可执行文件，并发布到 GitHub Releases。

```bash
git tag v1.0.0
git push origin v1.0.0
```

推送后，打开 [Actions 页面](https://github.com/aiyangdie/train-monitor/actions) 即可看到构建进度，约 3-5 分钟完成。

## 故障排查

- **看不到 GPU 数据**：先在命令行跑 `nvidia-smi`，确保命令能正常输出
- **进程列表为空**：检查训练程序是否正在运行（python.exe / weclone-cli.exe）
- **窗口白屏/报错**：尝试用 `python monitor_gui.py` 运行，查看控制台错误信息
- **macOS 打不开**：系统设置 → 隐私与安全性 → 仍要打开

## 许可证

MIT
