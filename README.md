# 训练实时监控面板 (Train Monitor)

一个轻量级的 Windows 桌面 GUI 工具，实时显示 GPU 利用率、显存、温度、功耗、训练进程、Checkpoint 输出等信息，不需要安装任何第三方依赖。

![preview](https://img.shields.io/badge/Windows-10%2F11-blue)
![python](https://img.shields.io/badge/Python-3.8+-yellow)
![nvidia](https://img.shields.io/badge/NVIDIA-GPU-green)

## 功能特性

- 🌡️ **GPU 实时状态**：利用率 / 显存 / 温度 / 功耗（通过 nvidia-smi）
- 📈 **利用率趋势图**：在界面内绘制实时折线，直观观察波动
- 🧠 **训练进程监测**：自动识别 Python / weclone 等训练进程，显示 PID、内存、CPU 时间、启动时间
- 📁 **Checkpoint 自动发现**：扫描项目目录，自动列出最近生成的 checkpoint / safetensors / adapter 文件
- 🖼️ **纯 GUI 窗口**：暗色主题 + Tkinter 原生控件，无需安装任何 pip 包
- ⚠️ **预警提示**：显存占用过高 / 温度过高时高亮提醒

## 快速开始

### 环境要求

- Windows 10 / 11
- NVIDIA GPU + 已安装驱动（`nvidia-smi` 命令可用）
- Python 3.8+（Tkinter 随 Python 一起安装，无需额外 pip）

### 运行

```bash
python monitor_gui.py
```

或者用后台模式运行（不显示控制台窗口）：

```bash
pythonw monitor_gui.py
```

### 自定义目录扫描

默认扫描项目所在目录，可在 `monitor_gui.py` 中修改：

```python
SEARCH_DIRS = [
    Path(r"你的项目路径"),
    Path.home() / ".cache" / "huggingface",
]
```

## 工作原理

```
monitor_gui.py
  ├── get_gpu_info()             # 调用 nvidia-smi 查询 GPU
  ├── list_training_processes()  # 用 Windows API (Toolhelp32) 枚举进程
  ├── find_ckpt_files()          # 扫描 checkpoint / 模型文件
  └── MonitorApp (Tkinter)       # 每 3 秒刷新界面
```

零第三方依赖，所有 Windows API 调用走标准库 `ctypes`。

## 故障排查

- **看不到 GPU 数据**：先在命令行跑 `nvidia-smi`，确保命令能正常输出
- **进程列表为空**：检查训练程序是否正在运行（python.exe / weclone-cli.exe）
- **窗口白屏/报错**：尝试用 `python monitor_gui.py`（非 pythonw）运行，查看控制台错误信息

## 许可证

MIT
