<p align="center"><img src="assets/covers/28-train-monitor.jpg" alt="训练监控 项目封面" width="100%"></p>

<h1 align="center">训练实时监控面板 · Train Monitor</h1>

<p align="center">训练时盯着 GPU、显存、温度、功耗和 Checkpoint 输出的桌面小面板</p>

## 简介

训练跑起来之后最想知道的三件事：卡有没有闲着、显存会不会爆、Checkpoint 有没有落盘。这个轻量桌面 GUI 把这些指标放一屏，不依赖第三方库，下载可执行文件直接跑。

## 功能

- **硬件指标**：GPU 利用率、显存占用、温度、功耗
- **进程视图**：识别训练进程及其资源占用
- **输出跟踪**：实时显示 Checkpoint 与日志输出
- **自定义目录**：可指定扫描路径，适配不同项目的输出结构
- **免安装**：提供 Windows / macOS 可执行文件

## 快速开始

**方式一：下载可执行文件**（不需要 Python）

前往 [Releases](https://github.com/aiyangdie/train-monitor/releases) 下载对应平台的文件，双击运行。

**方式二：源码运行**

```bash
git clone https://github.com/aiyangdie/train-monitor.git
cd train-monitor
pip install -r requirements.txt
python main.py
```

## 自行打包

```bash
# Windows
pyinstaller --onefile --windowed main.py    # 产物 dist\train-monitor.exe
# macOS
pyinstaller --onefile --windowed main.py    # 产物 dist/train-monitor
```

仓库带有 GitHub Actions 自动构建，推送标签会产出各平台产物。

## 工作原理

定时采样显卡驱动暴露的指标与训练进程输出，聚合成面板视图；不注入训练代码，不改动你的工程。

## 许可证

仓库未附 LICENSE 文件，使用与分发请先联系作者确认。
