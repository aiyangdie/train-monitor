# -*- coding: utf-8 -*-
"""
训练监控 - GUI 可视化版（Win 原生窗口，无需安装依赖）
运行：python monitor_gui.py
"""
import os
import subprocess
import platform
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import tkinter as tk
from tkinter import ttk, font

# ======== 配置 ========
# 把项目所在目录自动识别为当前脚本所在目录（不再写死路径）
_PROJECT_ROOT = Path(__file__).resolve().parent
REFRESH_MS = 3000   # 每 3 秒刷新一次
SEARCH_DIRS = [
    _PROJECT_ROOT,
    _PROJECT_ROOT.parent,
    Path.home() / "Documents",
    Path.home() / ".cache" / "huggingface",
]
CKPT_KEYWORDS = ["checkpoint", "adapter", "safetensors", "pytorch_model",
                 "optimizer", "rng_state", "trainer_state", "training_args",
                 "weclone", "sft", "lora", "adapter_model"]

# nvidia-smi 的常见安装路径（Windows）
_NVIDIA_SMI_CANDIDATES_WIN = [
    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    r"C:\Windows\System32\nvidia-smi.exe",
    r"C:\Windows\SysWOW64\nvidia-smi.exe",
]
# NVIDIA 驱动目录会随大版本号变化，通过枚举 C:\Program Files\NVIDIA Corporation\ 找 NVSMI 目录
_NVIDIA_ROOT = Path(r"C:\Program Files\NVIDIA Corporation")

# 缓存 nvidia-smi 路径，避免每次刷新都去扫描
_CACHED_NVIDIA_SMI = None
# =====================


def find_nvidia_smi():
    """
    找到 nvidia-smi 可执行文件的完整路径。
    优先顺序：PATH -> 常见安装路径 -> 扫描 C:/Program Files/NVIDIA Corporation/
    返回 (path_or_None, message)
    """
    global _CACHED_NVIDIA_SMI
    if _CACHED_NVIDIA_SMI and Path(_CACHED_NVIDIA_SMI).exists():
        return _CACHED_NVIDIA_SMI, "ok"

    # 1) PATH
    import shutil
    found = shutil.which("nvidia-smi")
    if found:
        _CACHED_NVIDIA_SMI = found
        return found, "ok"

    # 2) 常见固定路径
    for p in _NVIDIA_SMI_CANDIDATES_WIN:
        if Path(p).exists():
            _CACHED_NVIDIA_SMI = p
            return p, "ok"

    # 3) 扫描 C:\Program Files\NVIDIA Corporation 下所有 *NVSMI* 目录
    try:
        if _NVIDIA_ROOT.exists():
            for sub in _NVIDIA_ROOT.iterdir():
                if sub.is_dir() and "NVSMI" in sub.name.upper():
                    candidate = sub / "nvidia-smi.exe"
                    if candidate.exists():
                        _CACHED_NVIDIA_SMI = str(candidate)
                        return str(candidate), "ok"
    except Exception:
        pass

    return None, (
        "未找到 nvidia-smi。请确认：\n"
        "  1) 电脑已安装 NVIDIA 显卡驱动\n"
        "  2) C:/Program Files/NVIDIA Corporation/.../NVSMI/ 目录下存在 nvidia-smi.exe\n"
        "  3) 或者把 nvidia-smi 所在目录加入系统环境变量 PATH"
    )


def run_cmd(cmd, timeout=5):
    try:
        # 传进来的可能是 list（推荐，避免 shell 解析问题）
        if isinstance(cmd, (list, tuple)):
            r = subprocess.run(list(cmd), shell=False, capture_output=True,
                               text=True, timeout=timeout, errors="ignore")
        else:
            r = subprocess.run(cmd, shell=True, capture_output=True,
                               text=True, timeout=timeout, errors="ignore")
        return r.stdout.strip()
    except Exception:
        return ""


def get_gpu_info():
    """
    检测 GPU 状态。
    返回 {"name", "gpu_util", "mem_used_mb", "mem_total_mb",
          "temp_c", "power_w", "power_limit_w", "gpu_count", "error"}
    任一失败时，error 字段会包含用户可读的中文错误提示。
    """
    smi_path, msg = find_nvidia_smi()
    if smi_path is None:
        return {"error": msg}

    # 多 GPU 环境：每行一块卡，取总利用率/总显存/平均温度/第一张卡的名字
    out = run_cmd([
        smi_path,
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,"
        "temperature.gpu,power.draw,power.limit,count",
        "--format=csv,noheader,nounits"
    ])
    if not out:
        # 再试一次不带 count（部分老驱动不支持）
        out = run_cmd([
            smi_path,
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,"
            "temperature.gpu,power.draw,power.limit",
            "--format=csv,noheader,nounits"
        ])
        if not out:
            return {"error": "nvidia-smi 已找到但无输出，请检查驱动状态。"}

    lines = [l for l in out.splitlines() if l.strip()]
    if not lines:
        return {"error": "未检测到 NVIDIA GPU。"}

    gpus = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            gpus.append({
                "name": parts[0],
                "gpu_util": float(parts[1]) if parts[1] else 0.0,
                "mem_used_mb": float(parts[2]) if parts[2] else 0.0,
                "mem_total_mb": float(parts[3]) if parts[3] else 0.0,
                "temp_c": float(parts[4]) if parts[4] else 0.0,
                "power_w": parts[5] if parts[5] else "?",
                "power_limit_w": parts[6] if parts[6] else "?",
            })
        except (ValueError, IndexError):
            continue

    if not gpus:
        return {"error": "nvidia-smi 输出异常，无法解析。"}

    # 汇总：若有多 GPU，展示总利用率/总显存
    if len(gpus) == 1:
        g = gpus[0]
        return {
            "name": g["name"],
            "gpu_util": g["gpu_util"],
            "mem_used_mb": g["mem_used_mb"],
            "mem_total_mb": g["mem_total_mb"],
            "temp_c": g["temp_c"],
            "power_w": g["power_w"],
            "power_limit_w": g["power_limit_w"],
            "gpu_count": 1,
            "error": None,
        }
    # 多 GPU：汇总
    total_util = sum(g["gpu_util"] for g in gpus) / len(gpus)
    total_mem_used = sum(g["mem_used_mb"] for g in gpus)
    total_mem_total = sum(g["mem_total_mb"] for g in gpus)
    avg_temp = sum(g["temp_c"] for g in gpus) / len(gpus)
    names = "，".join(g["name"] for g in gpus[:2]) + (f" 等 {len(gpus)} 张卡" if len(gpus) > 2 else "")
    return {
        "name": names,
        "gpu_util": total_util,
        "mem_used_mb": total_mem_used,
        "mem_total_mb": total_mem_total,
        "temp_c": avg_temp,
        "power_w": "见各卡",
        "power_limit_w": "见各卡",
        "gpu_count": len(gpus),
        "error": None,
    }


def list_training_processes():
    """用 Windows 原生 API 枚举进程（零依赖，不需要 psutil）"""
    procs = []
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi

        class PROCESSENTRY32(ctypes.Structure):
            _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                        ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_void_p),
                        ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                        ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG),
                        ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_char * 260)]

        class FILETIME(ctypes.Structure):
            _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        h_snap = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)  # TH32CS_SNAPPROCESS
        if h_snap == -1:
            return []
        pe = PROCESSENTRY32()
        pe.dwSize = ctypes.sizeof(PROCESSENTRY32)
        PROCESS_QUERY_INFORMATION = 0x0400
        PROCESS_VM_READ = 0x0010

        ok = kernel32.Process32First(h_snap, ctypes.byref(pe))
        while ok:
            try:
                name = pe.szExeFile.decode("gbk", errors="ignore").strip()
            except Exception:
                name = ""
            if "python" in name.lower() or "weclone" in name.lower():
                pid = pe.th32ProcessID
                mem_mb = 0
                cpu_s = 0.0
                start_str = ""
                h_proc = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, 0, pid)
                if h_proc:
                    creation = FILETIME()
                    exit_t = FILETIME()
                    kernel_t = FILETIME()
                    user_t = FILETIME()
                    if kernel32.GetProcessTimes(h_proc, ctypes.byref(creation), ctypes.byref(exit_t),
                                                ctypes.byref(kernel_t), ctypes.byref(user_t)):
                        def ft_to_sec(ft):
                            return (ft.dwHighDateTime * (2**32) + ft.dwLowDateTime) / 10_000_000
                        creation_sec = ft_to_sec(creation) - 11644473600
                        cpu_s = ft_to_sec(kernel_t) + ft_to_sec(user_t)
                        if creation_sec > 0:
                            start_str = datetime.fromtimestamp(creation_sec).strftime("%Y/%m/%d %H:%M:%S")
                    pmc = PROCESS_MEMORY_COUNTERS()
                    pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
                    if psapi.GetProcessMemoryInfo(h_proc, ctypes.byref(pmc), ctypes.sizeof(pmc)):
                        mem_mb = pmc.WorkingSetSize / 1024 / 1024
                    kernel32.CloseHandle(h_proc)
                procs.append({
                    "pid": str(pid), "name": name,
                    "cpu_s": cpu_s, "mem_mb": mem_mb, "start_time": start_str,
                })
            ok = kernel32.Process32Next(h_snap, ctypes.byref(pe))
        kernel32.CloseHandle(h_snap)
        return procs
    except Exception:
        return procs


def find_ckpt_files(since_hours=24):
    cutoff = datetime.now() - timedelta(hours=since_hours)
    found = []
    for d in SEARCH_DIRS:
        if not d or not d.exists():
            continue
        try:
            for p in d.rglob("*"):
                try:
                    if p.is_dir():
                        continue
                    if p.stat().st_mtime < cutoff.timestamp():
                        continue
                    lower = p.name.lower()
                    if not any(k in lower for k in CKPT_KEYWORDS):
                        continue
                    size_mb = p.stat().st_size / 1024 / 1024
                    if size_mb < 1 and "checkpoint" not in lower and "adapter" not in lower:
                        continue
                    found.append((p, size_mb, datetime.fromtimestamp(p.stat().st_mtime)))
                except Exception:
                    pass
        except Exception:
            pass
    found.sort(key=lambda x: x[2], reverse=True)
    return found


def fmt_duration(seconds):
    if seconds is None or seconds < 0:
        return "—"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}时{m}分{sec}秒"
    if m:
        return f"{m}分{sec}秒"
    return f"{sec}秒"


# ==================== GUI ====================

class MonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("训练实时监控")
        self.root.geometry("780x620")
        self.root.configure(bg="#1e1e2e")
        self.root.minsize(720, 540)

        # 字体
        self.font_title = font.Font(family="Microsoft YaHei", size=16, weight="bold")
        self.font_label = font.Font(family="Microsoft YaHei", size=11)
        self.font_big = font.Font(family="Microsoft YaHei", size=24, weight="bold")
        self.font_mono = font.Font(family="Consolas", size=10)

        self.start_time = datetime.now()
        self.gpu_history = []  # 存最近的 GPU% 画小趋势图用
        self.last_ckpt_count = -1
        self._build_ui()

        # 首次刷新 + 启动循环
        self.root.after(200, self.refresh)

    # ---------- UI 构建 ----------
    def _build_ui(self):
        # 顶部标题栏
        header = tk.Frame(self.root, bg="#11111b", height=60)
        header.pack(fill="x")
        tk.Label(header, text="🔥 训练实时监控面板", fg="#f9e2af", bg="#11111b",
                 font=self.font_title).pack(side="left", padx=20, pady=12)
        self.lbl_time = tk.Label(header, text="", fg="#a6adc8", bg="#11111b",
                                 font=self.font_label)
        self.lbl_time.pack(side="right", padx=20, pady=12)

        # 主体 = 左侧 GPU 卡片 + 右侧训练状态
        body = tk.Frame(self.root, bg="#1e1e2e")
        body.pack(fill="both", expand=True, padx=10, pady=10)

        # ===== 左侧：GPU 大卡 =====
        left = tk.Frame(body, bg="#313244", padx=15, pady=15)
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))

        tk.Label(left, text="GPU 状态", fg="#89b4fa", bg="#313244",
                 font=self.font_label, anchor="w").pack(fill="x")
        self.lbl_gpu_name = tk.Label(left, text="—", fg="#cdd6f4", bg="#313244",
                                      font=("Microsoft YaHei", 12, "bold"), anchor="w")
        self.lbl_gpu_name.pack(fill="x", pady=(2, 10))

        # GPU 利用率
        self.lbl_gpu_pct = tk.Label(left, text="--%", fg="#a6e3a1", bg="#313244",
                                    font=self.font_big)
        self.lbl_gpu_pct.pack(anchor="w", pady=(0, 4))
        self.pb_gpu = self._mk_progress(left, total=100, color_fill="#a6e3a1")
        self.pb_gpu.pack(fill="x", pady=(0, 12))

        # 显存
        row = tk.Frame(left, bg="#313244"); row.pack(fill="x", pady=4)
        tk.Label(row, text="显存", fg="#fab387", bg="#313244",
                 font=self.font_label, width=8, anchor="w").pack(side="left")
        self.lbl_mem_text = tk.Label(row, text="-- / -- MB", fg="#cdd6f4",
                                     bg="#313244", font=self.font_label)
        self.lbl_mem_text.pack(side="left")
        self.pb_mem = self._mk_progress(left, total=100, color_fill="#fab387")
        self.pb_mem.pack(fill="x", pady=(4, 12))

        # 温度 / 功耗
        infos = tk.Frame(left, bg="#313244"); infos.pack(fill="x", pady=4)
        self.lbl_temp = tk.Label(infos, text="温度: -- °C", fg="#f9e2af",
                                 bg="#313244", font=self.font_label, anchor="w")
        self.lbl_temp.pack(side="left", expand=True, fill="x")
        self.lbl_power = tk.Label(infos, text="功耗: -- / -- W", fg="#cba6f7",
                                  bg="#313244", font=self.font_label, anchor="w")
        self.lbl_power.pack(side="left", expand=True, fill="x")

        # GPU 小趋势图（Canvas）
        tk.Label(left, text="GPU 利用率趋势（最近 60 次刷新）",
                 fg="#a6adc8", bg="#313244", font=("Microsoft YaHei", 9)).pack(
            anchor="w", pady=(14, 2))
        self.chart = tk.Canvas(left, height=80, bg="#181825", highlightthickness=0)
        self.chart.pack(fill="x", pady=2)

        # ===== 右侧：训练状态 + Checkpoint =====
        right = tk.Frame(body, bg="#313244", padx=15, pady=15)
        right.pack(side="left", fill="both", expand=True, padx=(5, 0))

        tk.Label(right, text="训练状态", fg="#89b4fa", bg="#313244",
                 font=self.font_label, anchor="w").pack(fill="x")
        self.lbl_runtime = tk.Label(right, text="已运行：--", fg="#a6e3a1",
                                    bg="#313244", font=("Microsoft YaHei", 16, "bold"))
        self.lbl_runtime.pack(anchor="w", pady=(4, 8))

        tk.Label(right, text="进程列表", fg="#f5c2e7", bg="#313244",
                 font=self.font_label, anchor="w").pack(fill="x", pady=(6, 2))
        self.txt_procs = tk.Text(right, height=7, bg="#181825", fg="#cdd6f4",
                                 font=self.font_mono, relief="flat", padx=8, pady=6)
        self.txt_procs.pack(fill="x")

        tk.Label(right, text="Checkpoint / 模型输出（最近 24 小时）",
                 fg="#f5c2e7", bg="#313244", font=self.font_label, anchor="w").pack(
            fill="x", pady=(12, 2))
        self.txt_ckpt = tk.Text(right, height=8, bg="#181825", fg="#cdd6f4",
                                font=self.font_mono, relief="flat", padx=8, pady=6)
        self.txt_ckpt.pack(fill="both", expand=True, pady=(0, 4))

        # ===== 底部：状态条 + 警告 =====
        bottom = tk.Frame(self.root, bg="#11111b", height=38)
        bottom.pack(fill="x", side="bottom")
        self.lbl_status = tk.Label(bottom, text="● 监控中", fg="#a6e3a1",
                                   bg="#11111b", font=self.font_label)
        self.lbl_status.pack(side="left", padx=16, pady=8)
        self.lbl_warn = tk.Label(bottom, text="", fg="#f38ba8", bg="#11111b",
                                 font=("Microsoft YaHei", 10, "bold"))
        self.lbl_warn.pack(side="right", padx=16, pady=8)
        self.lbl_monitor_uptime = tk.Label(bottom, text="", fg="#a6adc8",
                                           bg="#11111b", font=self.font_label)
        self.lbl_monitor_uptime.pack(side="right", padx=16, pady=8)

    def _mk_progress(self, parent, total, color_fill, height=18):
        """自定义进度条（因为 ttk 进度条在暗色主题里不好看）"""
        canvas = tk.Canvas(parent, height=height, bg="#181825",
                           highlightthickness=0, bd=0)
        canvas._total = total
        canvas._color = color_fill
        canvas._fill_id = None
        canvas.bind("<Configure>",
                    lambda e: self._draw_progress(canvas, getattr(canvas, "_value", 0)))
        setattr(canvas, "_value", 0)
        return canvas

    def _draw_progress(self, canvas, value):
        canvas.delete("all")
        w = canvas.winfo_width() or 200
        h = canvas.winfo_height() or 18
        ratio = max(0.0, min(1.0, value / canvas._total))
        fill_w = int(w * ratio)
        # 圆角背景
        canvas.create_rectangle(0, 0, w, h, fill="#181825", outline="")
        # 填充条
        if fill_w > 0:
            canvas.create_rectangle(0, 0, fill_w, h, fill=canvas._color, outline="")
        # 百分比文字
        canvas.create_text(w - 8, h / 2, text=f"{value:.0f}%",
                           fill="#cdd6f4", anchor="e",
                           font=("Microsoft YaHei", 9, "bold"))

    # ---------- 刷新逻辑 ----------
    def refresh(self):
        # 跑在子线程里拿数据，避免卡 UI
        t = threading.Thread(target=self._collect_and_update, daemon=True)
        t.start()

    def _collect_and_update(self):
        try:
            gpu = get_gpu_info()
            procs = list_training_processes()
            ckpts = find_ckpt_files(since_hours=24)
            # 回主线程更新 UI
            self.root.after(0, lambda: self._update_ui(gpu, procs, ckpts))
        except Exception as e:
            self.root.after(0, lambda: self.lbl_status.config(
                text=f"⚠ 刷新失败: {e}", fg="#f38ba8"))
        finally:
            # 下一轮刷新
            self.root.after(REFRESH_MS, self.refresh)

    def _update_ui(self, gpu, procs, ckpts):
        now = datetime.now()
        self.lbl_time.config(text=now.strftime("%Y-%m-%d %H:%M:%S"))

        # 监控运行时长
        up = fmt_duration((now - self.start_time).total_seconds())
        self.lbl_monitor_uptime.config(text=f"监控运行: {up}")

        warnings = []

        # GPU
        if gpu and gpu.get("error"):
            # 检测失败：显示友好提示
            self.lbl_gpu_name.config(text="⚠ GPU 检测失败", fg="#f38ba8")
            self.lbl_gpu_pct.config(text="--%", fg="#a6adc8")
            self._draw_progress(self.pb_gpu, 0)
            self.lbl_mem_text.config(text="(无法读取)")
            self._draw_progress(self.pb_mem, 0)
            self.lbl_temp.config(text="温度: -- °C")
            self.lbl_power.config(text="功耗: -- / -- W")
            warnings.append("⚠ " + gpu["error"])
        elif gpu and "name" in gpu and gpu.get("name"):
            self.lbl_gpu_name.config(text=gpu["name"])
            util = gpu["gpu_util"]
            self.lbl_gpu_pct.config(text=f"{util:.0f}%")
            self._draw_progress(self.pb_gpu, util)

            mem_pct = gpu["mem_used_mb"] / max(gpu["mem_total_mb"], 1) * 100
            self.lbl_mem_text.config(
                text=f"{gpu['mem_used_mb']:.0f} / {gpu['mem_total_mb']:.0f} MB  ({mem_pct:.1f}%)")
            self._draw_progress(self.pb_mem, mem_pct)
            if mem_pct > 95:
                warnings.append(f"⚠ 显存 {mem_pct:.0f}%，有 OOM 风险")

            self.lbl_temp.config(text=f"温度: {gpu['temp_c']:.0f} °C")
            if gpu["temp_c"] > 85:
                warnings.append(f"🌡 温度 {gpu['temp_c']:.0f}°C 偏高")
            self.lbl_power.config(text=f"功耗: {gpu['power_w']} / {gpu['power_limit_w']} W")

            # 记录趋势（最多 60 个点）
            self.gpu_history.append(util)
            if len(self.gpu_history) > 60:
                self.gpu_history = self.gpu_history[-60:]
            self._draw_chart()
        else:
            self.lbl_gpu_name.config(text="未检测到 GPU", fg="#f38ba8")
            warnings.append("⚠ 未检测到 GPU（请确认是否为 NVIDIA 显卡且已安装驱动")

        # 进程 + 运行时长
        if procs:
            self.txt_procs.config(state="normal")
            self.txt_procs.delete("1.0", "end")
            # 找主进程（CPU最高的那个）
            main_p = max(procs, key=lambda p: p.get("cpu_s", 0))
            try:
                st = datetime.strptime(main_p["start_time"], "%Y/%m/%d %H:%M:%S")
                runtime = fmt_duration((datetime.now() - st).total_seconds())
                self.lbl_runtime.config(text=f"已运行 {runtime}", fg="#a6e3a1")
            except Exception:
                self.lbl_runtime.config(text="已运行（未知）", fg="#f38ba8")

            # 排序（按 CPU 时间倒序）
            procs_sorted = sorted(procs, key=lambda p: p.get("cpu_s", 0), reverse=True)
            for p in procs_sorted:
                try:
                    st = datetime.strptime(p["start_time"], "%Y/%m/%d %H:%M:%S")
                    rt = fmt_duration((datetime.now() - st).total_seconds())
                except Exception:
                    rt = "—"
                line = (f"  PID {p['pid']:>6}  {p['name']:<18} "
                        f"内存 {p['mem_mb']:>7.0f} MB  CPU {p['cpu_s']/60:>5.1f} 分  "
                        f"运行 {rt}\n")
                self.txt_procs.insert("end", line)
            self.txt_procs.config(state="disabled")
            self.lbl_status.config(text="● 监控中 · 训练正常", fg="#a6e3a1")
        else:
            self.lbl_runtime.config(text="没找到训练进程！", fg="#f38ba8")
            self.txt_procs.config(state="normal")
            self.txt_procs.delete("1.0", "end")
            self.txt_procs.insert("end", "  ❌ 没有 python / weclone 进程 —— 训练可能已经停了")
            self.txt_procs.config(state="disabled")
            self.lbl_status.config(text="⚠ 未发现训练进程", fg="#f38ba8")
            warnings.append("❌ 训练进程未找到")

        # Checkpoint 列表
        self.txt_ckpt.config(state="normal")
        self.txt_ckpt.delete("1.0", "end")
        if not ckpts:
            self.txt_ckpt.insert("end",
                "  ⏳ 还没有 checkpoint / adapter 输出文件，训练还在跑第一个周期...\n"
                "     （通常每训练完一个 epoch 或固定步数才会保存一次）")
        else:
            for p, size_mb, mtime in ckpts[:8]:
                self.txt_ckpt.insert("end",
                    f"  [{mtime.strftime('%H:%M:%S')}]  {size_mb:>7.2f} MB   {p}\n")
            if len(ckpts) > 8:
                self.txt_ckpt.insert("end", f"  ... 还有 {len(ckpts) - 8} 个\n")
            if self.last_ckpt_count >= 0 and len(ckpts) > self.last_ckpt_count:
                self.lbl_status.config(
                    text=f"✨ 新增 checkpoint! 现在共 {len(ckpts)} 个", fg="#f9e2af")
        self.last_ckpt_count = len(ckpts)
        self.txt_ckpt.config(state="disabled")

        # 警告
        self.lbl_warn.config(text="  ".join(warnings))

    # ---------- 画小趋势图 ----------
    def _draw_chart(self):
        self.chart.delete("all")
        w = self.chart.winfo_width() or 300
        h = self.chart.winfo_height() or 80
        if len(self.gpu_history) < 2:
            return
        data = self.gpu_history
        n = len(data)
        # Y = 100 - value（越高利用率越靠上）；X 从右往左画最新在最右
        step = max(1, w // max(n, 1))
        # 先画一条基准线
        self.chart.create_line(0, h - 1, w, h - 1, fill="#45475a", width=1)
        # 画填充折线
        points = []
        for i, v in enumerate(data):
            x = i * step
            y = int(h - 2 - (h - 4) * v / 100)
            points.extend([x, y])
        # 画填充区（从底部到折线）
        fill_pts = [0, h - 2] + points + [(n - 1) * step, h - 2]
        if len(fill_pts) >= 6:
            try:
                self.chart.create_polygon(fill_pts, fill="#a6e3a1",
                                          outline="", stipple="gray25")
            except Exception:
                pass
        # 折线
        if len(points) >= 4:
            self.chart.create_line(*points, fill="#a6e3a1", width=2, smooth=True)


def main():
    root = tk.Tk()
    # 试一下让 DPI 更清晰（Win10+）
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    app = MonitorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
