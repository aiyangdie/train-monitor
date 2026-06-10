# -*- coding: utf-8 -*-
"""
训练实时监控面板 v1.2.0

核心特性:
  - 自动检测 NVIDIA GPU（nvidia-smi 多级搜索）
  - 自动发现训练进程（Python / WeClone / 自定义关键词）
  - 自动定位 model_output 目录并解析 trainer_log.jsonl
  - 实时显示训练进度、loss、学习率、剩余时间（倒计时）
  - 用户可手动选择项目目录，配置持久化
  - 友好错误诊断：五路独立检测，任一失败不影响其他
  - 零外部依赖：仅标准库 + tkinter
"""
import os
import sys
import json
import time
import threading
import subprocess
import platform
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque

import tkinter as tk
from tkinter import ttk, font, filedialog, messagebox

# ======================================================================
# 全局配置
# ======================================================================
REFRESH_MS = 3000
CHART_POINTS = 60
CONFIG_FILE = Path.home() / ".train_monitor_config.json"

CKPT_KEYWORDS = [
    "checkpoint", "adapter", "safetensors", "pytorch_model",
    "optimizer", "trainer_state", "training_args",
    "weclone", "sft", "lora", "adapter_model",
    ".bin", ".ckpt",
]

TRAIN_PROC_KEYWORDS = [
    "python", "weclone", "train", "finetune", "fine-tune",
    "sft", "lora", "accelerate", "deepspeed", "torch",
]

_NVIDIA_SMI_CANDIDATES = [
    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    r"C:\Windows\System32\nvidia-smi.exe",
    r"C:\Windows\SysWOW64\nvidia-smi.exe",
]

# 训练日志文件名（LLaMA Factory 生成）
_TRAINER_LOG_NAME = "trainer_log.jsonl"
_OUTPUT_DIR_CANDIDATES = ["model_output", "output", "outputs", "checkpoints", "ckpt"]


# ======================================================================
# 基础工具
# ======================================================================
def run_cmd(args, timeout=6):
    try:
        si = None
        if platform.system() == "Windows":
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        r = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="ignore",
            startupinfo=si,
        )
        return (r.stdout or "").strip(), r.returncode
    except Exception:
        return "", -1


def fmt_duration(seconds):
    if seconds is None or seconds < 0:
        return "-"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return "%d时%d分%d秒" % (h, m, sec)
    if m:
        return "%d分%d秒" % (m, sec)
    return "%d秒" % sec


def load_config():
    try:
        if CONFIG_FILE.exists():
            with open(str(CONFIG_FILE), "r", encoding="utf-8", errors="ignore") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def save_config(data):
    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(str(CONFIG_FILE), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ======================================================================
# nvidia-smi 路径搜索
# ======================================================================
def find_nvidia_smi():
    """返回 (path_or_None, message)"""
    p = shutil.which("nvidia-smi")
    if p:
        return p, "ok"
    for c in _NVIDIA_SMI_CANDIDATES:
        if Path(c).exists():
            return c, "ok"
    # 扫描 C:\Program Files\NVIDIA Corporation 下所有含 NVSMI 的目录
    try:
        base = Path(r"C:\Program Files\NVIDIA Corporation")
        if base.exists():
            for sub in base.iterdir():
                try:
                    if not sub.is_dir():
                        continue
                    if "NVSMI" in sub.name.upper():
                        candidate = sub / "nvidia-smi.exe"
                        if candidate.exists():
                            return str(candidate), "ok"
                except Exception:
                    pass
    except Exception:
        pass
    msg = ("未找到 nvidia-smi 可执行文件。\n\n"
           "请尝试：\n"
           "  1) 确认电脑已安装 NVIDIA 显卡驱动\n"
           "  2) 把 nvidia-smi.exe 所在目录加入系统环境变量 PATH\n"
           "  3) 或安装最新版 NVIDIA 驱动")
    return None, msg


# ======================================================================
# GPU 检测
# ======================================================================
def get_gpu_info():
    smi_path, msg = find_nvidia_smi()
    if smi_path is None:
        return {"error": msg}
    out, rc = run_cmd([
        smi_path,
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,"
        "temperature.gpu,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ])
    if not out or rc != 0:
        return {"error": "nvidia-smi 调用失败（返回码 %s）" % rc}
    lines = [l for l in out.splitlines() if l.strip()]
    if not lines:
        return {"error": "nvidia-smi 返回空结果（未检测到 NVIDIA GPU）"}

    gpus = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            def _to_float(s):
                if not s or "[Not" in s.upper() or s.upper() == "N/A":
                    return 0.0
                return float(s)
            def _to_str(s):
                if not s or "[Not" in s.upper() or s.upper() == "N/A":
                    return "?"
                return s
            gpus.append({
                "name": parts[0],
                "gpu_util": _to_float(parts[1]),
                "mem_used_mb": _to_float(parts[2]),
                "mem_total_mb": _to_float(parts[3]),
                "temp_c": _to_float(parts[4]),
                "power_w": _to_str(parts[5]) if len(parts) > 5 else "?",
                "power_limit_w": _to_str(parts[6]) if len(parts) > 6 else "?",
            })
        except Exception:
            continue

    if not gpus:
        return {"error": "nvidia-smi 输出无法解析"}

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
    total_util = sum(g["gpu_util"] for g in gpus) / len(gpus)
    total_mem_used = sum(g["mem_used_mb"] for g in gpus)
    total_mem_total = sum(g["mem_total_mb"] for g in gpus)
    avg_temp = sum(g["temp_c"] for g in gpus) / len(gpus)
    name_str = "、".join(g["name"] for g in gpus)
    return {
        "name": name_str,
        "gpu_util": total_util,
        "mem_used_mb": total_mem_used,
        "mem_total_mb": total_mem_total,
        "temp_c": avg_temp,
        "power_w": "见各卡",
        "power_limit_w": "见各卡",
        "gpu_count": len(gpus),
        "error": None,
    }


# ======================================================================
# 进程检测（Windows 原生 API）
# ======================================================================
def list_training_processes():
    procs = []
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi

        class PROCESSENTRY32(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_void_p),
                ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", wintypes.LONG),
                ("dwFlags", wintypes.DWORD),
                ("szExeFile", ctypes.c_char * 260),
            ]

        class FILETIME(ctypes.Structure):
            _fields_ = [
                ("dwLowDateTime", wintypes.DWORD),
                ("dwHighDateTime", wintypes.DWORD),
            ]

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pe = PROCESSENTRY32()
        pe.dwSize = ctypes.sizeof(PROCESSENTRY32)
        h_snap = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
        if h_snap == -1:
            return procs
        # PROCESS_QUERY_LIMITED_INFORMATION (0x1000) 在 Vista+ 更通用，即使非管理员也能查询时间信息
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        PROCESS_VM_READ = 0x0010
        ok = kernel32.Process32First(h_snap, ctypes.byref(pe))
        while ok:
            try:
                exe_name = pe.szExeFile.decode("gbk", errors="ignore").strip()
            except Exception:
                exe_name = ""
            name_lower = exe_name.lower()
            is_train = any(k in name_lower for k in TRAIN_PROC_KEYWORDS)
            if not is_train:
                ok = kernel32.Process32Next(h_snap, ctypes.byref(pe))
                continue

            pid = pe.th32ProcessID
            mem_mb = 0.0
            cpu_s = 0.0
            start_time = ""
            # 先用 LIMITED 权限（兼容非管理员），失败再用旧权限
            h_proc = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid)
            opened_ok = h_proc
            if not h_proc:
                h_proc = kernel32.OpenProcess(
                    PROCESS_VM_READ | PROCESS_QUERY_LIMITED_INFORMATION, 0, pid
                )
            if h_proc:
                creation = FILETIME()
                exit_t = FILETIME()
                kernel_t = FILETIME()
                user_t = FILETIME()
                try:
                    if kernel32.GetProcessTimes(
                        h_proc,
                        ctypes.byref(creation),
                        ctypes.byref(exit_t),
                        ctypes.byref(kernel_t),
                        ctypes.byref(user_t),
                    ):
                        creation_sec = (
                            (creation.dwHighDateTime * (2**32) + creation.dwLowDateTime)
                            / 10_000_000 - 11644473600
                        )
                        k_s = (
                            (kernel_t.dwHighDateTime * (2**32) + kernel_t.dwLowDateTime)
                            / 10_000_000
                        )
                        u_s = (
                            (user_t.dwHighDateTime * (2**32) + user_t.dwLowDateTime)
                            / 10_000_000
                        )
                        cpu_s = k_s + u_s
                        if creation_sec > 0:
                            start_time = datetime.fromtimestamp(creation_sec).strftime(
                                "%Y/%m/%d %H:%M:%S"
                            )
                except Exception:
                    pass
                try:
                    pmc = PROCESS_MEMORY_COUNTERS()
                    pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
                    if psapi.GetProcessMemoryInfo(
                        h_proc, ctypes.byref(pmc), ctypes.sizeof(pmc)
                    ):
                        mem_mb = float(pmc.WorkingSetSize) / 1024.0 / 1024.0
                except Exception:
                    pass
                kernel32.CloseHandle(h_proc)
            procs.append({
                "pid": str(pid),
                "name": exe_name,
                "cpu_s": cpu_s,
                "mem_mb": mem_mb,
                "start_time": start_time,
            })
            ok = kernel32.Process32Next(h_snap, ctypes.byref(pe))
        kernel32.CloseHandle(h_snap)
        return procs
    except Exception:
        return procs


# ======================================================================
# 训练日志解析 + model_output 定位
# ======================================================================
def _parse_hms(s):
    """解析 'H:MM:SS' 或 'HH:MM:SS' 字符串为秒数"""
    try:
        parts = s.strip().split(":")
        if len(parts) == 3:
            h, m, sec = parts
            return int(h) * 3600 + int(m) * 60 + int(sec)
        if len(parts) == 2:
            m, sec = parts
            return int(m) * 60 + int(sec)
    except Exception:
        pass
    return None


def _format_hms(total_seconds):
    if total_seconds is None or total_seconds < 0:
        return "--:--:--"
    total = int(total_seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return "%02d:%02d:%02d" % (h, m, s)


def find_trainer_log(base_dirs):
    """在多个目录下查找 trainer_log.jsonl。返回路径或 None，并附带最后一次修改时间"""
    candidates = []
    for d in base_dirs:
        try:
            p = Path(d)
            if not p.exists():
                continue
            # 直接搜
            f = p / _TRAINER_LOG_NAME
            if f.exists():
                candidates.append(f)
            # 在子目录里搜
            for subname in _OUTPUT_DIR_CANDIDATES:
                sub = p / subname
                if sub.exists():
                    ff = sub / _TRAINER_LOG_NAME
                    if ff.exists():
                        candidates.append(ff)
                    # 再深一层
                    for deep in sub.rglob(_TRAINER_LOG_NAME):
                        candidates.append(deep)
            # 深度搜索
            try:
                for deep in p.rglob(_TRAINER_LOG_NAME):
                    candidates.append(deep)
            except Exception:
                pass
        except Exception:
            pass
    if not candidates:
        return None
    # 去重 + 选最新修改的
    seen = set()
    uniq = []
    for c in candidates:
        try:
            k = str(c.resolve())
            if k in seen:
                continue
            seen.add(k)
            uniq.append(c)
        except Exception:
            pass
    uniq.sort(key=lambda x: x.stat().st_mtime if x.exists() else 0, reverse=True)
    return uniq[0]


def parse_training_progress(log_path, history_max=30):
    """
    解析 trainer_log.jsonl，返回最新训练进度数据。
    - 会区分不同 total_steps 的运行，取"最新一次运行"的最新记录
    - 返回 dict: {latest_rows, log_path, error, current_steps, total_steps, percentage,
                   elapsed_time, remaining_time, elapsed_sec, remaining_sec,
                   loss, lr, epoch, it_s, etas, finish_time_est}
    """
    try:
        lp = Path(log_path)
        if not lp.exists():
            return {"error": "未找到 trainer_log.jsonl", "log_path": str(lp)}

        rows = []
        with open(str(lp), "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue

        if not rows:
            return {"error": "trainer_log.jsonl 为空", "log_path": str(lp)}

        # 按 total_steps 分组（不同训练运行可能混在一个文件里）
        groups = {}
        for r in rows:
            try:
                ts = r["total_steps"]
                groups.setdefault(ts, []).append(r)
            except Exception:
                pass

        if not groups:
            return {"error": "日志格式异常（缺 total_steps）", "log_path": str(lp)}

        # 选最新一组：按各组最后一条记录的 current_steps 最大的那个
        best_key = max(groups.keys(), key=lambda k: groups[k][-1].get("current_steps", 0))
        latest_rows = groups[best_key]

        last = latest_rows[-1]
        # 基础字段
        current_steps = int(last.get("current_steps", 0))
        total_steps = int(last.get("total_steps", 0))
        percentage = float(last.get("percentage", 0))
        loss = float(last.get("loss", 0))
        lr = float(last.get("lr", 0))
        epoch = float(last.get("epoch", 0)) if last.get("epoch") is not None else 0

        # 解析 elapsed_time / remaining_time 字符串
        elapsed_sec = _parse_hms(last.get("elapsed_time", "")) or 0
        remaining_sec = _parse_hms(last.get("remaining_time", "")) or None

        # 计算 it/s（取最近 30 条 + 第一条做基准，排除冷启动阶段的速度波动）
        it_s = None
        etas = None
        if len(latest_rows) >= 2:
            try:
                # 用第一条 vs 最后一条算整体平均速度
                steps_delta = int(latest_rows[-1]["current_steps"]) - int(latest_rows[0]["current_steps"])
                t0 = _parse_hms(latest_rows[0].get("elapsed_time", "")) or 0
                t1 = _parse_hms(latest_rows[-1].get("elapsed_time", "")) or 0
                time_delta = t1 - t0
                if steps_delta > 0 and time_delta > 0:
                    it_s = steps_delta / time_delta
                    remaining_steps = total_steps - current_steps
                    if it_s > 0:
                        etas = remaining_steps / it_s
            except Exception:
                pass

        # 估算结束时间（本地时间）
        finish_time_est = None
        if etas:
            try:
                finish_time_est = datetime.now() + timedelta(seconds=etas)
            except Exception:
                pass

        # 最近 history_max 条 loss/百分比，用于趋势
        history = []
        for r in latest_rows[-history_max:]:
            try:
                history.append({
                    "percentage": float(r.get("percentage", 0)),
                    "loss": float(r.get("loss", 0)),
                    "lr": float(r.get("lr", 0)),
                    "elapsed": _parse_hms(r.get("elapsed_time", "")) or 0,
                })
            except Exception:
                continue

        return {
            "error": None,
            "log_path": str(lp),
            "current_steps": current_steps,
            "total_steps": total_steps,
            "percentage": percentage,
            "elapsed_time": last.get("elapsed_time", "--"),
            "remaining_time": last.get("remaining_time", "--"),
            "elapsed_sec": elapsed_sec,
            "remaining_sec": remaining_sec,
            "loss": loss,
            "lr": lr,
            "epoch": epoch,
            "it_s": it_s,
            "etas": etas,
            "finish_time_est": finish_time_est,
            "history": history,
            "total_rows": len(latest_rows),
        }
    except Exception as e:
        return {"error": "解析日志异常: %s" % e, "log_path": str(log_path)}


# ======================================================================
# 项目目录识别 + Checkpoint 搜索
# ======================================================================
def _scan_dir_for_projects(base_path, max_depth=3, max_total=30):
    """在 base_path 下搜索可能的项目目录（含"数字人/weclone/train/model_output"等关键词）
    - 无论是否匹配关键词，都继续向下（最大 max_depth）
    - 匹配关键词的目录加入结果
    - 结果数量限制 max_total，避免在大型目录下太慢
    """
    results = []
    try:
        base = Path(base_path)
        if not base.exists():
            return results
        # 检查 base 自身是否在跳过列表
        base_name_lower = base.name.lower()
        skip_base = [".git", "$recycle.bin", "system volume information",
                     "program files", "programdata", "windows",
                     "appdata", "temp", "tmp", "cache", "node_modules",
                     "dist-packages", "site-packages", ".venv", "venv",
                     "__pycache__", "assets", "public", "software",
                     "games", "game", "steam", "tencent",
                     "wechat", "thunder", "bilibili", "baidu",
                     "uucloud", "music", "picture",
                     "电影", "视频", "音乐", "图片", "$av", "recycler",
                     "pagefile.sys", "swapfile.sys", "hiberfil.sys",
                     "$windows.~bt", "bootmgr", "msocache", "intel",
                     "nvidia", "amd", "perflogs", "recovery", "$winre",
                     "valorant", "onmyoji", "riot games", "lol",
                     "league of legends", "honkai", "原神", "王者荣耀"]
        if any(s in base_name_lower for s in skip_base):
            return results
        queue = [(base, 0)]
        visited = set()
        found_count = 0
        while queue and found_count < max_total:
            current, depth = queue.pop(0)
            try:
                key = str(current.resolve())
                if key in visited:
                    continue
                visited.add(key)
                name = current.name.lower()
                matched = False
                if depth > 0:
                    for kw in ["数字人", "weclone", "train", "model_output", "output",
                               "sft", "lora", "llm", "finetune", "digital_human"]:
                        if kw in name:
                            matched = True
                            break
                if matched:
                    results.append(current)
                    found_count += 1
                    sub = current / "model_output"
                    if sub.exists():
                        results.append(sub)
                        found_count += 1
                if depth < max_depth:
                    try:
                        for child in current.iterdir():
                            try:
                                if not child.is_dir():
                                    continue
                                cname = child.name.lower()
                                # 第一层遇到系统/游戏目录直接跳过
                                if depth == 0 and any(s in cname for s in skip_base):
                                    continue
                                if any(sk in cname for sk in skip_base):
                                    continue
                                queue.append((child, depth + 1))
                                if len(queue) > 100:
                                    break
                            except Exception:
                                pass
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception:
        pass
    return results


_DIR_CACHE_FILE = Path.home() / ".train_monitor_dir_cache.json"
_DIR_CACHE_TTL = 3600  # 1 小时

def _load_dir_cache():
    try:
        if _DIR_CACHE_FILE.exists():
            import time as _t
            data = json.loads(_DIR_CACHE_FILE.read_text(encoding="utf-8"))
            if _t.time() - data.get("ts", 0) < _DIR_CACHE_TTL:
                return data.get("dirs", [])
    except Exception:
        pass
    return None

def _save_dir_cache(dirs):
    try:
        import time as _t
        data = {"ts": _t.time(), "dirs": [str(d) for d in dirs]}
        _DIR_CACHE_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def detect_project_dirs(custom_dir=None):
    """返回候选项目目录列表。优先用户指定目录，其次自动扫描常见位置。带 1 小时缓存。"""
    # 用户自定义目录：不用缓存
    if custom_dir:
        dirs = []
        seen = set()

        def _add(d_path):
            try:
                p = Path(d_path)
                if not p.exists():
                    return
                key = str(p.resolve())
                if key in seen:
                    return
                seen.add(key)
                dirs.append(p)
            except Exception:
                pass

        _add(custom_dir)
        return dirs

    # 尝试读缓存
    cached = _load_dir_cache()
    if cached is not None:
        # 把缓存的字符串恢复成 Path
        return [Path(p) for p in cached if Path(p).exists()]

    dirs = []
    seen = set()

    def _add(d_path):
        try:
            p = Path(d_path)
            if not p.exists():
                return
            key = str(p.resolve())
            if key in seen:
                return
            seen.add(key)
            dirs.append(p)
        except Exception:
            pass

    if custom_dir:
        _add(custom_dir)
    try:
        script_dir = Path(__file__).resolve().parent
        _add(script_dir)
        _add(script_dir.parent)
        try:
            _add(script_dir.parent.parent)
        except Exception:
            pass
    except Exception:
        pass
    _add(Path.cwd())
    _add(Path.cwd().parent)
    _add(Path.home() / "Documents")
    _add(Path.home() / "Desktop")
    _add(Path.home() / "Downloads")

    # 用户主目录下 2 层深度搜索
    for base in [
        Path.home(),
        Path.home() / "Documents",
        Path.home() / "Downloads",
        Path.home() / "Desktop",
    ]:
        try:
            for p in _scan_dir_for_projects(base, max_depth=2, max_total=15):
                _add(p)
        except Exception:
            pass

    # C 盘：仅扫主目录（避免扫 Windows/Program Files 太慢）
    try:
        for p in _scan_dir_for_projects(Path.home(), max_depth=3, max_total=15):
            _add(p)
    except Exception:
        pass

    # D/E/F/G/H 盘：从盘根扫 3 层（下载文件夹通常在数据盘根下）
    try:
        for drive_letter in "DEFGH":
            try:
                drive = Path(drive_letter + ":/")
                if not drive.exists():
                    continue
                # 第一层排除系统/游戏/缓存目录
                for sub in drive.iterdir():
                    try:
                        if not sub.is_dir():
                            continue
                        cname = sub.name.lower()
                        skip_top = ["$recycle.bin", "system volume information",
                                    "program files", "programdata", "windows",
                                    "appdata", "system32", "syswow64",
                                    "recycler", "recovery", "$winre", "perflogs",
                                    "msocache", "intel", "amd", "nvidia",
                                    ".git", "node_modules", "__pycache__",
                                    "pagefile.sys", "swapfile.sys", "hiberfil.sys",
                                    "config", "$windows.~bt", "boot", "bootmgr"]
                        if any(s in cname for s in skip_top):
                            continue
                        # 在这里深入扫描 3 层
                        for deep in _scan_dir_for_projects(sub, max_depth=3, max_total=15):
                            _add(deep)
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception:
        pass
    # 保存到缓存（避免下次启动全量扫描）
    try:
        _save_dir_cache(dirs)
    except Exception:
        pass
    return dirs


def find_ckpt_files(search_dirs, since_hours=24):
    cutoff = datetime.now() - timedelta(hours=since_hours)
    found = []
    seen_paths = set()
    for d in search_dirs:
        try:
            d_path = Path(d)
            if not d_path.exists():
                continue
            for p in d_path.rglob("*"):
                try:
                    if p.is_dir():
                        continue
                    sp = str(p.resolve())
                    if sp in seen_paths:
                        continue
                    lower = p.name.lower()
                    if not any(k in lower for k in CKPT_KEYWORDS):
                        continue
                    try:
                        st = p.stat()
                        mtime = st.st_mtime
                        if mtime < cutoff.timestamp():
                            continue
                        size_mb = float(st.st_size) / 1024.0 / 1024.0
                    except Exception:
                        continue
                    if size_mb < 0.5 and "checkpoint" not in lower \
                            and "adapter" not in lower and "safetensors" not in lower:
                        continue
                    found.append((p, size_mb, datetime.fromtimestamp(mtime)))
                    seen_paths.add(sp)
                except Exception:
                    pass
        except Exception:
            pass
    found.sort(key=lambda x: x[2], reverse=True)
    return found[:12]


# ======================================================================
# GUI
# ======================================================================
class MonitorApp:
    BG_DARK = "#1e1e2e"
    BG_CARD = "#313244"
    BG_HEADER = "#11111b"
    BG_BAR = "#181825"
    FG_MAIN = "#cdd6f4"
    FG_SUB = "#a6adc8"
    FG_ACCENT = "#f9e2af"
    FG_BLUE = "#89b4fa"
    FG_RED = "#f38ba8"
    FG_GREEN = "#a6e3a1"
    FG_ORANGE = "#fab387"
    FG_PINK = "#f5c2e7"

    def __init__(self, root):
        self.root = root
        self.root.title("训练实时监控面板 v1.2.0")
        self.root.geometry("900x680")
        self.root.minsize(780, 580)
        self.root.configure(bg=self.BG_DARK)
        self.start_time = datetime.now()
        self.gpu_history = deque(maxlen=CHART_POINTS)

        self.cfg = load_config()
        self.custom_project_dir = self.cfg.get("project_dir", "")
        self._build_ui()
        self.root.after(500, self.refresh)

    def _build_ui(self):
        # 顶部栏
        header = tk.Frame(self.root, bg=self.BG_HEADER, height=72)
        header.pack(fill="x")

        tk.Label(
            header,
            text="训练实时监控",
            fg=self.FG_ACCENT,
            bg=self.BG_HEADER,
            font=("Microsoft YaHei", 16, "bold"),
        ).pack(side="left", padx=20, pady=18)

        dir_frame = tk.Frame(header, bg=self.BG_HEADER)
        dir_frame.pack(side="right", padx=10, pady=12)
        tk.Label(
            dir_frame,
            text="项目目录:",
            fg=self.FG_SUB,
            bg=self.BG_HEADER,
            font=("Microsoft YaHei", 9),
        ).pack(side="left")

        display_text = self.custom_project_dir if self.custom_project_dir else "(自动检测)"
        self.lbl_project = tk.Label(
            dir_frame,
            text=display_text,
            fg=self.FG_BLUE,
            bg=self.BG_DARK,
            anchor="w",
            width=48,
            font=("Microsoft YaHei", 9),
            padx=8,
            pady=4,
        )
        self.lbl_project.pack(side="left", padx=6)

        btn_choose = tk.Button(
            dir_frame,
            text="选择...",
            command=self._on_choose_dir,
            bg=self.BG_CARD,
            fg=self.FG_MAIN,
            relief="flat",
            activebackground=self.BG_BAR,
            borderwidth=0,
            padx=12,
            pady=4,
            cursor="hand2",
        )
        btn_choose.pack(side="left", padx=4)
        btn_auto = tk.Button(
            dir_frame,
            text="自动检测",
            command=self._on_auto_detect,
            bg=self.BG_CARD,
            fg=self.FG_MAIN,
            relief="flat",
            activebackground=self.BG_BAR,
            borderwidth=0,
            padx=12,
            pady=4,
            cursor="hand2",
        )
        btn_auto.pack(side="left")

        # 主体
        body = tk.Frame(self.root, bg=self.BG_DARK)
        body.pack(fill="both", expand=True, padx=10, pady=10)

        # ----- 左侧 GPU 卡片
        left = tk.Frame(body, bg=self.BG_CARD, padx=15, pady=15)
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))

        tk.Label(
            left,
            text="GPU 状态",
            fg=self.FG_BLUE,
            bg=self.BG_CARD,
            font=("Microsoft YaHei", 12, "bold"),
            anchor="w",
        ).pack(fill="x")
        self.lbl_gpu_name = tk.Label(
            left,
            text="检测中...",
            fg=self.FG_MAIN,
            bg=self.BG_CARD,
            font=("Microsoft YaHei", 11, "bold"),
            anchor="w",
            justify="left",
        )
        self.lbl_gpu_name.pack(fill="x", pady=(2, 10))

        self.lbl_gpu_pct = tk.Label(
            left, text="--%", fg=self.FG_GREEN, bg=self.BG_CARD,
            font=("Microsoft YaHei", 26, "bold"),
        )
        self.lbl_gpu_pct.pack(anchor="w", pady=(0, 4))
        self.pb_gpu = self._mk_progress(left, total=100, color_fill=self.FG_GREEN)
        self.pb_gpu.pack(fill="x", pady=(0, 12))

        tk.Label(
            left,
            text="显存 (已用 / 总量)",
            fg=self.FG_ORANGE,
            bg=self.BG_CARD,
            font=("Microsoft YaHei", 10),
            anchor="w",
        ).pack(fill="x")
        self.lbl_mem_text = tk.Label(
            left,
            text="-- / -- MB (--%)",
            fg=self.FG_MAIN,
            bg=self.BG_CARD,
            font=("Microsoft YaHei", 11),
        )
        self.lbl_mem_text.pack(anchor="w", pady=(0, 4))
        self.pb_mem = self._mk_progress(left, total=100, color_fill=self.FG_ORANGE)
        self.pb_mem.pack(fill="x", pady=(0, 12))

        row = tk.Frame(left, bg=self.BG_CARD)
        row.pack(fill="x", pady=4)
        self.lbl_temp = tk.Label(
            row,
            text="温度: -- °C",
            fg=self.FG_ACCENT,
            bg=self.BG_CARD,
            font=("Microsoft YaHei", 11),
            anchor="w",
        )
        self.lbl_temp.pack(side="left", expand=True, fill="x")
        self.lbl_power = tk.Label(
            row,
            text="功耗: -- / -- W",
            fg="#cba6f7",
            bg=self.BG_CARD,
            font=("Microsoft YaHei", 11),
            anchor="w",
        )
        self.lbl_power.pack(side="left", expand=True, fill="x")

        tk.Label(
            left,
            text="GPU 利用率趋势（最近 %d 次刷新）" % CHART_POINTS,
            fg=self.FG_SUB,
            bg=self.BG_CARD,
            font=("Microsoft YaHei", 9),
        ).pack(anchor="w", pady=(14, 2))
        self.chart = tk.Canvas(left, height=90, bg=self.BG_BAR, highlightthickness=0)
        self.chart.pack(fill="x", pady=2)
        self.chart.bind("<Configure>", lambda e: self._draw_chart())

        # ----- 右侧 训练状态
        right = tk.Frame(body, bg=self.BG_CARD, padx=15, pady=15)
        right.pack(side="left", fill="both", expand=True, padx=(5, 0))

        tk.Label(
            right,
            text="训练状态（倒计时）",
            fg=self.FG_BLUE,
            bg=self.BG_CARD,
            font=("Microsoft YaHei", 12, "bold"),
            anchor="w",
        ).pack(fill="x")

        # 大字显示剩余时间
        self.lbl_remaining = tk.Label(
            right,
            text="剩余 --:--:--",
            fg=self.FG_GREEN,
            bg=self.BG_CARD,
            font=("Microsoft YaHei", 22, "bold"),
        )
        self.lbl_remaining.pack(anchor="w", pady=(4, 2))

        # 训练进度百分比 + 总步数
        self.lbl_progress = tk.Label(
            right,
            text="进度 --% (0 / 0 步)",
            fg=self.FG_MAIN,
            bg=self.BG_CARD,
            font=("Microsoft YaHei", 11, "bold"),
        )
        self.lbl_progress.pack(anchor="w", pady=(0, 4))

        # 进度条
        self.pb_train = self._mk_progress(right, total=100, color_fill=self.FG_GREEN, height=20)
        self.pb_train.pack(fill="x", pady=(0, 10))

        # 已用时间 + 预计结束时间
        self.lbl_timeinfo = tk.Label(
            right,
            text="已运行 --   预计结束 --",
            fg=self.FG_MAIN,
            bg=self.BG_CARD,
            font=("Microsoft YaHei", 10),
        )
        self.lbl_timeinfo.pack(anchor="w")

        # loss / lr / it/s
        self.lbl_metrics = tk.Label(
            right,
            text="loss --    lr --    it/s --    epoch --",
            fg=self.FG_ACCENT,
            bg=self.BG_CARD,
            font=("Microsoft YaHei", 10),
        )
        self.lbl_metrics.pack(anchor="w", pady=(4, 8))

        # 日志路径提示
        self.lbl_logpath = tk.Label(
            right,
            text="",
            fg=self.FG_SUB,
            bg=self.BG_CARD,
            font=("Microsoft YaHei", 8),
            wraplength=380,
            justify="left",
        )
        self.lbl_logpath.pack(anchor="w", pady=(0, 4))

        # 进程列表
        tk.Label(
            right,
            text="进程列表",
            fg=self.FG_PINK,
            bg=self.BG_CARD,
            font=("Microsoft YaHei", 11),
            anchor="w",
        ).pack(fill="x", pady=(6, 2))
        self.txt_procs = tk.Text(
            right,
            height=5,
            bg=self.BG_BAR,
            fg=self.FG_MAIN,
            font=("Consolas", 10),
            relief="flat",
            wrap="word",
        )
        self.txt_procs.pack(fill="x")

        tk.Label(
            right,
            text="Checkpoint / 模型输出（最近 24 小时）",
            fg=self.FG_PINK,
            bg=self.BG_CARD,
            font=("Microsoft YaHei", 11),
            anchor="w",
        ).pack(fill="x", pady=(12, 2))
        self.txt_ckpt = tk.Text(
            right,
            height=6,
            bg=self.BG_BAR,
            fg=self.FG_MAIN,
            font=("Consolas", 10),
            relief="flat",
            wrap="word",
        )
        self.txt_ckpt.pack(fill="both", expand=True, pady=(0, 4))

        # 底部状态栏
        bottom = tk.Frame(self.root, bg=self.BG_HEADER, height=40)
        bottom.pack(fill="x", side="bottom")
        self.lbl_status = tk.Label(
            bottom,
            text="监控中",
            fg=self.FG_GREEN,
            bg=self.BG_HEADER,
            font=("Microsoft YaHei", 10),
        )
        self.lbl_status.pack(side="left", padx=16, pady=10)
        self.lbl_warn = tk.Label(
            bottom,
            text="",
            fg=self.FG_RED,
            bg=self.BG_HEADER,
            font=("Microsoft YaHei", 10, "bold"),
            wraplength=600,
            justify="left",
        )
        self.lbl_warn.pack(side="right", padx=16, pady=10)
        self.lbl_monitor_uptime = tk.Label(
            bottom,
            text="",
            fg=self.FG_SUB,
            bg=self.BG_HEADER,
            font=("Microsoft YaHei", 10),
        )
        self.lbl_monitor_uptime.pack(side="right", padx=16, pady=10)

    def _mk_progress(self, parent, total, color_fill, height=16):
        canvas = tk.Canvas(parent, height=height, bg=self.BG_BAR,
                           highlightthickness=0, bd=0)
        canvas._total = total
        canvas._color = color_fill
        canvas._value = 0.0
        canvas.bind("<Configure>", lambda e: self._draw_progress(canvas))
        return canvas

    def _draw_progress(self, canvas):
        canvas.delete("all")
        w = canvas.winfo_width() or 200
        h = canvas.winfo_height() or 16
        ratio = max(0.0, min(1.0, getattr(canvas, "_value", 0) / canvas._total))
        fill_w = int(w * ratio)
        canvas.create_rectangle(0, 0, w, h, fill=self.BG_BAR, outline="")
        if fill_w > 0:
            canvas.create_rectangle(0, 0, fill_w, h, fill=canvas._color, outline="")
        canvas.create_text(
            w - 8, h / 2,
            text="%.0f%%" % getattr(canvas, "_value", 0),
            fill=self.FG_MAIN,
            anchor="e",
            font=("Microsoft YaHei", 9, "bold"),
        )

    def _on_choose_dir(self):
        d = filedialog.askdirectory(title="选择项目根目录（训练脚本所在目录）")
        if d:
            self.custom_project_dir = d
            self.lbl_project.config(text=d)
            cfg = load_config()
            cfg["project_dir"] = d
            save_config(cfg)
            messagebox.showinfo(
                "已保存",
                "项目目录已设置为：\n%s\n\n下次启动将自动使用此目录。" % d,
            )

    def _on_auto_detect(self):
        self.custom_project_dir = ""
        cfg = load_config()
        cfg.pop("project_dir", None)
        save_config(cfg)
        self.lbl_project.config(text="(自动检测)")

    # ---------- 刷新逻辑
    def refresh(self):
        t = threading.Thread(target=self._collect_and_update, daemon=True)
        t.start()

    def _collect_and_update(self):
        gpu_info = None
        procs = []
        ckpts = []
        dirs_searched = []
        progress = None
        try:
            gpu_info = get_gpu_info()
        except Exception as e:
            gpu_info = {"error": "GPU 检测异常: %s" % e}
        try:
            procs = list_training_processes()
        except Exception:
            procs = []
        try:
            if self.custom_project_dir:
                dirs_searched = [Path(self.custom_project_dir)]
            else:
                dirs_searched = detect_project_dirs()
            # checkpoint 搜索
            ckpts = find_ckpt_files(dirs_searched, since_hours=24)
            # 训练日志解析
            log_path = find_trainer_log(dirs_searched)
            if log_path:
                progress = parse_training_progress(log_path)
            else:
                progress = {
                    "error": "未找到 trainer_log.jsonl（请在顶部选择项目目录）",
                    "log_path": None,
                }
        except Exception as e:
            progress = {"error": "查找日志异常: %s" % e, "log_path": None}
        try:
            self.root.after(
                0,
                lambda: self._update_ui(gpu_info, procs, progress, ckpts, dirs_searched),
            )
        finally:
            self.root.after(REFRESH_MS, self.refresh)

    def _update_ui(self, gpu, procs, progress, ckpts, dirs_searched):
        now = datetime.now()
        up = fmt_duration((now - self.start_time).total_seconds())
        self.lbl_monitor_uptime.config(text="监控运行: %s" % up)

        warnings = []

        # GPU
        if gpu and gpu.get("error"):
            self.lbl_gpu_name.config(
                text="GPU 检测失败\n" + gpu["error"],
                fg=self.FG_RED,
            )
            self.lbl_gpu_pct.config(text="--%", fg=self.FG_SUB)
            self.pb_gpu._value = 0.0
            self._draw_progress(self.pb_gpu)
            self.lbl_mem_text.config(text="(无法读取)")
            self.pb_mem._value = 0.0
            self._draw_progress(self.pb_mem)
            self.lbl_temp.config(text="温度: -- C")
            self.lbl_power.config(text="功耗: -- / -- W")
            warnings.append("GPU 检测失败")
        elif gpu and gpu.get("name"):
            cnt = gpu.get("gpu_count", 1)
            name_text = gpu["name"]
            if cnt > 1:
                name_text = "%s (共 %d 张卡)" % (name_text, cnt)
            self.lbl_gpu_name.config(text=name_text, fg=self.FG_MAIN)
            util = gpu["gpu_util"]
            self.lbl_gpu_pct.config(text="%.0f%%" % util, fg=self.FG_GREEN)
            self.pb_gpu._value = util
            self._draw_progress(self.pb_gpu)
            mem_pct = gpu["mem_used_mb"] / max(gpu["mem_total_mb"], 1) * 100
            self.lbl_mem_text.config(
                text="%.0f / %.0f MB (%.1f%%)" % (
                    gpu["mem_used_mb"], gpu["mem_total_mb"], mem_pct,
                )
            )
            self.pb_mem._value = mem_pct
            self._draw_progress(self.pb_mem)
            if mem_pct > 95:
                warnings.append("显存 %.0f%%，有 OOM 风险" % mem_pct)
            self.lbl_temp.config(text="温度: %.0f C" % gpu["temp_c"])
            if gpu["temp_c"] > 85:
                warnings.append("温度 %.0fC 偏高" % gpu["temp_c"])
            self.lbl_power.config(
                text="功耗: %s / %s W" % (gpu["power_w"], gpu["power_limit_w"])
            )
            self.gpu_history.append(util)
            self._draw_chart()

        # 训练进度 / 倒计时（从 trainer_log.jsonl 解析）
        if progress and not progress.get("error") and progress.get("total_steps", 0) > 0:
            # 优先使用训练框架自己写入的 remaining_time（更准）
            remaining_sec = progress.get("remaining_sec")
            if not remaining_sec:
                # 退而求其次：用我们自己基于最近速度的估算
                remaining_sec = progress.get("etas")
            remaining_text = _format_hms(remaining_sec)
            self.lbl_remaining.config(
                text="剩余 %s" % remaining_text,
                fg=self.FG_GREEN,
            )
            # 进度条
            pct = progress.get("percentage", 0)
            self.lbl_progress.config(
                text="进度 %.2f%% (%d / %d 步)"
                % (pct, progress.get("current_steps", 0), progress.get("total_steps", 0)),
                fg=self.FG_MAIN,
            )
            self.pb_train._value = pct
            self._draw_progress(self.pb_train)

            # 已用时间 + 预计结束时间
            elapsed = progress.get("elapsed_time", "--")
            finish = progress.get("finish_time_est")
            finish_text = finish.strftime("%Y-%m-%d %H:%M:%S") if finish else "--"
            self.lbl_timeinfo.config(
                text="已运行 %s   |   预计结束 %s" % (elapsed, finish_text),
                fg=self.FG_MAIN,
            )

            # loss / lr / it/s / epoch
            loss = progress.get("loss", 0)
            lr = progress.get("lr", 0)
            it_s = progress.get("it_s")
            epoch = progress.get("epoch", 0)
            it_s_text = "%.2f" % it_s if it_s else "--"
            self.lbl_metrics.config(
                text="loss %.4f    lr %.2e    it/s %s    epoch %.2f"
                % (loss, lr, it_s_text, epoch),
                fg=self.FG_ACCENT,
            )

            # 日志路径
            lp = progress.get("log_path") or ""
            self.lbl_logpath.config(text="日志: %s" % lp)

        else:
            # 无日志或解析失败
            self.lbl_remaining.config(text="剩余 --:--:--", fg=self.FG_SUB)
            self.lbl_progress.config(text="进度 --% (0 / 0 步)", fg=self.FG_SUB)
            self.pb_train._value = 0.0
            self._draw_progress(self.pb_train)
            self.lbl_timeinfo.config(text="已运行 --   预计结束 --", fg=self.FG_SUB)
            self.lbl_metrics.config(text="loss --    lr --    it/s --    epoch --", fg=self.FG_SUB)
            msg = ""
            if progress and progress.get("error"):
                msg = progress["error"]
            if not msg:
                msg = "未发现训练日志。请在顶部选择项目目录（训练脚本所在目录）"
            self.lbl_logpath.config(text="提示: " + msg)
            warnings.append("训练日志未找到")

        # 进程
        if procs:
            self.txt_procs.config(state="normal")
            self.txt_procs.delete("1.0", "end")
            for p in sorted(procs, key=lambda p: p.get("cpu_s", 0), reverse=True):
                try:
                    st = datetime.strptime(p["start_time"], "%Y/%m/%d %H:%M:%S")
                    rt = fmt_duration((datetime.now() - st).total_seconds())
                except Exception:
                    rt = "-"
                line = (
                    "  PID %6s  %-22s  内存 %7.0f MB  CPU %5.1f 分  运行 %s\n"
                    % (p["pid"], p["name"], p["mem_mb"], p["cpu_s"] / 60.0, rt)
                )
                self.txt_procs.insert("end", line)
            self.txt_procs.config(state="disabled")
            self.lbl_status.config(text="监控中 - 训练正常", fg=self.FG_GREEN)
        else:
            self.txt_procs.config(state="normal")
            self.txt_procs.delete("1.0", "end")
            self.txt_procs.insert(
                "end",
                "  未发现 python / weclone 训练进程。\n"
                "  如果训练已停止或尚未启动，这是正常现象。\n"
                "  （你也可以在顶部手动指定项目目录来搜索 checkpoint）",
            )
            self.txt_procs.config(state="disabled")
            self.lbl_status.config(text="未发现训练进程", fg=self.FG_RED)
            warnings.append("未发现训练进程")

        # Checkpoint
        self.txt_ckpt.config(state="normal")
        self.txt_ckpt.delete("1.0", "end")
        if not ckpts:
            hint = "  尚未发现 checkpoint / adapter 输出文件。\n"
            hint += "  （训练可能还未完成第一次保存，或尚未开始训练。）\n"
            hint += "  搜索目录: " + "、".join(str(d) for d in (dirs_searched or []))
            self.txt_ckpt.insert("end", hint)
        else:
            for p, size_mb, mtime in ckpts:
                self.txt_ckpt.insert(
                    "end",
                    "  [%s]  %7.2f MB   %s\n" % (
                        mtime.strftime("%H:%M:%S"), size_mb, str(p),
                    ),
                )
            if len(ckpts) > 12:
                self.txt_ckpt.insert("end", "  ... 还有更多未展示\n")
        self.txt_ckpt.config(state="disabled")

        self.lbl_warn.config(text="  ".join(warnings))

    def _draw_chart(self):
        self.chart.delete("all")
        w = self.chart.winfo_width() or 300
        h = self.chart.winfo_height() or 90
        if len(self.gpu_history) < 2:
            return
        data = list(self.gpu_history)
        n = len(data)
        step = max(1, w // max(n, 1))
        self.chart.create_line(0, h - 1, w, h - 1, fill="#45475a", width=1)
        points = []
        for i, v in enumerate(data):
            x = i * step
            y = int(h - 2 - (h - 4) * v / 100.0)
            points.extend([x, y])
        fill_pts = [0, h - 2] + points + [(n - 1) * step, h - 2]
        if len(fill_pts) >= 6:
            try:
                self.chart.create_polygon(
                    fill_pts, fill=self.FG_GREEN, outline="", stipple="gray25"
                )
            except Exception:
                pass
        if len(points) >= 4:
            try:
                self.chart.create_line(
                    *points, fill=self.FG_GREEN, width=2, smooth=True
                )
            except Exception:
                pass


def main():
    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    app = MonitorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
