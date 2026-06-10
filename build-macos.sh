#!/usr/bin/env bash
# 训练实时监控 - macOS 构建脚本
# 用法: chmod +x build-macos.sh && ./build-macos.sh
# 产物: dist/train-monitor (intel/arm 根据本机架构)

set -euo pipefail
cd "$(dirname "$0")"

echo -e "\n\033[1;36m[1/3] 安装/更新 PyInstaller ...\033[0m"
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade pyinstaller

echo -e "\n\033[1;36m[2/3] 清理旧产物 ...\033[0m"
rm -rf build dist train-monitor.spec

echo -e "\n\033[1;36m[3/3] PyInstaller 打包 (onefile + windowed) ...\033[0m"
python3 -m PyInstaller --onefile --windowed --name "train-monitor" --noconfirm monitor_gui.py

APP_PATH="dist/train-monitor"
if [ -f "$APP_PATH" ]; then
    SIZE_MB=$(du -m "$APP_PATH" | cut -f1)
    ARCH=$(file "$APP_PATH" | grep -oE 'x86_64|arm64|universal' | head -n1)
    echo -e "\n\033[1;32m✅ 构建成功: $APP_PATH ($SIZE_MB MB, arch=$ARCH)\033[0m"
else
    echo -e "\n\033[1;31m❌ 构建失败，未找到产物\033[0m"
    exit 1
fi
