# 训练实时监控 - Windows 构建脚本
# 用法: powershell -ExecutionPolicy Bypass -File build-windows.ps1
# 产物: dist/train-monitor.exe

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "`n[1/3] 安装/更新 PyInstaller ..." -ForegroundColor Cyan
python -m pip install --upgrade pip
python -m pip install --upgrade pyinstaller

Write-Host "`n[2/3] 清理旧产物 ..." -ForegroundColor Cyan
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue
Remove-Item -Force "train-monitor.spec" -ErrorAction SilentlyContinue

Write-Host "`n[3/3] PyInstaller 打包 (onefile + windowed) ..." -ForegroundColor Cyan
python -m PyInstaller --onefile --windowed --name "train-monitor" --noconfirm monitor_gui.py

$exe = Join-Path $ProjectRoot "dist\train-monitor.exe"
if (Test-Path $exe) {
    $sizeMB = [math]::Round((Get-Item $exe).Length / 1MB, 2)
    Write-Host "`n✅ 构建成功: $exe ($sizeMB MB)`n" -ForegroundColor Green
} else {
    Write-Host "`n❌ 构建失败，未找到 exe" -ForegroundColor Red
    exit 1
}
