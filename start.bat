@echo off
:: ===========================================================
::  Qwen Image Edit 3D 资产生产工具 - 统一启动脚本 (Windows)
::  功能：虚拟环境检测/创建 + 依赖自动安装 + CPU优化选项 + 启动服务
:: ===========================================================
chcp 65001 >nul
color 0A
pushd "%~dp0"

echo.
echo ===========================================================
echo   🎬 Qwen Image Edit 3D 资产生产工具 - Windows 启动向导
echo ===========================================================
echo.

:: --- Step 1: 检查虚拟环境 ---
if not exist .venv (
    echo [1/3] ⚠️  未检测到虚拟环境，正在创建...
    echo.

    :: 检查 uv 是否安装
    where uv >nul 2>&1
    if %errorlevel% neq 0 (
        echo ❌ 错误：未找到 uv 包管理器
        echo    请先安装 uv: pip install uv 或访问 https://github.com/astral-sh/uv
        pause
        exit /b 1
    )

    echo    正在创建 Python 3.10 虚拟环境...
    uv venv --python 3.10
    if %errorlevel% neq 0 (
        echo ❌ 虚拟环境创建失败
        pause
        exit /b 1
    )
    echo    ✅ 虚拟环境创建完成
) else (
    echo [1/3] ✅ 检测到已有虚拟环境 (.venv)
)

:: --- Step 2: 激活虚拟环境并检查依赖 ---
echo.
echo [2/3] 🔍 检查依赖完整性...

call .venv\Scripts\activate

:: 检查核心依赖是否存在（通过 pip show）
pip show gradio >nul 2>&1
if %errorlevel% neq 0 (
    echo    ⚠️  检测到缺失依赖，正在自动安装...
    echo    (这可能需要几分钟，请耐心等待)
    echo.

    :: 安装 pip（如果缺失）
    pip show pip >nul 2>&1
    if %errorlevel% neq 0 (
        uv pip install pip >nul 2>&1
    )

    :: 安装 requirements.txt 中的所有依赖
    uv pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ❌ 依赖安装失败，请检查 requirements.txt
        pause
        exit /b 1
    )

    :: 检查并安装 rembg（抠图功能依赖）
    pip show rembg >nul 2>&1
    if %errorlevel% neq 0 (
        echo    正在安装 rembg (一键抠图库)...
        uv pip install rembg >nul 2>&1
    )

    echo    ✅ 依赖安装完成
) else (
    echo    ✅ 核心依赖已就绪

    :: 确保 rembg 也存在
    pip show rembg >nul 2>&1
    if %errorlevel% neq 0 (
        echo    正在补装 rembg (一键抠图库)...
        uv pip install rembg >nul 2>&1
    )
)

:: --- Step 3: CPU 优化选项 ---
echo.
echo [3/3] ⚙️  显存优化配置
echo.
echo 请选择是否开启 CPU 内存优化 (模型卸载):
echo   [1] 开启 (默认) - 适合 8G~16G 显存，防止爆显存报错
echo   [2] 关闭        - 适合 24G 及以上大显存 (如 RTX3090/4090)，全速运行
echo.

set OPTION=1
set /p OPTION="请输入你的选择 (1 或 2，按回车默认选 1): "

if "%OPTION%"=="2" (
    set DISABLE_CPU_OFFLOAD=1
    echo    🟢 已选择：关闭 CPU 优化，释放全速性能
) else (
    set DISABLE_CPU_OFFLOAD=0
    echo    🟡 已选择：开启 CPU 优化，保护显存不溢出
)

:: --- 启动服务 ---
echo.
echo ===========================================================
echo   🚀 正在启动 WebUI 服务器...
echo ===========================================================
echo.
echo    访问地址: http://127.0.0.1:6006
echo    (首次启动需下载约 13GB 模型文件)
echo.

python app.py

:: 暂停以便查看错误信息
echo.
echo ===========================================================
echo   服务已停止
echo ===========================================================
pause