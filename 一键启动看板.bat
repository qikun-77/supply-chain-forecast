@echo off
chcp 65001 >nul
title 📦 电商供应链智能决策系统
echo.
echo ╔══════════════════════════════════════════╗
echo ║   📦 电商供应链需求预测与库存优化系统      ║
echo ║   正在启动...                              ║
echo ╚══════════════════════════════════════════╝
echo.
echo 🔧 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 未找到Python！请先安装Python
    pause
    exit
)

echo 📦 检查依赖...
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo 📥 正在安装Streamlit...
    pip install streamlit -q
)

echo 🚀 启动看板服务器...
echo.
echo ╔══════════════════════════════════════════╗
echo ║  ✅ 浏览器将自动打开                        ║
echo ║  如未自动打开，请访问:                      ║
echo ║  http://localhost:8501                     ║
echo ║  关闭此窗口即可停止服务                     ║
echo ╚══════════════════════════════════════════╝
echo.
start http://localhost:8501
streamlit run 05_dashboard.py --server.headless true
pause