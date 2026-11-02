@echo off
setlocal
cd /d "%~dp0"

rem Windows 原生数据导入入口：不依赖 WSL/bash。
rem 默认数据根 = scripts 的上一级（本机 D:\paper-factor-data）；
rem 复制到 G 盘时也自动使用 G:\paper-factor-data。
if not defined PAPER_FACTOR_DATA_ROOT (
    for %%I in ("%~dp0..") do set "PAPER_FACTOR_DATA_ROOT=%%~fI"
)

where python >nul 2>nul
if errorlevel 1 (
    py -3 getdata.py %*
) else (
    python getdata.py %*
)

if errorlevel 1 (
    echo.
    echo getdata 执行失败，请查看上面的输出。
    pause
)
endlocal
