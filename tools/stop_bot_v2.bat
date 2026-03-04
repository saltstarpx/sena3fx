@echo off
chcp 65001 > nul
echo ============================================================
echo   sena3fx Bot v2.0 - 停止
echo ============================================================
echo.

if not exist "bot_v2.pid" (
    echo ボット v2 は実行されていません (bot_v2.pid が見つかりません)
    pause
    exit /b 0
)

set /p BOT_PID=<bot_v2.pid
echo ボット v2 プロセス (PID: %BOT_PID%) を停止中...

tasklist /FI "PID eq %BOT_PID%" 2>nul | find "%BOT_PID%" > nul
if %errorlevel% neq 0 (
    echo ボットはすでに停止しています。
    del bot_v2.pid
    pause
    exit /b 0
)

taskkill /PID %BOT_PID% > nul 2>&1
timeout /t 3 /nobreak > nul

tasklist /FI "PID eq %BOT_PID%" 2>nul | find "%BOT_PID%" > nul
if %errorlevel% equ 0 (
    echo 強制停止中...
    taskkill /F /PID %BOT_PID% > nul 2>&1
)

if exist "bot_v2.pid" del bot_v2.pid

echo [OK] ボット v2 を停止しました。
echo.
echo ※ オープン中のポジションは自動でクローズされません。
echo   OANDAダッシュボードでポジションを確認してください。
echo.
pause
