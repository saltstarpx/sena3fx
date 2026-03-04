@echo off
chcp 65001 > nul
echo ============================================================
echo   sena3fx Live Trading Bot - 停止
echo ============================================================
echo.

if not exist "bot.pid" (
    echo ボットは実行されていません (bot.pid が見つかりません)
    echo.
    pause
    exit /b 0
)

REM PIDを読み込む
set /p BOT_PID=<bot.pid

echo ボットプロセス (PID: %BOT_PID%) を停止中...

REM プロセスが存在するか確認
tasklist /FI "PID eq %BOT_PID%" 2>nul | find "%BOT_PID%" > nul
if %errorlevel% neq 0 (
    echo ボットはすでに停止しています。
    del bot.pid
    pause
    exit /b 0
)

REM グレースフルに停止 (SIGTERM相当)
taskkill /PID %BOT_PID% > nul 2>&1

REM 3秒待って確認
timeout /t 3 /nobreak > nul

tasklist /FI "PID eq %BOT_PID%" 2>nul | find "%BOT_PID%" > nul
if %errorlevel% equ 0 (
    REM まだ動いていたら強制終了
    echo 強制停止中...
    taskkill /F /PID %BOT_PID% > nul 2>&1
)

REM PIDファイル削除
if exist "bot.pid" del bot.pid

echo [OK] ボットを停止しました。
echo.
echo ※ 現在オープン中のポジションは自動ではクローズされません。
echo   OANDAダッシュボードでポジションを確認してください。
echo.
pause
