@echo off
chcp 65001 > nul
echo ============================================================
echo   sena3fx Live Trading Bot - 起動
echo ============================================================
echo.

REM === 設定ファイル確認 ===
if not exist "config.yaml" (
    echo [ERROR] config.yaml が見つかりません。
    echo.
    echo まず setup.bat を実行してセットアップを完了してください。
    pause
    exit /b 1
)

REM === APIキー未設定チェック ===
findstr /C:"ENTER_YOUR_API_KEY" config.yaml > nul 2>&1
if %errorlevel% equ 0 (
    echo [ERROR] config.yaml のAPIキーが未設定です。
    echo.
    echo config.yaml を編集して api_key と account_id を入力してください。
    pause
    exit /b 1
)

REM === 仮想環境確認 ===
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] 仮想環境が見つかりません。
    echo setup.bat を先に実行してください。
    pause
    exit /b 1
)

REM === 既存ボットの確認 ===
if exist "bot.pid" (
    set /p EXISTING_PID=<bot.pid
    tasklist /FI "PID eq %EXISTING_PID%" 2>nul | find "%EXISTING_PID%" > nul
    if %errorlevel% equ 0 (
        echo [WARNING] ボットはすでに実行中です (PID: %EXISTING_PID%)
        echo 停止する場合は stop_bot.bat を実行してください。
        pause
        exit /b 1
    ) else (
        REM PIDファイルが残っているだけで実際には停止済み
        del bot.pid
    )
)

REM === 仮想環境アクティベート ===
call venv\Scripts\activate.bat

echo ボットを起動しています...
echo.
echo  ┌─────────────────────────────────────────────────────┐
echo  │  ログはリアルタイムでこの画面と logs\ フォルダに    │
echo  │  保存されます。                                      │
echo  │                                                      │
echo  │  停止方法:                                           │
echo  │    1. このウィンドウで Ctrl+C を押す                 │
echo  │    2. または stop_bot.bat を実行                     │
echo  └─────────────────────────────────────────────────────┘
echo.

REM === ボット起動 ===
python live\bot.py

echo.
echo ボットが終了しました。
echo.
pause
