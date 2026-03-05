@echo off
chcp 65001 > nul
echo ============================================================
echo   sena3fx Bot v2.0 (4H x 15m デュアルTF) - 起動
echo ============================================================
echo.

REM === 設定ファイル確認 ===
if not exist "config.yaml" (
    echo [ERROR] config.yaml が見つかりません。
    echo.
    echo config_v2.example.yaml を config.yaml にコピーしてください:
    echo   copy config_v2.example.yaml config.yaml
    echo.
    echo その後、config.yaml を編集して account_id と api_key を設定してください。
    pause
    exit /b 1
)

REM === APIキー未設定チェック ===
findstr /C:"ENTER_YOUR_API_KEY" config.yaml > nul 2>&1
if %errorlevel% equ 0 (
    echo [ERROR] config.yaml の APIキーが未設定です。
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

REM === 既存ボット確認 ===
if exist "bot_v2.pid" (
    set /p EXISTING_PID=<bot_v2.pid
    tasklist /FI "PID eq %EXISTING_PID%" 2>nul | find "%EXISTING_PID%" > nul
    if %errorlevel% equ 0 (
        echo [WARNING] Bot v2 はすでに実行中です (PID: %EXISTING_PID%)
        echo 停止する場合は stop_bot_v2.bat を実行してください。
        pause
        exit /b 1
    ) else (
        del bot_v2.pid
    )
)

REM === 仮想環境アクティベート ===
call venv\Scripts\activate.bat

echo Bot v2.0 を起動中...
echo.
echo  ┌─────────────────────────────────────────────────────┐
echo  │  戦略: 4H EMA21 トレンド + 15m DC ブレイクアウト    │
echo  │  SL:   4H スイング安値/高値 (やがみルール準拠)       │
echo  │  RR:   最小 2.0 (デフォルト 2.5)                    │
echo  │  禁止: 年末年始/祝日/土曜日 (CMEクローズ)            │
echo  │                                                      │
echo  │  ログ: logs\bot_v2_YYYYMMDD.log                     │
echo  │  停止: Ctrl+C または stop_bot_v2.bat                 │
echo  └─────────────────────────────────────────────────────┘
echo.

REM === ボット起動 ===
python live\bot_v2.py

echo.
echo Bot v2.0 が終了しました。
echo.
pause
