@echo off
chcp 65001 > nul
echo ============================================================
echo   sena3fx Live Trading Bot - 初回セットアップ
echo ============================================================
echo.

REM === Python 確認 ===
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python が見つかりません。
    echo.
    echo Python 3.10 以上をインストールしてください:
    echo   https://www.python.org/downloads/
    echo.
    echo インストール時に "Add Python to PATH" にチェックを入れてください。
    pause
    exit /b 1
)
echo [OK] Python が見つかりました:
python --version

REM === Python バージョン確認 (3.8以上) ===
python -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)" > nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.8 以上が必要です。
    pause
    exit /b 1
)

echo.
echo --- 仮想環境セットアップ ---

REM === 仮想環境作成 ===
if not exist "venv" (
    echo 仮想環境を作成中...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] 仮想環境の作成に失敗しました。
        pause
        exit /b 1
    )
    echo [OK] 仮想環境を作成しました: venv\
) else (
    echo [OK] 仮想環境は既に存在します: venv\
)

REM === 仮想環境アクティベート ===
call venv\Scripts\activate.bat

REM === pip アップグレード ===
echo.
echo pip をアップグレード中...
python -m pip install --upgrade pip --quiet

REM === 依存パッケージインストール ===
echo.
echo --- パッケージインストール ---
pip install -r requirements_live.txt
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] パッケージのインストールに失敗しました。
    echo インターネット接続を確認してください。
    pause
    exit /b 1
)
echo [OK] 全パッケージのインストール完了

REM === config.yaml コピー ===
echo.
echo --- 設定ファイル ---
if not exist "config.yaml" (
    echo config.example.yaml を config.yaml にコピー中...
    copy config.example.yaml config.yaml > nul
    echo [OK] config.yaml を作成しました
    echo.
    echo ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    echo   重要: 次のステップとして config.yaml を編集してください！
    echo   - account_id: OANDA口座ID
    echo   - api_key:    OANDAのAPIキー
    echo   - environment: "practice" (デモ) or "live" (本番)
    echo ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
) else (
    echo [OK] config.yaml は既に存在します (スキップ)
)

REM === ディレクトリ作成 ===
if not exist "logs"        mkdir logs
if not exist "data\ohlc"   mkdir data\ohlc

echo.
echo ============================================================
echo   セットアップ完了！
echo.
echo   次のステップ:
echo     1. config.yaml をテキストエディタで開いて編集
echo        - account_id と api_key を入力
echo        - environment: "practice" でデモテスト推奨
echo.
echo     2. start_bot.bat でボット起動
echo ============================================================
echo.
pause
