## Claude Code指示書v19：OANDA Practice接続・動作検証（USDJPY）

### 背景

XAUUSD用のAPIトークン取得に時間がかかるため、先にFX口座（`demobare`）用のAPIトークンを使って、OANDA APIへの接続と実発注の基本動作を検証します。対象銘柄は**USD_JPY**です。

### 重要：環境変数の設定

プロジェクトのルートディレクトリにある`.env`ファイルに必要な情報を設定済みです。このファイルは`.gitignore`で管理されており、GitHubにはプッシュされません。

```
OANDA_API_KEY="b3c7db048d5b6d1ac77e4263bd8bfb8b-1222fbcaf7d9ffe642692a226f7e7467"
OANDA_ACCOUNT_ID="101-009-38652105-001"
OANDA_ENVIRONMENT="practice"
```

### Task 1: OANDA API 発注クライアントの実装

**目的:** `oandapyV20` を使用して、実際の発注・決済を行うためのクライアントを実装する。

1.  `lib/` に `oanda_client.py` を新設します。
2.  このクライアントは、`.env`ファイルから環境変数を読み込んでAPIに接続するクラス（例: `OandaClient`）を実装します。`python-dotenv`ライブラリの使用を推奨します。
3.  以下の機能を持つメソッドを実装してください。
    - `create_order(instrument, units)`: 新規注文（成行）を発注する。
    - `close_trade(trade_id)`: 指定したトレードIDのポジションを決済する。
    - `get_open_trades(instrument)`: 指定した銘柄のオープンなポジション一覧を取得する。

### Task 2: 接続・発注テストスクリプトの作成

**目的:** `OandaClient` を使って、API接続から発注・決済までの一連の流れを検証する最小限のスクリプトを作成する。

1.  `scripts/` に `test_oanda_connection.py` を新設します。
2.  このスクリプトは、実行すると以下の処理を順に行います。
    1.  `OandaClient`をインポートし、インスタンスを作成する。
    2.  `create_order("USD_JPY", 100)` を実行して、USDJPYを100通貨（最小ロット）で成行買い注文を出す。
    3.  発注に成功したら、返ってきたトレードIDをコンソールに表示する。
    4.  5秒待機する。
    5.  取得したトレードIDを使って `close_trade()` を実行し、ポジションを決済する。
    6.  決済に成功したら、「OANDA API connection and order test successful.」と表示して終了する。

### 完了報告

全タスク完了後、変更した全ファイルを `claude/add-trading-backtest-ePJat` ブランチにプッシュし、完了報告を行ってください。特に、`python3 scripts/test_oanda_connection.py` を実行した結果、**OANDA Practice口座で実際にUSDJPYの取引が実行され、成功メッセージが表示されたこと**を報告してください。
