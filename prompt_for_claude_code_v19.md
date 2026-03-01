## Claude Code指示書v19：本番稼働準備 - OANDA Practice環境での実発注テスト

### 背景

v18で最終戦略 `XAUUSD+Kelly(f=0.25)+ADX(>25)` が確立され、バックテスト上の全作業が完了しました。いよいよ次のフェーズ、**本番環境での自動売買**に移行します。

v19では、その最終ステップとして、**OANDAのPractice（デモ）環境**に接続し、実際にシグナル通りに発注・決済・ポジション管理を行う機能を実装します。

### Task 1: OANDA API 発注クライアントの実装

**目的:** `oandapyV20` を使用して、実際の発注・決済を行うためのクライアントを実装する。

1.  `lib/` に `oanda_client.py` を新設します。
2.  このクライアントは、以下の機能を持つクラス（例: `OandaClient`）を実装します。
    - 環境変数（`OANDA_API_KEY`, `OANDA_ACCOUNT_ID`, `OANDA_ENVIRONMENT`）を読み込んでAPIに接続する。
    - `create_order(instrument, units, stop_loss_price)`: 新規注文（成行）を発注する。`units` はKelly計算後のロット数。
    - `close_trade(trade_id)`: 指定したトレードIDのポジションを決済する。
    - `get_open_trades(instrument)`: 指定した銘柄のオープンなポジション一覧を取得する。
3.  APIキーのハードコーディングは絶対に避け、必ず環境変数から読み込むようにしてください。

### Task 2: 監視スクリプトへの発注機能の統合

**目的:** `forward_main_strategy.py` がシグナルを検出した際に、`OandaClient` を使って実発注を行うようにする。

1.  `monitors/forward_main_strategy.py` を修正します。
2.  スクリプトに `--live` と `--dry-run` のコマンドライン引数を追加します。
    - `--dry-run`（デフォルト）: 従来通りシグナルをログファイルに記録するだけ。
    - `--live`: 実際に `OandaClient` を呼び出して発注・決済を行う。
3.  シグナル検出時のロジックを以下のように変更します。
    - **エントリーシグナル:**
        1.  `OandaClient.get_open_trades('XAUUSD')` で既にポジションがないことを確認する。
        2.  `main_strategy.py` のKelly計算結果から `units` を決定する。
        3.  `OandaClient.create_order()` で発注する。
    - **決済シグナル:**
        1.  `OandaClient.get_open_trades('XAUUSD')` で決済対象のポジションIDを取得する。
        2.  `OandaClient.close_trade()` で決済する。

### Task 3: systemdによるデーモン化

**目的:** 完成した監視・発注スクリプトをサーバー上で永続的に実行させるための準備を行う。

1.  `deploy/` ディレクトリに `sena3fx.service` というsystemdのサービスファイルを作成します。
2.  このサービスファイルは、`monitors/forward_main_strategy.py` を `--live` モードで実行するように設定します。
    - 例: `ExecStart=/usr/bin/python3 /home/ubuntu/sena3fx/monitors/forward_main_strategy.py --live`
3.  環境変数を読み込めるように、`EnvironmentFile` ディレクティブで設定ファイルを指定できるようにしてください。

### 完了報告

全タスク完了後、変更した全ファイルを `claude/add-trading-backtest-ePJat` ブランチにプッシュし、完了報告を行ってください。特に、**Practice環境で以下の動作確認ができたこと**を報告してください。

- `--dry-run` モードでシグナルがログに記録されること。
- `--live` モードで実際にOANDA Practice口座にポジションが建ち、決済されること。
- `systemctl start sena3fx.service` でサービスが正常に起動すること。
