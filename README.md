# sena3fx — FX自動売買システム v76

USDJPY / EURJPY / GBPJPYを対象とした、**二番底・二番天井パターン**に基づくFX自動売買システム。v76戦略ロジックをGoogle Cloud Run上でペーパートレーディング稼働中。

---

## 現在の稼働状況

| 項目 | 内容 |
|------|------|
| **稼働環境** | Google Cloud Run（asia-northeast1） |
| **稼働モード** | ペーパートレーディング（実弾未使用） |
| **取引ペア** | USDJPY / EURJPY / GBPJPY |
| **実行間隔** | 毎分（Cloud Scheduler） |
| **定期レポート** | 4時間ごと（JST 9/13/17/21/1/5時）Discord通知 |
| **データ取得** | OANDA v20 API（1分・15分・1時間・4時間足） |
| **状態管理** | Google Cloud Storage（positions.json, trade_history.json） |
| **コスト** | 無料枠内（月額$0想定） |

---

## 戦略概要 — v76（完成版）

### コンセプト

やがみメソッドに基づく**二番底・二番天井パターン**の自動検出。マルチタイムフレーム（4時間足・1時間足・15分足・1分足）を組み合わせ、高確率なエントリーポイントを機械的に判定する。

### v76の主要改善点（v75からの変更）

v75では、スプレッドをエントリー価格に加算した後にリスク幅を計算していたため、スプレッドの大小が損益にほとんど影響しないバグが存在した。v76ではSL/TPをチャートレベル（始値基準）で固定し、スプレッドをエントリーコストとして正しく損益に反映する実装に修正した。

```
■ v76の損益計算（実環境に近い正しい実装）
  - SL/TP: チャートレベル（始値基準）で固定
  - 実際のエントリー価格: ep = 始値 + spread
  - ロング SL到達時: -(risk + spread) × 100  ← スプレッドが損失に加算
  - ロング TP到達時: (risk × RR - spread) × 100  ← スプレッドが利益を圧縮
```

### アウト・オブ・サンプル（OOS）検証結果

| 指標 | インサンプル（IS） | アウト・オブ・サンプル（OOS） |
|------|-------------------|-------------------------------|
| 期間 | 2024/7〜2025/2 | 2025/3〜2026/2 |
| プロフィットファクター | 4.68 | 3.89 |
| 統計的有意性（p値） | — | p = 0.12 |

OOS期間でもPF 3.89を維持しており、過学習リスクは低い。

---

## ディレクトリ構成

```
sena3fx/
├── README.md                       # このファイル
├── CLAUDE.md                       # AI開発ガイド（Claude Code用）
├── MANUS_CHANGELOG.md              # AI変更履歴
├── requirements.txt                # 共通依存パッケージ
├── requirements_live.txt           # ライブ環境用依存パッケージ
│
├── cloud_run/                      # ★ Cloud Run 本番コード
│   ├── main.py                     #   FastAPI HTTPエンドポイント
│   ├── Dockerfile                  #   コンテナ定義
│   ├── requirements.txt            #   Cloud Run用依存パッケージ
│   └── strategies/
│       └── yagami_mtf_v76.py       #   本番稼働中の戦略ロジック（v76）
│
├── paper_trading/                  # ペーパートレーディングBot
│   ├── paper_trader.py             #   Bot本体（GCS連携・OANDA API）
│   └── logs/                       #   ローカルログ（paper_trades.csv等）
│
├── strategies/                     # 戦略ロジック
│   ├── current/
│   │   └── yagami_mtf_v76.py       #   ★ 完成版（現行戦略）
│   └── archive/                    #   v1〜v75 全過去版（次バージョン開発時の学習データ）
│
├── data/                           # ローソク足データ（品質チェック済み）
│   ├── usdjpy_is_1m.csv            #   USDJPY 1分足 IS（2024/7〜2025/2）
│   ├── usdjpy_is_15m.csv           #   USDJPY 15分足 IS
│   ├── usdjpy_is_1h.csv            #   USDJPY 1時間足 IS
│   ├── usdjpy_is_4h.csv            #   USDJPY 4時間足 IS（重複修正済み）
│   ├── usdjpy_oos_1m.csv           #   USDJPY 1分足 OOS（2025/3〜2026/2）
│   ├── usdjpy_oos_15m.csv          #   USDJPY 15分足 OOS
│   ├── usdjpy_oos_1h.csv           #   USDJPY 1時間足 OOS
│   ├── usdjpy_oos_4h.csv           #   USDJPY 4時間足 OOS（重複修正済み）
│   ├── eurjpy_1m.csv               #   EURJPY 1分足（2025/1〜2026/2）
│   ├── eurjpy_15m.csv              #   EURJPY 15分足
│   ├── eurjpy_4h.csv               #   EURJPY 4時間足（重複修正済み）
│   ├── gbpjpy_1m.csv               #   GBPJPY 1分足（2025/1〜2026/2）
│   ├── gbpjpy_15m.csv              #   GBPJPY 15分足
│   ├── gbpjpy_4h.csv               #   GBPJPY 4時間足（重複修正済み）
│   ├── ohlc/                       #   その他通貨ペア・旧データ
│   ├── oanda_ticks/                #   USDJPY ティックデータ（週次）
│   ├── oanda_ticks_eurjpy/         #   EURJPY ティックデータ
│   ├── oanda_ticks_gbpjpy/         #   GBPJPY ティックデータ
│   └── archive/                    #   旧版・異常ファイル（参照用保管）
│
├── scripts/                        # バックテスト・分析スクリプト
│   ├── check_ohlc_quality.py       #   ★ データ品質チェックツール
│   ├── fix_ohlc_data.py            #   データ修正スクリプト（2026/3実施分）
│   ├── run_backtest_v76_*.py       #   v76バックテスト
│   ├── sl_sensitivity_v76.py       #   SL感度分析
│   ├── sensitivity_tolerance.py    #   トレランス感度分析
│   └── backtest_full_oanda.py      #   OANDA全期間バックテスト
│
├── docs/                           # ドキュメント
│   ├── strategy_development_log_v60_v76.md  # ★ v60〜v76 開発議事録
│   ├── sena3fx_v75_trading_strategy.md      # 戦略詳細仕様書
│   ├── report_to_claude_v76.md              # v76 OOS検証レポート
│   ├── learning_materials/         #   やがみメソッドPDF教材（7冊）
│   ├── prompts/                    #   Claude Code向けプロンプト履歴
│   ├── config_examples/            #   設定ファイルサンプル
│   └── slides_*/                   #   プレゼンテーション資料
│
├── results/                        # バックテスト結果（CSV・PNG）
├── trade_logs/                     # トレード記録・シミュレーション結果
├── lib/                            # コアエンジン（バックテスト・指標等）
├── tests/                          # テストコード
├── tools/                          # Windows用起動スクリプト
└── archive/                        # 旧システム（v1/v2）
```

---

## データ仕様

### 通貨ペア別データ

| ファイル | 期間 | 足数 | 用途 |
|---------|------|------|------|
| `usdjpy_is_1m.csv` | 2024/7〜2025/2 | 248,731 | USDJPY インサンプル |
| `usdjpy_oos_1m.csv` | 2025/3〜2026/2 | 371,480 | USDJPY アウト・オブ・サンプル |
| `usdjpy_is_4h.csv` | 2024/7〜2025/2 | 1,042 | USDJPY IS 4時間足 |
| `usdjpy_oos_4h.csv` | 2025/3〜2026/2 | 1,553 | USDJPY OOS 4時間足 |
| `eurjpy_1m.csv` | 2025/1〜2026/2 | 431,796 | EURJPY |
| `gbpjpy_1m.csv` | 2025/1〜2026/2 | 431,898 | GBPJPY |

### データ品質チェック（2026年3月実施）

新規データ取得後は必ず以下のスクリプトで品質チェックを実施すること。

```bash
python scripts/check_ohlc_quality.py
```

**過去に発見・修正した問題:**

| 問題 | 対象ファイル | 原因 | 対処 |
|------|------------|------|------|
| カラム27個の異常ファイル | `USDJPY_15m.csv`, `USDJPY_4h.csv` | データ結合バグ | `data/archive/`へ移動 |
| 月末タイムスタンプ重複 | `*_4h.csv` 全4ファイル | OANDA APIが月末最終足と翌月最初足を二重返却 | 重複行を削除 |

> **注意:** OANDAのAPIは月末16:00 UTCのタイムスタンプを重複して返すことがある。4時間足データを新規取得した際は必ず重複チェックを行うこと。

---

## Cloud Run エンドポイント

| エンドポイント | メソッド | 説明 |
|--------------|---------|------|
| `/run` | POST | 取引ロジックを1回実行（毎分Cloud Schedulerが呼び出す） |
| `/report` | POST | 定期レポートをDiscordに送信（4時間ごと） |
| `/health` | GET | ヘルスチェック |
| `/status` | GET | 現在のポジション・損益状況を返す |
| `/notify_test` | POST | Discord通知テスト |

---

## セットアップ

### 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 環境変数（`.env`ファイルまたはCloud Runシークレット）

```
OANDA_ACCOUNT_ID=xxx-xxx-xxxxxxxx-xxx
OANDA_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
GCS_BUCKET_NAME=sena3fx-paper-trading
```

### バックテスト実行

```bash
# v76 OOS検証
python scripts/run_backtest_v76_wall.py

# データ品質チェック
python scripts/check_ohlc_quality.py
```

---

## 開発履歴

戦略の開発経緯・各バージョンの変更点・学習事項は以下を参照。

- **[docs/strategy_development_log_v60_v76.md](docs/strategy_development_log_v60_v76.md)** — v60〜v76の開発議事録（決定根拠・失敗事例・学習事項を含む）
- **[docs/sena3fx_v75_trading_strategy.md](docs/sena3fx_v75_trading_strategy.md)** — 戦略詳細仕様書
- **[strategies/archive/](strategies/archive/)** — v1〜v75の全過去版（次バージョン開発時の学習データ）

---

## 注意事項

- 本システムは現在**ペーパートレーディングモード**で稼働中。実弾投入前に十分な検証期間（目安: 3ヶ月以上・150トレード以上）を設ける。
- `gcp-key.json`はGitで管理されていない（`.gitignore`で除外）。Cloud Runのシークレットマネージャーで管理すること。
- 1分足データ（`*_1m.csv`）は容量が大きいためGitで管理されていない。必要に応じてOANDA APIから再取得すること。
