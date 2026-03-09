# sena3fx — YAGAMI改 MTF自動取引エージェント

やがみメソッドをベースに定量・計量分析で再設計した **YAGAMI改** 戦略の自動化システム。
4時間足EMA20によるトレンドフィルター + 1時間足の二番底・二番天井パターン検出。
Google Cloud Run上でペーパートレードを24時間稼働。

---

## 現在の稼働状況

| 項目 | 内容 |
|------|------|
| **稼働環境** | Google Cloud Run（asia-northeast1） |
| **稼働モード** | ペーパートレーディング（実弾未使用） |
| **戦略名** | **YAGAMI改**（FX: v79BC / METALS: v79A / USDJPY: v77） |
| **取引ペア** | USDJPY / XAUUSD（採用済） / GBPUSD（候補） |
| **実行間隔** | 毎分（Cloud Scheduler） |
| **定期レポート** | 4時間ごと（JST 9/13/17/21/1/5時）Discord通知 |
| **データ取得** | OANDA v20 API（1分・15分・1時間・4時間足） |
| **状態管理** | Google Cloud Storage（positions.json, trade_history.json） |

---

## 戦略概要 — YAGAMI改

### コンセプト

やがみメソッドに基づく**二番底・二番天井パターン**の自動検出。
マルチタイムフレーム（4時間足・1時間足・1分足）を組み合わせ、高確率なエントリーポイントを機械的に判定する。
v77の KMID + KLOW フィルターを核に、カテゴリ別定量分析で過学習なく改善。

### カテゴリ別フィルター

| カテゴリ | 銘柄 | バリアント | フィルター | OOS PF |
|------|------|:---:|------|:---:|
| **FX** | EURUSD / GBPUSD / AUDUSD | v79BC | ADX≥20 + 直近4H足Streak≥4 | avg **1.98** |
| **METALS** | XAUUSD | v79A | 日足EMA20方向一致 | **2.16** |
| **USDJPY** | USDJPY | v77 | KMID + KLOW（cloud_run保持） | **4.96** |

### エントリー条件（全条件AND）

| 条件 | 内容 |
|------|------|
| トレンド一致 | 4H EMA20より上→ロングのみ、下→ショートのみ |
| パターン成立 | 二番底/二番天井の安値/高値が ATR×0.3 以内 |
| 確認足 | パターン形成後の足が方向一致の実体 |
| **KMID** | 直前4H足の実体方向がエントリー方向と一致 |
| **KLOW** | 直前4H足の下ヒゲ比率 < 0.15%（`< 0.0015`） |
| リスク幅 | SLまでの距離が ATR×2（1H基準）以内 |

### SL/TP設定

- **SL**: 二番底/天井の安値・高値から ATR×0.15 外側
- **TP**: SLリスク幅 × 2.5倍（RR=2.5）
- **半利確**: 1R到達でポジション半分決済 → SLをBEへ移動

---

## バックテスト結果（OOS: 2025/03〜2026/02）

| 銘柄 | 戦略 | PF | 勝率 | MDD | Kelly | プラス月 |
|------|------|:---:|:---:|:---:|:---:|:---:|
| **USDJPY** | v77 | **4.96** | 76.1% | 222.6pips | 0.608 | 12/12 |
| **XAUUSD** | v79A | **2.16** | — | — | — | — |
| **GBPUSD** | v79BC | **2.17** | — | — | — | — |
| EURUSD | v79BC | 1.87 | — | — | — | — |
| AUDUSD | v79BC | 1.90 | — | — | — | — |

---

## ディレクトリ構成

```
sena3fx/
├── README.md
├── CLAUDE.md                        # AI開発ガイド（Claude Code用）
│
├── strategies/
│   ├── current/
│   │   ├── yagami_mtf_v79.py        # ★ YAGAMI改 本体（FX/METALS カテゴリ別フィルター）
│   │   └── yagami_mtf_v78.py        # 参照用（過学習チェック済）
│   └── archive/                     # v1〜v77（履歴）
│
├── cloud_run/                       # Cloud Run 本番コード
│   ├── main.py                      # FastAPI エンドポイント
│   ├── Dockerfile
│   ├── requirements.txt
│   └── strategies/
│       ├── yagami_mtf_v79.py        # 本番用（FX/METALS）
│       ├── yagami_mtf_v78.py        # 参照用
│       ├── yagami_mtf_v77.py        # USDJPY本番用
│       └── archive/
│
├── data/                            # バックテスト用CSVデータ
│   ├── {symbol}_is_*.csv            # IS期間（2024/7〜2025/2）
│   ├── {symbol}_oos_*.csv           # OOS期間（2025/3〜2026/2）
│   └── ohlc/                        # 全期間1ファイル（大文字銘柄名）
│
├── scripts/                         # アクティブスクリプト（データ取得・ユーティリティ）
│   ├── fetch_*.py                   # データ取得（OANDA API）
│   ├── generate_htf_from_1m.py      # 1mから上位足生成
│   ├── main_loop.py
│   └── archive/                     # 旧バックテスト・分析スクリプト全件
│
├── utils/
│   ├── risk_manager.py              # AdaptiveRiskManager / SYMBOL_CONFIG
│   └── position_manager.py
│
├── trade_logs/                      # ペーパートレード実績ログ
├── results/                         # バックテスト結果（PNG・CSV）
├── docs/
│   ├── strategy_development_log_v60_v76.md
│   └── learning_materials/          # やがみ氏PDF教材
└── tests/
```

---

## Cloud Run エンドポイント

| エンドポイント | メソッド | 説明 |
|---|---|---|
| `/run` | POST | 1サイクル実行（シグナル判定・注文） |
| `/report` | GET | 定時レポートをDiscordへ送信 |
| `/health` | GET | ヘルスチェック |
| `/status` | GET | 現在のオープンポジション一覧 |
| `/notify_test` | GET | Discord通知テスト |

---

## セットアップ

```bash
pip install -r requirements.txt
```

### 環境変数

```
OANDA_ACCOUNT_ID=xxx-xxx-xxxxxxxx-xxx
OANDA_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
GCS_BUCKET_NAME=sena3fx-paper-trading
```

---

## 注意事項

- 本システムは現在**ペーパートレーディングモード**で稼働中
- `gcp-key.json` は `.gitignore` 対象。コミット禁止
- USDJPYに1分足データは存在しない（15m/1h/4hのみ）
- 旧バックテストスクリプトは `scripts/archive/`、旧戦略は `strategies/archive/` を参照
- `strategies/current/` と `cloud_run/strategies/` は常に同一内容を保つこと
