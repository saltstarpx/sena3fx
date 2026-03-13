# sena3fx — YAGAMI改 MTF自動取引エージェント

やがみメソッドをベースに定量・計量分析で再設計した **YAGAMI改** 戦略の自動化システム。
4時間足EMA20によるトレンドフィルター + 1時間足の二番底・二番天井パターン検出。
Google Cloud Run + Exness MT5（MetaApi経由）で本番取引を24時間稼働。

---

## 現在の稼働状況

| 項目 | 内容 |
|------|------|
| **稼働環境** | Google Cloud Run（asia-northeast1、min-instances=1） |
| **稼働モード** | **本番取引（Exness MT5 / MetaApi経由）** |
| **戦略名** | **YAGAMI改**（Logic-C / Logic-A / Logic-B 銘柄別） |
| **取引ペア** | 全7銘柄（下記ポートフォリオ参照） |
| **リスク** | Phase2: 1.0% / 銘柄 |
| **実行間隔** | 毎分（Cloud Scheduler） |
| **データ取得** | MetaApi REST API（1分・15分・4時間足） |
| **状態管理** | Google Cloud Storage（positions.json, trade_history.json） |

---

## 採用7銘柄 × ポートフォリオ統合結果

| 優先 | 銘柄 | ロジック | OOS PF | Sharpe | 運用フェーズ |
|:---:|------|:---:|:---:|:---:|---|
| 1 | **USDJPY** | Logic-C（オーパーツ） | 2.15 | 6.18 | Phase1（即導入） |
| 2 | **GBPUSD** | Logic-A（GOLD） | 1.86 | 7.12 | Phase1（即導入） |
| 3 | **EURUSD** | Logic-C（オーパーツ） | 1.81 | 6.18 | Phase1（即導入） |
| 4 | **USDCAD** | Logic-A（GOLD） | 2.02 | 5.62 | Phase1（即導入） |
| 5 | **NZDUSD** | Logic-A（GOLD） | 1.98 | 5.45 | Phase2（3ヶ月後） |
| 6 | **XAUUSD** | Logic-A（GOLD） | 3.10 | 3.42 | Phase2（別カテゴリ） |
| 7 | **AUDUSD** | Logic-B（ADX+Streak） | 2.03 | 3.66 | Phase2（月次安定確認後） |

### ポートフォリオ合成指標（OOS: 2025-06〜2026-02）

| Phase | リスク | 銘柄数 | トレード数 | Sharpe | MDD | 月次+ |
|---|---|:---:|:---:|:---:|:---:|:---:|
| Phase1 | 0.5% × 4 | 4 | 2,537 | 10.46 | 7.0% | 9/9 |
| **Phase2** | **1.0% × 7** | **7** | **3,634** | **6.95** | **8.2%** | **9/9** |
| Phase3 | 2.0% × 7 | 7 | 3,634 | 7.32 | 13.0% | 9/9 |

---

## 戦略概要 — YAGAMI改

### コンセプト

やがみメソッドに基づく**二番底・二番天井パターン**の自動検出。
マルチタイムフレーム（4時間足・1時間足・1分足）を組み合わせ、高確率なエントリーポイントを機械的に判定する。
KMID + KLOW フィルターを核に、銘柄別ロジック（Logic-A/B/C）で最適化。

### 3ロジック体系

| ロジック | 名称 | フィルター | エントリー | 対象銘柄 |
|:---:|---|---|---|---|
| **Logic-C** | オーパーツYAGAMI | KMID + KLOW + 1H確認足方向 | E0（即時2分以内） | USDJPY, EURUSD |
| **Logic-A** | GOLDYAGAMI | KMID + KLOW + 日足EMA20方向一致 + EMA距離≥ATR×1.0 | E2（スパイク除外3分以内） | GBPUSD, USDCAD, NZDUSD, XAUUSD |
| **Logic-B** | ADX+Streak | KMID + KLOW + ADX≥20 + 直近4H Streak≥4 | E1（方向一致待ち5分以内） | AUDUSD |

### エントリー条件（全ロジック共通 AND）

| 条件 | 内容 |
|------|------|
| トレンド一致 | 4H EMA20より上→ロングのみ、下→ショートのみ |
| パターン成立 | 二番底/二番天井の安値/高値が ATR×0.3 以内 |
| **KMID** | 直前4H足の実体方向がエントリー方向と一致 |
| **KLOW** | 直前4H足の下ヒゲ比率 < 0.15%（`< 0.0015`） |
| **EMA距離** | 4H終値とEMA20の距離 ≥ ATR×1.0 |
| **パターン** | 1H足 二番底/二番天井（ATR×0.30以内） |
| **SL** | パターンの安値/高値 ± ATR×0.15 |
| **TP** | リスク幅 × 2.5倍（RR=2.5） |
| **半利確** | 1R到達でポジション50%決済 → SLをBE移動 |

---

## 3つのロジック

### Logic-A: GOLDYAGAMI（Goldロジック）

> **「引き算の美学 + 足し算の効果」**で生まれた現行主力ロジック。
> ADXやセッションフィルターを除去（引き算）し、日足EMA20アライメントとE2エントリーを追加（足し算）。
> 定量・計量分析で過学習なしを確認済み（IS-0.52 / OOS+0.13）。

| 項目 | 設定 |
|------|------|
| トレンドフィルター追加 | **日足EMA20方向一致**（クローズ > 日足EMA20） |
| エントリー方式 | **E2方式**: スパイク足除外（レンジ > ATR×2.0をスキップ）、足確定後2〜3分以内 |
| ADX | 不使用（過学習リスクで除去） |
| セッション | 不使用（過学習リスクで除去） |
| **採用銘柄** | **GBPUSD / USDCAD / NZDUSD / XAUUSD** |

**OOS成績（2025/03〜2026/02）**

| 銘柄 | PF | Sharpe | MDD | 月次+ |
|:---:|:---:|:---:|:---:|:---:|
| GBPUSD | 1.86 | 7.12 | 18.3% | 9/9 |
| USDCAD | 2.02 | 5.62 | 22.7% | 8/9 |
| NZDUSD | 1.98 | 5.45 | 20.5% | 8/9 |
| XAUUSD | 3.10 | 3.42 | 8.7% | 9/9 |

---

## リスクステージング

| Phase | リスク | 対象 | 昇格条件 |
|---|---|---|---|
| Phase1 | 0.5% × 4銘柄 | USDJPY/GBPUSD/EURUSD/USDCAD | — |
| **Phase2（現在）** | **1.0% × 7銘柄** | **全7銘柄** | **20トレード+PF≥1.30+MDD≤8%** |
| Phase3 | 2.0% × 7銘柄 | 全7銘柄 | 40トレード+PF≥1.35+MDD≤10% |

### ストップルール

- **日次**: -2R で新規エントリー凍結
- **週次**: -4R で週間凍結
- **月次DD**: -8% でPhase降格（Phase3→2→1→停止）

---

## ディレクトリ構成

```
sena3fx/
├── README.md
├── CLAUDE.md                        # AI開発ガイド（Claude Code用）
├── TRADING_MEMO.md                  # 本番運用ルール書
│
├── cloud_run/                       # ★ Cloud Run 本番コード
│   ├── main.py                      #   FastAPI エンドポイント（Phase2 全7銘柄）
│   ├── broker_metaapi.py            #   Exness MT5 ブローカー（MetaApi経由）
│   ├── broker_base.py               #   抽象ブローカーインターフェース
│   ├── Dockerfile
│   ├── requirements.txt
│   └── strategies/
│       ├── yagami_mtf_v79.py        #   Logic-A / Logic-B 本番用
│       ├── yagami_mtf_v77.py        #   Logic-C 本番用（USDJPY/EURUSD）
│       └── yagami_mtf_v78.py        #   参照用（過学習チェック済）
│
├── strategies/
│   └── current/
│       ├── yagami_mtf_v79.py        # cloud_run/strategies/ と同期必須
│       └── yagami_mtf_v78.py        # 参照用
│
├── utils/
│   └── risk_manager.py              # AdaptiveRiskManager / SYMBOL_CONFIG
│
├── scripts/                         # バックテスト・データ取得
│   ├── backtest_portfolio_integration.py  # ★ ポートフォリオ統合BT
│   ├── backtest_logic_comparison.py       # ★ Logic-A/B/C 比較BT
│   ├── backtest_all_symbols.py
│   ├── backtest_final_optimized.py
│   ├── fetch_*.py                   # データ取得（OANDA API、13本）
│   ├── generate_htf_from_1m.py
│   └── main_loop.py
│
├── deploy/
│   ├── deploy_gcp.sh               # GCPデプロイ（min-instances=1）
│   └── .env.example
│
├── data/                            # バックテスト用CSVデータ
│   ├── {symbol}_is_*.csv            # IS期間（2024/7〜2025/2）
│   ├── {symbol}_oos_*.csv           # OOS期間（2025/3〜2026/2）
│   └── ohlc/                        # 全期間1ファイル（大文字銘柄名）
│
├── results/                         # バックテスト結果（PNG・CSV）
├── trade_logs/                      # トレード実績ログ
├── reports/                         # PDCAレポート
├── docs/                            # ドキュメント・やがみ氏PDF教材
├── tests/                           # テスト
└── tools/                           # Windows用batファイル
```

※ 旧ファイル（v1〜v77戦略、旧スクリプト、lib/、live/等）は全て削除済み。git履歴から復元可能。

---

## Cloud Run エンドポイント

| エンドポイント | メソッド | 説明 |
|---|---|---|
| `/run` | POST | 1サイクル実行（シグナル判定・注文） |
| `/report` | POST | 手動レポート送信 |
| `/health` | GET | ヘルスチェック |
| `/status` | GET | 現在のオープンポジション一覧 |
| `/weekly_feedback` | POST | 週次フィードバック記録 |

---

## セットアップ

### 環境変数（deploy/.env）

```
BROKER=exness
METAAPI_TOKEN=<MetaApi APIトークン>
METAAPI_ACCOUNT_ID=<MetaApi アカウントID>
EQUITY_JPY=<口座残高(円)>
DISCORD_WEBHOOK=<Discord Webhook URL>
GCS_BUCKET=sena3fx-paper-trading
GCP_PROJECT=aiyagami
```

### デプロイ

```bash
cd ~/sena3fx
bash deploy/deploy_gcp.sh
```

---

## 注意事項

- `gcp-key.json` / `deploy/.env` は `.gitignore` 対象。コミット禁止
- `strategies/current/` と `cloud_run/strategies/` は常に同一内容を保つこと
- USDJPYに1分足データは存在しない（15m/1h/4hのみ）
- 詳細な運用ルールは `TRADING_MEMO.md` を参照
